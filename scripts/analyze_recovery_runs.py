"""Compare v3 recovery experiments against the frozen v2 official runs."""
from collections import Counter,defaultdict
import json
from pathlib import Path
from analyze_review_runs import load
ROOT=Path(__file__).resolve().parents[1]
MODEL_REASONS={"model_hint_request","model_query","model_candidate_selection","model_retrieval_request"}
BASELINE={"validation":"e13_review_risk","test":"e15_review_selected"}
EXPERIMENTS=["e20_recovery4","e21_recovery6","e22_recovery_selected"]


def details(experiment,split):
    item=load(experiment,split)
    if not item:return None
    result,pred,score=item
    run=ROOT/result["source"]
    decisions=[json.loads(s) for s in (run/"decisions.jsonl").read_text().splitlines()]
    logs=[json.loads(s) for s in (run/"generations.jsonl").read_text().splitlines()]
    truth={p["id"]:p["passed"] for p in score["per_question"]}
    by_task=defaultdict(list)
    for row in decisions:by_task[row["task_id"]].append(row)
    dup={r["task_id"] for r in decisions if r["guard_reason"]=="duplicate_action"}
    fallback_reasons={"completion_cap_fallback","model_budget_fallback","recovery_limit_fallback"}
    early={r["task_id"] for r in decisions if r["guard_reason"] in fallback_reasons and r.get("accepted_action",{}).get("tool")=="submit" and r["attempt"]<result["config"]["max_completions"] and r["remaining"]>0}
    retry=[r for r in decisions if r["guard_reason"]=="recovery_retry"]
    recovered=sum(any(d["attempt"]==r["attempt"]+1 and d["guard_reason"] in MODEL_REASONS for d in by_task[r["task_id"]]) for r in retry)
    # Prefix matching would misclassify model_budget_fallback as a model action.
    result["model_selected_actions"]=dict(Counter(r["accepted_action"]["tool"] for r in decisions if r["guard_reason"] in MODEL_REASONS))
    result["model_accepted_proposals"]=sum(r["guard_reason"] in MODEL_REASONS for r in decisions)
    counts=Counter(r["task_id"] for r in logs)
    result.update({"duplicate_tasks":len(dup),"duplicate_task_failures":sum(not truth[k] for k in dup),
                   "duplicate_proposals":sum(r["guard_reason"]=="duplicate_action" for r in decisions),
                   "early_fallback_tasks":len(early),"early_fallback_failures":sum(not truth[k] for k in early),
                   "recovery_retries":len(retry),"accepted_recovery_retries":recovered,
                   "recovery_tasks":len({r["task_id"] for r in retry}),
                   "generation_cap_tasks":sum(n==result["config"]["max_completions"] for n in counts.values()),
                   "early_fallback_ids":sorted(early)})
    return result,pred,score


def main():
    results=[];pairs=[];baseline={}
    for split,exp in BASELINE.items():
        baseline[split]=details(exp,split)
        results.append(baseline[split][0])
    for exp in EXPERIMENTS:
        for split in ["validation","test"]:
            item=details(exp,split)
            if not item:continue
            result,pred,score=item;results.append(result)
            b,bp,bs=baseline[split]
            old={p["id"]:p["passed"] for p in bs["per_question"]};new={p["id"]:p["passed"] for p in score["per_question"]}
            assert set(old)==set(new)
            wins=[k for k in old if not old[k] and new[k]];losses=[k for k in old if old[k] and not new[k]]
            pairs.append({"experiment":exp,"split":split,"wins":wins,"losses":losses,"net_correct":len(wins)-len(losses),
                          "rescued_baseline_early_fallbacks":[k for k in b["early_fallback_ids"] if not old[k] and new[k]]})
    (ROOT/"reports/v3_results.json").write_text(json.dumps({"results":results,"paired_against_v2":pairs},ensure_ascii=False,indent=2)+"\n")
    lines=["# 第三轮恢复实验汇总","","全部成绩来自原样官方评分器，每题三个实例全部通过。当前只列出已完成运行。","",
           "| 实验 | 划分 | 正确 | 准确率 | 预测秒 | 模型调用 | 重复提议 | 恢复重试/接受 | 预算未尽回退提交/错 | 平均工具 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| [{r['experiment']}/{r['run']}](../{r['source']}/score.json) | {r['split']} | {r['correct']}/{r['n']} | {r['accuracy']:.2f}% | {r['prediction_seconds']:.2f} | {r['model_calls']} | {r['duplicate_proposals']} | {r['recovery_retries']}/{r['accepted_recovery_retries']} | {r['early_fallback_tasks']}/{r['early_fallback_failures']} | {r['mean_tool_calls']:.3f} |")
    (ROOT/"reports/v3_results.md").write_text("\n".join(lines)+"\n")
    print("Completed recovery comparisons:",len(results))


if __name__=="__main__":main()
