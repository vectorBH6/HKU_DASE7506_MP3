"""Summarize unchanged official result files; never modify predictions or scores."""
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

def time_field(path, field):
    if not path.exists():
        return 0.
    match = re.search(re.escape(field) + r":\s*([0-9.]+)", path.read_text())
    return float(match.group(1)) if match else 0.


def main():
    results = []
    for path in sorted((ROOT / "runs").glob("*/*/r*/run_metadata.json")):
        metadata = json.loads(path.read_text())
        if metadata["status"] != "completed":
            continue
        run = path.parent
        score = json.loads((run / "score.json").read_text())
        document = json.loads((run / "predictions.json").read_text())
        assert hashlib.sha256((run / "predictions.json").read_bytes()).hexdigest() == score["predictions_sha256"]
        records = [json.loads(s) for s in (run / "generations.jsonl").read_text().splitlines()] if (run / "generations.jsonl").exists() else []
        calls = {}
        for record in records:
            calls[record["task_id"]] = calls.get(record["task_id"], 0) + 1
        assert max(calls.values(), default=0) <= 6
        assert max((r["new_tokens"] for r in records), default=0) <= 2048
        assert max(p["tool_calls"] for p in document["predictions"]) <= 8
        public_matched, false_accept = 0, 0
        for prediction, item in zip(document["predictions"], score["per_question"]):
            assert prediction["id"] == item["id"]
            matched = any(e["tool"] == "check" and e["sql"] == prediction["sql"] and e["observation"].get("matches_example") for e in prediction["trace"])
            public_matched += bool(matched)
            false_accept += bool(matched and not item["passed"])
        result = {"experiment": metadata["experiment"], "split": metadata["split"], "run": run.name,
                  "correct": score["correct"], "n": score["n"], "accuracy": score["accuracy"],
                  "prediction_seconds": document["prediction_seconds"], "scoring_seconds": score["scoring_seconds"],
                  "mean_tool_calls": score["mean_tool_calls"], "model_calls": len(records),
                  "input_tokens": sum(r["input_tokens"] for r in records), "output_tokens": sum(r["new_tokens"] for r in records),
                  "max_calls_per_task": max(calls.values(), default=0), "max_new_tokens": max((r["new_tokens"] for r in records), default=0),
                  "cpu_user_seconds": time_field(run / "prediction_resources.txt", "User time (seconds)"),
                  "cpu_system_seconds": time_field(run / "prediction_resources.txt", "System time (seconds)"),
                  "prediction_max_rss_mib": time_field(run / "prediction_resources.txt", "Maximum resident set size (kbytes)") / 1024,
                  "process_tree_peak_rss_mib": metadata.get("peak_process_tree_rss_bytes", 0) / 1024 ** 2,
                  "checked_public_matches": public_matched, "public_match_but_final_failed": false_accept,
                  "device": metadata.get("device", "cpu"), "cpu_affinity": metadata["cpu_affinity"],
                  "source": str(run.relative_to(ROOT))}
        results.append(result)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    if results:
        with (out / "results.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    lines = ["# 自动汇总的官方实验成绩", "", "来源为每次实验保存的完整官方 score.json；所有准确率均要求三实例全部通过。", "",
             "| 实验 | 划分 | 答对/总数 | 准确率 | 预测秒 | 评分秒 | 平均工具次数 | 模型调用 | 生成 token | 峰值 RSS MiB |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| [{r['experiment']}/{r['run']}](../{r['source']}/score.json) | {r['split']} | {r['correct']}/{r['n']} | {r['accuracy']:.2f}% | {r['prediction_seconds']:.2f} | {r['scoring_seconds']:.2f} | {r['mean_tool_calls']:.3f} | {r['model_calls']} | {r['output_tokens']} | {r['prediction_max_rss_mib']:.1f} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("Summarized",len(results),"completed official runs")

if __name__ == "__main__":
    main()
