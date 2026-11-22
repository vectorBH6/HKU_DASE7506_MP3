"""Write the frozen v3 experiment comparison after the official full test."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    data=json.loads((ROOT/"reports/v3_results.json").read_text())
    selection=json.loads((ROOT/"reports/v3_selection.json").read_text())
    def get(exp,split):return next(r for r in data["results"] if r["experiment"]==exp and r["split"]==split)
    old=get("e15_review_selected","test");new=get("e22_recovery_selected","test")
    pair=next(p for p in data["paired_against_v2"] if p["experiment"]=="e22_recovery_selected" and p["split"]=="test")
    lines=["# SQL Repair Agent 第三轮：拒绝提议后的恢复实验","",
           f"新方案测试 **{new['correct']}/240（{new['accuracy']:.2f}%）**，v2 为 **{old['correct']}/240（{old['accuracy']:.2f}%）**，净变化 **{new['correct']-old['correct']:+d}题、{new['accuracy']-old['accuracy']:+.2f}个百分点**。", "",
           f"只依据验证集锁定 `{selection['new_experiment']}` 后才运行240题测试；是否替换默认：**{'是' if selection['promote'] else '否，保留v2'}**。选择记录见 [v3_selection.json](v3_selection.json)，本轮事先方案见 [v3_plan_cn.md](v3_plan_cn.md)。","",
           "## 改了什么","",
           "本轮没有训练新权重。沿用第二轮同一基座、同一单个未合并 LoRA，仅新增 recovery_v3 控制器。首次生成及没有拒绝的路径保持原 v2 提示。发生重复、无效或不可用提议后，Agent 内部记录最近两次拒绝、最近官方 SQL 错误和剩余生成预算，在同一观察下最多立即恢复一次。第二次生成输入已改变，不是重复相同输入。", "",
           "拒绝和恢复均计入模型生成上限，不会产生伪造的工具反馈；未执行的提议不进入官方 trace。模型仍通过原有六种官方动作与环境交互。原稿精确 check、风险复核、schema、提交保护、候选排序和最终有限回退沿用 v2。", "",
           "恢复次数用尽时仍可能预算未尽就回退，故本轮假设是改善提前结束，不是彻底消灭所有回退。模型接受新提议只表示通过协议校验，不代表 SQL 正确。", "",
           "## 验证集：受控比较","",
           "| 配置 | 正确/120 | 准确率 | 模型调用 | 重复提议 | 恢复/被接受 | 预测秒 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for exp in ["e13_review_risk","e20_recovery4","e21_recovery6"]:
        r=get(exp,"validation")
        lines.append(f"| {exp} | {r['correct']}/120 | {r['accuracy']:.2f}% | {r['model_calls']} | {r['duplicate_proposals']} | {r['recovery_retries']}/{r['accepted_recovery_retries']} | {r['prediction_seconds']:.2f} |")
    a=get("e20_recovery4","validation");b=get("e21_recovery6","validation")
    lines += ["",f"相同四次生成上限下，恢复机制相对 v2 变化 {a['correct']-87:+d}题；相同恢复机制下，六次上限相对四次变化 {b['correct']-a['correct']:+d}题。两组新实验 adapter、控制器代码和其余配置相同，仅 max_completions 不同。", "",
              f"六次组相对四次组多用 {b['model_calls']-a['model_calls']}次生成、重复提议增加 {b['duplicate_proposals']-a['duplicate_proposals']}次，正确题数没有增加。六次组 {b['recovery_retries']}次恢复仅 {b['accepted_recovery_retries']}次得到可接受提议，说明增加预算不能替代利用反馈改写SQL的训练。四次组‘预算未尽回退’降为0，也不能解读为错误已解决：有些路径只是恢复后耗尽生成额度，再由相同回退提交；是否救回必须看逐题全实例评分。", "",
              "按正确题数最多、并列模型调用更少选择新方案；只有验证严格超过87/120才替换默认。测试结果不用于重新选配置。单种子、单划分结果不作统计显著性或普遍泛化的声明。", "",
              "## 完整测试、成本与恢复行为","",
              "| 指标 | v2 | 本轮锁定方案 |","|---|---:|---:|"]
    for label,key in [("正确题数","correct"),("准确率%","accuracy"),("模型调用","model_calls"),("输入token","input_tokens"),("输出token","output_tokens"),("重复提议","duplicate_proposals"),("有重复提议的题数","duplicate_tasks"),("预算未尽回退提交","early_fallback_tasks"),("其中错误","early_fallback_failures"),("公开匹配但最终错误","public_match_final_failed"),("平均工具次数","mean_tool_calls"),("预测秒","prediction_seconds"),("评分秒","scoring_seconds"),("CPU秒","cpu_seconds"),("峰值RSS MiB","peak_rss_mib")]:
        values=[f"{r[key]:.2f}" if isinstance(r[key],float) else str(r[key]) for r in [old,new]]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines += ["",f"本轮共 {new['recovery_retries']} 次立即恢复，其中 {new['accepted_recovery_retries']} 次下一提议被接受；涉及 {new['recovery_tasks']} 题。模型实际动作：`{json.dumps(new['model_selected_actions'],ensure_ascii=False)}`。所有 model_budget_fallback 等回退均从模型自主动作统计中排除。", "",
              f"逐题比较：改善 {len(pair['wins'])}题，退步 {len(pair['losses'])}题，净变化 {pair['net_correct']:+d}。救回 v2 预算未尽回退失败题 {len(pair['rescued_baseline_early_fallbacks'])}题；这些集合有重叠，不能相加。", "",
              f"预算实测：最多 {new['max_model_calls']}次模型生成、单次最多 {new['max_new_tokens']}新token、最多 {new['max_tool_calls']}次收费工具。仍有 {new['early_fallback_failures']}道提前回退的失败、{new['public_match_final_failed']}道公开匹配但最终错误。", "",
              "两组新验证实验各固定四个P核并行，最终测试单独固定CPU0–3四核。v2历史验证含异构核/迁移，整机并发和CPU状态会影响耗时；成本表是本机测量，不作为严格独占环境的速度结论。硬件与第二轮相同，见 [v2_hardware.json](v2_hardware.json)。", ""]
    oldp={p["id"]:p for p in json.loads((ROOT/old["source"]/"predictions.json").read_text())["predictions"]}
    newp={p["id"]:p for p in json.loads((ROOT/new["source"]/"predictions.json").read_text())["predictions"]}
    tasks={p["id"]:p for p in map(json.loads,(ROOT/"mp3_sql/data/test.jsonl").read_text().splitlines())}
    for label,ids in [("改善案例",pair["rescued_baseline_early_fallbacks"] or pair["wins"]),("退步案例",pair["losses"])]:
        if ids:
            key=ids[0]
            lines += [f"**{label} `{key}`**：{tasks[key]['question']}","","v2 SQL：","```sql",oldp[key]["sql"],"```","本轮 SQL：","```sql",newp[key]["sql"],"```",""]
    lines += ["## 合规、复现与局限","",
              "官方框架、数据、observation与trace均未改；推理不读评分参考或额外数据库，数据库访问仅通过官方工具。本轮无训练；沿用固定 Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca 和 adapter SHA-256 43d91619f16b1a7e4cb3358ad93cfc5f8ee447ff124640a8b5e2ff8ee44de613。基座596,049,920参数，加LoRA共606,142,464参数，只有A/B可训练且本轮未更新。", "",
              "最多6次生成、每次384新token，所有恢复/规划均记录；CPU四线程、workers=1、离线safetensors、trust_remote_code=False，模型只加载一次。预测2小时、评分180秒、16GiB RSS由运行器监测。仍使用同一个 .venv，未创建 peer_reviews 或 dist，也未上传。", "",
              "19项单元检查通过，涵盖旧回归及恢复输入变化、重试边界、任务记忆隔离和官方观察不变。官方 scorer 对全量验证和测试重放轨迹。原样文件与模型哈希核验见 [v3_artifact_verification.json](v3_artifact_verification.json)，最终运行审计见 [v3_runtime_audit.json](v3_runtime_audit.json)。", "",
              "问题、工具结果、提议SQL和拒绝记录中的文本作为不可信数据处理；固定system规则与协议保护继续生效。未做独立对抗注入评测，不声称已验证注入防御。控制器恢复不能替代SQL知识；没有新增execute/retrieve训练课程或语义排序器，错误表列、聚合、连接与候选误改仍需后续在train上采集当前策略失败来训练。", "",
              "后续训练前需先适配采集流程：training/prepare_onpolicy.py 仍用旧纯SQL提取器处理 generated_text，scripts/score_onpolicy_candidates.py 仍固定五组check候选。不能直接把旧脚本指向本轮动作JSON日志。应只在train上采集当前策略的真实状态、被拒绝提议、错误反馈与随后有效修正，加入有用途的execute/retrieve示例，再与旧SQL修复样本混合微调，保留validation选型与test后验分析的边界。", "",
              "[完整结果表](v3_results.md) · [案例与结果边界](v3_case_notes.md) · [第三轮复现说明](../submission/REPRODUCE_V3.md)。原始预测、完整评分、提示/生成/模型身份、恢复与拒绝日志都保存在独立 runs 目录。本机自测成绩不等于教师已验证。Codex 辅助实现、分析和报告已披露，NOTICE归属沿用。测试案例只用于报告，不作为训练数据或硬编码答案。"]
    (ROOT/"reports/v3_report.md").write_text("\n".join(lines)+"\n")
    print("Wrote reports/v3_report.md")


if __name__=="__main__":main()
