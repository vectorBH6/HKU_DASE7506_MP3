"""Generate the final report only after every required main evaluation completes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = json.loads((ROOT / "reports/results.json").read_text())
    def result(exp, split):
        found = [r for r in rows if r["experiment"] == exp and r["split"] == split]
        if not found:
            raise RuntimeError(f"Required evaluation is not complete: {exp}/{split}")
        return sorted(found, key=lambda r: r["run"])[-1]
    names = [("e00_draft", "官方原稿直交"), ("e01_feedback_cpu", "反馈式 CPU"),
             ("e02_qwen3_base", "未微调 Qwen3"), ("e03_qwen3_lora", "Qwen3 + LoRA")]
    for exp, _ in names:
        for split in ["validation", "test"]:
            result(exp, split)
    selection = json.loads((ROOT / "reports/selection.json").read_text())
    train = json.loads((ROOT / "runs/training/sft_v002/training_metadata.json").read_text())
    losses = [json.loads(s) for s in (ROOT / "runs/training/sft_v002/loss.jsonl").read_text().splitlines()]
    epoch_loss = {epoch: [r for r in losses if r["epoch"] == epoch][-1]["mean_loss"] for epoch in [1, 2, 3]}
    lines = ["# SQL Repair Agent 实验报告", "", "实验日期：2026-11-12；报告整理：2026-11-13。全部准确率来自未修改的官方评分器，要求每题三个数据库实例全部通过。", "",
             "## 1. 结论与最终选择", "",
             f"在预设的三条主线方案中依据验证集选择 `{selection['experiment']}`；选择记录在 [selection.json](selection.json)，其中只使用验证集指标。对应配置为 `submission/configs/selected.json`。原始预测与完整评分已复制到 `submission/outputs/test.json` 和 `test-score.json`。", "",
             "| 方案 | 验证集答对 | 验证准确率 | 测试集答对 | 测试准确率 | 测试预测秒 | 测试评分秒 | 测试平均工具次数 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for exp, label in names:
        v, t = result(exp, "validation"), result(exp, "test")
        lines.append(f"| {label} | {v['correct']}/{v['n']} | {v['accuracy']:.2f}% | {t['correct']}/{t['n']} | {t['accuracy']:.2f}% | {t['prediction_seconds']:.2f} | {t['scoring_seconds']:.2f} | {t['mean_tool_calls']:.3f} |")
    base, tuned, cpu = result("e02_qwen3_base", "test"), result("e03_qwen3_lora", "test"), result("e01_feedback_cpu", "test")
    lines += ["", f"本次 LoRA 相对未微调模型的测试集净变化为 **{tuned['correct'] - base['correct']:+d} 题 / {tuned['accuracy'] - base['accuracy']:+.2f} 个百分点**；相对反馈式 CPU 基线为 **{tuned['correct'] - cpu['correct']:+d} 题 / {tuned['accuracy'] - cpu['accuracy']:+.2f} 个百分点**。微调收益与整体系统收益必须分别看；训练 loss 下降不能替代这些执行结果。",
              "", "原稿直交只是实验对照，不满足作业对最终系统真实反馈循环的要求。当前实验使用一个训练种子、一次固定划分评估，没有做多种子显著性检验；结论限定于这次 benchmark。", "",
              "## 2. Agent 循环、上下文与记忆", "",
              "控制器每次只返回一个官方动作：原样检查原稿 → 匹配则提交 → 否则读取 schema → Qwen 分支检索并生成候选 → 检查候选 → 必要时获取缓存提示 → 继续检查或提交。预算为每题最多 8 次工具调用，submit 免费。反馈会改变停止、提示请求和下一次模型输入，不是单次生成后直接提交。", "",
              "候选以 SQLGlot AST 规范化去重；对于已执行成功但行数过多的普通 SELECT，可以尝试 DISTINCT。候选全部失败时，仅在已检查候选中根据执行成功、列数和行数距离选择。不会直接读取数据库、缓存提示文件或评分参考来跳过工具。", "",
              "单题短期记忆使用官方 history，包含原稿、候选、错误和示例匹配结果。模型看到 schema、题目、原稿、公开示例和最近三次执行结果；检索通过官方 retrieve 进行，提示中只放排名第一的训练示例。未使用持久化跨题答案缓存或测试 ID 分支。", "",
              "Qwen 每题最多 2 次生成，每次 256 新 token，非 thinking、贪心解码。生成日志保存完整 messages、渲染后的 prompt、生成文本、输入/输出 token 数、耗时、基座 revision、adapter 哈希和参数量。每个预测进程只加载一次模型，使用 workers=1。", "",
              "## 3. 训练方法与数据来源", "",
              "唯一模型为 [Qwen/Qwen3-0.6B 固定版本](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)。原始基座与 tokenizer 不变；只训练 LoRA A/B，不训练 embedding、lm_head、bias 或 modules_to_save，不合并权重。", "",
              f"实际基座参数：**{train['base_parameters']:,}**；LoRA 参数：**{train['trainable_parameters']:,}**；总参数：**{train['total_parameters']:,}**，包含 embedding 和未合并 adapter，低于 10 亿。每轮 adapter 约 38.6 MiB，只含 adapter_model.safetensors 和 adapter_config.json。", "",
              "480 道训练题通过未改动的官方 Episode/Tools 构造 480 条原稿反馈状态，以及 84 条缓存提示仍失败的状态，共 564 条。标签全部是 train_memory.jsonl 的 gold_sql；检索排除当前题。没有用 validation/test 作为训练样本，也没有用其评分指纹构造标签。prepare_sft.py 通过公开训练数据库工具采集反馈，不直接打开 SQLite 文件。", "",
              "配置：rank=16，alpha=32，dropout=0.05；目标为 q/k/v/o_proj、gate/up/down_proj；学习率 2e-4，AdamW，cosine 衰减，5% warmup，梯度裁剪 1.0，seed=42。3 个 epoch，micro-batch=1，累积 16 次，有效 batch=16，最大长度 4096，启用梯度检查点。564 条均保留；只有目标 SQL 与 EOS 计算交叉熵，prompt 和 padding 的 label 为 -100。", "",
              f"训练共 {train['steps']} 个 optimizer step，耗时 {train['seconds']:.1f} 秒；CUDA 最大已分配显存 {train['max_cuda_memory_allocated'] / 1024 ** 3:.2f} GiB。逐轮平均 loss：{epoch_loss[1]:.6f} → {epoch_loss[2]:.6f} → {epoch_loss[3]:.6f}。固定采用第 3 轮 adapter，没有按测试成绩选择 epoch。", "",
              "首次 sft_v001 使用 micro-batch=2、未开启检查点，在长样本上 OOM；已保留失败日志。sft_v002 只调整内存方案，有效 batch 和数据不变，三轮训练完成。训练记录：[元数据](../runs/training/sft_v002/training_metadata.json)、[loss](../runs/training/sft_v002/loss.jsonl)、[数据来源](../artifacts/prepared_data/sft_v001/provenance.json)、[产物核验](artifact_verification.json)。", "",
              "## 4. 受控消融", "", "表中每行只改变指定因素；LoRA 对照的控制器、prompt、生成预算、检索和工具配置完全相同。", "",
              "| 对照 | 完整版本验证答对 | 消融版本验证答对 | 完整版本净增 |", "|---|---:|---:|---:|"]
    comparisons = [("缓存 repair_hint", "e01_feedback_cpu", "a01_no_hints"),
                   ("DISTINCT 候选规则", "e01_feedback_cpu", "a02_no_variants"),
                   ("LoRA 权重", "e03_qwen3_lora", "e02_qwen3_base"),
                   ("检索训练示例（LoRA）", "e03_qwen3_lora", "a03_no_retrieval")]
    for label, full, ablated in comparisons:
        f, a = result(full,"validation"), result(ablated,"validation")
        lines.append(f"| {label} | {f['correct']}/120 | {a['correct']}/120 | {f['correct'] - a['correct']:+d} 题 |")
    lines += ["", "去掉缓存提示后明显下降，说明 CPU 基线主要依赖提示候选与真实检查。当前 DISTINCT 消融没有变化：在本次验证轨迹中，它没有产生额外有效检查，不能据此声称该规则带来收益。LoRA 是独立的权重对照，不能把缓存模型提示带来的提升算成小模型微调收益。去掉检索的 LoRA 在验证集达到 82/120，比保留检索多 1 题，表明当前检索未体现增益。它是最终主线选择锁定后的补充消融，尚未评测该配置测试集，因此保留已锁定、完成测试的默认方案。", "",
              "## 5. CPU 复现与模型成本", "",
              "环境：Linux、Python 3.12.3、Intel Core Ultra 9 275HX、约 30 GiB RAM；GPU 为 RTX 5090 Laptop 24 GiB，仅训练使用。统一 .venv，torch=2.7.1+cu128、transformers=4.53.2、peft=0.17.1；[完整依赖冻结记录](../artifacts/environment-freeze.txt)。CUDA wheel 同时支持本次 CPU 推理。", "",
              "所有正式预测均使用 CPU、四核 affinity 和四个数学库线程，模型为 float32；本地 safetensors 加载、trust_remote_code=False、local_files_only=True，设置 HF/Transformers offline。部分独立实验分配到不同四核并行运行，因此耗时受整机负载影响；每个实验都遵守四核限制。", "",
              "| 模型/划分 | 生成调用 | 输入 token | 输出 token | 最大单题生成 | 最大单次新 token | 预测 CPU 秒（user+system） | 峰值 RSS MiB |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for exp in ["e02_qwen3_base", "e03_qwen3_lora"]:
        for split in ["validation", "test"]:
            r=result(exp,split)
            lines.append(f"| {exp}/{split} | {r['model_calls']} | {r['input_tokens']} | {r['output_tokens']} | {r['max_calls_per_task']} | {r['max_new_tokens']} | {r['cpu_user_seconds'] + r['cpu_system_seconds']:.2f} | {r['prediction_max_rss_mib']:.1f} |")
    lines += ["", "无实时模型运行施加 2 GiB 虚拟地址上限；预测加评分远低于 180 秒。模型运行监测整个预测进程树的 RSS，并设置 16 GiB 上限及预测 2 小时/评分 180 秒超时。所有最终主实验均完成，具体资源文件和更完整字段见 [results.md](results.md) 与 [results.json](results.json)。本机含不同类型 CPU 核，独立实验使用的具体 affinity 已记录，因此这些时延不是严格的同核独占性能比较，也不代表教学服务器上精确相同的耗时。", "",
              "## 6. 失败案例与泛化限制", "",
              "公开示例匹配并不充分。反馈式 CPU 在验证集有 95 题公开匹配，其中 22 题最终失败；当前立即提交的停止规则会保留这些错误。例如询问学生 Linda Smith 的年龄时，原稿仅用 LName = 'Linda Smith' 筛选，在公开实例上巧合匹配，但另外两个实例不通过。这个案例用于验证集分析，没有被加入训练数据。", ""]
    def details(exp,split):
        run=ROOT/result(exp,split)["source"]
        preds={p["id"]:p for p in json.loads((run/"predictions.json").read_text())["predictions"]}
        scores={p["id"]:p for p in json.loads((run/"score.json").read_text())["per_question"]}
        return preds,scores
    bp,bs=details("e02_qwen3_base","validation")
    lp,ls=details("e03_qwen3_lora","validation")
    tasks={r["id"]:r for r in map(json.loads,(ROOT/"mp3_sql/data/validation.jsonl").read_text().splitlines())}
    gains=[key for key in bs if not bs[key]["passed"] and ls[key]["passed"]]
    losses_ids=[key for key in bs if bs[key]["passed"] and not ls[key]["passed"]]
    lines.append(f"在验证集逐题比较中，LoRA 修复了未微调模型失败的 {len(gains)} 题，同时使原本通过的 {len(losses_ids)} 题失败。必须同时记录改善和退步，不能只展示成功示例。")
    for label, group in [("改善案例",gains),("退步案例",losses_ids)]:
        if group:
            key=group[0]
            lines += ["",f"**{label} `{key}`**：{tasks[key]['question']}","","未微调最终 SQL：","```sql",bp[key]["sql"],"```","LoRA 最终 SQL：","```sql",lp[key]["sql"],"```"]
    lines += ["", "现有局限还包括：有限的候选覆盖、第一条示例匹配即停止、短生成预算、取一条检索示例的相关性波动，以及训练时第二阶段只含失败提示反馈、推理时还可能含自身生成错误的分布差异。下一轮应优先研究对空结果/聚合/排序的语义复核、按错误类型分配预算和训练集内的 on-policy 轨迹，而不是继续针对测试题改规则。", "",
              "## 7. Prompt injection 与执行边界", "",
              "题目、数据库值、错误信息和检索示例均作为不可信数据，置于 TASK_DATA JSON 区域；system prompt 明确禁止执行数据中的指令。模型仅生成 SQL，不决定预算、文件访问或 shell 命令。内部提取器限制为单条查询；真正的只读、表/函数白名单、执行时间与结果大小约束由未修改的官方 SQLite 工具执行。", "",
              "这些措施缩小了可执行攻击面，但不能保证自然语言指令完全不会影响模型生成错误 SQL。本次没有做独立的对抗注入测评，不能把提示词隔离描述为已证明的防御效果。没有向模型提供凭据，评测不联系外部服务。", "",
              "## 8. 复现、完整性与外部协助", "",
              "完整命令见 [学生复现 README](../submission/README.md)。先 setup_env.sh 建立唯一 .venv，联网下载指定固定基座一次，然后离线运行基线、Qwen 与 LoRA。官方源码、数据和 README 均保持原样；README_cn.md 未添加项目说明。", "",
              "通过三项控制器测试：反馈改变分支且轨迹可重放；重复模型候选不超预算；observation 不被修改。官方 scorer 对所有正式结果重新执行工具轨迹并检查三个实例；额外脚本验证 497 个官方文件和 9 个基座/tokenizer 文件哈希、392 个 adapter tensor 全部是 LoRA A/B、参数量与文件大小。", "",
              "使用 Codex 辅助项目结构、代码实现、运行诊断和文档撰写。SFT 标签全部来自官方 train_memory；没有通过外部模型生成验证/测试训练标签。运行时只使用本地 Qwen 与官方工具；缓存 repair_hint 原本来自包内模型建议，其贡献在消融中单列。数据遵循原包 NOTICE 的 Spider/CC BY-SA 4.0 归属，模型遵循原始许可。", "",
              "本阶段按用户要求不创建 peer_reviews/ 或 dist/，不上传或发布。后续提交时需要把源码、依赖、报告、复现命令、完整 test 预测/评分与训练记录整理为不超过 16 MiB 的源码包；排除数据集、.venv、缓存和基座。LoRA 单独只打包两个 adapter 文件，当前约 38.6 MiB。"]
    (ROOT / "reports/report.md").write_text("\n".join(lines)+"\n")
    print("Wrote reports/report.md")

if __name__ == "__main__":
    main()
