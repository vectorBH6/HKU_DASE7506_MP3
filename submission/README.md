# SQL Repair Agent：学生实现与复现

本目录包含学生Agent；根README.md和README_cn.md保留官方原文及逐项翻译。最新实验见[第五轮报告](../reports/v5/v5_report.md)，复现见[REPRODUCE_V5.md](REPRODUCE_V5.md)。所有命令从项目根目录执行，共用.venv，暂不创建peer_reviews和dist。历史报告与日志保留。

## 目录与命名

```text
submission/                  学生入口、控制器、模型、提示词、配置
training/                    训练数据构造、LoRA 训练与版本化配置
scripts/                     环境安装、固定模型下载、实验运行和资源限制
artifacts/models/            原始基座和下载哈希清单（不提交）
artifacts/prepared_data/      仅训练集派生样本和真实工具轨迹（不提交）
artifacts/adapters/           每轮单独保存的未合并 LoRA
runs/<实验>/<划分>/r001/       预测、完整官方评分、日志、源码快照、耗时与内存
runs/training/<训练版本>/     配置、loss、参数白名单与数据来源
reports/                     实验报告与结果汇总
```

`e00_draft` 是官方原稿直交，`e01_feedback_cpu` 是反馈基线，`e02_qwen3_base` 是未微调模型，`e03_qwen3_lora` 是同一控制器加 LoRA。`a01_no_hints`、`a02_no_variants` 和 `a03_no_retrieval` 是第一轮单变量消融。第二轮 `e10` 采集 train 轨迹，`e11` 对照旧 adapter，`e12/e13/e14` 比较原稿复核策略，`e15` 是验证选型后锁定的测试。第三轮 `e20/e21` 比较恢复机制下4/6次生成，`e22` 是验证锁定后的完整测试。第三轮沿用第二轮权重。第四轮e30采集train轨迹，e31/e33/e38组成诊断与权重对照，e36/e37/e39比较通用约束；e34确认开发候选，e35是初次锁定后的完整测试；e40/e41为本次未启动的预留名称，e42为后续固定旧权重约束对照。GPU开发实验用dev_v4前缀，训练版本为feedback_v4和feedback_v4_extra。每次运行自动递增 `r001/r002/...`，不覆盖旧结果。

## 环境与模型

本机采用 Python 3.12，RTX 5090 Laptop GPU，CUDA 版 PyTorch 2.7.1；GPU 用于训练与开发筛选。正式结果使用四个 CPU 核、离线推理。`nvidia-smi` 的 CUDA 13.2 表示驱动支持上限，并不要求安装 PyTorch CUDA 13.2 wheel。PyTorch 2.7 的 CUDA 12.8 wheel 支持 Blackwell，见 [PyTorch 官方说明](https://pytorch.org/blog/pytorch-2-7/)。

```bash
# 本机 GPU 训练环境；仍然可以直接做 CPU 推理。
bash scripts/setup_env.sh --cuda
# 另一台只有 CPU 的复现机器可改用 --cpu，仍只建一个 .venv。

# ROS 的全局 PYTHONPATH 会混入系统包；本项目命令清除它。
unset PYTHONPATH
source .venv/bin/activate
python -m mp3_sql.check_data
python -m unittest discover -s tests -v
# 训练完成后可运行 python scripts/verify_artifacts.py --run-name sft_v002，实验完成后运行 python scripts/summarize_results.py

# 模型准备阶段的联网步骤：下载指定不可变版本。
python scripts/download_model.py
export MP3_BASE_MODEL="$PWD/artifacts/models/Qwen3-0.6B"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

唯一基座为 `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`。下载文件 SHA-256 在 `artifacts/models/base_manifest.json`。基座与 tokenizer 不修改、不合并 adapter，推理加载使用 `local_files_only=True`、`trust_remote_code=False` 和 safetensors。

## 已完成的官方评测

| 方案 | CPU验证 | 完整CPU测试 |
|---|---:|---:|
| 第一轮 | 81/120（67.50%） | 175/240（72.92%） |
| 第二轮 | 87/120（72.50%） | 181/240（75.42%） |
| 第三轮 | 89/120（74.17%） | 182/240（75.83%） |
| 第四轮初次训练候选 | 95/120（79.17%） | 180/240（75.00%） |
| 第四轮旧权重约束对照 | 92/120（76.67%） | 184/240（76.67%） |
| 第五轮架构挑战者 | 93/120 | 182/240 |
| 第五轮GRPO挑战者 | 92/120 | 182/240 |

当前规范outputs/test.json、test-score.json是`runs/e42_contract_control/test/r001`原始预测和完整官方评分的逐字节副本，正确184/240。V5两组挑战者及所有负结果均保留；[结果记录](../reports/v5/v5_results.json)披露公开test聚合分数参与最终保留。

## 已选方案的快速复现

默认控制器`contract_v4`，adapter为`artifacts/adapters/onpolicy_v001/epoch_02`。

```bash
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/onpolicy_v001/epoch_02"
bash scripts/run_selected.sh test
```

完整环境、权重与对照命令见[第五轮复现指南](REPRODUCE_V5.md)。此前各轮的配置和权重仍保留。

## 依次复现实验（第一轮历史对照）

```bash
# 1. 原稿直交对照；不满足最终 Agent 的反馈要求。
bash scripts/run_baseline.sh

# 2. 反馈式 CPU Agent。
for split in validation test; do
  python scripts/run_experiment.py --experiment e01_feedback_cpu \
    --split "$split" --config submission/configs/feedback_cpu.json
done

# 3. 未微调 Qwen3，默认在 CPU 上运行，worker=1。
for split in validation test; do
  python scripts/run_experiment.py --experiment e02_qwen3_base \
    --split "$split" --config submission/configs/qwen3_base.json
done

# 4. 仅用 train 生成训练样本并训练 LoRA。
python training/prepare_sft.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python training/train_lora.py \
    --config training/configs/lora_sft_v002.json --run-name sft_v002_reproduction

# 5. 加载第 3 轮 adapter；更换训练 run-name 后同步调整路径。
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/sft_v002_reproduction/epoch_03"
for split in validation test; do
  python scripts/run_experiment.py --experiment e03_qwen3_lora \
    --split "$split" --config submission/configs/qwen3_lora.json
done

# 6. 两个单变量消融（验证集）。
python scripts/run_experiment.py --experiment a01_no_hints --split validation \
  --config submission/configs/ablation_no_hints.json
python scripts/run_experiment.py --experiment a02_no_variants --split validation \
  --config submission/configs/ablation_no_variants.json
# 在最终选择锁定后补充的检索消融，不参与重新选型。
python scripts/run_experiment.py --experiment a03_no_retrieval --split validation \
  --config submission/configs/ablation_no_retrieval.json
```

本次保留的 adapter 在 `artifacts/adapters/sft_v002/epoch_03`。`sft_v001` 因长序列显存不足而失败；`sft_v002` 把 micro-batch 改为 1，累积 16 次并开启梯度检查点。有效 batch 仍为 16。训练器拒绝覆盖同名训练目录，重跑请换 run-name；随后校验时同样传入 `python scripts/verify_artifacts.py --run-name sft_v002_reproduction`。

如直接调用官方 CLI，需设置 `PYTHONPATH="$PWD/submission:$PWD"`、`MP3_GENERATION_LOG`、四线程环境变量并自行测量资源；推荐使用上面的运行器，它会完成这些设置。模型运行默认 CPU；`--device cuda` 仅用于开发，不能把其速度当作 CPU 复现成绩。`--cpu-offset` 只用于将独立实验分配到不同的四个可用 CPU 核。

## 第一轮 Agent 如何工作（历史对照）

1. 原样 `check(issue_sql)`，保证真实反馈和提示前置条件。
2. 示例匹配则停止；否则读取 schema。该停止规则存在“公开实例巧合匹配”的风险，报告会统计它。
3. Qwen 分支通过官方 retrieve 取训练示例，提示中只放排名第一条，避免上下文过长；在原稿反馈基础上生成首个 SQL。
4. `check` 新候选。仍不匹配时调用缓存 `repair_hint`，候选经 AST 规范化去重后逐个检查。
5. 对成功执行但行数过多的普通 SELECT 尝试 DISTINCT；Qwen 还可根据更新后的失败反馈生成第二个候选。
6. 匹配则提交；用尽候选或预算则按执行成功、列数、行数接近程度选择已检查候选。

每题最多 8 次官方工具调用；Qwen 最多 2 次生成，每次最多 256 新 token。生成用非 thinking、贪心解码，两组模型共用完全相同的控制器和提示词。每次调用的提示、文本、token 数、模型和 adapter 身份均记录在 `generations.jsonl`。

SQL AST 用于提取单条只读候选、去重和有限结构编辑；真正的可执行性和结果由官方工具判定。内部 SQL 处理函数不新增官方 action，不绕过工具次数，也不直接打开数据库。原始 observation 和官方轨迹保持不变。

## 第二轮 Agent 如何工作

Python 先原样 check 原稿；低风险且已匹配的原稿可直接提交，其余读取 schema。Qwen 随后根据题意、schema、全部已检查候选和真实反馈提出动作 JSON，包含 check、execute、repair_hint、retrieve 或提交候选。Python 负责格式、去重、预算与有界回退，不将模型自评当作正确证明。

每题最多 4 次生成、每次最多 384 新 token、8 次收费工具；生成规划和复核均计数。动作训练实际覆盖 check、repair_hint、submit，不能把允许的工具列表当作全部能力已经学会的证据。模型、数据、消融和新实验命令均见 [第二轮复现说明](REPRODUCE_V2.md)。

## 第三轮恢复机制

在第二轮基础上，模型提议被协议保护拒绝后，内部保存最近两次拒绝、最近SQL错误和剩余生成预算，再立即尝试一次。新上下文会变化；每次尝试均计入四次生成上限，不改官方观察或工具轨迹。恢复后仍可能回退，接受提议不等于SQL正确。完整行为、成本与局限见 [第三轮报告](../reports/v3/v3_report.md) 和 [案例说明](../reports/v3/v3_case_notes.md)。

## 第四轮诊断与约束修正

比较SQL AST/schema诊断、真实反馈微调、公开列名投影与显式去重规则，以及数值边界和短plan课程。新课程分别含5871/2740条train样本，共三个新检查点；初次选中的新权重方案验证95/120但测试180/240，未超过旧版182/240。随后完整测试了已验证的旧权重基础约束方案。

最终保留`contract_v4`与旧权重。工具仍为原官方工具，所有新候选真实check，最多8次收费调用，模型最多4次生成/384新token。模型自评、绑定诊断和公开示例匹配都不能证明三个实例均正确。收益、退步、全部训练负结果与实际自主工具次数见[最终报告](../reports/v4b/v4b_report.md)。

## 第五轮搜索与GRPO

五组架构开发对照比较检索恢复、额外复查、JOIN探查、始终复查和提示候选复查。训练集四轨迹诊断覆盖480题；GRPO在train上实际尝试264组，完成64组有效更新，额外评分实例只用于离线训练奖励，不进入Agent观察。

e50为CPU旧方案复现，e51确认架构，e52/e53比较同一GRPO权重的旧/新架构，e54/e55是预锁定的两组完整CPU测试。当前最佳184/240。权重、提示和工具是否改进，均以[第五轮报告](../reports/v5/v5_report.md)的真实对照为准；训练完成不等于能力或分数提高。

## 学习重点

- **Agent 与模型的分工**：模型提出修复 SQL 和下一步动作，控制器校验协议与预算，官方执行反馈验证候选。
- **LoRA 的作用**：通过训练 A/B 低秩矩阵改变生成倾向；微调 loss 下降不能替代跨 schema 的验证集和测试集成绩。
- **反馈 SFT**：输入含真实失败信息，只有目标 SQL/动作JSON和 EOS 参与 loss；schema、题目和工具输出作为条件，标签全部来自训练集。
- **公开示例与泛化**：一个实例匹配仍可能是错 SQL，最终以三实例官方判分为准。
- **消融与成本**：一次只关闭一个模块，比较准确率、工具次数、模型 token、CPU 时间和内存。

实验文件和完整日志由本地保留；后续打包应排除 `.venv`、数据集、基座、缓存和训练派生数据。每个 LoRA 目录只含作业要求的两份 adapter 文件。当前未执行上传、发布或互评。
