# 第二轮：动作决策与语义复核的复现

全部命令在项目根目录执行，共用 `.venv`。官方文件不修改；不创建 peer_reviews 或 dist。第一轮环境、固定基座和 `sft_v002/epoch_03` 的准备见 [README](README.md)。最终结论见 [第二轮报告](../reports/v2/v2_report.md)，全部已完成对照见 [结果表](../reports/v2/v2_results.md)。

## 环境与模型

```bash
unset PYTHONPATH
source .venv/bin/activate
export MP3_BASE_MODEL="$PWD/artifacts/models/Qwen3-0.6B"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
python -m unittest discover -s tests -v
```

基座固定为 `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`。推理只加载一个未合并的 adapter。新训练从第一轮 LoRA 继续更新 A/B，不叠加两个 adapter。

## 训练材料与新增训练

全新机器若没有第一轮 adapter，在完成基座下载后先执行以下步骤（同名目录必须尚不存在）：

```bash
python training/prepare_sft.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python training/train_lora.py --config training/configs/lora_sft_v002.json --run-name sft_v002
```

以下步骤只使用 train。采集可以用 GPU；GPU 用时不作为正式推理成绩。

```bash
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/sft_v002/epoch_03"
python scripts/run_experiment.py --experiment e10_onpolicy_collection --split train \
  --config submission/configs/qwen3_lora.json --device cuda
python scripts/score_onpolicy_candidates.py
python training/prepare_onpolicy.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python training/train_lora.py --config training/configs/lora_onpolicy_v001.json \
  --run-name onpolicy_v001_reproduction
python scripts/verify_artifacts.py --run-name onpolicy_v001_reproduction \
  --output reports/v2/v2_reproduction_verification.json
```

候选评分脚本按原始轨迹的各个 check 截断并合法 submit，原始反馈不改写，随后交给官方评分器重放。训练构造器默认读取 `e10_onpolicy_collection/train/r001`；采集已有其他编号时，先通过脚本的 `--source` 指向新预测，并为训练构造器传入 `--source-run runs/e10_onpolicy_collection/train/rXXX`。若只复现本机既有数据对应的训练，可直接复用已保存的 `artifacts/prepared_data/onpolicy_v001/train.jsonl` 和 provenance。

本轮固定 2 个 epoch，使用最终 epoch_02，不按验证集挑 epoch。训练器拒绝覆盖同名训练 run；复现采用新 run-name。2,566 条样本包含 2,002 条动作标签和 564 条 SQL 保留样本；动作标签为 check、repair_hint、submit。execute/retrieve 的接口允许模型提出，但本轮没有专门的正例训练。train 三实例判分只用于离线监督，不向推理 Agent 提供。

## 完整 CPU 验证与消融

```bash
# 新控制器使用旧 adapter；与新 adapter 对照。
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/sft_v002/epoch_03"
python scripts/run_experiment.py --experiment e11_review_old_adapter --split validation \
  --config submission/configs/review_v2_old_adapter.json

export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/onpolicy_v001_reproduction/epoch_02"
python scripts/run_experiment.py --experiment e12_review_onpolicy --split validation \
  --config submission/configs/review_v2_onpolicy.json
python scripts/run_experiment.py --experiment e13_review_risk --split validation \
  --config submission/configs/review_v2_risk.json
python scripts/run_experiment.py --experiment e14_review_off --split validation \
  --config submission/configs/review_v2_off.json
python scripts/analyze_review_runs.py
# 所有验证完成后，仅在本次尚未选型时执行；已有选择记录会拒绝覆盖。
python scripts/select_review.py
```

always/risk/off 只改变“原稿已经匹配公开示例时，是否继续复核”；后续候选仍交给模型比较。不同实验可用 `--cpu-offset 4`、`--cpu-offset 8` 分配不同四核，但并发会影响计时。本机正式评测均为 CPU、workers=1、离线、有内存和时间监测。

## 固定选择后测试

先根据验证集固定新方案配置，记录 `reports/v2/v2_selection.json`，再运行测试。默认配置是否替换第一轮，同样仅由验证集决定；测试结果不能用于倒推改规则。

```bash
# 本机完成选择后，可直接使用已保存的锁定配置。
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/onpolicy_v001/epoch_02"
python scripts/run_experiment.py --experiment e15_review_selected --split test \
  --config submission/configs/review_v2_selected.json
python scripts/analyze_review_runs.py
python scripts/write_review_report.py
```

每次运行自动创建新编号，保留 predictions、完整 score、源码快照、资源记录、完整模型提示及生成日志、模型动作与协议拦截记录。最后一个脚本汇总本次固定训练版本的报告；复现训练采用新 run-name 时，应在自有报告中引用对应元数据。

初始 check/schema 和格式、去重、预算校验由 Python 保证，Qwen 决定后续动作及 SQL；模型无效输出触发有限回退。每题最多 4 次模型生成、每次最多 384 新 token、8 次收费工具，submit 免费。所有生成包括复核和无效提议都计数。公开 check 匹配只是一个实例上的证据，最终成绩仍要求三个实例全部通过。

源码体积检查可执行 `python scripts/check_review_delivery.py`。它只生成未来导出清单及完整日志的无损 gzip 副本，不创建 dist、源码归档或 adapter 归档，不上传。原始 JSONL 保留；`gzip -dc <日志.gz>` 可还原完整检查日志。
