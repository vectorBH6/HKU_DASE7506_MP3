# 第四轮复现

本轮CPU成绩见[报告](../reports/v4/v4_report.md)，配置与权重在[选择记录](../reports/v4/v4_selection.json)锁定。新机器先解压原始官方学生包，再将学生源码按导出清单保留相对目录放到包根目录（含mp3_sql），官方文件以原包为准。所有命令从该根目录执行，共用.venv。

## 仅复现已选Agent成绩

按[学生README](README.md)运行bash scripts/setup_env.sh --cpu准备CPU环境，再运行python scripts/download_model.py准备固定版本基座；这两步允许联网，正式预测离线。将所选adapter两文件放到 `artifacts/adapters/feedback_v4/epoch_01`。不需要先重新训练历史adapter。

```bash
unset PYTHONPATH
source .venv/bin/activate
export MP3_BASE_MODEL="$PWD/artifacts/models/Qwen3-0.6B"
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/feedback_v4/epoch_01"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m unittest discover -s tests -v
python scripts/run_experiment.py --experiment reproduction_v4 --split test \
  --config submission/configs/feedback_v4_selected.json
```

使用独立递增目录，不覆盖原始分数。新运行的predictions.json与score.json由官方程序生成，不修改。CPU四线程、workers1、所有模型调用记录并监测预算。

## 从训练起复现本轮

训练初始化需要先按[第一轮](../reports/v1/report.md)、[第二轮](REPRODUCE_V2.md)生成onpolicy_v001/epoch_02；推理复现只需要上面的所选adapter。训练需本机GPU，正式评测仍CPU。

```bash
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/onpolicy_v001/epoch_02"
python scripts/run_experiment.py --experiment e30_feedback_collection --split train \
  --config submission/configs/recovery_v3_selected.json --device cuda
python training/prepare_feedback_v4.py \
  --source-run runs/e30_feedback_collection/train/r001 \
  --output-dir artifacts/prepared_data/feedback_v4
python training/finalize_feedback_v4.py
python training/train_lora.py --config training/configs/lora_feedback_v4.json \
  --run-name feedback_v4
```

同名数据/训练目录拒绝覆盖。已有运行时应使用新的实验和训练版本名，并同步配置路径；原始数据SHA见报告。初始数据含5871条train样本，诊断先于SQL，保留旧SQL/动作样本；没有validation/test训练标签。追加课程独立生成2740条样本，并重新从旧adapter初始化单轮LoRA：

```bash
python training/prepare_synthesis_v4.py
python training/train_lora.py --config training/configs/lora_synthesis_v4.json   --run-name feedback_v4_extra
```

辅助SQL合成任务不声称是工具轨迹；真实失败修复与辅助目标在provenance中区分。追加课程配置与最终选中权重按选择记录核对，不能默认新训练版本必然胜出。

## 验证比较与选型

GPU开发筛选脚本为scripts/screen_feedback_v4.py，追加训练及四项开发对照由scripts/train_synthesis_v4.py记录，正式CPU核心对照使用recovery_v3_selected.json与feedback_v4.json并分别加载旧/新adapter。semantic_v4.json、contract_v4.json和constraint_v4.json比较提示与规则收益，plan_v4.json/plan_constraint_v4.json比较追加合成课程。完整命令、硬件分配、源码哈希和adapter身份记录在每个runs目录的run_metadata.json与generations.jsonl。

python scripts/analyze_feedback_v4.py 汇总已完成正式结果；scripts/select_feedback_v4.py只可执行一次且拒绝覆盖选择记录。复现现有方案不应重新执行选择脚本，也不应根据test结果更改配置。

原始预测/评分、完整提示/生成、拒绝/恢复、工具轨迹、资源与训练记录均保留。未来导出清单见../reports/v4/v4_delivery_manifest.json；不包含数据集、基座、.venv、缓存或未选权重。未创建dist或peer_reviews，也未上传。
