# V5复现

当前正式结果为184/240；完整对照、负结果、预算与数据来源见[报告](../reports/v5/v5_report.md)。原始官方学生包必须位于项目根目录，保持mp3_sql及数据不变。全部使用同一个.venv。

```bash
bash scripts/setup_env.sh --cpu
unset PYTHONPATH
source .venv/bin/activate
python scripts/download_model.py
export MP3_BASE_MODEL="$PWD/artifacts/models/Qwen3-0.6B"
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/onpolicy_v001/epoch_02"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash scripts/run_selected.sh test
```

将所选adapter的adapter_model.safetensors和adapter_config.json放在上面的目录。它们独立于源码交付，不包含基座或合并权重。默认读取submission/configs/selected.json，每次产生新的runs/reproduction_selected/test/rNNN，不覆盖历史结果。运行器记录单次模型加载、完整提示和生成、CPU四线程、RSS及原样官方评分。

CPU矩阵对照：contract_v4_selected.json搭配旧adapter为A；search_v5_selected.json搭配旧adapter为B。将adapter改为reports/v5/v5_grpo_selection.json所记录检查点，分别运行两个配置得到C/D。固定测试挑战者为e54_v5_arch_test和e55_v5_grpo_test；结果逐字节保留在submission/outputs/v5-arch-test.json与v5-grpo-test.json及各自score文件。

以下命令复现两组挑战者；若只跑验证，将--split test改成--split validation。需要分别准备旧adapter和GRPO的update_016两份独立权重目录；两次运行各加载一个模型，不做集成。

```bash
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/onpolicy_v001/epoch_02"
python scripts/run_experiment.py --experiment reproduction_v5_arch --split test --device cpu --config submission/configs/search_v5_selected.json
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/grpo_v5/update_016"
python scripts/run_experiment.py --experiment reproduction_v5_grpo --split test --device cpu --config submission/configs/grpo_v5_selected.json
```

本机GPU训练仍使用相同.venv；已有依赖即可，不依赖TRL。训练从旧onpolicy_v001/epoch_02开始，其来源见REPRODUCE_V2.md。

```bash
export MP3_BASE_MODEL="$PWD/artifacts/models/Qwen3-0.6B"
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/onpolicy_v001/epoch_02"
export MP3_DEVICE=cuda
python training/collect_v5.py --output runs/v5_train_coverage_reproduction --report reports/v5/v5_train_coverage_reproduction.json
python training/train_grpo_v5.py --config training/configs/grpo_v5.json --run-name grpo_v5_reproduction
```

GRPO配置固定了train采样课程，不依赖后续覆盖诊断更新。训练脚本读取原有runs/e30_feedback_collection/train/r001/score.json（train派生辅助记录，提供在源码清单中），其最终hard_task_ids以配置为准。训练日志和源快照在runs/training/grpo_v5；训练命令中的新run-name避免覆盖旧目录。GPU采样数值可能随硬件或依赖不同而变化；推理复现应直接使用提交adapter。

所有训练仅使用train；额外实例由独立训练评分器提供终局奖励，不向模型暴露。正式推理保持离线、一个模型、官方工具接口；不导出数据集、.venv或基座。未打包到dist，未上传或进行互评。
