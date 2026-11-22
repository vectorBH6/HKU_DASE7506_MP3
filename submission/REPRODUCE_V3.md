# 第三轮：拒绝提议后恢复实验复现

全部命令从项目根目录运行，同一个 `.venv`，官方包保持原样。本轮不训练权重，使用第二轮已保存的单个 LoRA。环境与模型准备见 [README](README.md) 和 [第二轮复现说明](REPRODUCE_V2.md)。

本轮锁定四次生成方案，验证89/120、测试182/240；默认配置已更新，旧配置与原始结果保留。选择依据只用验证集，见 [选择记录](../reports/v3/v3_selection.json)。最终运行审计见 [v3_runtime_audit.json](../reports/v3/v3_runtime_audit.json)，未来导出清单见 [v3_delivery_manifest.json](../reports/v3/v3_delivery_manifest.json)（没有创建压缩包或dist）。

```bash
unset PYTHONPATH
source .venv/bin/activate
export MP3_BASE_MODEL="$PWD/artifacts/models/Qwen3-0.6B"
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/onpolicy_v001/epoch_02"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m unittest discover -s tests -v
python scripts/run_experiment.py --experiment e20_recovery4 --split validation \
  --config submission/configs/recovery_v3_4.json
python scripts/run_experiment.py --experiment e21_recovery6 --split validation \
  --config submission/configs/recovery_v3_6.json
python scripts/analyze_recovery_runs.py
```

每次运行递增编号，不覆盖旧结果。两组独立验证可分别使用 `--cpu-offset 0`、`--cpu-offset 4` 分配不同四核并行；并发影响计时。两组模型、权重、复核策略和恢复控制器均相同，只改变最多4/6次生成。仍是每次最多384新token和每题8次收费工具。

完整验证后才执行一次 `python scripts/select_recovery.py`，锁定新方案并记录是否替换默认。该脚本拒绝覆盖已有 v3_selection.json。已有锁定记录时直接复现所选测试：

```bash
python scripts/run_experiment.py --experiment e22_recovery_selected --split test \
  --config submission/configs/recovery_v3_selected.json
python scripts/analyze_recovery_runs.py
python scripts/write_recovery_report.py
```

当前官方数据、基座和adapter可检查：

```bash
python scripts/verify_artifacts.py --run-name onpolicy_v001 \
  --output reports/v3/v3_reproduction_artifact_verification.json
```

拒绝后的新上下文位于模型日志 messages 的 controller_feedback 中，只取自已发生的官方反馈和自己的被拒绝提议。立即恢复最多一次；无效提议及恢复均消耗模型生成次数，不消耗未执行的工具次数。拒绝历史保存在每题新建的Agent对象中，不修改官方观察或伪造工具结果。

正式评测CPU四核、workers=1、离线且有资源监测。原始 generations.jsonl、decisions.jsonl、predictions.json、score.json、资源记录和源码快照保留；未来导出可使用无损gzip日志副本，原始本地文件不删改。最终报告见 [v3_report.md](../reports/v3/v3_report.md)，选择规则见 [v3_plan_cn.md](../reports/v3/v3_plan_cn.md)。仍不创建 peer_reviews 或 dist，也不上传。
