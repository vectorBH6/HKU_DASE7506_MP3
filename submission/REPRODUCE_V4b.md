# 第四轮最终方案复现

最终结果见[报告](../reports/v4b/v4b_report.md)，[部署记录](../reports/v4/v4_deployment.json)记录全部公开完整测试分数及最终选择。初次新权重方案180/240保留在[初次复现说明](REPRODUCE_V4.md)。

新机器先解压原始官方学生包，将本源码保留相对目录放到包根目录（含mp3_sql），官方文件以原包为准。所有命令在该根目录执行，只使用一个.venv。准备环境与基座阶段可以联网，正式推理离线。

```bash
bash scripts/setup_env.sh --cpu
unset PYTHONPATH
source .venv/bin/activate
python scripts/download_model.py
```

将最终adapter的两文件放到`artifacts/adapters/onpolicy_v001/epoch_02`，核对adapter_model.safetensors的SHA-256为`43d91619f16b1a7e4cb3358ad93cfc5f8ee447ff124640a8b5e2ff8ee44de613`。推理复现直接加载提交的两文件，不需要先重新训练。

```bash
export MP3_BASE_MODEL="$PWD/artifacts/models/Qwen3-0.6B"
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/onpolicy_v001/epoch_02"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m unittest discover -s tests -v
python scripts/run_experiment.py --experiment reproduction_v4b --split test \
  --config submission/configs/contract_v4_selected.json
```

默认快捷命令为`bash scripts/run_selected.sh test`，读取configs/selected.json并使用同一旧adapter默认路径；显式设置MP3_ADAPTER_DIR可避免终端遗留路径混淆。每次自动生成独立r001/r002目录，不覆盖原始成绩。官方预测器与评分器原样运行，全240题、每题三个实例；CPU四线程、workers1、模型加载一次、16GiB RSS、预测7200秒/评分180秒限制均由运行器记录。

训练从头复现：最终adapter沿用[第二轮训练](REPRODUCE_V2.md)，初始化和数据过程也见[第一轮报告](../reports/v1/report.md)。本轮5871条和2740条新课程不是最终权重来源，复现它们的负结果见REPRODUCE_V4.md。没有validation/test训练样本，未将测试逐题标签写入规则。

对照复现只需分别设置旧adapter与contract_v4_selected.json、新adapter与feedback_v4_selected.json；完整验证/测试结果见../reports/v4/v4_results.md。v4b_selection.json在后续对照测试前锁定，v4_deployment.json明确披露聚合公开分数参与最终保留，不将其宣称为隐藏测试选择。

原始日志均保留，未来导出使用../reports/v4/v4_delivery_manifest.json及../reports/v4/v4_reproduction_logs/index.json的无损压缩日志映射。保持NOTICE归属；数据集、基座、.venv、缓存不导出，adapter另列两文件。不创建peer_reviews或dist，未上传。
