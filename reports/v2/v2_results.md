# 第二轮官方实验汇总

准确率要求每题三个实例全部通过；仅列出已完成、官方轨迹重放通过的运行。

| 实验 | 划分 | 正确 | 准确率 | 预测秒 | 平均工具 | 模型调用 | 输出 token | RSS MiB | 公开匹配但最终错 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [e03_qwen3_lora/r001](../../runs/e03_qwen3_lora/validation/r001/score.json) | validation | 81/120 | 67.50% | 363.35 | 2.883 | 71 | 2123 | 5018.2 | 25 |
| [e03_qwen3_lora/r001](../../runs/e03_qwen3_lora/test/r001/score.json) | test | 175/240 | 72.92% | 658.84 | 3.442 | 169 | 4879 | 5018.7 | 31 |
| [e11_review_old_adapter/r002](../../runs/e11_review_old_adapter/validation/r002/score.json) | validation | 81/120 | 67.50% | 1943.82 | 3.575 | 295 | 10433 | 4989.3 | 22 |
| [e12_review_onpolicy/r001](../../runs/e12_review_onpolicy/validation/r001/score.json) | validation | 87/120 | 72.50% | 1986.44 | 3.625 | 299 | 9353 | 4989.7 | 17 |
| [e13_review_risk/r001](../../runs/e13_review_risk/validation/r001/score.json) | validation | 87/120 | 72.50% | 2171.65 | 3.533 | 288 | 9166 | 4990.0 | 17 |
| [e14_review_off/r001](../../runs/e14_review_off/validation/r001/score.json) | validation | 79/120 | 65.83% | 2287.65 | 2.758 | 195 | 6685 | 4989.8 | 25 |
| [e15_review_selected/r001](../../runs/e15_review_selected/test/r001/score.json) | test | 181/240 | 75.42% | 2707.85 | 3.629 | 598 | 18591 | 4990.3 | 28 |
