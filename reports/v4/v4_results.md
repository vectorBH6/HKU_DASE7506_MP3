# 第四轮结果：正式CPU验证与测试

全部CPU成绩由原样官方评分器重放，按每题三个实例全部通过计分。GPU开发筛选单独记录。

| 实验 | 划分 | 正确 | 准确率 | 模型调用 | 输出token | 预测秒 | RSS MiB | 公开匹配仍错 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [e20_recovery4/r001](../../runs/e20_recovery4/validation/r001/score.json) | validation | 89/120 | 74.17% | 293 | 9536 | 1869.56 | 4990.0 | 15 |
| [e22_recovery_selected/r001](../../runs/e22_recovery_selected/test/r001/score.json) | test | 182/240 | 75.83% | 610 | 19275 | 2655.31 | 4990.3 | 28 |
| [e31_feedback_old_weights/r002](../../runs/e31_feedback_old_weights/validation/r002/score.json) | validation | 83/120 | 69.17% | 281 | 8670 | 1702.42 | 4990.6 | 19 |
| [e33_feedback_new_weights/r001](../../runs/e33_feedback_new_weights/validation/r001/score.json) | validation | 89/120 | 74.17% | 265 | 9152 | 1859.59 | 4990.6 | 18 |
| [e34_feedback_variant/r001](../../runs/e34_feedback_variant/validation/r001/score.json) | validation | 95/120 | 79.17% | 254 | 8953 | 1765.64 | 4990.4 | 16 |
| [e36_semantic_old_weights/r001](../../runs/e36_semantic_old_weights/validation/r001/score.json) | validation | 87/120 | 72.50% | 269 | 8472 | 3190.34 | 4990.7 | 19 |
| [e37_contract_old_weights/r001](../../runs/e37_contract_old_weights/validation/r001/score.json) | validation | 92/120 | 76.67% | 278 | 8915 | 1488.66 | 4990.5 | 15 |
| [e38_recovery_epoch1/r001](../../runs/e38_recovery_epoch1/validation/r001/score.json) | validation | 86/120 | 71.67% | 253 | 9017 | 1432.88 | 4989.8 | 20 |
| [e39_constraint_old_weights/r001](../../runs/e39_constraint_old_weights/validation/r001/score.json) | validation | 92/120 | 76.67% | 278 | 8917 | 1840.63 | 4990.7 | 15 |
| [e35_feedback_selected/r001](../../runs/e35_feedback_selected/test/r001/score.json) | test | 180/240 | 75.00% | 552 | 19902 | 3172.87 | 4991.0 | 30 |
| [e42_contract_control/r001](../../runs/e42_contract_control/test/r001/score.json) | test | 184/240 | 76.67% | 601 | 19085 | 2481.34 | 4990.9 | 27 |
