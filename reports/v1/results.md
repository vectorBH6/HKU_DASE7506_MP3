# 自动汇总的官方实验成绩

来源为每次实验保存的完整官方 score.json；所有准确率均要求三实例全部通过。

| 实验 | 划分 | 答对/总数 | 准确率 | 预测秒 | 评分秒 | 平均工具次数 | 模型调用 | 生成 token | 峰值 RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [a01_no_hints/r001](../../runs/a01_no_hints/validation/r001/score.json) | validation | 48/120 | 40.00% | 0.54 | 2.02 | 1.517 | 0 | 0 | 35.6 |
| [a02_no_variants/r001](../../runs/a02_no_variants/validation/r001/score.json) | validation | 73/120 | 60.83% | 1.01 | 2.26 | 2.467 | 0 | 0 | 35.2 |
| [a03_no_retrieval/r001](../../runs/a03_no_retrieval/validation/r001/score.json) | validation | 82/120 | 68.33% | 540.44 | 3.17 | 2.517 | 73 | 1937 | 5018.2 |
| [e00_draft/r001](../../runs/e00_draft/test/r001/score.json) | test | 88/240 | 36.67% | 0.01 | 2.72 | 0.000 | 0 | 0 | 32.1 |
| [e00_draft/r001](../../runs/e00_draft/validation/r001/score.json) | validation | 47/120 | 39.17% | 0.01 | 1.37 | 0.000 | 0 | 0 | 31.6 |
| [e01_feedback_cpu/r001](../../runs/e01_feedback_cpu/test/r001/score.json) | test | 158/240 | 65.83% | 2.00 | 4.50 | 2.750 | 0 | 0 | 38.6 |
| [e01_feedback_cpu/r001](../../runs/e01_feedback_cpu/validation/r001/score.json) | validation | 73/120 | 60.83% | 0.99 | 2.20 | 2.467 | 0 | 0 | 35.3 |
| [e02_qwen3_base/r001](../../runs/e02_qwen3_base/test/r001/score.json) | test | 158/240 | 65.83% | 895.50 | 6.69 | 3.517 | 182 | 6807 | 5018.3 |
| [e02_qwen3_base/r001](../../runs/e02_qwen3_base/validation/r001/score.json) | validation | 72/120 | 60.00% | 387.20 | 4.33 | 3.075 | 82 | 3131 | 5017.9 |
| [e03_qwen3_lora/r001](../../runs/e03_qwen3_lora/test/r001/score.json) | test | 175/240 | 72.92% | 658.84 | 4.50 | 3.442 | 169 | 4879 | 5018.7 |
| [e03_qwen3_lora/r001](../../runs/e03_qwen3_lora/validation/r001/score.json) | validation | 81/120 | 67.50% | 363.35 | 2.67 | 2.883 | 71 | 2123 | 5018.2 |
