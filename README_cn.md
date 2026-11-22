# DASE7506 小型项目 3：SQL 修复 Agent

**个人项目 · 占课程成绩 10% · CPU 优先 · 竞争性排行榜**

项目页面与提交入口：<https://xudongwu-0.github.io/courses/dase7506/mp3/>

**提交截止时间：2026 年 11 月 30 日 23:59，北京时间（UTC+8）。同伴评审：2026 年 12 月 1—7 日，12 月 7 日 23:59（UTC+8）截止。** 本文档是完整的学生项目规范；起始项目压缩包中没有单独的指南。

## 1. 你的任务

构建一个修复 SQL 的 Agent。每道任务包含一个自然语言问题、一个数据库模式、一份不完善的 SQL 草稿和一个公开示例结果。你的 Agent 观察反馈、决定调用哪些工具、更新上下文，并提交一条最终的 SQLite 查询。你可以编写新的 SQL；**不局限于从提供的候选中选择**。

学习目标是**感知—规划—行动循环**、工具使用、上下文/记忆管理，以及对可靠性、成本和安全性的评估。最终系统必须在决策过程中使用真实的执行/检查反馈。固定答案查表、没有工具反馈循环的单次模型生成，或仅进行模型训练，都不构成完整项目。

## 2. 数据与官方指标

提供的 Spider 衍生基准包含 **480 道训练任务、120 道验证任务和 240 道测试任务**，不同划分之间的数据库模式互不重叠。

- 使用训练任务/参考资料来拟合模型、检索索引和其他学习组件。
- 使用验证集选择设计和超参数，不得将其用作额外的训练标签。
- **完整的测试评测包是公开的：** 包括输入、每道任务的全部三个数据库实例、预期结果指纹和评分程序。在本地运行官方评测并报告自己的结果。
- 只有当提交的查询在**全部三个评分数据库实例**上都返回所要求的结果时，该任务才算正确。通过一个公开示例不能保证正确。
- 官方指标：`100 x 正确测试任务数 / 240`。评分器会重放工具轨迹并检查每个实例。提交其完整的评分 JSON 和预测结果；教师可能会复现结果并检查合规性。

允许反复自我评测。这是一个**公开基准**，并不是对隐藏测试集泛化能力的声明。常规设计选择应使用验证集。不得在测试任务或其评分指纹上训练、硬编码针对具体任务的答案，或让 Agent 读取评分参考资料/额外评测数据库。额外实例和指纹仅供评分器使用；不提供测试集参考 SQL。

公开提示是所有人共享的、由未经额外训练的 LLM 生成的固定输出。它们**不是标准答案**，可能有误。`repair_hint` 读取该缓存；不会调用在线模型。更改你的提示词或工具描述**不会**重新生成缓存。使用这些提示的 CPU 控制器属于 LLM 辅助的工具工作流，但不一定是实时 LLM 规划器。

## 3. 从 CPU 开始

使用 Python 3.11 或 3.12。在解压后的压缩包顶层目录中执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m mp3_sql.check_data

python -m mp3_sql.predict --split validation --output outputs/validation.json
python -m mp3_sql.score --split validation \
  --predictions outputs/validation.json --output outputs/validation-score.json

python -m mp3_sql.predict --split test --output outputs/test.json
python -m mp3_sql.score --split test \
  --predictions outputs/test.json --output outputs/test-score.json
```

默认入口 `mp3_sql/submission.py` 只会提交原始草稿。这是可复现的**起始基线**，不是参考解，也不足以作为最终 Agent。不需要下载模型、API 密钥、PyTorch 或 GPU。

创建自己的导出目录，其中仅包含你的实现和资产，例如：

```text
submission/
  submission.py
  config.json
  report.pdf
```

你的模块必须提供 `create_agent(checkpoint=None)`。返回的对象必须提供 `act(observation)`，并且每次恰好返回一个工具动作字典。每道任务都会创建一个新的 Agent。辅助模块应使用包导入；评分运行器会将导出目录加入 `PYTHONPATH`。

```bash
PYTHONPATH="$PWD/submission" python -m mp3_sql.predict \
  --split validation --method submission/submission.py \
  --checkpoint submission/config.json --output outputs/my-validation.json
python -m mp3_sql.score --split validation \
  --predictions outputs/my-validation.json --output outputs/my-validation-score.json
```

如果你的实现不需要配置文件，则省略 `--checkpoint`。未经修改的起始入口不接受 checkpoint。

提交时，使用 `--split test`、你自己的 `--method`/可选的 `--checkpoint`，以及输出路径 `outputs/test.json` 和 `outputs/test-score.json` 运行相同命令。将完整、未经修改的 **test-score JSON** 上传到提交表单；表单会自动读取 `correct`、`n` 和 `accuracy`。不要报告验证集分数或单实例公开示例的通过率。

## 4. 工具与预算

每个 observation 包含 `task`、`issue_sql`、`example`、`history` 和剩余预算。工具结果会出现在后续 observation 中。

| 动作 | 精确返回值 | 用途 |
|---|---|---|
| 模式 | `{"tool":"schema"}` | 读取数据库模式。 |
| 执行 | `{"tool":"execute","sql":"..."}` | 执行只读 SQL，查看结果/错误。 |
| 检查 | `{"tool":"check","sql":"..."}` | 执行 SQL，并与公开示例比较。 |
| 检索 | `{"tool":"retrieve","query":"..."}` | 检索训练示例，排除当前任务。 |
| 提示 | `{"tool":"repair_hint"}` | 读取针对原始草稿的固定共享建议。 |
| 提交 | `{"tool":"submit","sql":"..."}` | 以一条 SQL 查询结束任务。 |

**每道任务有 8 次计费工具调用**；`submit` 免费。请求提示前，先执行或检查**完全相同的原始草稿**。不能重复相同的工具动作。SQL 长度限制为 16,000 个 UTF-8 字节。执行器允许只读查询，并实施自身的执行限制。

只有提供的工具可以访问基准数据库。不得绕过工具预算、编辑 observation/轨迹、直接访问数据库文件，或修改用于评分的官方环境、数据、预测器或评分器。本地实验可以使用副本，但教师会在未经修改的官方框架中重新运行你导出的 Agent。

## 5. 可以优化的内容

**允许优化运行框架：** 改进自己的控制器、工具描述/包装、提示词、工具调用决策、上下文、记忆/检索、SQL 编辑和最终答案选择。保留底层工具语义和预算。可以在训练集上训练非 LLM 组件。

**权重优化是可选的：** 如果训练 LLM，**只能对 `Qwen/Qwen3-0.6B` 使用 LoRA**；提交适配器，而不是完整或合并后的模型权重。不提供 GPU。

- 可以使用实际参数量最多为 1,000,000,000 的未经修改的公开模型进行推理，需声明其精确仓库和不可变 revision。嵌入层和任何未合并的适配器都计入这一上限。
- LoRA 训练须使用基础 revision `c1899de289a04d12100db370d81485cdf75e47ca`，对应模型为 `Qwen/Qwen3-0.6B`。
- 只能训练 LoRA A/B 矩阵。不得训练或提交嵌入层、`lm_head`、偏置、`modules_to_save`、完整模型 checkpoint、合并后的 checkpoint 或其他适配器类型。未规定 LoRA 的秩。
- 对于训练过的模型，只提交 `adapter_model.safetensors` 和 `adapter_config.json`，并附上加载配置、训练配置/日志和数据来源记录。原始基础模型/tokenizer 必须保持不变。
- 提供加载实现。教师加载的是**你提交的适配器**，不是课程参考适配器。使用本地加载，仅加载 safetensors，并设置 `trust_remote_code=False`；不接受自定义远程模型代码和 pickle checkpoint。
- 外部 LLM 可以辅助开发；须披露其使用情况。评分期间不得联系它们。任何针对具体任务生成的训练材料都只能来自训练集。
- 量化不会减少模型的实际参数量。不得在源码/配置文件中嵌入额外模型权重、测试答案或未声明的运行时依赖。

可选模型依赖位于 `requirements-model.txt`；只有选择实时模型路线时才安装。标准加载使用 `AutoTokenizer.from_pretrained(..., local_files_only=True, trust_remote_code=False)` 和 `AutoModelForCausalLM.from_pretrained(..., local_files_only=True, trust_remote_code=False, use_safetensors=True)`；对于 LoRA，使用 `PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)`。自行设计提示词和 Agent 集成。使用 `MP3_BASE_MODEL`，以及适用时的 `MP3_ADAPTER_DIR` 指定本地模型路径，并记录精确加载命令。实时模型预测使用 `--workers 1`，模型只加载一次，并在任务间复用。

### 运行与提交限制

| 路线 | 导出资产 | 复现预算 |
|---|---|---|
| 无实时 LLM | 源码/配置/非 LLM 资产最多 16 MiB | 4 个 CPU 线程，2 GiB 虚拟地址空间，含评分在内共 180 秒 |
| 一个获准的实时 LLM | 同样的 16 MiB 导出限制；单独的 LoRA 导出最多 512 MiB | 4 个 CPU 线程，16 GiB 峰值常驻内存，预测最多 2 小时，评分另有 180 秒 |

不要上传基础权重。运行前准备好声明的公开基础模型；执行/复现必须能在 Linux 上离线完成，并使用获准的工具接口。不提供或保证 GPU。如果使用实时 LLM，限制为**每道任务最多 6 次生成**，每次最多 **2,048 个新 token**，并记录提示词、生成文本、token 数量以及模型/适配器身份，供教师检查。在报告中写明实测 CPU 时间和内存。这些是复现要求，并不表示网站会执行提交程序或自动强制执行模型预算。墙钟时间不是排行榜同分时的决胜依据。

## 6. 在本地评测并提交结果

1. 使用提供的预测器/评分器，在**全部 240 道测试任务**上评测导出的 Agent。保留 `outputs/test.json` 和完整的 `outputs/test-score.json`。正确题数必须要求每道任务的全部三个实例均通过。
2. 准备一个内容不可变、可通过 HTTPS 访问的 ZIP，包含源码导出、配置、依赖列表、报告、复现命令和两个输出文件。不要包含数据集、基础权重、`.venv` 或缓存。导出内容须符合上述限制。
3. 对于 LoRA，提供单独的、内容不可变的 HTTPS ZIP，其中恰好包含两个适配器文件。报告/训练记录应放在源码导出中。无需计算或填写 ZIP 的 SHA-256 值。
4. 在项目页面选择 `outputs/test-score.json`，然后输入学号和源码 ZIP 链接。只有训练了 LoRA 才添加可选的适配器链接。在报告中说明所用模型的精确仓库、不可变 revision、实际参数量和加载方法；无需选择模型路线。页面会检查评分文件的划分、基准身份和计数，并自动提取评测指纹；这**不能证明提交的程序复现了该成绩**。
5. 点击 **Prepare score submission**，然后点击 **Submit on GitHub**，并实际创建 issue。仅准备链接不会保存提交。更新时使用相同的 GitHub 账户和学号；以新的 issue 和新的不可变资产提交改进结果。
6. 成绩以 **self-reported（自报）** 状态进入排行榜，接受教师复现、抽查和合规审查。无需事先请求教师评测。只有教师可以将结果标记为已验证或判为无效。网站本身不运行代码或模型。

在**提交截止时间之前**提交最终成绩和所有必需的资产链接。截止后的新成绩属于迟交；截止后编辑不能替换已冻结的成绩或原始源码/适配器提交。

学号、报告的成绩和评测元数据会公开；GitHub 也会保留提交账户。资产 URL 在转交 GitHub 前会被加密。截止后，教师会公布提交的源码和可选适配器链接，用于同伴评审。在评分结束前，保持它们可访问且内容不变。提交前移除凭据、私人数据和无关文件；不要在任何 URL 中包含凭据。

报告必须涵盖 Agent 循环、上下文/记忆/检索选择、**至少两组受控消融实验**、验证集与测试集结果、运行时间/工具/模型成本、失败案例、来自不可信问题/工具/检索内容的**提示注入风险**及防护措施、外部协助、数据来源和精确复现说明。如果进行了训练，附上训练记录。这些是参评资格要求，不是额外的加权评分项。

### 七天同伴评审

在 **12 月 1—7 日**期间，使用公开的不可变源码/适配器链接，在完全相同的官方项目包和评测器下复现其他同学的结果。遵循提交的命令，评测全部 240 道测试任务及每道任务的全部三个实例。

使用排行榜上的 **Peer Review Report** 按钮填写复现准确率；Markdown 说明和证据链接是可选项。创建准备好的 GitHub issue 以保存报告，或使用提供的邮件报告链接。包含目标提交编号、环境和复现命令。不要评审自己的提交。

评审期间成绩冻结；评审不是额外的调参/提交期。报告必须在 **2026 年 12 月 7 日 23:59（UTC+8）** 前提交。报告本身不会改变成绩或等级：教师检查差异并确定参评资格，然后进行最终排名。

## 7. 排名与等级

**所有方法在同一个排行榜上竞争；不设单独的模型或训练赛道。** 在符合资格的个人提交中，每位学生以截止前提交的、可复现的最佳测试准确率参与排名。实时榜单包含自报成绩；最终等级使用同伴评审后由教师确认的有效排名。**前 35% 获得 A，接下来的 50% 获得 B，最后的 15% 获得 C**。

排名使用精确的正确测试题数，而不是四舍五入后的显示值。同分采用竞赛排名（1、1、3）。对于 `N` 名符合资格的学生，A 的边界为 `ceil(0.35N)`，A 或 B 的边界为 `ceil(0.85N)`；跨越边界的并列者获得较高等级，因此实际比例可能略有变化。缺交、无效、不可复现或不合规的提交不属于有效排名项，将按课程考核规则处理。

不得伪造成绩、轨迹或模型声明。能够访问公开评测器，并不意味着可以向 Agent 泄露评分指纹、拟合测试标签、记忆测试答案或篡改评测。

## 8. 归属与诚信

Spider 衍生数据和课程修改依据 **CC BY-SA 4.0** 分发，归属信息见 `NOTICE`。Qwen 模型输出/基础模型保留各自适用的模型条款；不包含模型权重。执行框架使用 SQLite 和 SQLGlot。参见 `NOTICE` 和 `PACKAGE_MANIFEST.json`。

清单列出了确切的公开文件及其哈希。公开测试评测材料已包含在内；不包含教师解答、优化后的运行框架、训练好的适配器和内部优化结果。
