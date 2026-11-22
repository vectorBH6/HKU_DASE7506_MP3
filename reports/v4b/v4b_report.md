# 第四轮最终结论：训练候选回退与通用约束对照

最终保留 **184/240（76.67%）**，相对上一版182/240为 **+2题、+0.83个百分点**。默认升级为旧权重contract_v4。所有成绩均是本机原样官方评分器重放、每题三个实例全部通过的自报成绩。

## 完整结果

| 方案 | CPU验证 | 完整CPU测试 | 模型调用 | 预测秒 |
|---|---:|---:|---:|---:|
| 第三轮旧方案 | 89/120（74.17%） | 182/240（75.83%） | 610 | 2655.31 |
| 新权重＋诊断＋约束（首次锁定） | 95/120（79.17%） | 180/240（75.00%） | 552 | 3172.87 |
| 旧权重＋基础约束（后续固定对照） | 92/120（76.67%） | 184/240（76.67%） | 601 | 2481.34 |

首次按CPU验证选中新权重方案95/120，但测试仅180/240，因此恢复已有182/240默认。随后仅补测此前已完成验证的旧权重约束方案：配置、提示、权重和实现原样沿用e37/r001，选择在e42测试前锁定；没有追加训练或根据逐题测试标签改写规则。见[首次选择](../v4/v4_selection.json)、[后续预锁定](v4b_selection.json)。

最终保留参考了公开test的聚合分数，这一点在[部署记录](../v4/v4_deployment.json)明确披露。作业允许重复公开自评；所有训练材料仍只来自train，额外评分实例和指纹没有进入Agent。不能将这里的最终选择宣称为完全独立的隐藏测试泛化评估。

## 哪些尝试有效，哪些没有

旧权重下，丰富诊断83/120、基础约束87/120（含诊断）与92/120（保留旧提示），表明诊断内容的增加可能造成负收益。新权重不加诊断为86/120，诊断加新权重89/120，完整组合95/120；首次测试180/240说明验证提升未稳定迁移。

两套train课程共训练三个新检查点：5871条真实反馈/专家扰动课程两轮；2740条SQL合成/短plan课程从旧adapter独立训练一轮。追加课程四个GPU开发配置最高86/120，未胜出。本轮最终没有启用这些新权重；训练投入和loss下降不能代替最终评分。完整15组GPU筛选、核心CPU对照、训练日志与数据SHA见[初次实验留档](../v4/v4_report.md)及[全部正式结果](../v4/v4_results.md)。

旧权重基础约束只增加公开示例列名对应的投影顺序、题目显式去重候选与已匹配候选选择；保留第三轮原模型提示。每个候选都通过官方check，最多两次规则检查，共享原8次收费预算。扩展数值边界/COUNT去重的验证也是92/120，没有额外收益，最终没有采用。

约束对照流程：先原样check原稿，必要时schema；基于公开题意/列名生成规则候选并check；Qwen读取问题、schema、已检查候选及真实反馈提出SQL/下一动作。拒绝提议保留最近两次单题记忆，最多立即再试一次，所有生成计数；预算耗尽后有界回退。跨题只复用模型，Agent记忆重新创建。检索沿用官方retrieve，实际使用次数见下方动作统计。

## 最终测试变化与成本

| 指标 | 第三轮 | 基础约束对照 |
|---|---:|---:|
| 正确题数 | 182 | 184 |
| 模型生成次数 | 610 | 601 |
| 输入token | 555690 | 549627 |
| 输出token | 19275 | 19085 |
| 公开匹配仍最终错误 | 28 | 27 |
| 重复提议 | 63 | 61 |
| 平均收费工具 | 3.65 | 3.65 |
| 预测秒 | 2655.31 | 2481.34 |
| 评分秒 | 4.50 | 4.53 |
| CPU秒 | 10563.34 | 9860.11 |
| 峰值RSS MiB | 4990.26 | 4990.90 |

基础约束对照改善2题、退步0题，净变化+2题。实际模型自主动作：`{"repair_hint": 154, "check": 199, "submit": 182}`。自动原稿检查、schema、静态约束与回退不计为模型自主动作。全部官方动作：`{"check": 494, "schema": 229, "repair_hint": 154, "submit": 240}`。

对照实测每题最多4次生成、单次最多174新token、最多8次收费工具。静态修正不能证明SQL语义正确；公开匹配假阳性仍按三实例结果统计。若动作统计中没有execute/retrieve，则不能宣称已学会主动值探查/检索规划。

**改善案例 test-fa097b64a47cd1c2**：What is the number of employees from each city?

第三轮：
```sql
SELECT City, COUNT(*) AS Employee_count
FROM employee
JOIN hiring ON employee.Employee_ID = hiring.Employee_ID
GROUP BY City;
```
基础约束对照：
```sql
SELECT COUNT(*), City FROM employee GROUP BY City
```

本题公开列顺序是`count(*)、City`。新轨迹先调换投影，check仍失败；读取schema/提示后，中途shop查询也被check淘汰，最终生成employee本表统计并通过三个实例。这里既修正了输出顺序，也改变了计数涉及的表；收益属于整个反馈轨迹，不能单独归因于某一次编辑或新的模型权重。

**改善案例 test-9d0333af61df8e61**：What are the opening year and staff number of the museum named Plaza Museum?

第三轮：
```sql
SELECT museum.Open_Year, museum.Num_of_Staff
FROM museum
WHERE museum.Name = 'Plaza Museum';
```
基础约束对照：
```sql
SELECT museum.Num_of_Staff, museum.Open_Year FROM museum WHERE museum.Name = 'Plaza Museum'
```

本题公开示例有0行，列名顺序却明确为`Num_of_Staff、Open_Year`。原SQL与改后的SQL在公开库都返回空表，所以两次check都显示匹配；错误列顺序被空结果掩盖。通用规则按公开列名调整输出后，最终三个实例通过。这是“公开匹配不能保证最终正确”的实际案例，Agent没有读取另外两个评分实例。

## 运行、合规与复现

最终配置：`submission/configs/contract_v4_selected.json`；adapter：`artifacts/adapters/onpolicy_v001/epoch_02`，SHA-256 `43d91619f16b1a7e4cb3358ad93cfc5f8ee447ff124640a8b5e2ff8ee44de613`。唯一基座Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca，共606,142,464实际参数（含embedding与未合并LoRA）。原基座、tokenizer、官方497个文件核验不变；训练只更新A/B。

正式CPU四核、workers1、离线，模型加载一次并跨任务复用；每题Agent实例独立记忆。问题、schema、提示、检索文本视为不可信数据，固定system规则、只读SQL、动作校验、去重和预算保护维持。未完成独立对抗注入评测，不能宣称完全防御。

Codex辅助工程、课程、实验分析与报告；正式推理不联系外部模型。NOTICE归属保留；参考思路见初次报告，最终adapter训练配置、日志与数据来源见[第二轮报告](../v2/v2_report.md)。全部共用.venv，未创建peer_reviews或dist，未上传。

新训练没有形成稳定测试提升。后续值得研究跨schema稳定的SQL生成与真正有用的工具探查；本轮结果不足以证明增加诊断文本或短plan课程能解决这些问题。

[复现指南](../../submission/REPRODUCE_V4b.md) · [后续运行审计](v4b_runtime_audit.json) · [最终权重核验](v4b_artifact_verification.json) · [初次运行审计留档](../v4/v4_runtime_audit_initial.json) · [未来导出清单](../v4/v4_delivery_manifest.json)
