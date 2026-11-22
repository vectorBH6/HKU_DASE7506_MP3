# V5 实验：搜索控制器与多轮 LoRA-GRPO

当前最佳 **184/240（76.67%）**，相对V4b为+0题、+0.00个百分点。默认方案：`contract_v4`；adapter：`artifacts/adapters/onpolicy_v001/epoch_02`。

所有正式结果由原样官方predict/score在CPU四线程完成，逐题三个实例全部通过才计分。最终选择参考公开test聚合分数；两组挑战者均在test运行前锁定，不将这里的选择宣称为独立隐藏测试评估。

## CPU四组对照

| 实验 | CPU验证 | 模型生成 | 预测秒 |
|---|---:|---:|---:|
| A_old_weights_old_arch | 92/120 | 278 | 1480.49 |
| B_old_weights_search | 93/120 | 371 | 2467.19 |
| C_grpo_weights_old_arch | 92/120 | 259 | 1694.19 |
| D_grpo_weights_search | 92/120 | 370 | 2389.20 |

## GPU开发消融

| 架构变体（旧权重） | 验证正确数 | 模型生成 |
|---|---:|---:|
| 拒绝后的检索恢复 | 94/120 | 322 |
| 额外独立复查 | 91/120 | 462 |
| 复查加JOIN探查 | 91/120 | 458 |
| 始终复查 | 94/120 | 334 |
| 提示候选复查 | 95/120 | 367 |

| GRPO检查点 | 旧架构GPU验证 | 新架构GPU验证 |
|---|---:|---:|
| 8组有效更新 | 91/120 | 92/120 |
| 16组有效更新 | 91/120 | 92/120 |
| 32组有效更新 | 88/120 | 88/120 |
| 64组有效更新 | 91/120 | 90/120 |

GPU仅用于开发筛选。五个架构对照分别为：拒绝重复后的有界检索恢复、额外独立语义复查、聚合/JOIN探查、取消低风险原稿匹配后的模型跳过，以及提交前获取并检查不同提示候选。GPU筛选结果及CPU矩阵均保留，不能把GPU速度当作正式CPU复现成本。独立实验可能并行，墙钟成本包含当时的机器竞争。

GPU开发前两名另做CPU确认，再确定架构。协调器替换曾导致两项验证失去监控，原记录标为interrupted并保留；两项均用标准运行器重新完成，不采用不完整运行。第8次GRPO更新重跑前后的完整预测和工具轨迹一致，见v5_interruption_audit.json。

## 预锁定后的完整测试

| 方案 | 正确题数 | 相对184：改善/退步 | 模型生成 | 预测秒 | RSS MiB |
|---|---:|---:|---:|---:|---:|
| V4b基线 | 184/240 | — | 601 | 2481.95 | 4777.36 |
| architecture | 182/240 | 2/4 | 784 | 4189.65 | 4087.27 |
| grpo | 182/240 | 3/5 | 577 | 3069.85 | 3982.05 |

## Train候选覆盖诊断

当前旧策略在480道train题上独立采样1920条真实轨迹；其中至少一条最终正确的题有459道，已check候选中至少一个正确的题有461道。混合成功/失败组95个，全部失败组21个。这里只是train采样覆盖诊断，不是test上限，也不是部署时允许运行四套完整工具预算。

完整诊断中，20道train题没有调用模型，均未出现全错；零模型调用的全错任务数为0。因此早期把少工具调用推断成‘没有模型参与导致失败’不成立。该结论限于train诊断，不能推断测试同样如此。

训练诊断实例：`train-1a1ca6ce36878e21`只问年龄最大主持人的姓名和国籍，四条轨迹都保留了原稿中多余的party_host关联，全部失败；参考SQL只需`SELECT Name, Nationality FROM host ORDER BY Age DESC LIMIT 1`。这表现为正确候选生成不足。四条奖励全0时，组内优势无法告诉模型应删除哪段SQL。完整train实例见v5_train_failure_cases.json；它们只用于事后诊断，没有追加训练或改写推理规则。

## GRPO训练记录

初始化为旧onpolicy_v001/epoch_02；同一题每组4条独立完整轨迹，最多尝试320组、最多64次有效更新。实际尝试264组、覆盖250道不同train题，64组奖励有差异并更新，每组做2次clipped优化，共128个优化器步；训练4800.57秒，CUDA最大allocated 2.89GiB。

组内优势按二元终局奖励标准化；所有train三个实例通过奖励1，否则0。同分组跳过更新，没有对工具调用次数提供正奖励。clip=0.2，KL beta=0.03，lr=5e-06。冻结初始化SFT模型为参考策略。采样temperature=1、top_p=1、top_k=0，生成概率与策略损失一致；所有提示、工具反馈、历史token均不计入动作损失，拒绝提议的生成仍计数并参与所属轨迹的信用分配。

仅指定Qwen3-0.6B基座的LoRA A/B可训练；没有价值头、embedding、lm_head、bias或额外可训练模块。参考模型只在训练中存在；正式推理仅加载一个未合并模型，总计606,142,464参数。训练过程中关闭dropout以保证旧/新策略概率比可比，小规模验证的首轮最大log-ratio差异为0。

课程采样偏好由train失败轨迹决定，并固化成配置中的hard_task_ids；不会随诊断文件后续写入而改变。train三实例评分仅在离线训练奖励侧执行，参考SQL、额外实例和指纹均不出现在actor观察、提示或部署文件中。validation/test不进入训练、采样课程或梯度。

## 实际工具能力与局限

- architecture：模型自主动作`{"repair_hint": 154, "check": 209, "submit": 180}`；全部官方动作`{"check": 536, "schema": 229, "repair_hint": 229, "submit": 240, "retrieve": 37}`；公开匹配却最终错误28题，重复提议152次。
- grpo：模型自主动作`{"repair_hint": 128, "check": 184, "submit": 171}`；全部官方动作`{"check": 492, "schema": 229, "repair_hint": 133, "submit": 240}`；公开匹配却最终错误28题，重复提议91次。

## 固定评测后的案例

以下案例只用于事后分析，不进入训练、提示或控制器规则。

**architecture 改善：test-437441d03fd77a78** — Which grade has the most high schoolers?

基线：
```sql
SELECT grade FROM Highschooler ORDER BY grade DESC LIMIT 1;
```
挑战者：
```sql
SELECT grade FROM Highschooler GROUP BY grade ORDER BY COUNT(*) DESC LIMIT 1
```

**architecture 退步：test-cd72991919cf7fa2** — Find the id of the pet owned by student whose last name is ‘Smith’.

基线：
```sql
SELECT Has_Pet.PetID
FROM Has_Pet
JOIN Student ON Has_Pet.StuID = Student.StuID
WHERE Student.LName = 'Smith';
```
挑战者：
```sql
SELECT PetID FROM pets WHERE pet_age = 10
```

**grpo 改善：test-574711dd6944dfd6** — List the name of the conductor with the most years of work.

基线：
```sql
SELECT conductor.Name
FROM conductor
WHERE conductor.Conductor_ID IN (
    SELECT conductor.Conductor_ID
    FROM conductor
    JOIN performance ON conductor.Conductor_ID = performance.Orchestra_ID
    JOIN orchestra ON performance.Orchestra_ID = orchestra.Orchestra_ID
    GROUP BY conductor.Conductor_ID
    ORDER BY conductor.Age DESC
    LIMIT 1
);
```
挑战者：
```sql
SELECT T1.Name FROM conductor AS T1 JOIN orchestra AS T2 ON T1.Conductor_ID = T2.Conductor_ID GROUP BY T1.Name ORDER BY MAX(T1.Year_of_Work) DESC LIMIT 1
```

**grpo 退步：test-3cc4ebb971dc3959** — Show the names of students who have a grade higher than 5 and have at least 2 friends.

基线：
```sql
SELECT T1.name FROM Highschooler AS T1 JOIN Friend AS T2 ON T1.id = T2.student_id WHERE T1.grade > 5 GROUP BY T1.id HAVING COUNT(T2.friend_id) >= 2
```
挑战者：
```sql
SELECT name FROM Highschooler WHERE grade > 5 GROUP BY name HAVING COUNT(*) >= 2
```

**最佳方案仍然失败：test-167f97f7d5a2eda6** — Which professionals have operated a treatment that costs less than the average? Give me theor first names and last names.

```sql
SELECT DISTINCT first_name, last_name FROM Professionals JOIN Treatments ON Professionals.professional_id = Treatments.professional_id WHERE cost_of_treatment < (SELECT AVG(cost_of_treatment) FROM Treatments)
```

最终SQL通过公开check，但三个实例的通过标记为`[True, False, True]`。这是有限公开反馈不能证明最终正确的实际反例，不能仅凭该标记确定全部语义根因。


控制器在重复拒绝后触发的retrieve、在提交前触发的multiplicity execute，与Qwen自主选择动作分开统计。调用过工具不能证明学会有效规划；公开库的探查也无法证明额外实例上的语义正确。GRPO同分组没有区分正确策略的信号，且固定控制器跳过模型的路径无法依靠模型梯度修复。

本轮二元终局奖励没有区分模型修复与控制器回退的贡献：如果不合格提议之后的回退恰好成功，整条轨迹仍获奖励。因此不能仅凭训练成功率声称模型学会了SQL推理、自我反思或主动验证；需要结合拒绝率、候选覆盖和真正自主动作检查。

训练集只有480题且schema与验证/测试不同；奖励增长和训练loss变化不能替代正式成绩。已进行多轮公开验证与公开test自评，存在选择偏差。固定三实例评分仍是有限检验，不能宣称SQL对任意数据库都等价。

## 后续实验优先级（尚未运行）

1. 先补强正确SQL候选生成。仅从train轨迹整理失败SQL、题意、schema和真实反馈，以train参考答案及三个实例检验得到修正目标；重点覆盖多余JOIN、聚合粒度、条件丢失和去重。保留普通成功样本混合训练，避免只学难题后破坏已有能力。四条采样全错的题优先用监督信号补足，暂不直接扩大同类GRPO训练时长。

2. 再做候选比较的受控训练。用train中“公开check匹配但额外实例失败”与“三实例通过”的候选构造对比，让同一个LoRA模型选择候选并指出可由题意/schema支持的差异；标签只由train离线评分产生。不能只要求模型多反思一次，也不能把缓存提示当答案。新增比较仍计入正式六次生成和八次工具预算。

3. 最后尝试小规模GRPO复验。先检查组内奖励是否有差异，以及成功是否来自模型提议或控制器回退；候选覆盖改善后再比较GRPO是否带来额外收益。以train内部按schema留出的开发集初筛，官方validation确认，锁定一个候选后再做完整CPU测试。上述方向是基于本轮记录的待验证假设，不代表已取得提分。

## 合规、风险与复现

官方497个文件、基座9个文件、根README.md及README_cn.md核验不变。仍只使用一个.venv，没有安装TRL或升级依赖，没有peer_reviews、dist或上传。所有正式结果在8次收费工具、6次模型生成、每次2048新token、7200秒预测、180秒评分、16GiB RSS限制内；模型加载一次，单题记忆重新创建。

问题、schema、检索内容、缓存提示均为不可信输入。固定system规则、只读SQL提取、动作字段校验、重复保护、预算限制及官方执行器继续约束行为；未完成独立对抗提示注入评测，不声称完全防御。检索可能引入不相关标识符，复查也可能强化同一模型的错误判断。

Codex辅助设计、工程、实验与报告；正式推理不联系外部模型。参考DeepSeekMath（https://arxiv.org/abs/2402.03300）、ReEx-SQL（https://arxiv.org/abs/2505.12768）、CHASE-SQL（https://arxiv.org/abs/2410.01943）；只借鉴方法，不移用其成绩。保留NOTICE归属。

[复现指南](../../submission/REPRODUCE_V5.md) · [CPU矩阵](v5_cpu_matrix.json) · [预锁定记录](v5_test_lock.json) · [完整结果](v5_results.json) · [运行核验](v5_runtime_audit.json) · [train覆盖](v5_train_coverage.json) · [权重核验](v5_grpo_artifact_audit.json) · [未来源码清单](v5_delivery_manifest.json)
