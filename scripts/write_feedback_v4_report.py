"""Write the final v4 report and selected inference reproduction instructions."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    data=json.loads((ROOT/'reports/v4_results.json').read_text());selection=json.loads((ROOT/'reports/v4_selection.json').read_text())
    audit=json.loads((ROOT/'reports/v4_data_audit.json').read_text());provenance=json.loads((ROOT/'artifacts/prepared_data/feedback_v4/provenance.json').read_text())
    results=data['CPU_results'];old=next(r for r in results if r['experiment']=='e22_recovery_selected' and r['split']=='test');new=next(r for r in results if r['experiment']=='e35_feedback_selected' and r['split']=='test')
    pair=next(p for p in data['paired_against_v3'] if p['experiment']=='e35_feedback_selected' and p['split']=='test')
    diagnostic=provenance['diagnostic']
    matrix=json.loads((ROOT/'reports/v4_cpu_matrix.json').read_text())
    synthesis=json.loads((ROOT/'reports/v4_synthesis_data_audit.json').read_text())
    lines=['# 第四轮：反馈训练、结构诊断与通用约束修正','',
        f"完整CPU测试 **{new['correct']}/240（{new['accuracy']:.2f}%）**，第三轮 **{old['correct']}/240（{old['accuracy']:.2f}%）**，变化 **{new['correct']-old['correct']:+d}题、{new['accuracy']-old['accuracy']:+.2f}个百分点**。",'',
        f"按CPU验证集锁定 `{selection['selected_experiment']}`；默认配置{'已升级' if selection['promote'] else '保留第三轮'}。新测试成绩没有参与选型。参见 [事先方案](v4_plan_cn.md)、[不可覆盖的选择记录](v4_selection.json)。",'',
        '## 先诊断失分来源','',
        f"用第三轮策略在480道train采集真实动作：最终通过 {diagnostic['final_correct']}题；已检查候选中存在三实例正确SQL的有 {diagnostic['checked_candidate_oracle_correct']}题。因此选择损失只有 {len(diagnostic['selection_losses'])}题，另有 {len(diagnostic['no_correct_checked_candidate'])}题没有生成正确的已检查候选，主要投入放在生成与反馈修复。采集使用GPU开发推理，CPU正式结果另列。",'',
        '## 本轮改动','',
        '1. `feedback_v4`：在第三轮协议保护和有限恢复上增加SQL AST、聚合/连接/过滤/投影摘要，以及schema绑定诊断。绑定检查对SQLite双引号文本值保留歧义提示；全部诊断均是辅助证据，不是语义正确性标签。',
        '2. 训练课程：当前策略真实调用与拒绝状态，加上train gold派生的边界、去重、过滤、投影和排序扰动，全部使用真实工具反馈。候选数量动态确定，通过原样官方train scorer重放并检查三个实例。',
        '3. 新目标先输出短reason_code再输出SQL/候选ID，诊断标签取自train gold AST差异和真实错误。部分上下文随机排列schema/候选摘要，ID绑定与官方观察、轨迹不变。',
        '4. `semantic_v4`独立对照：根据公开示例列名对齐投影，并处理题目显式distinct/unique要求；最多两次额外check，计入原8次工具预算。不是任务ID答案映射，也不读取评分参考。',
        '5. `contract_v4`保留第三轮模型提示词，仅启用投影/显式去重修正，隔离诊断提示的影响；`constraint_v4`进一步比较题意明确的数值边界和COUNT(column)去重。规则只生成通用候选，仍需真实check，最多三次规则检查共享原工具预算。',
        '6. 追加SQL合成课程：每个train题的辅助plan+SQL生成任务，加上真实失败状态修复与旧样本保留。辅助任务不是伪造的工具轨迹；plan只是提议，不能证明SQL正确。`plan_v4`/`plan_constraint_v4`比较在同一JSON中先写短plan再写SQL，进入官方动作前去掉plan。',
        '7. 继续复用单个未合并Qwen与LoRA，所有诊断、修复和选择的模型生成计入同一上限。', '',
        f"训练数据 **{audit['records']}条，480个train任务，零validation/test训练样本**；最长目标 {audit['max_target_tokens']}token。数据SHA-256：`{audit['prepared_sha256']}`。480道gold均通过三个train实例。原始采集、扰动与探查均为真实执行；这是专家增强的监督动作训练，不是强化学习。来源见 [数据核验](v4_data_audit.json)。", '',
        f"追加课程 **{synthesis['records']}条、{synthesis['unique_train_tasks']}个train题**，最长目标{synthesis['max_target_tokens']}token，推理生成上限512；其中480条是根据训练gold AST构造的辅助SQL合成任务，其他修复状态来自真实反馈。数据与初始5871条课程分别核验，均无validation/test标签。见[追加课程审计](v4_synthesis_data_audit.json)。",'',
        '## 训练记录','', '| 训练版本 | 轮数 | 使用样本 | 优化步 | 秒 | 最后轮loss |','|---|---:|---:|---:|---:|---:|']
    for name in ['feedback_v4','feedback_v4_extra']:
        p=ROOT/'runs/training'/name/'training_metadata.json'
        if not p.exists():continue
        m=json.loads(p.read_text())
        if m['status']=='completed':lines.append(f"| {name} | {m['config']['epochs']} | {m['examples_used']} | {m['steps']} | {m['seconds']:.2f} | {m['final_epoch_mean_loss']:.4f} |")
    lines += ['', '两套课程均从第二轮onpolicy_v001/epoch_02独立初始化；feedback_v4训练两轮，feedback_v4_extra训练一轮，合计三个新检查点，不是连续追加第三轮。只训练LoRA A/B，使用目标token交叉熵，问题、schema、工具反馈和padding均不计loss。不同课程的loss不能直接横向比较，loss下降也不能替代验证集与测试集成绩。所有检查点与训练日志分版本保留。', '',
              '## 正式CPU对照','', '| 实验 | 划分 | 正确 | 准确率 | 模型调用 | 预测秒 |','|---|---|---:|---:|---:|---:|']
    for r in results:lines.append(f"| [{r['experiment']}/{r['run']}](../{r['source']}/score.json) | {r['split']} | {r['correct']}/{r['n']} | {r['accuracy']:.2f}% | {r['model_calls']} | {r['prediction_seconds']:.2f} |")
    lookup={r['experiment']:r for r in results if r['split']=='validation'}
    base=lookup['e20_recovery4']['correct']
    lines += ['', '核心对照结论：']
    for exp,label in [('e31_feedback_old_weights','仅丰富诊断'),(matrix['old_controller_new_weights'],'仅替换新权重'),('e33_feedback_new_weights','诊断与新权重组合'),('e37_contract_old_weights','保留旧提示并增加基础约束'),('e39_constraint_old_weights','增加显式数值/去重计数约束')]:
        if exp in lookup:
            r=lookup[exp];lines.append(f"- {label}：{r['correct']}/120，相对89题为{r['correct']-base:+d}题。")
    lines += ['', '这些结果用于区分提示、权重和规则的收益，不能将组合的全部提升归因于LoRA。GPU预算增至6次、优先新候选等未进入正式选型的对照也完整列出。']
    lines += ['', f"四个核心条件为第三轮旧控制器/旧权重、e31新诊断控制器/旧权重、{matrix['old_controller_new_weights']}旧控制器/新权重、e33新诊断控制器/新权重。e36为丰富诊断+基础约束/旧权重，e37为旧提示+基础约束/旧权重，e39为旧提示+扩展约束/旧权重；e34及可选e40/e41是GPU筛选后的完整CPU确认。旧控制器实现保留原样，入口仅增加新分支。", '',
              '日志身份说明：decisions.jsonl的controller字段沿用recovery_v3基类标签，实际策略以run_metadata.json中的config.controller和对应源码哈希为准；模型身份则来自每次generation的adapter SHA。', '',
              '每个CPU预测进程固定四核、workers=1。e36使用CPU8–11，其余核心对照使用P核组；独立实验和GPU训练存在并发，耗时只作本机测量，不据此宣称严格速度提升。e31/r001因SQLite双引号误导性警告而中止，不计正式成绩。', '',
              '## GPU开发筛选','', 'GPU筛选只用于缩小验证候选，不作为正式CPU成本或教师复现成绩。同一配置已观察到GPU/CPU相差1题，因此第一阶段最高分提前CPU确认，最后再确认GPU最高分附近尚未完整CPU评测的候选。全部已启动的正式CPU验证完成后才锁定，所有开发结果保留。','',
              '| 实验 | 轮数 | 正确/120 | 模型调用 |','|---|---:|---:|---:|']
    for r in data['GPU_development']:lines.append(f"| {r['experiment']} | {r.get('epoch','-')} | {r['correct']}/120 | {r['calls']} |")
    gpu={r['experiment']:r for r in data['GPU_development']}
    observations=[]
    if 'dev_v4_e1_semantic' in gpu and 'dev_v4_e2_semantic' in gpu:
        observations.append(f"同一semantic控制器下，第一轮/第二轮检查点的GPU验证为{gpu['dev_v4_e1_semantic']['correct']}/120与{gpu['dev_v4_e2_semantic']['correct']}/120，继续训练没有改善该组合。")
    extra_scores=[r['correct'] for r in data['GPU_development'] if r['experiment'].startswith('dev_v4_extra_')]
    if extra_scores:
        observations.append(f"追加SQL合成课程的四组GPU开发分数为{extra_scores}，最高{max(extra_scores)}/120，未超过第一套课程的最佳组合，因此没有以最新训练版本替换已选候选。")
    lines += ['', '开发对照的结论：'+''.join(observations)+'训练/推理上下文差异和模型容量可能影响迁移，但本轮没有把原因逐项隔离，不能把退步确定归因于某一个因素。以上均为GPU筛选，正式结论以CPU验证和锁定测试为准。']
    lines += ['', '## 测试变化与成本','',f"逐题改善 {len(pair['wins'])}题，退步 {len(pair['losses'])}题，净变化 {pair['net_correct']:+d}题。",'',
              '| 指标 | 第三轮 | 本轮锁定方案 |','|---|---:|---:|']
    for label,key in [('模型调用','model_calls'),('输入token','input_tokens'),('输出token','output_tokens'),('重复提议','duplicate_proposals'),('公开匹配仍最终错误','public_match_final_failed'),('原稿公开匹配仍最终错误','original_match_final_failed'),('公开匹配原稿被改写','original_match_final_changed'),('预算未尽回退失败','early_fallback_failures'),('平均收费工具','mean_tool_calls'),('CPU预测秒','prediction_seconds'),('评分秒','scoring_seconds'),('CPU秒','cpu_seconds'),('峰值RSS MiB','peak_rss_mib')]:
        vals=[f'{r[key]:.2f}' if isinstance(r[key],float) else str(r[key]) for r in [old,new]];lines.append(f'| {label} | {vals[0]} | {vals[1]} |')
    lines += ['',f"本轮实测每题最多 {new['max_model_calls']}次生成、单次最多 {new['max_new_tokens']}新token、最多 {new['max_tool_calls']}次收费工具。模型自主接受动作：`{json.dumps(new['model_selected_actions'],ensure_ascii=False)}`。静态修正与回退不计入模型自主动作。",'']
    static={k:v for k,v in new['guard_reasons'].items() if k.startswith('static_')}
    lines += [f"全部官方工具/提交次数：`{json.dumps(new['tool_actions'],ensure_ascii=False)}`；静态约束决策：`{json.dumps(static,ensure_ascii=False)}`。这是混合控制器，自动原稿检查/schema/规则与Qwen自主提议分别统计；不能把全部工具动作都归因于模型。",'']
    unused=[name for name in ['execute','retrieve'] if not new['model_selected_actions'].get(name)]
    if unused:
        lines += ['能力边界：本轮训练包含execute/retrieve课程，但所选测试策略没有自主使用`'+'`、`'.join(unused)+'`。因此本次成绩不能证明已经学会主动值探查或检索规划；工具允许调用与模型实际学会调用是两回事。','']
    tasks={r['id']:r for r in map(json.loads,(ROOT/'mp3_sql/data/test.jsonl').read_text().splitlines())}
    oldp={r['id']:r for r in json.loads((ROOT/old['source']/'predictions.json').read_text())['predictions']};newp={r['id']:r for r in json.loads((ROOT/new['source']/'predictions.json').read_text())['predictions']}
    for label,ids in [('改善案例',pair['wins'][:2]),('退步案例',pair['losses'][:2])]:
        for key in ids:
            lines += [f"**{label} {key}**：{tasks[key]['question']}",'','第三轮：','```sql',oldp[key]['sql'],'```','本轮：','```sql',newp[key]['sql'],'```','']
    lines += ['## 实际推理流程','',
              f"所选控制器为`{new['config']['controller']}`，加载单一adapter。模型直接生成SQL及下一步动作，Python按协议接纳动作；通用规则候选与模型候选都通过同一官方工具验证。",'',
              '```mermaid','flowchart TD','  A[题目、issue_sql、公开示例] --> B[原样check issue_sql]',
              '  B --> C[真实history与剩余预算]','  C --> D{有未验证的通用约束候选?}',
              '  D -->|是且有工具预算| E[官方check，更新history]','  E --> C',
              '  D -->|否| F{可直接提交或预算耗尽?}','  F -->|是| Z[选择已检查SQL并submit]',
              '  F -->|否| G[需要时调用schema]','  G --> H[Qwen根据真实反馈提出动作]',
              '  H --> I{格式、只读、去重与预算保护}','  I -->|接纳工具动作| J[官方execute/check/hint/retrieve]',
              '  J --> C','  I -->|接纳提交| Z','  I -->|拒绝| K[内部记录拒绝，最多立即再试一次]',
              '  K --> H','  K -->|恢复或生成预算耗尽| L[有界回退与候选选择]','  L --> C','```','',
              'schema提供结构；check只比较公开实例；execute提供真实查询结果；repair_hint提供不保证正确的缓存建议；retrieve只返回官方训练记忆。允许某动作不代表本次已学会使用，实际自主动作次数以上表为准。规则检查、模型查询都计入8次收费工具，拒绝提议也计入模型生成次数。', '',
              '低风险原稿仍可能在公开匹配后直接提交；其余情况下也没有办法在推理时读取另外两个评分实例。内部reason_code、plan或静态绑定成功均不能代替验证。']
    lines += ['## 合规与局限','',
              '官方框架、预测器、评分器、数据、观察与工具轨迹保持原样，数据库仅通过官方工具访问。离线推理只加载一个Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca与一个未合并LoRA，共606,142,464参数。所有规划生成计数，最多6次、每次不超过2048新token。正式预测四线程、16GiB RSS、两小时，评分180秒。', '',
              f"选中adapter：`{selection['adapter_path']}`，SHA-256 `{selection['adapter_sha256']}`。本轮训练仅A/B矩阵，基座、tokenizer、embedding与lm_head均未更新。导出只含所选adapter的两个文件，基座不导出。",'',
              '问题、schema文本、提示和检索内容均为不可信数据。固定system规则、只读SQL、协议保护、去重与预算检查继续生效；没有独立对抗注入评测，不宣称注入防御已验证。静态诊断、规则和模型自评均不能证明任意数据库上的语义正确，公开匹配假阳性仍按完整评分统计。', '',
              '训练只来自train，validation用于开发筛选和选型，test只在锁定后运行并用于报告分析。单种子与公开固定划分不构成普遍泛化或统计显著性结论。Codex辅助实现、数据课程、训练、分析与报告；正式推理不联系外部模型或服务，NOTICE归属保留。', '',
              '启发来源：[CHESS](https://arxiv.org/abs/2405.16755)的生成/评审分工、[CHASE-SQL](https://arxiv.org/abs/2410.01943)的候选生成与选择、[ReEx-SQL](https://arxiv.org/abs/2505.12768)的执行反馈训练、[STaR](https://arxiv.org/abs/2203.14465)的验证后迭代训练。这里是受预算限制的适配，不是论文全套系统复现。', '',
              '[追加权重核验](v4_synthesis_verification.json) · [完整结果](v4_results.md) · [复现说明](../submission/REPRODUCE_V4.md) · [运行审计](v4_runtime_audit.json) · [产物核验](v4_artifact_verification.json)。所有原始日志本地保留，后续清单提供无损压缩副本。仍共用一个.venv，未创建peer_reviews或dist，未上传；本机成绩未经过教师核验。']
    (ROOT/'reports/v4_report.md').write_text('\n'.join(lines)+'\n')
    adapter=selection['adapter_path']
    repro=f'''# 第四轮复现

本轮CPU成绩见[报告](../reports/v4_report.md)，配置与权重在[选择记录](../reports/v4_selection.json)锁定。新机器先解压原始官方学生包，再将学生源码按导出清单保留相对目录放到包根目录（含mp3_sql），官方文件以原包为准。所有命令从该根目录执行，共用.venv。

## 仅复现已选Agent成绩

按[学生README](README.md)运行bash scripts/setup_env.sh --cpu准备CPU环境，再运行python scripts/download_model.py准备固定版本基座；这两步允许联网，正式预测离线。将所选adapter两文件放到 `{adapter}`。不需要先重新训练历史adapter。

```bash
unset PYTHONPATH
source .venv/bin/activate
export MP3_BASE_MODEL="$PWD/artifacts/models/Qwen3-0.6B"
export MP3_ADAPTER_DIR="$PWD/{adapter}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m unittest discover -s tests -v
python scripts/run_experiment.py --experiment reproduction_v4 --split test \\
  --config submission/configs/feedback_v4_selected.json
```

使用独立递增目录，不覆盖原始分数。新运行的predictions.json与score.json由官方程序生成，不修改。CPU四线程、workers1、所有模型调用记录并监测预算。

## 从训练起复现本轮

训练初始化需要先按[第一轮](../reports/report.md)、[第二轮](REPRODUCE_V2.md)生成onpolicy_v001/epoch_02；推理复现只需要上面的所选adapter。训练需本机GPU，正式评测仍CPU。

```bash
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/onpolicy_v001/epoch_02"
python scripts/run_experiment.py --experiment e30_feedback_collection --split train \\
  --config submission/configs/recovery_v3_selected.json --device cuda
python training/prepare_feedback_v4.py \\
  --source-run runs/e30_feedback_collection/train/r001 \\
  --output-dir artifacts/prepared_data/feedback_v4
python training/finalize_feedback_v4.py
python training/train_lora.py --config training/configs/lora_feedback_v4.json \\
  --run-name feedback_v4
```

同名数据/训练目录拒绝覆盖。已有运行时应使用新的实验和训练版本名，并同步配置路径；原始数据SHA见报告。初始数据含5871条train样本，诊断先于SQL，保留旧SQL/动作样本；没有validation/test训练标签。追加课程独立生成2740条样本，并重新从旧adapter初始化单轮LoRA：

```bash
python training/prepare_synthesis_v4.py
python training/train_lora.py --config training/configs/lora_synthesis_v4.json \
  --run-name feedback_v4_extra
```

辅助SQL合成任务不声称是工具轨迹；真实失败修复与辅助目标在provenance中区分。追加课程配置与最终选中权重按选择记录核对，不能默认新训练版本必然胜出。

## 验证比较与选型

GPU开发筛选脚本为scripts/screen_feedback_v4.py，追加训练及四项开发对照由scripts/train_synthesis_v4.py记录，正式CPU核心对照使用recovery_v3_selected.json与feedback_v4.json并分别加载旧/新adapter。semantic_v4.json、contract_v4.json和constraint_v4.json比较提示与规则收益，plan_v4.json/plan_constraint_v4.json比较追加合成课程。完整命令、硬件分配、源码哈希和adapter身份记录在每个runs目录的run_metadata.json与generations.jsonl。

python scripts/analyze_feedback_v4.py 汇总已完成正式结果；scripts/select_feedback_v4.py只可执行一次且拒绝覆盖选择记录。复现现有方案不应重新执行选择脚本，也不应根据test结果更改配置。

原始预测/评分、完整提示/生成、拒绝/恢复、工具轨迹、资源与训练记录均保留。未来导出清单见../reports/v4_delivery_manifest.json；不包含数据集、基座、.venv、缓存或未选权重。未创建dist或peer_reviews，也未上传。
'''
    (ROOT/'submission/REPRODUCE_V4.md').write_text(repro)
    print('Wrote fourth-round report and reproduction instructions')

if __name__=='__main__':main()
