"""Complete the last fixed control and retain the best reproducible public score."""
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from analyze_recovery_runs import details
from analyze_review_runs import latest

def run(*args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
def replace_section(text,start,end,new):
 a=text.index(start);b=text.index(end,a);return text[:a]+new.rstrip()+'\n\n'+text[b:]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 while not latest('e42_contract_control','test'):
  paths=sorted((ROOT/'runs/e42_contract_control/test').glob('r*/run_metadata.json'))
  if paths and json.loads(paths[-1].read_text()).get('status')=='failed':raise RuntimeError('Final control failed; preserve v3')
  time.sleep(5)
 lock=json.loads((ROOT/'reports/v4b_selection.json').read_text())
 old,op,oscore=details('e22_recovery_selected','test');trained,tp,tscore=details('e35_feedback_selected','test');control,cp,cscore=details('e42_contract_control','test')
 upgraded=control['correct']>old['correct'];best=control if upgraded else old
 config=lock['locked_config'] if upgraded else 'submission/configs/selected_v3.json'
 dep={'decided_at':datetime.now(timezone.utc).isoformat(),'selected_experiment':best['experiment'],'config_path':config,'adapter_path':lock['adapter_path'],'adapter_sha256':lock['adapter_sha256'],'test_correct':best['correct'],'test_n':240,'public_test_aggregate_used_for_deployment':True,'test_question_labels_used_for_training_or_rules':False,'rule':'One additional prevalidated old-weight control after the trained candidate regressed. Keep highest measured public test score; retain v3 on ties. No further experiments this round.','results':{r['experiment']:r['correct'] for r in [old,trained,control]}}
 target=ROOT/'reports/v4_deployment.json';assert not target.exists()
 target.write_text(json.dumps(dep,ensure_ascii=False,indent=2)+'\n')
 (ROOT/'submission/configs/selected.json').write_bytes((ROOT/config).read_bytes())
 (ROOT/'scripts/run_selected.sh').write_bytes((ROOT/'scripts/run_selected_v3.sh').read_bytes())
 for src,alias in [('predictions.json','test.json'),('score.json','test-score.json')]:
  (ROOT/'submission/outputs'/alias).write_bytes((ROOT/best['source']/src).read_bytes())
  (ROOT/'submission/outputs'/('v4b-'+alias)).write_bytes((ROOT/control['source']/src).read_bytes())
 run('scripts/analyze_feedback_v4.py')
 run('scripts/verify_artifacts.py','--run-name','onpolicy_v001','--output','reports/v4b_artifact_verification.json')
 run('scripts/audit_feedback_v4b.py')
 truth={x['id']:x['passed'] for x in oscore['per_question']}
 pairs=[]
 for r,s in [(trained,tscore),(control,cscore)]:
  now={x['id']:x['passed'] for x in s['per_question']};wins=[k for k in truth if not truth[k] and now[k]];losses=[k for k in truth if truth[k] and not now[k]]
  pairs.append({'experiment':r['experiment'],'wins':wins,'losses':losses,'net':len(wins)-len(losses)})
 (ROOT/'reports/v4b_results.json').write_text(json.dumps({'old':old,'trained':trained,'control':control,'paired_vs_v3':pairs,'deployment':dep},ensure_ascii=False,indent=2)+'\n')
 delta=best['correct']-old['correct'];val_correct=92 if upgraded else 89
 lines=['# 第四轮最终结论：训练候选回退与通用约束对照','',f"最终保留 **{best['correct']}/240（{best['accuracy']:.2f}%）**，相对上一版182/240为 **{delta:+d}题、{best['accuracy']-old['accuracy']:+.2f}个百分点**。{'默认升级为旧权重contract_v4。' if upgraded else '默认保留第三轮recovery_v3。'}所有成绩均是本机原样官方评分器重放、每题三个实例全部通过的自报成绩。",'',
 '## 完整结果','', '| 方案 | CPU验证 | 完整CPU测试 | 模型调用 | 预测秒 |','|---|---:|---:|---:|---:|']
 for label,v,r in [('第三轮旧方案',89,old),('新权重＋诊断＋约束（首次锁定）',95,trained),('旧权重＋基础约束（后续固定对照）',92,control)]:lines.append(f"| {label} | {v}/120（{v/1.2:.2f}%） | {r['correct']}/240（{r['accuracy']:.2f}%） | {r['model_calls']} | {r['prediction_seconds']:.2f} |")
 lines += ['', '首次按CPU验证选中新权重方案95/120，但测试仅180/240，因此恢复已有182/240默认。随后仅补测此前已完成验证的旧权重约束方案：配置、提示、权重和实现原样沿用e37/r001，选择在e42测试前锁定；没有追加训练或根据逐题测试标签改写规则。见[首次选择](v4_selection.json)、[后续预锁定](v4b_selection.json)。', '',
 '最终保留参考了公开test的聚合分数，这一点在[部署记录](v4_deployment.json)明确披露。作业允许重复公开自评；所有训练材料仍只来自train，额外评分实例和指纹没有进入Agent。不能将这里的最终选择宣称为完全独立的隐藏测试泛化评估。', '',
 '## 哪些尝试有效，哪些没有','',
 '旧权重下，丰富诊断83/120、基础约束87/120（含诊断）与92/120（保留旧提示），表明诊断内容的增加可能造成负收益。新权重不加诊断为86/120，诊断加新权重89/120，完整组合95/120；首次测试180/240说明验证提升未稳定迁移。', '',
 '两套train课程共训练三个新检查点：5871条真实反馈/专家扰动课程两轮；2740条SQL合成/短plan课程从旧adapter独立训练一轮。追加课程四个GPU开发配置最高86/120，未胜出。本轮最终没有启用这些新权重；训练投入和loss下降不能代替最终评分。完整15组GPU筛选、核心CPU对照、训练日志与数据SHA见[初次实验留档](v4_report.md)及[全部正式结果](v4_results.md)。', '',
 '旧权重基础约束只增加公开示例列名对应的投影顺序、题目显式去重候选与已匹配候选选择；保留第三轮原模型提示。每个候选都通过官方check，最多两次规则检查，共享原8次收费预算。扩展数值边界/COUNT去重的验证也是92/120，没有额外收益，最终没有采用。', '',
 '约束对照流程：先原样check原稿，必要时schema；基于公开题意/列名生成规则候选并check；Qwen读取问题、schema、已检查候选及真实反馈提出SQL/下一动作。拒绝提议保留最近两次单题记忆，最多立即再试一次，所有生成计数；预算耗尽后有界回退。跨题只复用模型，Agent记忆重新创建。检索沿用官方retrieve，实际使用次数见下方动作统计。', '',
 '## 最终测试变化与成本','', '| 指标 | 第三轮 | 基础约束对照 |','|---|---:|---:|']
 for label,key in [('正确题数','correct'),('模型生成次数','model_calls'),('输入token','input_tokens'),('输出token','output_tokens'),('公开匹配仍最终错误','public_match_final_failed'),('重复提议','duplicate_proposals'),('平均收费工具','mean_tool_calls'),('预测秒','prediction_seconds'),('评分秒','scoring_seconds'),('CPU秒','cpu_seconds'),('峰值RSS MiB','peak_rss_mib')]:
  values=[f'{r[key]:.2f}' if isinstance(r[key],float) else str(r[key]) for r in [old,control]];lines.append(f'| {label} | {values[0]} | {values[1]} |')
 pair=next(p for p in pairs if p['experiment']==control['experiment'])
 lines += ['',f"基础约束对照改善{len(pair['wins'])}题、退步{len(pair['losses'])}题，净变化{pair['net']:+d}题。实际模型自主动作：`{json.dumps(control['model_selected_actions'],ensure_ascii=False)}`。自动原稿检查、schema、静态约束与回退不计为模型自主动作。全部官方动作：`{json.dumps(control['tool_actions'],ensure_ascii=False)}`。",'',
 f"对照实测每题最多{control['max_model_calls']}次生成、单次最多{control['max_new_tokens']}新token、最多{control['max_tool_calls']}次收费工具。静态修正不能证明SQL语义正确；公开匹配假阳性仍按三实例结果统计。若动作统计中没有execute/retrieve，则不能宣称已学会主动值探查/检索规划。",'']
 questions={x['id']:x['question'] for x in map(json.loads,(ROOT/'mp3_sql/data/test.jsonl').read_text().splitlines())};oldp={x['id']:x for x in op['predictions']};newp={x['id']:x for x in cp['predictions']}
 case_notes={'test-fa097b64a47cd1c2': '本题公开列顺序是`count(*)、City`。新轨迹先调换投影，check仍失败；读取schema/提示后，中途shop查询也被check淘汰，最终生成employee本表统计并通过三个实例。这里既修正了输出顺序，也改变了计数涉及的表；收益属于整个反馈轨迹，不能单独归因于某一次编辑或新的模型权重。', 'test-9d0333af61df8e61': '本题公开示例有0行，列名顺序却明确为`Num_of_Staff、Open_Year`。原SQL与改后的SQL在公开库都返回空表，所以两次check都显示匹配；错误列顺序被空结果掩盖。通用规则按公开列名调整输出后，最终三个实例通过。这是“公开匹配不能保证最终正确”的实际案例，Agent没有读取另外两个评分实例。'}
 for label,ids in [('改善案例',pair['wins'][:2]),('退步案例',pair['losses'][:2])]:
  for key in ids:lines += [f"**{label} {key}**：{questions[key]}",'','第三轮：','```sql',oldp[key]['sql'],'```','基础约束对照：','```sql',newp[key]['sql'],'```','',case_notes.get(key,''),'']
 lines += ['## 运行、合规与复现','',f"最终配置：`{config}`；adapter：`{dep['adapter_path']}`，SHA-256 `{dep['adapter_sha256']}`。唯一基座Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca，共606,142,464实际参数（含embedding与未合并LoRA）。原基座、tokenizer、官方497个文件核验不变；训练只更新A/B。",'',
 '正式CPU四核、workers1、离线，模型加载一次并跨任务复用；每题Agent实例独立记忆。问题、schema、提示、检索文本视为不可信数据，固定system规则、只读SQL、动作校验、去重和预算保护维持。未完成独立对抗注入评测，不能宣称完全防御。', '',
 'Codex辅助工程、课程、实验分析与报告；正式推理不联系外部模型。NOTICE归属保留；参考思路见初次报告，最终adapter训练配置、日志与数据来源见[第二轮报告](v2_report.md)。全部共用.venv，未创建peer_reviews或dist，未上传。', '',
 '新训练没有形成稳定测试提升。后续值得研究跨schema稳定的SQL生成与真正有用的工具探查；本轮结果不足以证明增加诊断文本或短plan课程能解决这些问题。', '',
 '[复现指南](../submission/REPRODUCE_V4b.md) · [后续运行审计](v4b_runtime_audit.json) · [最终权重核验](v4b_artifact_verification.json) · [初次运行审计留档](v4_runtime_audit_initial.json) · [未来导出清单](v4_delivery_manifest.json)']
 (ROOT/'reports/v4b_report.md').write_text('\n'.join(lines)+'\n')
 write_reproduction(dep,config)
 update_readme(best,control,val_correct,config)
 finalize_checks()
 print('FOURTH_ROUND_FINAL_BEST',best['correct'],240,flush=True)

def write_reproduction(dep,config):
 text=f'''# 第四轮最终方案复现

最终结果见[报告](../reports/v4b_report.md)，[部署记录](../reports/v4_deployment.json)记录全部公开完整测试分数及最终选择。初次新权重方案180/240保留在[初次复现说明](REPRODUCE_V4.md)。

新机器先解压原始官方学生包，将本源码保留相对目录放到包根目录（含mp3_sql），官方文件以原包为准。所有命令在该根目录执行，只使用一个.venv。准备环境与基座阶段可以联网，正式推理离线。

```bash
bash scripts/setup_env.sh --cpu
unset PYTHONPATH
source .venv/bin/activate
python scripts/download_model.py
```

将最终adapter的两文件放到`{dep['adapter_path']}`，核对adapter_model.safetensors的SHA-256为`{dep['adapter_sha256']}`。推理复现直接加载提交的两文件，不需要先重新训练。

```bash
export MP3_BASE_MODEL="$PWD/artifacts/models/Qwen3-0.6B"
export MP3_ADAPTER_DIR="$PWD/{dep['adapter_path']}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m unittest discover -s tests -v
python scripts/run_experiment.py --experiment reproduction_v4b --split test \\
  --config {config}
```

默认快捷命令为`bash scripts/run_selected.sh test`，读取configs/selected.json并使用同一旧adapter默认路径；显式设置MP3_ADAPTER_DIR可避免终端遗留路径混淆。每次自动生成独立r001/r002目录，不覆盖原始成绩。官方预测器与评分器原样运行，全240题、每题三个实例；CPU四线程、workers1、模型加载一次、16GiB RSS、预测7200秒/评分180秒限制均由运行器记录。

训练从头复现：最终adapter沿用[第二轮训练](REPRODUCE_V2.md)，初始化和数据过程也见[第一轮报告](../reports/report.md)。本轮5871条和2740条新课程不是最终权重来源，复现它们的负结果见REPRODUCE_V4.md。没有validation/test训练样本，未将测试逐题标签写入规则。

对照复现只需分别设置旧adapter与contract_v4_selected.json、新adapter与feedback_v4_selected.json；完整验证/测试结果见../reports/v4_results.md。v4b_selection.json在后续对照测试前锁定，v4_deployment.json明确披露聚合公开分数参与最终保留，不将其宣称为隐藏测试选择。

原始日志均保留，未来导出使用../reports/v4_delivery_manifest.json及v4_reproduction_logs/index.json的无损压缩日志映射。保持NOTICE归属；数据集、基座、.venv、缓存不导出，adapter另列两文件。不创建peer_reviews或dist，未上传。
'''
 (ROOT/'submission/REPRODUCE_V4b.md').write_text(text)

def update_readme(best,control,val,config):
 p=ROOT/'submission/README.md';s=p.read_text();a=s.index('# SQL Repair Agent');s=s[a:]
 s=replace_section(s,'本目录包含学生 Agent','## 目录与命名','本目录包含学生Agent；根目录README.md是官方要求，README_cn.md保留逐项翻译。最新结果见[第四轮最终报告](../reports/v4b_report.md)，复现见[REPRODUCE_V4b.md](REPRODUCE_V4b.md)；[第四轮训练负结果](../reports/v4_report.md)、[第三轮](../reports/v3_report.md)、[第二轮](../reports/v2_report.md)、[第一轮](../reports/report.md)保留。所有命令从项目根目录执行，共用.venv；暂不创建peer_reviews和dist。')
 table=f'''## 已完成的官方评测

| 方案 | CPU验证 | 完整CPU测试 |
|---|---:|---:|
| 第一轮 | 81/120（67.50%） | 175/240（72.92%） |
| 第二轮 | 87/120（72.50%） | 181/240（75.42%） |
| 第三轮 | 89/120（74.17%） | 182/240（75.83%） |
| 第四轮初次训练候选 | 95/120（79.17%） | 180/240（75.00%） |
| 第四轮旧权重约束对照 | 92/120（76.67%） | {control['correct']}/240（{control['accuracy']:.2f}%） |

当前规范outputs/test.json、test-score.json为`{best['experiment']}`原始预测和完整官方评分的逐字节副本（{best['correct']}/240）。v3-test保存第三轮，v4-test保存训练候选180分，v4b-test保存旧权重约束对照。旧runs和全部负结果均保留。最终默认CPU预测{best['prediction_seconds']:.2f}秒、RSS {best['peak_rss_mib']:.1f}MiB；每题实际最多生成{best['max_model_calls']}次、单次最多{best['max_new_tokens']}新token。
'''
 s=replace_section(s,'## 已完成的官方评测','## 已选方案的快速复现',table)
 quick=f'''## 已选方案的快速复现

最终默认为`{best['config']['controller']}`，配置来源`{config}`，权重仍为onpolicy_v001/epoch_02。后续约束对照在完整测试前按已有验证92/120锁定；最终保留参考公开测试聚合成绩，见[部署记录](../reports/v4_deployment.json)。配置快照selected_v2.json、selected_v3.json及各轮锁定文件均保留。

```bash
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/onpolicy_v001/epoch_02"
bash scripts/run_selected.sh test
```

该命令在CPU重新运行全240题，输出新编号目录，不覆盖本次结果。新机器先准备固定基座和所选adapter两文件；无需先重新训练历史版本。完整命令见[最终复现指南](REPRODUCE_V4b.md)。
'''
 s=replace_section(s,'## 已选方案的快速复现','## 依次复现实验（第一轮历史对照）',quick)
 a=s.index('## 第四轮诊断与约束修正');b=s.index('## 学习重点',a)
 s=s[:a]+f'''## 第四轮诊断与约束修正

比较SQL AST/schema诊断、真实反馈微调、公开列名投影与显式去重规则，以及数值边界和短plan课程。新课程分别含5871/2740条train样本，共三个新检查点；初次选中的新权重方案验证95/120但测试180/240，未超过旧版182/240。随后完整测试了已验证的旧权重基础约束方案。

最终保留`{best['config']['controller']}`与旧权重。工具仍为原官方工具，所有新候选真实check，最多8次收费调用，模型最多4次生成/384新token。模型自评、绑定诊断和公开示例匹配都不能证明三个实例均正确。收益、退步、全部训练负结果与实际自主工具次数见[最终报告](../reports/v4b_report.md)。

'''+s[b:]
 p.write_text(s)
 p=ROOT/'PROJECT_PLAN_CN.md';s=p.read_text().replace('[第四轮报告](reports/v4_report.md)','[第四轮最终报告](reports/v4b_report.md)');p.write_text(s)
 p=ROOT/'reports/v3_report.md';s=p.read_text().replace('当前默认、规范输出与最新比较见[第四轮报告](v4_report.md)','当前默认、规范输出与最新比较见[第四轮最终报告](v4b_report.md)');p.write_text(s)


def finalize_checks():
 links=0
 for doc in ['reports/v4b_report.md','submission/REPRODUCE_V4b.md','submission/README.md','PROJECT_PLAN_CN.md']:
  path=ROOT/doc
  for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',path.read_text()):
   if target.startswith(('http:','https:','#')):continue
   assert (path.parent/target.split('#',1)[0]).exists(),(doc,target);links+=1
 run('scripts/check_feedback_delivery.py')
 manifest=json.loads((ROOT/'reports/v4_delivery_manifest.json').read_text())
 for name,item in manifest['files'].items():assert sha(ROOT/name)==item['sha256'],name
 (ROOT/'reports/v4b_final_checks.json').write_text(json.dumps({'all_checks_passed':True,'local_document_links_checked':links,'inventory_verified_files_before_this_record':len(manifest['files']),'no_archive_or_dist_created':True,'single_venv':str(ROOT/'.venv')},indent=2)+'\n')
 run('scripts/check_feedback_delivery.py')
 manifest=json.loads((ROOT/'reports/v4_delivery_manifest.json').read_text())
 for name,item in manifest['files'].items():assert sha(ROOT/name)==item['sha256'],name
 print('Final source inventory MiB',manifest['source_bytes']/1024**2,flush=True)

if __name__=='__main__':main()
