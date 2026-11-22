"""Finalize the locked fourth-round test, documentation and future inventory."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(*args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
def replace_section(text,start,end,new):
 a=text.index(start);b=text.index(end,a);return text[:a]+new.rstrip()+'\n\n'+text[b:]

while True:
 paths=sorted((ROOT/'runs/e35_feedback_selected/test').glob('r*/run_metadata.json'))
 try:
  if paths:
   meta=json.loads(paths[-1].read_text())
   if meta['status']=='completed':break
   if meta['status']=='failed':raise RuntimeError('The locked full test failed')
 except json.JSONDecodeError:pass
 time.sleep(5)
raw=paths[-1].parent;sel=json.loads((ROOT/'reports/v4_selection.json').read_text())
score=json.loads((raw/'score.json').read_text());assert score['n']==240
assert all(len(x['cases'])==3 for x in score['per_question'])
for src,alias in [('predictions.json','v4-test.json'),('score.json','v4-test-score.json')]:
 (ROOT/'submission/outputs'/alias).write_bytes((raw/src).read_bytes())
 if sel['promote']:(ROOT/'submission/outputs'/('test.json' if src=='predictions.json' else 'test-score.json')).write_bytes((raw/src).read_bytes())
run('scripts/verify_artifacts.py','--run-name','feedback_v4_extra','--output','reports/v4_synthesis_verification.json')
run('scripts/analyze_feedback_v4.py')
run('scripts/audit_feedback_v4.py')
run('scripts/write_feedback_v4_report.py')
data=json.loads((ROOT/'reports/v4_results.json').read_text())
current=next(r for r in data['CPU_results'] if r['experiment']=='e35_feedback_selected' and r['split']=='test')
chosen=next(r for r in sel['validation'] if r['experiment']==sel['selected_experiment'])
p=ROOT/'submission/README.md';s=p.read_text()
s=replace_section(s,'本目录包含学生 Agent','## 目录与命名',
 '本目录包含学生 Agent；官方规则以根目录 `README.md` 为准，`README_cn.md` 保留逐项翻译。第四轮结论见 [v4_report.md](../reports/v4_report.md)，复现见 [REPRODUCE_V4.md](REPRODUCE_V4.md)。[第三轮](../reports/v3_report.md)、[第二轮](../reports/v2_report.md)、[第一轮](../reports/report.md)保留为历史对照。所有命令从项目根目录执行，共用 `.venv`；暂不创建 `peer_reviews/` 或 `dist/`。')
s=s.replace('本轮沿用第二轮权重，不再训练。','第三轮沿用第二轮权重。第四轮e30采集train轨迹，e31/e33/e38组成诊断与权重对照，e36/e37/e39比较通用约束；e34/e40/e41按实际启动情况确认开发候选，e35为锁定后的完整测试。GPU开发实验用dev_v4前缀，训练版本为feedback_v4和feedback_v4_extra。')
s=s.replace('GPU 只用于训练。正式结果使用四个 CPU 核','GPU 用于训练与开发筛选。正式结果使用四个 CPU 核')
summary=f'''## 已完成的官方评测

| 方案 | 验证集 | 测试集 |
|---|---:|---:|
| 第一轮 | 81/120（67.50%） | 175/240（72.92%） |
| 第二轮 | 87/120（72.50%） | 181/240（75.42%） |
| 第三轮 | 89/120（74.17%） | 182/240（75.83%） |
| 第四轮锁定方案 | {chosen['correct']}/120（{chosen['accuracy']:.2f}%） | {score['correct']}/240（{score['accuracy']:.2f}%） |

当前 `outputs/test.json`、`outputs/test-score.json` 是{'第四轮' if sel['promote'] else '第三轮'}原始预测及完整官方评分的逐字节副本。第四轮另存 `v4-test.json`、`v4-test-score.json`，第三轮另存 `v3-test.json`、`v3-test-score.json`，历史runs均保留。第四轮CPU预测 {current['prediction_seconds']:.2f} 秒、峰值RSS {current['peak_rss_mib']:.1f} MiB；每题实际最多生成 {current['max_model_calls']} 次、单次最多实际输出 {current['max_new_tokens']} token。所有题按三个实例全部通过计分。
'''
s=replace_section(s,'## 已完成的官方评测','## 已选方案的快速复现',summary)
quick=f'''## 已选方案的快速复现

按完整CPU验证锁定 `{sel['selected_experiment']}`（{chosen['correct']}/120），控制器 `{chosen['config']['controller']}`，权重 `{sel['adapter_path']}`。选择先于第四轮test，见[不可覆盖的选择记录](../reports/v4_selection.json)。默认 `create_agent()` 读取 `configs/selected.json`，历史默认保存在 `configs/selected_v3.json`。本机可直接执行：

```bash
export MP3_ADAPTER_DIR="$PWD/{sel['adapter_path']}"
bash scripts/run_selected.sh test
```

该命令重新运行全部240题，输出新编号目录，不覆盖原始结果。新机器先准备固定版本基座和所选adapter两文件；推理无需重训历史版本。完整训练与对照命令见[第四轮复现](REPRODUCE_V4.md)。
'''
s=replace_section(s,'## 已选方案的快速复现','## 依次复现实验（第一轮历史对照）',quick)
s=s.replace('只有目标 SQL 和 EOS 参与 loss','只有目标 SQL/动作JSON和 EOS 参与 loss')
needle='## 学习重点';detail=f'''## 第四轮诊断与约束修正

新增SQL AST与schema绑定摘要、真实失败反馈SFT、公开列名对应的输出顺序修正和显式去重候选；另比较数值边界及短plan课程。所有候选仍调用官方check，规则不读取任务答案或额外数据库，不增加工具预算。训练数据分别为5871条反馈动作和2740条SQL合成/修复样本，均来自train；三份新检查点与旧权重公平比较。

最终选择 `{chosen['config']['controller']}`。提示诊断、模型规划和静态检查都只是辅助证据，公开示例匹配仍不等于三个实例都正确。真实模型自主动作、规则动作、消融、改善和退步见[第四轮报告](../reports/v4_report.md)。

'''
assert needle in s;s=s.replace(needle,detail+needle);p.write_text(s)
p=ROOT/'PROJECT_PLAN_CN.md';s=p.read_text().replace('[第三轮报告](reports/v3_report.md)（[第二轮]', '[第四轮报告](reports/v4_report.md)（[第三轮](reports/v3_report.md)、[第二轮]');p.write_text(s)
p=ROOT/'reports/v3_report.md';s=p.read_text();note='> 第三轮历史记录：当前默认、规范输出与最新比较见[第四轮报告](v4_report.md)和[学生复现README](../submission/README.md)；第三轮原始runs与v3-test副本保留。\n\n';p.write_text(s if s.startswith(note) else note+s)
import re
checked_links=0
for doc in ['reports/v4_report.md','reports/v4_results.md','submission/REPRODUCE_V4.md','submission/README.md','PROJECT_PLAN_CN.md']:
    path=ROOT/doc
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',path.read_text()):
        if target.startswith(('http:','https:','#')):continue
        target=target.split('#',1)[0]
        assert (path.parent/target).exists(),(doc,target)
        checked_links+=1
run('scripts/check_feedback_delivery.py')
manifest=json.loads((ROOT/'reports/v4_delivery_manifest.json').read_text())
for name,item in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==item['sha256'],name
sys.path.insert(0,str(ROOT/'submission'));import submission
agent=submission.create_agent();other=submission.create_agent();assert agent is not other and agent.completions==other.completions==0
assert agent.config==json.loads((ROOT/'submission/configs/selected.json').read_text())
assert not (ROOT/'dist').exists() and not (ROOT/'peer_reviews').exists()
(ROOT/'reports/v4_final_checks.json').write_text(json.dumps({'all_checks_passed':True,'full_test_three_instances':True,'default_factory':type(agent).__name__,'fresh_task_memory':True,'inventory_before_final_check_record_bytes':manifest['source_bytes'],'inventory_hashes_verified':len(manifest['files']),'canonical_raw_copies':True,'local_document_links_checked':checked_links,'single_venv':str(ROOT/'.venv'),'test_correct':score['correct'],'test_n':240},ensure_ascii=False,indent=2)+'\n')
# Include the final check record itself in the future inventory.
run('scripts/check_feedback_delivery.py')
final_inventory=json.loads((ROOT/'reports/v4_delivery_manifest.json').read_text())
for name,item in final_inventory['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==item['sha256'],name
print('FOURTH_ROUND_DELIVERABLES_COMPLETE',score['correct'],score['n'],flush=True)
