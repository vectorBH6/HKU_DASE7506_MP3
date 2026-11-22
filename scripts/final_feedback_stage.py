"""CPU-confirm the top development alternatives, freeze selection, then full test."""
import json,os
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_feedback_v4 import CPU

def latest_meta(exp,split='validation'):
    paths=sorted((ROOT/'runs'/exp/split).glob('r*/run_metadata.json'))
    if not paths:return None
    try:return json.loads(paths[-1].read_text())
    except json.JSONDecodeError:return None

def key(config,adapter):return json.dumps(config,sort_keys=True),str(Path(adapter).resolve())
while not (ROOT/'reports/v4_all_screen_shortlist.json').exists():time.sleep(5)
rows=json.loads((ROOT/'reports/v4_gpu_screen.json').read_text())
seen=set()
for exp in CPU:
    m=latest_meta(exp)
    if m and m['status'] in ['completed','running']:seen.add(key(m['config'],m['adapter_path']))
ranked=sorted(rows,key=lambda r:(-r['correct'],r['calls'],str(r['epoch']),r['experiment']));threshold=ranked[0]['correct']-1
candidates=[]
for r in ranked:
    if r['correct']<threshold:break
    m=json.loads((ROOT/r['source']/'run_metadata.json').read_text());signature=key(m['config'],m['adapter_path'])
    if signature in seen:continue
    seen.add(signature);candidates.append({**r,'adapter_path':m['adapter_path']})
    if len(candidates)==2:break
(ROOT/'reports/v4_CPU_shortlist.json').write_text(json.dumps({'threshold':threshold,'candidates':candidates,'test_metrics_used':False},indent=2)+'\n')
queue=list(zip(['e40_feedback_variant2','e41_feedback_variant3'],candidates));active={}
while queue or active:
    for offset,(exp,proc,log) in list(active.items()):
        code=proc.poll()
        if code is not None:
            log.close();del active[offset]
            if code:raise RuntimeError(exp+' failed CPU verification')
    busy=set(active)
    for exp in CPU:
        m=latest_meta(exp)
        if m and m['status']=='running' and m['device']=='cpu':
            for offset in [0,4]:
                if m['cpu_affinity']==list(range(offset,offset+4)):busy.add(offset)
    for offset in [0,4]:
        if not queue:break
        if offset in busy:continue
        exp,item=queue.pop(0);env=os.environ.copy();env['MP3_ADAPTER_DIR']=item['adapter_path']
        log=(ROOT/f'artifacts/{exp}.log').open('w')
        cmd=[sys.executable,'scripts/run_experiment.py','--experiment',exp,'--split','validation','--config',str(ROOT/item['config']),'--cpu-offset',str(offset)]
        print('CPU_FINALIST',exp,item['experiment'],offset,flush=True)
        active[offset]=(exp,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log)
    time.sleep(2)
while True:
    incomplete=[]
    for exp in CPU:
        if exp=='e35_feedback_selected':continue
        m=latest_meta(exp)
        if m and m['status']!='completed':incomplete.append((exp,m['status']))
    if not incomplete:break
    if any(status=='failed' for _,status in incomplete):raise RuntimeError('Validation failed: '+repr(incomplete))
    time.sleep(5)
subprocess.run([sys.executable,'scripts/analyze_feedback_v4.py'],cwd=ROOT,check=True)
subprocess.run([sys.executable,'scripts/select_feedback_v4.py'],cwd=ROOT,check=True)
sel=json.loads((ROOT/'reports/v4_selection.json').read_text());env=os.environ.copy();env['MP3_ADAPTER_DIR']=str(ROOT/sel['adapter_path'])
print('LOCKED_FULL_TEST',sel['selected_experiment'],sel['adapter_path'],flush=True)
subprocess.run([sys.executable,'scripts/run_experiment.py','--experiment','e35_feedback_selected','--split','test','--config',str(ROOT/sel['locked_config']),'--cpu-offset','0'],cwd=ROOT,env=env,check=True)
subprocess.run([sys.executable,'scripts/analyze_feedback_v4.py'],cwd=ROOT,check=True)
print('FOURTH_ROUND_FULL_TEST_COMPLETE',flush=True)
