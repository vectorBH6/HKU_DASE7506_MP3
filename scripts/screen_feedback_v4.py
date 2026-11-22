"""Two independent GPU development jobs; CPU scores determine final selection."""
from concurrent.futures import ThreadPoolExecutor
import json,os
from pathlib import Path
import queue,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1]
started=time.monotonic()
while True:
    meta=ROOT/'runs/training/feedback_v4/training_metadata.json'
    try:
        if meta.exists() and json.loads(meta.read_text())['status']=='completed':break
    except json.JSONDecodeError:pass
    if time.monotonic()-started>4*3600:raise RuntimeError('Training wait exceeded four hours')
    time.sleep(5)
results=[];lock=threading.Lock();offsets=queue.Queue()
for offset in [12,16]:offsets.put(offset)

def save_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)

def run(job):
    exp,config,epoch=job;offset=offsets.get()
    try:
        env=os.environ.copy();env['MP3_ADAPTER_DIR']=str(ROOT/f'artifacts/adapters/feedback_v4/epoch_{epoch:02d}')
        print('GPU_START',exp,'epoch',epoch,'CPU offset',offset,flush=True)
        subprocess.run([sys.executable,'scripts/run_experiment.py','--experiment',exp,'--split','validation',
                        '--config',str(config),'--device','cuda','--cpu-offset',str(offset)],cwd=ROOT,env=env,check=True)
        directory=sorted((ROOT/'runs'/exp/'validation').glob('r*'))[-1]
        score=json.loads((directory/'score.json').read_text());calls=sum(1 for _ in (directory/'generations.jsonl').open())
        item={'experiment':exp,'epoch':epoch,'config':str(Path(config).relative_to(ROOT)),
              'correct':score['correct'],'n':score['n'],'calls':calls,'device':'cuda','source':str(directory.relative_to(ROOT))}
        with lock:
            results.append(item);save_json(ROOT/'reports/v4_gpu_screen.json',sorted(results,key=lambda x:x['experiment']))
        print('GPU_DONE',exp,score['correct'],flush=True);return item
    finally:offsets.put(offset)

jobs=[]
for epoch in [1,2]:
    for label,config in [('recovery','recovery_v3_selected.json'),('feedback','feedback_v4.json')]:
        jobs.append((f'dev_v4_e{epoch}_{label}',ROOT/'submission/configs'/config,epoch))
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,jobs))
best=sorted(results,key=lambda r:(-r['correct'],r['calls'],r['epoch'],r['experiment']))[0]
save_json(ROOT/'reports/v4_standard_screen.json',{'best':best,'development_only':True})
jobs=[]
for label in ['recovery','feedback']:
    config=json.loads((ROOT/'submission/configs'/('feedback_v4.json' if label=='feedback' else 'recovery_v3_selected.json')).read_text());config['max_completions']=6
    path=ROOT/f'submission/configs/feedback_screen_{label}6.json';path.write_text(json.dumps(config,indent=2)+'\n')
    jobs.append((f'dev_v4_e{best["epoch"]}_{label}6',path,best['epoch']))
config=json.loads((ROOT/'submission/configs/feedback_v4.json').read_text());config['prefer_model_candidate']=True
path=ROOT/'submission/configs/feedback_screen_prefer_model.json';path.write_text(json.dumps(config,indent=2)+'\n')
jobs.append((f'dev_v4_e{best["epoch"]}_prefer_model',path,best['epoch']))
for epoch in [1,2]:
    for label in ['semantic','contract']:jobs.append((f'dev_v4_e{epoch}_{label}',ROOT/f'submission/configs/{label}_v4.json',epoch))
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,jobs))
record={'standard_best_epoch':best['epoch'],'development_only':True,'formal_CPU_validation_required':True,
        'standard_best':best,'overall_best':sorted(results,key=lambda r:(-r['correct'],r['calls'],r['epoch'],r['experiment']))[0]}
save_json(ROOT/'reports/v4_screen_shortlist.json',record)
print('GPU_SCREEN_COMPLETE',json.dumps(record),flush=True)
