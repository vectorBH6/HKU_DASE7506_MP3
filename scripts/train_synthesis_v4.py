"""One synthesis-focused training adjustment after the first screening phase."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json,math,os
from pathlib import Path
import queue,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1]
# Preserve the ongoing CPU comparisons; use a separate four-core group for training CPU work.
os.sched_setaffinity(0,{20,21,22,23})
while not (ROOT/'artifacts/prepared_data/synthesis_v4/provenance.json').exists():time.sleep(5)
from transformers import AutoTokenizer
folder=ROOT/'artifacts/prepared_data/synthesis_v4';records=[json.loads(s) for s in (folder/'train.jsonl').read_text().splitlines()]
tokenizer=AutoTokenizer.from_pretrained(ROOT/'artifacts/models/Qwen3-0.6B',local_files_only=True,trust_remote_code=False)
lengths=[len(tokenizer.encode(r['target'],add_special_tokens=False)) for r in records]
cap=max(512,128*math.ceil((max(lengths)+16)/128));assert cap<=2048
assert all(r['id'].startswith('train-') for r in records) and len({r['id'] for r in records})==480
cfg=json.loads((ROOT/'training/configs/lora_feedback_v4.json').read_text());cfg.update({'epochs':1,'learning_rate':2e-5,'prepared_data':'artifacts/prepared_data/synthesis_v4','init_adapter':'artifacts/adapters/onpolicy_v001/epoch_02'})
(ROOT/'training/configs/lora_synthesis_v4.json').write_text(json.dumps(cfg,indent=2)+'\n')
report={'all_checks_passed':True,'train_only':True,'records':len(records),'unique_train_tasks':480,'max_target_tokens':max(lengths),'new_token_cap':cap,
        'actions':dict(Counter(json.loads(r['target'])['tool'] for r in records if r['target'].startswith('{'))),
        'note':'One alternative single-epoch initialization from the old adapter, not an extra epoch appended to feedback_v4.'}
(ROOT/'reports/v4_synthesis_data_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('SYNTHESIS_DATA',json.dumps(report),flush=True)
while not (ROOT/'reports/v4_screen_shortlist.json').exists():time.sleep(5)
print('SYNTHESIS_TRAIN_START',flush=True)
subprocess.run([sys.executable,'training/train_lora.py','--config','training/configs/lora_synthesis_v4.json','--run-name','feedback_v4_extra'],cwd=ROOT,check=True)
# Broaden CPU affinity for the independent GPU development runners' core offsets.
os.sched_setaffinity(0,set(range(24)))
existing=json.loads((ROOT/'reports/v4_gpu_screen.json').read_text());lock=threading.Lock();offsets=queue.Queue()
for n in [12,16]:offsets.put(n)
jobs=[]
for label,name in [('plan','plan_v4.json'),('plan_constraint','plan_constraint_v4.json'),('constraint','constraint_v4.json'),('recovery','recovery_v3_selected.json')]:
    config=json.loads((ROOT/'submission/configs'/name).read_text());config['max_new_tokens']=cap
    path=ROOT/f'submission/configs/synthesis_{label}_v4.json';path.write_text(json.dumps(config,indent=2)+'\n')
    jobs.append((label,path))

def run(job):
    label,path=job;offset=offsets.get()
    try:
        exp='dev_v4_extra_'+label;env=os.environ.copy();env['MP3_ADAPTER_DIR']=str(ROOT/'artifacts/adapters/feedback_v4_extra/epoch_01')
        subprocess.run([sys.executable,'scripts/run_experiment.py','--experiment',exp,'--split','validation','--config',str(path),'--device','cuda','--cpu-offset',str(offset)],cwd=ROOT,env=env,check=True)
        run=sorted((ROOT/'runs'/exp/'validation').glob('r*'))[-1];score=json.loads((run/'score.json').read_text())
        item={'experiment':exp,'epoch':'alt1','training_run':'feedback_v4_extra','adapter_path':'artifacts/adapters/feedback_v4_extra/epoch_01',
              'config':str(path.relative_to(ROOT)),'correct':score['correct'],'n':score['n'],'calls':sum(1 for _ in (run/'generations.jsonl').open()),'device':'cuda','source':str(run.relative_to(ROOT))}
        with lock:
            existing.append(item);p=ROOT/'reports/v4_gpu_screen.json';temporary=p.with_suffix('.json.tmp');temporary.write_text(json.dumps(sorted(existing,key=lambda r:r['experiment']),indent=2)+'\n');temporary.replace(p)
        print('SYNTHESIS_GPU',json.dumps(item),flush=True)
    finally:offsets.put(offset)

with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,jobs))
best=sorted(existing,key=lambda r:(-r['correct'],r['calls'],str(r['epoch']),r['experiment']))[0]
(ROOT/'reports/v4_all_screen_shortlist.json').write_text(json.dumps({'overall_best':best,'formal_CPU_required':True,'development_only':True},indent=2)+'\n')
print('ALL_DEVELOPMENT_SCREENING_COMPLETE',json.dumps(best),flush=True)
