"""Overlap an early epoch-one CPU validation with training and GPU screening."""
import json,os
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]

def wait_done(experiment):
    while True:
        paths=sorted((ROOT/'runs'/experiment/'validation').glob('r*/run_metadata.json'))
        try:
            if paths and json.loads(paths[-1].read_text())['status']!='running':return
        except json.JSONDecodeError:pass
        time.sleep(5)

def launch(exp,config,epoch,offset):
    env=os.environ.copy();env['MP3_ADAPTER_DIR']=str(ROOT/f'artifacts/adapters/feedback_v4/epoch_{epoch:02d}')
    log=(ROOT/f'artifacts/{exp}.log').open('w')
    command=[sys.executable,'scripts/run_experiment.py','--experiment',exp,'--split','validation','--config',str(ROOT/'submission/configs'/config),'--cpu-offset',str(offset)]
    print('CPU_START',exp,'epoch',epoch,flush=True)
    return subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log

wait_done('e37_contract_old_weights')
while not (ROOT/'artifacts/adapters/feedback_v4/epoch_01/adapter_model.safetensors').exists():time.sleep(5)
early,earlylog=launch('e38_recovery_epoch1','recovery_v3_selected.json',1,0)
while True:
    path=ROOT/'reports/v4_standard_screen.json'
    try:
        if path.exists():best=json.loads(path.read_text())['best'];break
    except json.JSONDecodeError:pass
    time.sleep(5)
epoch=best['epoch'];new,newlog=launch('e33_feedback_new_weights','feedback_v4.json',epoch,4)
if epoch==1:
    old=early;oldlog=earlylog;old_exp='e38_recovery_epoch1'
else:
    assert early.wait()==0;earlylog.close()
    old,oldlog=launch('e32_recovery_new_weights','recovery_v3_selected.json',epoch,0);old_exp='e32_recovery_new_weights'
(ROOT/'reports/v4_cpu_matrix.json').write_text(json.dumps({'epoch':epoch,'old_controller_new_weights':old_exp,'new_controller_new_weights':'e33_feedback_new_weights'},indent=2)+'\n')
assert old.wait()==0;oldlog.close();assert new.wait()==0;newlog.close()
subprocess.run([sys.executable,'scripts/analyze_feedback_v4.py'],cwd=ROOT,check=True)
print('Controlled CPU matrix complete',flush=True)
