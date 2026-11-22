"""One predeclared routed-distillation GPU trial, paired against the unmodified V7 adapter."""
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'runs/routed_distill'


def main():
    os.chdir(ROOT);os.sched_setaffinity(0,{12,13,14,15})
    work=WORK/'pipeline';work.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ)
    env.pop('PYTHONPATH',None)
    env.update(MP3_BASE_MODEL=str(ROOT/'artifacts/models/Qwen3-0.6B'),PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    eval_script='scripts/probe_routed_distill.py';prep='training/prepare_routed_distill.py'
    stages=[('teacher_pass',[eval_script,'teacher_pass']),('prepare_pass',[prep,'pass']),('combine',[prep,'combine']),
            ('holdout_before',[eval_script,'holdout_before']),
            ('train',['training/train_lora.py','--config','training/configs/lora_routed_distill_v1.json',
                      '--run-name','routed_distill_v1']),
            ('validation_after',[eval_script,'validation_after']),('holdout_after',[eval_script,'holdout_after']),
            ('compare',[eval_script,'compare'])]
    for stage,argv in stages:
        print('starting',stage,flush=True)
        with (work/(stage+'.log')).open('x') as output:
            subprocess.run([sys.executable,'-B',*argv],cwd=ROOT,env=env,check=True,stdout=output,stderr=subprocess.STDOUT)
        if stage=='combine':
            rows=[json.loads(l) for l in (ROOT/'artifacts/prepared_data/routed_distill/train.jsonl').open()]
            by_state=Counter(r['state'] for r in rows);print('selected teacher tasks',dict(by_state),flush=True)
            assert len({r['id'] for r in rows})==len(rows) and by_state['fail_check']>0
        print('completed',stage,flush=True)


if __name__=='__main__':main()
