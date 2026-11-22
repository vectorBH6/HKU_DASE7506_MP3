"""Wait for current-policy train data, then build curriculum and train LoRA."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
meta=ROOT/'runs/e30_feedback_collection/train/r001/run_metadata.json'
while True:
    state=json.loads(meta.read_text())['status']
    if state=='failed':raise RuntimeError('Current-policy train collection failed')
    if state=='completed':break
    time.sleep(5)
commands=[]
if not (ROOT/'artifacts/prepared_data/feedback_v4/train.jsonl').exists():
    commands.append(['training/prepare_feedback_v4.py','--source-run','runs/e30_feedback_collection/train/r001','--output-dir','artifacts/prepared_data/feedback_v4'])
commands += [['training/finalize_feedback_v4.py'],
             ['training/train_lora.py','--config','training/configs/lora_feedback_v4.json','--run-name','feedback_v4']]
for command in commands:
    print('RUN',command,flush=True)
    subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
print('Feedback training complete',flush=True)
