"""CPU-confirm the completed first-stage best while the alternative trains."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
while True:
 p=ROOT/'runs/e33_feedback_new_weights/validation/r001/run_metadata.json'
 try:
  m=json.loads(p.read_text())
  if m['status']=='completed':break
  if m['status']=='failed':raise RuntimeError('e33 failed')
 except json.JSONDecodeError:pass
 time.sleep(5)
env=os.environ.copy();env['MP3_ADAPTER_DIR']=str(ROOT/'artifacts/adapters/feedback_v4/epoch_01')
subprocess.run([sys.executable,'scripts/run_experiment.py','--experiment','e34_feedback_variant','--split','validation','--config','submission/configs/semantic_v4.json','--cpu-offset','4'],cwd=ROOT,env=env,check=True)
