"""Train-only post-hoc contract-edit diagnostic with real tool replay."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'submission')]
from mp3_sql.common import load_bank,write_json
from mp3_sql.environment import Tools,Episode
from sql_repair_agent.semantic_agent import contract_edits
from sql_repair_agent.sql_utils import sql_key
from training.prepare_feedback_v4 import grade


def main():
    out=ROOT/'runs/contract_train_diagnostic';out.mkdir(exist_ok=False)
    rows,manifest,folder=load_bank('train');tools=Tools(folder,manifest,cache={})
    source=json.loads((ROOT/'runs/e30_feedback_collection/train/r001/predictions.json').read_text());preds={r['id']:r for r in source['predictions']}
    started=time.perf_counter()
    def run(row):
        original=preds[row['id']];ep=Episode(row,tools)
        for event in original['trace'][:-1]:
            ep.step({k:v for k,v in event.items() if k!='observation'})
            assert ep.trace[-1]==event
        sql=original['sql'];used={sql_key(e['sql']) for e in ep.trace if e['tool']=='check'}
        for _ in range(2):
            if not ep.remaining:break
            choices=[(s,label) for s,label in contract_edits(ep.observation(),sql) if sql_key(s) not in used]
            if not choices:break
            candidate,_=choices[0];used.add(sql_key(candidate));ep.step({'tool':'check','sql':candidate})
            if ep.trace[-1]['observation']['matches_example']:sql=candidate
        ep.step({'tool':'submit','sql':sql})
        return {'id':row['id'],'sql':sql,'trace':ep.trace,'tool_calls':8-ep.remaining}
    with ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(run,rows))
    score=grade({**source,'predictions':result,'prediction_seconds':time.perf_counter()-started},out/'contracts')
    baseline=json.loads((ROOT/'runs/e30_feedback_collection/train/r001/score.json').read_text());old={r['id']:r['passed'] for r in baseline['per_question']};new={r['id']:r['passed'] for r in score['per_question']}
    summary={'train_only':True,'baseline_correct':baseline['correct'],'contract_overlay_correct':score['correct'],
             'wins':[k for k in old if not old[k] and new[k]],'losses':[k for k in old if old[k] and not new[k]],
             'note':'Post-hoc diagnostic, not a live-model policy result. Real tool calls appended within original eight-call budget; no grading labels used for decisions.'}
    write_json(ROOT/'reports/v4_contract_train_diagnostic.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
