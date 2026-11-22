"""Compare replayed official runs; keep old evidence and derived v2 metrics separate."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from summarize_results import time_field
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS=['e03_qwen3_lora','e11_review_old_adapter','e12_review_onpolicy','e13_review_risk','e14_review_off','e15_review_selected','e16_old_controller_new_adapter']

def latest(exp,split):
    runs=[]
    for p in (ROOT/'runs'/exp/split).glob('r*/run_metadata.json'):
        m=json.loads(p.read_text())
        if m['status']=='completed': runs.append((p.parent,m))
    return sorted(runs,key=lambda x:x[0].name)[-1] if runs else None


def load(exp,split):
    found=latest(exp,split)
    if not found: return None
    run,meta=found
    pred=json.loads((run/'predictions.json').read_text())
    score=json.loads((run/'score.json').read_text())
    assert hashlib.sha256((run/'predictions.json').read_bytes()).hexdigest()==score['predictions_sha256']
    logs=[json.loads(s) for s in (run/'generations.jsonl').read_text().splitlines()] if (run/'generations.jsonl').exists() else []
    decisions=[json.loads(s) for s in (run/'decisions.jsonl').read_text().splitlines()] if (run/'decisions.jsonl').exists() else []
    calls=Counter(r['task_id'] for r in logs)
    assert max(calls.values(),default=0)<=6 and max((r['new_tokens'] for r in logs),default=0)<=2048
    assert max(r['tool_calls'] for r in pred['predictions'])<=8
    matched=wrong_matched=original_wrong=0
    for p,s in zip(pred['predictions'],score['per_question']):
        assert p['id']==s['id']
        ok=any(e['tool']=='check' and e['sql']==p['sql'] and e['observation']['matches_example'] for e in p['trace'])
        matched+=ok;wrong_matched+=ok and not s['passed']
        original_wrong+=bool(not s['passed'] and p['sql']==p['trace'][0]['sql'] and p['trace'][0]['observation']['matches_example'])
    result={'experiment':exp,'split':split,'run':run.name,'source':str(run.relative_to(ROOT)),
            'correct':score['correct'],'n':score['n'],'accuracy':score['accuracy'],
            'prediction_seconds':pred['prediction_seconds'],'scoring_seconds':score['scoring_seconds'],
            'mean_tool_calls':score['mean_tool_calls'],'model_calls':len(logs),
            'input_tokens':sum(r['input_tokens'] for r in logs),'output_tokens':sum(r['new_tokens'] for r in logs),
            'max_model_calls':max(calls.values(),default=0),'max_new_tokens':max((r['new_tokens'] for r in logs),default=0),
            'max_tool_calls':max(r['tool_calls'] for r in pred['predictions']),
            'peak_rss_mib':time_field(run/'prediction_resources.txt','Maximum resident set size (kbytes)')/1024,
            'cpu_seconds':time_field(run/'prediction_resources.txt','User time (seconds)')+time_field(run/'prediction_resources.txt','System time (seconds)'),
            'public_matches':matched,'public_match_final_failed':wrong_matched,'original_match_final_failed':original_wrong,
            'guard_reasons':dict(Counter(d['guard_reason'] for d in decisions)),
            'model_selected_actions':dict(Counter(d['accepted_action']['tool'] for d in decisions if d.get('accepted_action') and d['guard_reason'].startswith('model_'))),
            'tool_actions':dict(Counter(e['tool'] for p in pred['predictions'] for e in p['trace'])),
            'device':meta.get('device','cpu'),'config':meta['config'],
            'adapter':logs[0]['identity']['adapter'] if logs else None,
            'review_records':len(decisions),
            'model_accepted_proposals':sum(bool(d.get('accepted_action')) and d['guard_reason'].startswith('model_') for d in decisions),
            'original_match_tasks':sum(bool(p['trace'][0]['observation']['matches_example']) for p in pred['predictions']),
            'original_match_final_changed':sum(bool(p['trace'][0]['observation']['matches_example']) and p['sql']!=p['trace'][0]['sql'] for p in pred['predictions']),
            'affinity_changes':json.loads((run/'affinity_changes.json').read_text()) if (run/'affinity_changes.json').exists() else []}
    if logs: assert logs[0]['identity']['total_parameters']<=1000000000
    return result,pred,score


def main():
    results=[];loaded={}
    for exp in EXPERIMENTS:
        for split in ['validation','test']:
            item=load(exp,split)
            if item:
                result,pred,score=item;results.append(result);loaded[(exp,split)]=(pred,score)
    pairs=[]
    for (exp,split),(p,s) in loaded.items():
        if exp=='e03_qwen3_lora' or ('e03_qwen3_lora',split) not in loaded: continue
        bp,bs=loaded[('e03_qwen3_lora',split)]
        old={x['id']:x for x in bs['per_question']};new={x['id']:x for x in s['per_question']}
        wins=[k for k in old if not old[k]['passed'] and new[k]['passed']]
        losses=[k for k in old if old[k]['passed'] and not new[k]['passed']]
        base_preds={x['id']:x for x in bp['predictions']}
        early_false=[k for k,x in base_preds.items() if x['sql']==x['trace'][0]['sql'] and x['trace'][0]['observation']['matches_example'] and not old[k]['passed']]
        pairs.append({'experiment':exp,'split':split,'wins':wins,'losses':losses,'net_correct':len(wins)-len(losses),
                      'rescued_original_false_accepts':[k for k in early_false if new[k]['passed']]})
    data={'results':results,'paired_against_v1':pairs}
    (ROOT/'reports/v2_results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    lines=['# 第二轮官方实验汇总','','准确率要求每题三个实例全部通过；仅列出已完成、官方轨迹重放通过的运行。','',
           '| 实验 | 划分 | 正确 | 准确率 | 预测秒 | 平均工具 | 模型调用 | 输出 token | RSS MiB | 公开匹配但最终错 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| [{r['experiment']}/{r['run']}](../{r['source']}/score.json) | {r['split']} | {r['correct']}/{r['n']} | {r['accuracy']:.2f}% | {r['prediction_seconds']:.2f} | {r['mean_tool_calls']:.3f} | {r['model_calls']} | {r['output_tokens']} | {r['peak_rss_mib']:.1f} | {r['public_match_final_failed']} |")
    (ROOT/'reports/v2_results.md').write_text('\n'.join(lines)+'\n')
    print('Completed comparison runs:',len(results))

if __name__=='__main__': main()
