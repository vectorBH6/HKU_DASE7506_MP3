"""Separate GPU development screens from replayed CPU comparisons."""
import json
from pathlib import Path
from analyze_recovery_runs import details
ROOT=Path(__file__).resolve().parents[1]
CPU=['e31_feedback_old_weights','e32_recovery_new_weights','e33_feedback_new_weights','e34_feedback_variant','e36_semantic_old_weights','e37_contract_old_weights','e38_recovery_epoch1','e39_constraint_old_weights','e40_feedback_variant2','e41_feedback_variant3','e35_feedback_selected','e42_contract_control']


def main():
    results=[details('e20_recovery4','validation')[0],details('e22_recovery_selected','test')[0]];pairs=[]
    for exp in CPU:
        for split in ['validation','test']:
            item=details(exp,split)
            if not item:continue
            r,p,s=item;assert r['device']=='cpu';results.append(r)
            base=details('e20_recovery4' if split=='validation' else 'e22_recovery_selected',split)
            old={x['id']:x['passed'] for x in base[2]['per_question']};new={x['id']:x['passed'] for x in s['per_question']}
            wins=[k for k in old if not old[k] and new[k]];losses=[k for k in old if old[k] and not new[k]]
            pairs.append({'experiment':exp,'split':split,'wins':wins,'losses':losses,'net_correct':len(wins)-len(losses)})
    data={'CPU_results':results,'paired_against_v3':pairs,
          'GPU_development':json.loads((ROOT/'reports/v4_gpu_screen.json').read_text()) if (ROOT/'reports/v4_gpu_screen.json').exists() else []}
    (ROOT/'reports/v4_results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    lines=['# 第四轮结果：正式CPU验证与测试','','全部CPU成绩由原样官方评分器重放，按每题三个实例全部通过计分。GPU开发筛选单独记录。','',
           '| 实验 | 划分 | 正确 | 准确率 | 模型调用 | 输出token | 预测秒 | RSS MiB | 公开匹配仍错 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| [{r['experiment']}/{r['run']}](../{r['source']}/score.json) | {r['split']} | {r['correct']}/{r['n']} | {r['accuracy']:.2f}% | {r['model_calls']} | {r['output_tokens']} | {r['prediction_seconds']:.2f} | {r['peak_rss_mib']:.1f} | {r['public_match_final_failed']} |")
    (ROOT/'reports/v4_results.md').write_text('\n'.join(lines)+'\n');print('CPU comparisons',len(results))

if __name__=='__main__':main()
