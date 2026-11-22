"""Audit finished V5 experiments and write the Chinese report/reproduction guide."""
from collections import Counter
from datetime import datetime
import gzip
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'submission')]
from mp3_sql.common import digest, write_json
from safetensors import safe_open


def load(path):
    return json.loads((ROOT / path).read_text())


def details(item):
    out = ROOT / item['run']
    meta = json.loads((out / 'run_metadata.json').read_text())
    predictions = json.loads((out / 'predictions.json').read_text())
    score = json.loads((out / 'score.json').read_text())
    logs = [json.loads(s) for s in (out / 'generations.jsonl').read_text().splitlines()]
    decisions = [json.loads(s) for s in (out / 'decisions.jsonl').read_text().splitlines()]
    counts = Counter(g['task_id'] for g in logs)
    actual_model_reasons = {'model_hint_request', 'model_query', 'model_candidate_selection', 'model_retrieval_request'}
    autonomous = Counter(d['accepted_action']['tool'] for d in decisions
                        if d.get('accepted_action') and d.get('guard_reason') in actual_model_reasons)
    final_pass = {s['id']: s['passed'] for s in score['per_question']}
    public_wrong = sum(not final_pass[p['id']] and any(e['tool'] == 'check' and e['sql'] == p['sql']
                       and e['observation']['matches_example'] for e in p['trace']) for p in predictions['predictions'])
    return {'model_generations': len(logs), 'max_generations': max(counts.values(), default=0),
            'max_new_tokens': max((g['new_tokens'] for g in logs), default=0),
            'max_tools': max(p['tool_calls'] for p in predictions['predictions']),
            'model_actions': dict(autonomous),
            'all_actions': dict(Counter(e['tool'] for p in predictions['predictions'] for e in p['trace'])),
            'public_match_final_wrong': public_wrong,
            'duplicate_proposals': sum(d.get('guard_reason') == 'duplicate_action' for d in decisions)}, meta, logs, score


def audit(result):
    initial = load('reports/v5_initial_manifest.json')
    package = load('PACKAGE_MANIFEST.json')
    official_bad = [p for p, v in package['files'].items() if digest(ROOT / p) != v['sha256']]
    base_bad = [p for p, sha in initial['base_files'].items() if digest(ROOT / p) != sha]
    assert not official_bad and not base_bad
    assert digest(ROOT / 'README_cn.md') == initial['baseline']['README_cn.md']
    lock = load('reports/v5_test_lock.json')
    assert digest(ROOT / 'submission/configs/search_v5_selected.json') == lock['architecture_config_sha256']
    assert digest(ROOT / 'submission/configs/grpo_v5_selected.json') == lock['trained_config_sha256']
    records = {}
    for item in result['tests']:
        d, meta, logs, score = details(item)
        assert meta['status'] == 'completed' and meta['device'] == 'cpu' and meta['offline']
        assert len(meta['cpu_affinity']) == 4 and meta['python_executable'] == str(ROOT / '.venv/bin/python')
        assert datetime.fromisoformat(lock['locked_utc']) < datetime.fromisoformat(meta['started_at'])
        assert score['n'] == 240 and all(len(q['cases']) == 3 for q in score['per_question'])
        assert score['scorer_sha256'] == digest(ROOT / 'mp3_sql/score.py')
        assert score['executor_sha256'] == digest(ROOT / 'mp3_sql/sqlite_tool.py')
        assert score['predictions_sha256'] == digest(ROOT / item['run'] / 'predictions.json')
        assert item['prediction_seconds'] <= 7200 and item['scoring_seconds'] <= 180
        assert item['rss_bytes'] <= 16 * 1024 ** 3
        assert d['max_tools'] <= 8 and d['max_generations'] <= meta['config']['max_completions'] <= 6
        assert d['max_new_tokens'] <= meta['config']['max_new_tokens'] <= 2048
        assert (ROOT / item['run'] / 'prediction.log').read_text().count('Loaded model: ') == 1
        for g in logs:
            assert not g['generation']['do_sample'] and not g['generation']['enable_thinking']
            assert g['identity']['total_parameters'] == 606142464
            assert g['identity']['revision'] == 'c1899de289a04d12100db370d81485cdf75e47ca'
            assert g['identity']['adapter']['sha256'] == digest(ROOT / item['adapter'] / 'adapter_model.safetensors')
        for path, sha in meta['source_sha256'].items():
            if path == 'submission/submission.py' or path.startswith('submission/sql_repair_agent/'):
                assert digest(ROOT / path) == sha, path
        records[item['label']] = d
    training = load('runs/training/grpo_v5/training_metadata.json')
    assert training['status'] == 'completed' and training['updates'] > 0
    assert all('.lora_A.' in n or '.lora_B.' in n for n in training['trainable_names'])
    train_ids = {json.loads(s)['id'] for s in (ROOT / 'mp3_sql/data/train.jsonl').read_text().splitlines()}
    assert set(training['config']['hard_task_ids']) <= train_ids
    for line in (ROOT / 'runs/training/grpo_v5/groups.jsonl').read_text().splitlines():
        assert json.loads(line)['id'] in train_ids
    adapters = []
    for cp in training['checkpoints']:
        folder = ROOT / cp['path']; file = folder / 'adapter_model.safetensors'
        assert {p.name for p in folder.iterdir()} == {'adapter_config.json', 'adapter_model.safetensors'}
        assert digest(file) == cp['sha256']
        with safe_open(file, framework='pt', device='cpu') as f:
            keys = list(f.keys())
            assert all('.lora_A.' in k or '.lora_B.' in k for k in keys)
            count = sum(f.get_tensor(k).numel() for k in keys)
        assert count == 10092544 and file.stat().st_size <= 512 * 1024 ** 2
        adapters.append({**cp, 'tensors': len(keys), 'parameters': count, 'bytes': file.stat().st_size})
    best = result['best']
    assert digest(ROOT / 'submission/outputs/test.json') == digest(ROOT / best['run'] / 'predictions.json')
    assert digest(ROOT / 'submission/outputs/test-score.json') == digest(ROOT / best['run'] / 'score.json')
    assert load('submission/configs/selected.json') == best['config']
    import submission
    one = submission.create_agent(); two = submission.create_agent()
    assert one is not two and one.completions == two.completions == 0
    assert not (ROOT / 'dist').exists() and not (ROOT / 'peer_reviews').exists()
    report = {'all_checks_passed': True, 'official_files_checked': len(package['files']),
              'official_modified': official_bad, 'base_files_modified': base_bad,
              'README_cn_unchanged': True, 'fresh_task_memory': True,
              'single_venv': str(ROOT / '.venv'), 'train_only': True,
              'extra_instances_only_in_offline_train_grader': True,
              'two_challengers_locked_before_tests': True,
              'public_test_aggregate_used_for_deployment': True,
              'test_labels_used_for_training_or_rules': False,
              'runtime': records, 'adapters': adapters, 'selected': best['label']}
    write_json(ROOT / 'reports/v5_runtime_audit.json', report)
    return report


def main():
    result = load('reports/v5_results.json')
    audit_result = audit(result)
    matrix = load('reports/v5_cpu_matrix.json')
    training = load('runs/training/grpo_v5/training_metadata.json')
    coverage = load('reports/v5_train_coverage.json')
    architecture = load('reports/v5_arch_gpu_selection.json')
    gpu_weights = load('reports/v5_grpo_gpu_screen.json')
    groups = [json.loads(s) for s in (ROOT / 'runs/training/grpo_v5/groups.jsonl').read_text().splitlines()]
    best = result['best']; delta = best['correct'] - 184
    lines = ['# V5 实验：搜索控制器与多轮 LoRA-GRPO', '',
             f"当前最佳 **{best['correct']}/240（{100 * best['correct'] / 240:.2f}%）**，相对V4b为{delta:+d}题、{delta / 2.4:+.2f}个百分点。默认方案：`{best['config']['controller']}`；adapter：`{best['adapter']}`。", '',
             '所有正式结果由原样官方predict/score在CPU四线程完成，逐题三个实例全部通过才计分。最终选择参考公开test聚合分数；两组挑战者均在test运行前锁定，不将这里的选择宣称为独立隐藏测试评估。', '',
             '## CPU四组对照', '', '| 实验 | CPU验证 | 模型生成 | 预测秒 |', '|---|---:|---:|---:|']
    for r in matrix:
        lines.append(f"| {r['label']} | {r['correct']}/120 | {r['generations']} | {r['prediction_seconds']:.2f} |")
    lines += ['', '## GPU开发消融', '', '| 架构变体（旧权重） | 验证正确数 | 模型生成 |', '|---|---:|---:|']
    labels = {'recover': '拒绝后的检索恢复', 'review': '额外独立复查', 'probe': '复查加JOIN探查',
              'always': '始终复查', 'hints': '提示候选复查'}
    for item in architecture['candidates']:
        lines.append(f"| {labels[item['label']]} | {item['correct']}/120 | {item['generations']} |")
    lines += ['', '| GRPO检查点 | 旧架构GPU验证 | 新架构GPU验证 |', '|---|---:|---:|']
    for item in gpu_weights:
        cells = {r['label']: r['correct'] for r in item['results']}
        lines.append(f"| {item['checkpoint']['updates']}组有效更新 | {cells['contract']}/120 | {cells['search']}/120 |")
    lines += ['', 'GPU仅用于开发筛选。五个架构对照分别为：拒绝重复后的有界检索恢复、额外独立语义复查、聚合/JOIN探查、取消低风险原稿匹配后的模型跳过，以及提交前获取并检查不同提示候选。GPU筛选结果及CPU矩阵均保留，不能把GPU速度当作正式CPU复现成本。独立实验可能并行，墙钟成本包含当时的机器竞争。', '',
              'GPU开发前两名另做CPU确认，再确定架构。协调器替换曾导致两项验证失去监控，原记录标为interrupted并保留；两项均用标准运行器重新完成，不采用不完整运行。第8次GRPO更新重跑前后的完整预测和工具轨迹一致，见v5_interruption_audit.json。', '',
              '## 预锁定后的完整测试', '', '| 方案 | 正确题数 | 相对184：改善/退步 | 模型生成 | 预测秒 | RSS MiB |', '|---|---:|---:|---:|---:|---:|']
    base = result['baseline']
    lines.append(f"| V4b基线 | 184/240 | — | {base['generations']} | {base['prediction_seconds']:.2f} | {base['rss_bytes'] / 1024**2:.2f} |")
    for r in result['tests']:
        pair = result['paired_changes'][r['label']]
        lines.append(f"| {r['label']} | {r['correct']}/240 | {len(pair['wins'])}/{len(pair['losses'])} | {r['generations']} | {r['prediction_seconds']:.2f} | {r['rss_bytes'] / 1024**2:.2f} |")
    lines += ['', '## Train候选覆盖诊断', '',
              f"当前旧策略在{coverage['tasks']}道train题上独立采样{coverage['episodes']}条真实轨迹；其中至少一条最终正确的题有{coverage['tasks_with_correct_final']}道，已check候选中至少一个正确的题有{coverage['tasks_with_correct_checked_candidate']}道。混合成功/失败组{coverage['mixed_reward_groups']}个，全部失败组{len(coverage['all_failed_groups'])}个。这里只是train采样覆盖诊断，不是test上限，也不是部署时允许运行四套完整工具预算。", '',
              f"完整诊断中，{coverage.get('tasks_with_no_model_generations', 0)}道train题没有调用模型，均未出现全错；零模型调用的全错任务数为{len(coverage.get('all_failed_tasks_with_no_model_generations', []))}。因此早期把少工具调用推断成‘没有模型参与导致失败’不成立。该结论限于train诊断，不能推断测试同样如此。", '',
              '训练诊断实例：`train-1a1ca6ce36878e21`只问年龄最大主持人的姓名和国籍，四条轨迹都保留了原稿中多余的party_host关联，全部失败；参考SQL只需`SELECT Name, Nationality FROM host ORDER BY Age DESC LIMIT 1`。这表现为正确候选生成不足。四条奖励全0时，组内优势无法告诉模型应删除哪段SQL。完整train实例见v5_train_failure_cases.json；它们只用于事后诊断，没有追加训练或改写推理规则。', '',
              '## GRPO训练记录', '',
              f"初始化为旧onpolicy_v001/epoch_02；同一题每组4条独立完整轨迹，最多尝试{training['config']['max_groups']}组、最多{training['config']['max_updates']}次有效更新。实际尝试{training['groups']}组、覆盖{len({g['id'] for g in groups})}道不同train题，{training['updates']}组奖励有差异并更新，每组做{training['config']['num_iterations']}次clipped优化，共{training['updates']*training['config']['num_iterations']}个优化器步；训练{training['seconds']:.2f}秒，CUDA最大allocated {training['max_cuda_memory_allocated']/1024**3:.2f}GiB。", '',
              f"组内优势按二元终局奖励标准化；所有train三个实例通过奖励1，否则0。同分组跳过更新，没有对工具调用次数提供正奖励。clip={training['config']['clip_epsilon']}，KL beta={training['config']['beta']}，lr={training['config']['learning_rate']}。冻结初始化SFT模型为参考策略。采样temperature=1、top_p=1、top_k=0，生成概率与策略损失一致；所有提示、工具反馈、历史token均不计入动作损失，拒绝提议的生成仍计数并参与所属轨迹的信用分配。", '',
              '仅指定Qwen3-0.6B基座的LoRA A/B可训练；没有价值头、embedding、lm_head、bias或额外可训练模块。参考模型只在训练中存在；正式推理仅加载一个未合并模型，总计606,142,464参数。训练过程中关闭dropout以保证旧/新策略概率比可比，小规模验证的首轮最大log-ratio差异为0。', '',
              '课程采样偏好由train失败轨迹决定，并固化成配置中的hard_task_ids；不会随诊断文件后续写入而改变。train三实例评分仅在离线训练奖励侧执行，参考SQL、额外实例和指纹均不出现在actor观察、提示或部署文件中。validation/test不进入训练、采样课程或梯度。', '',
              '## 实际工具能力与局限', '']
    for label, stats in audit_result['runtime'].items():
        lines.append(f"- {label}：模型自主动作`{json.dumps(stats['model_actions'], ensure_ascii=False)}`；全部官方动作`{json.dumps(stats['all_actions'], ensure_ascii=False)}`；公开匹配却最终错误{stats['public_match_final_wrong']}题，重复提议{stats['duplicate_proposals']}次。")
    test_rows = {r['id']: r for r in map(json.loads, (ROOT / 'mp3_sql/data/test.jsonl').read_text().splitlines())}
    baseline_preds = {p['id']: p for p in load(result['baseline']['run'] + '/predictions.json')['predictions']}
    examples = []
    for candidate in result['tests']:
        pair = result['paired_changes'][candidate['label']]
        preds = {p['id']: p for p in load(candidate['run'] + '/predictions.json')['predictions']}
        for title, ids in [('改善', pair['wins']), ('退步', pair['losses'])]:
            if ids:
                key = ids[0]
                examples.append((f'{candidate["label"]} {title}', key, baseline_preds[key]['sql'], preds[key]['sql']))
    lines += ['', '## 固定评测后的案例', '', '以下案例只用于事后分析，不进入训练、提示或控制器规则。', '']
    for title, key, before, after in examples[:4]:
        lines += [f"**{title}：{key}** — {test_rows[key]['question']}", '', '基线：', '```sql', before, '```', '挑战者：', '```sql', after, '```', '']
    best_preds = {p['id']: p for p in load(best['run'] + '/predictions.json')['predictions']}
    best_scores = load(best['run'] + '/score.json')['per_question']
    wrong = next((q for q in best_scores if not q['passed'] and any(e['tool'] == 'check'
                 and e['sql'] == best_preds[q['id']]['sql'] and e['observation']['matches_example']
                 for e in best_preds[q['id']]['trace'])), None)
    if wrong:
        key = wrong['id']
        lines += [f"**最佳方案仍然失败：{key}** — {test_rows[key]['question']}", '', '```sql', best_preds[key]['sql'], '```', '',
                  f"最终SQL通过公开check，但三个实例的通过标记为`{[c['passed'] for c in wrong['cases']]}`。这是有限公开反馈不能证明最终正确的实际反例，不能仅凭该标记确定全部语义根因。", '']
    lines += ['', '控制器在重复拒绝后触发的retrieve、在提交前触发的multiplicity execute，与Qwen自主选择动作分开统计。调用过工具不能证明学会有效规划；公开库的探查也无法证明额外实例上的语义正确。GRPO同分组没有区分正确策略的信号，且固定控制器跳过模型的路径无法依靠模型梯度修复。', '',
              '本轮二元终局奖励没有区分模型修复与控制器回退的贡献：如果不合格提议之后的回退恰好成功，整条轨迹仍获奖励。因此不能仅凭训练成功率声称模型学会了SQL推理、自我反思或主动验证；需要结合拒绝率、候选覆盖和真正自主动作检查。', '',
              '训练集只有480题且schema与验证/测试不同；奖励增长和训练loss变化不能替代正式成绩。已进行多轮公开验证与公开test自评，存在选择偏差。固定三实例评分仍是有限检验，不能宣称SQL对任意数据库都等价。', '',
              '## 后续实验优先级（尚未运行）', '',
              '1. 先补强正确SQL候选生成。仅从train轨迹整理失败SQL、题意、schema和真实反馈，以train参考答案及三个实例检验得到修正目标；重点覆盖多余JOIN、聚合粒度、条件丢失和去重。保留普通成功样本混合训练，避免只学难题后破坏已有能力。四条采样全错的题优先用监督信号补足，暂不直接扩大同类GRPO训练时长。', '',
              '2. 再做候选比较的受控训练。用train中“公开check匹配但额外实例失败”与“三实例通过”的候选构造对比，让同一个LoRA模型选择候选并指出可由题意/schema支持的差异；标签只由train离线评分产生。不能只要求模型多反思一次，也不能把缓存提示当答案。新增比较仍计入正式六次生成和八次工具预算。', '',
              '3. 最后尝试小规模GRPO复验。先检查组内奖励是否有差异，以及成功是否来自模型提议或控制器回退；候选覆盖改善后再比较GRPO是否带来额外收益。以train内部按schema留出的开发集初筛，官方validation确认，锁定一个候选后再做完整CPU测试。上述方向是基于本轮记录的待验证假设，不代表已取得提分。', '',
              '## 合规、风险与复现', '',
              '官方497个文件、基座9个文件、根README.md及README_cn.md核验不变。仍只使用一个.venv，没有安装TRL或升级依赖，没有peer_reviews、dist或上传。所有正式结果在8次收费工具、6次模型生成、每次2048新token、7200秒预测、180秒评分、16GiB RSS限制内；模型加载一次，单题记忆重新创建。', '',
              '问题、schema、检索内容、缓存提示均为不可信输入。固定system规则、只读SQL提取、动作字段校验、重复保护、预算限制及官方执行器继续约束行为；未完成独立对抗提示注入评测，不声称完全防御。检索可能引入不相关标识符，复查也可能强化同一模型的错误判断。', '',
              'Codex辅助设计、工程、实验与报告；正式推理不联系外部模型。参考DeepSeekMath（https://arxiv.org/abs/2402.03300）、ReEx-SQL（https://arxiv.org/abs/2505.12768）、CHASE-SQL（https://arxiv.org/abs/2410.01943）；只借鉴方法，不移用其成绩。保留NOTICE归属。', '',
              '[复现指南](../submission/REPRODUCE_V5.md) · [CPU矩阵](v5_cpu_matrix.json) · [预锁定记录](v5_test_lock.json) · [完整结果](v5_results.json) · [运行核验](v5_runtime_audit.json) · [train覆盖](v5_train_coverage.json) · [权重核验](v5_grpo_artifact_audit.json) · [未来源码清单](v5_delivery_manifest.json)']
    (ROOT / 'reports/v5_report.md').write_text('\n'.join(lines) + '\n')
    reproduce = f'''# V5复现

当前正式结果为{best['correct']}/240；完整对照、负结果、预算与数据来源见[报告](../reports/v5_report.md)。原始官方学生包必须位于项目根目录，保持mp3_sql及数据不变。全部使用同一个.venv。

```bash
bash scripts/setup_env.sh --cpu
unset PYTHONPATH
source .venv/bin/activate
python scripts/download_model.py
export MP3_BASE_MODEL="$PWD/artifacts/models/Qwen3-0.6B"
export MP3_ADAPTER_DIR="$PWD/{best['adapter']}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash scripts/run_selected.sh test
```

将所选adapter的adapter_model.safetensors和adapter_config.json放在上面的目录。它们独立于源码交付，不包含基座或合并权重。默认读取submission/configs/selected.json，每次产生新的runs/reproduction_selected/test/rNNN，不覆盖历史结果。运行器记录单次模型加载、完整提示和生成、CPU四线程、RSS及原样官方评分。

CPU矩阵对照：contract_v4_selected.json搭配旧adapter为A；search_v5_selected.json搭配旧adapter为B。将adapter改为reports/v5_grpo_selection.json所记录检查点，分别运行两个配置得到C/D。固定测试挑战者为e54_v5_arch_test和e55_v5_grpo_test；结果逐字节保留在submission/outputs/v5-arch-test.json与v5-grpo-test.json及各自score文件。

以下命令复现两组挑战者；若只跑验证，将--split test改成--split validation。需要分别准备旧adapter和GRPO的update_016两份独立权重目录；两次运行各加载一个模型，不做集成。

```bash
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/onpolicy_v001/epoch_02"
python scripts/run_experiment.py --experiment reproduction_v5_arch --split test --device cpu --config submission/configs/search_v5_selected.json
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/grpo_v5/update_016"
python scripts/run_experiment.py --experiment reproduction_v5_grpo --split test --device cpu --config submission/configs/grpo_v5_selected.json
```

本机GPU训练仍使用相同.venv；已有依赖即可，不依赖TRL。训练从旧onpolicy_v001/epoch_02开始，其来源见REPRODUCE_V2.md。

```bash
export MP3_BASE_MODEL="$PWD/artifacts/models/Qwen3-0.6B"
export MP3_ADAPTER_DIR="$PWD/artifacts/adapters/onpolicy_v001/epoch_02"
export MP3_DEVICE=cuda
python training/collect_v5.py --output runs/v5_train_coverage_reproduction --report reports/v5_train_coverage_reproduction.json
python training/train_grpo_v5.py --config training/configs/grpo_v5.json --run-name grpo_v5_reproduction
```

GRPO配置固定了train采样课程，不依赖后续覆盖诊断更新。训练脚本读取原有runs/e30_feedback_collection/train/r001/score.json（train派生辅助记录，提供在源码清单中），其最终hard_task_ids以配置为准。训练日志和源快照在runs/training/grpo_v5；训练命令中的新run-name避免覆盖旧目录。GPU采样数值可能随硬件或依赖不同而变化；推理复现应直接使用提交adapter。

所有训练仅使用train；额外实例由独立训练评分器提供终局奖励，不向模型暴露。正式推理保持离线、一个模型、官方工具接口；不导出数据集、.venv或基座。未打包到dist，未上传或进行互评。
'''
    (ROOT / 'submission/REPRODUCE_V5.md').write_text(reproduce)
    readme = ROOT / 'submission/README.md'; text = readme.read_text()
    intro_start = text.index('本目录包含学生Agent')
    intro_end = text.index('## 目录与命名', intro_start)
    intro = '本目录包含学生Agent；根README.md和README_cn.md保留官方原文及逐项翻译。最新实验见[第五轮报告](../reports/v5_report.md)，复现见[REPRODUCE_V5.md](REPRODUCE_V5.md)。所有命令从项目根目录执行，共用.venv，暂不创建peer_reviews和dist。历史报告与日志保留。\n\n'
    text = text[:intro_start] + intro + text[intro_end:]
    start = text.index('当前规范outputs/test.json')
    end = text.index('## 依次复现实验', start)
    current = f'''当前规范outputs/test.json、test-score.json是`{best['run']}`原始预测和完整官方评分的逐字节副本，正确{best['correct']}/240。V5两组挑战者及所有负结果均保留；[结果记录](../reports/v5_results.json)披露公开test聚合分数参与最终保留。

## 已选方案的快速复现

默认控制器`{best['config']['controller']}`，adapter为`{best['adapter']}`。

```bash
export MP3_ADAPTER_DIR="$PWD/{best['adapter']}"
bash scripts/run_selected.sh test
```

完整环境、权重与对照命令见[第五轮复现指南](REPRODUCE_V5.md)。此前各轮的配置和权重仍保留。

'''
    text = text[:start] + current + text[end:]
    row = f"| 第五轮架构挑战者 | {matrix[1]['correct']}/120 | {result['tests'][0]['correct']}/240 |\n| 第五轮GRPO挑战者 | {max(matrix[2]['correct'], matrix[3]['correct'])}/120 | {result['tests'][1]['correct']}/240 |\n"
    text = '\n'.join(line for line in text.splitlines() if not line.startswith('| 第五轮')) + '\n'
    text = text.replace('\n\n当前规范outputs/test.json', '\n' + row + '\n当前规范outputs/test.json', 1)
    marker = '## 第五轮搜索与GRPO'
    if marker in text:
        start = text.index(marker); end = text.index('## 学习重点', start)
        text = text[:start] + text[end:]
    fifth = f'''## 第五轮搜索与GRPO

五组架构开发对照比较检索恢复、额外复查、JOIN探查、始终复查和提示候选复查。训练集四轨迹诊断覆盖480题；GRPO在train上实际尝试{training['groups']}组，完成{training['updates']}组有效更新，额外评分实例只用于离线训练奖励，不进入Agent观察。

e50为CPU旧方案复现，e51确认架构，e52/e53比较同一GRPO权重的旧/新架构，e54/e55是预锁定的两组完整CPU测试。当前最佳{best['correct']}/240。权重、提示和工具是否改进，均以[第五轮报告](../reports/v5_report.md)的真实对照为准；训练完成不等于能力或分数提高。

'''
    text = text.replace('## 学习重点', fifth + '## 学习重点', 1)
    readme.write_text(text.rstrip() + '\n')
    print('V5_REPORT_AND_AUDIT_COMPLETE', best['correct'], flush=True)


if __name__ == '__main__':
    main()
