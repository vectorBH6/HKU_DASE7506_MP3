import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'submission')]
from mp3_sql.environment import rollout, replay, Episode
from sql_repair_agent.search_agent import SearchAgent, multiplicity_probe
from test_agent import FakeTools


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.row = {'id': 'train-unit', 'db_id': 'unit', 'question': 'How many numbers?',
                    'issue_sql': 'SELECT 1', 'ordered': False,
                    'example': {'columns': ['x'], 'rows': [[1]], 'row_count': 1, 'truncated': False}}
        self.cfg = {'max_completions': 6, 'max_new_tokens': 384, 'review_matched': 'always',
                    'second_pass': True, 'recover_retrieve': True, 'max_retries_per_observation': 1}

    def test_independent_review_is_bounded_and_does_not_mutate_observation(self):
        agent = SearchAgent(self.cfg)
        ep = Episode(self.row, FakeTools())
        ep.step({'tool': 'check', 'sql': 'SELECT 1'}); ep.step({'tool': 'schema'})
        obs = ep.observation(); before = copy.deepcopy(obs)
        agent.completions = 1
        action, reason = agent.guard(obs, {'tool': 'submit', 'candidate': 'c0'})
        self.assertIsNone(action); self.assertEqual(reason, 'independent_semantic_review')
        self.assertIn('independent_review', agent.model_messages(obs)[1]['content'])
        action, _ = agent.guard(obs, {'tool': 'submit', 'candidate': 'c0'})
        self.assertEqual(action['tool'], 'submit'); self.assertEqual(obs, before)

    def test_real_tool_trajectory_with_rejected_duplicate_and_retrieval(self):
        agent = SearchAgent(dict(self.cfg, second_pass=False))
        proposals = iter([{'tool': 'check', 'sql': 'SELECT 1'}] * 2 + [{'tool': 'submit', 'candidate': 'c0'}])
        def generate(obs):
            agent.completions += 1
            return next(proposals, None)
        tools = FakeTools()
        with patch.object(agent, 'generate_proposal', side_effect=generate):
            result = rollout(self.row, tools, agent)
        replay(self.row, tools, result)
        self.assertIn('retrieve', [e['tool'] for e in result['trace']])
        self.assertLessEqual(result['tool_calls'], 8); self.assertLessEqual(agent.completions, 6)

    def test_probe_uses_only_seen_schema_and_preserves_join_filter(self):
        obs = {'history': [{'tool': 'schema', 'observation': [
            {'name': 'employee', 'columns': [{'name': 'id', 'pk': 1}]}]}]}
        sql = 'SELECT COUNT(*) FROM employee e JOIN hiring h ON e.id=h.employee_id WHERE h.salary>10'
        probe = multiplicity_probe(obs, sql)
        self.assertIsNotNone(probe)
        self.assertIn('DISTINCT', probe); self.assertIn('salary > 10', probe)
        self.assertNotIn('GROUP BY', probe)
        self.assertIsNone(multiplicity_probe(obs, 'SELECT * FROM employee'))

    def test_hint_review_adds_real_feedback_before_selection(self):
        agent = SearchAgent(dict(self.cfg, second_pass=False, hint_review=True))
        proposals = iter([{'tool': 'submit', 'candidate': 'c0'}, {'tool': 'submit', 'candidate': 'c0'}])
        def generate(obs):
            agent.completions += 1
            return next(proposals, None)
        tools = FakeTools()
        tools.repair_hint = lambda row: ['SELECT 2']
        with patch.object(agent, 'generate_proposal', side_effect=generate):
            result = rollout(self.row, tools, agent)
        replay(self.row, tools, result)
        self.assertIn('repair_hint', [e['tool'] for e in result['trace']])
        self.assertTrue(any(e['tool'] == 'check' and e['sql'] == 'SELECT 2' for e in result['trace']))
        self.assertEqual(result['sql'], 'SELECT 1')
        self.assertEqual(agent.completions, 2)
        self.assertLessEqual(result['tool_calls'], 8)


if __name__ == '__main__':
    unittest.main()
