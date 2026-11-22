import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'submission')]
from mp3_sql.environment import rollout, replay, Episode
from sql_repair_agent.review_agent import ReviewAgent, parse_proposal
from test_agent import FakeTools

class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.row = {'id':'train-unit','db_id':'unit','question':'Return one number','issue_sql':'SELECT 1','ordered':False,
                    'example':{'columns':['x'],'rows':[[1]],'row_count':1,'truncated':False}}
        self.cfg = {'mode':'qwen','controller':'review_v2','review_matched':'always','max_completions':4,
                    'max_new_tokens':384,'use_hints':True,'use_retrieval':True,'use_variants':True}

    def run_proposals(self, row, proposals, tools=None):
        agent = ReviewAgent(self.cfg)
        values = iter(proposals)
        def generate(obs):
            agent.completions += 1
            return next(values, None)
        tools = tools or FakeTools()
        with patch.object(agent,'generate_proposal',side_effect=generate):
            result = rollout(row,tools,agent)
        replay(row,tools,result)
        return result,agent

    def test_public_match_is_reviewed_before_submit(self):
        result,agent = self.run_proposals(self.row,[{'tool':'submit','candidate':'c0'}])
        self.assertEqual([x['tool'] for x in result['trace']],['check','schema','submit'])
        self.assertEqual(agent.completions,1)

    def test_model_can_select_a_different_checked_matching_candidate(self):
        tools = FakeTools()
        original = tools.query
        def query(row,sql,check):
            feedback = original(row,sql,check)
            feedback['matches_example'] = sql in ['SELECT 1','SELECT 2']
            return feedback
        tools.query = query
        result,agent = self.run_proposals(self.row,[{'tool':'check','sql':'SELECT 2'},
                                                  {'tool':'submit','candidate':'c1'}],tools)
        self.assertEqual(result['sql'],'SELECT 2')
        self.assertEqual(result['tool_calls'],3)

    def test_invalid_proposal_does_not_regenerate_on_unchanged_input(self):
        result,agent = self.run_proposals(self.row,[{'tool':'check','sql':'SELECT 1'},
                                                  {'tool':'submit','candidate':'c99'},
                                                  {'tool':'check','sql':'SELECT 1; DROP TABLE x'},
                                                  {'tool':'filesystem','path':'test_references.json'}])
        self.assertEqual(agent.completions,1)
        self.assertEqual(result['tool_calls'],2)
        self.assertEqual(result['sql'],'SELECT 1')

    def test_protocol_guard_rejects_unsafe_unavailable_or_unchecked_choices(self):
        ep=Episode(self.row,FakeTools())
        ep.step({'tool':'check','sql':'SELECT 1'})
        ep.step({'tool':'schema'})
        ep.step({'tool':'check','sql':'SELECT 2'})
        agent=ReviewAgent(self.cfg)
        for proposal in [
            {'tool':'check','sql':' select 1 '},
            {'tool':'submit','candidate':'c99'},
            {'tool':'check','sql':'SELECT 1; DROP TABLE x'},
            {'tool':'execute','sql':'DELETE FROM x'},
            {'tool':'filesystem','path':'test_references.json'},
            {'tool':'submit','candidate':'c1'},
        ]:
            with self.subTest(proposal=proposal):
                action,reason=agent.guard(ep.observation(),proposal)
                self.assertIsNone(action)
        ep.step({'tool':'repair_hint'})
        action,reason=agent.guard(ep.observation(),{'tool':'repair_hint'})
        self.assertIsNone(action)
        self.assertEqual(reason,'unavailable_action')

    def test_risk_policy_skips_only_low_risk_matching_original(self):
        low=ReviewAgent(dict(self.cfg,review_matched='risk'))
        result=rollout(self.row,FakeTools(),low)
        self.assertEqual(result['tool_calls'],1)
        self.assertEqual(low.completions,0)
        high_row=dict(self.row,question='How many different numbers?')
        ep=Episode(high_row,FakeTools())
        ep.step({'tool':'check','sql':'SELECT 1'})
        high=ReviewAgent(dict(self.cfg,review_matched='risk'))
        self.assertEqual(high.act(ep.observation()),{'tool':'schema'})

    def test_fallback_respects_eight_tool_budget(self):
        row = dict(self.row,issue_sql='broken')
        result,agent = self.run_proposals(row,[{'tool':'execute','sql':f'SELECT {i}'} for i in range(2,6)])
        self.assertEqual(result['tool_calls'],8)
        self.assertEqual(agent.completions,4)
        self.assertEqual(result['sql'],'SELECT 1')

    def test_review_ablation_can_skip_matching_original(self):
        agent = ReviewAgent(dict(self.cfg,review_matched='off'))
        result=rollout(self.row,FakeTools(),agent)
        self.assertEqual(result['tool_calls'],1)
        self.assertEqual(agent.completions,0)

    def test_observation_unchanged(self):
        ep=Episode(self.row,FakeTools())
        ep.step({'tool':'check','sql':self.row['issue_sql']})
        ep.step({'tool':'schema'})
        obs=ep.observation();before=copy.deepcopy(obs)
        agent=ReviewAgent(self.cfg)
        with patch.object(agent,'generate_proposal',return_value={'tool':'submit','candidate':'c0'}):
            agent.act(obs)
        self.assertEqual(obs,before)

    def test_json_and_sql_only_proposals(self):
        self.assertEqual(parse_proposal('{"tool":"submit","candidate":"c0"}')['candidate'],'c0')
        self.assertEqual(parse_proposal('SELECT 2')['tool'],'check')
        self.assertIsNone(parse_proposal('DROP TABLE x'))

if __name__=='__main__': unittest.main()
