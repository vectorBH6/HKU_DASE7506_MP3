import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'submission')]
from mp3_sql.environment import Episode,rollout,replay
from sql_repair_agent.recovery_agent import RecoveryAgent
from sql_repair_agent.review_prompts import messages_for
from test_agent import FakeTools


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.row={'id':'train-unit','db_id':'unit','question':'How many numbers?',
                  'issue_sql':'SELECT 1','ordered':False,
                  'example':{'columns':['x'],'rows':[[1]],'row_count':1,'truncated':False}}
        self.cfg={'mode':'qwen','controller':'recovery_v3','review_matched':'always',
                  'max_completions':4,'max_new_tokens':384,'max_retries_per_observation':1,
                  'use_hints':True,'use_retrieval':True,'use_variants':True}

    def run_actions(self,row,proposals,tools=None,config=None):
        agent=RecoveryAgent(config or self.cfg);messages=[];values=iter(proposals)
        def generate(obs):
            agent.completions+=1
            messages.append(agent.model_messages(obs))
            return next(values,None)
        tools=tools or FakeTools()
        with patch.object(agent,'generate_proposal',side_effect=generate):
            result=rollout(row,tools,agent)
        replay(row,tools,result)
        return result,agent,messages

    def test_prompt_is_v2_until_first_rejection(self):
        ep=Episode(self.row,FakeTools());ep.step({'tool':'check','sql':'SELECT 1'});ep.step({'tool':'schema'})
        agent=RecoveryAgent(self.cfg)
        self.assertEqual(agent.model_messages(ep.observation()),messages_for(ep.observation(),self.cfg))

    def test_rejected_duplicate_gets_new_input_and_can_repair(self):
        tools=FakeTools();query=tools.query
        def both_match(row,sql,check):
            result=query(row,sql,check);result['matches_example']=sql in ('SELECT 1','SELECT 2');return result
        tools.query=both_match
        result,agent,messages=self.run_actions(self.row,[{'tool':'check','sql':'SELECT 1'},
            {'tool':'check','sql':'SELECT 2'},{'tool':'submit','candidate':'c1'}],tools)
        self.assertEqual(result['sql'],'SELECT 2');self.assertEqual(agent.completions,3)
        self.assertEqual(result['tool_calls'],3)
        self.assertNotEqual(messages[0],messages[1])
        payload=json.loads(messages[1][1]['content'].split('TASK_DATA\n',1)[1].split('\nEND_TASK_DATA')[0])
        feedback=payload['controller_feedback']
        self.assertEqual(feedback['rejected_proposals'][-1]['reason'],'duplicate_action')
        self.assertEqual(feedback['model_generations_after_this'],2)

    def test_repeated_invalid_proposals_have_bounded_recovery(self):
        result,agent,messages=self.run_actions(self.row,[{'tool':'check','sql':'SELECT 1'}]*6)
        self.assertEqual(agent.completions,2)
        self.assertEqual(result['tool_calls'],2)
        self.assertEqual(result['sql'],'SELECT 1')
        self.assertEqual(len(agent.rejections),2)
        self.assertNotEqual(messages[0],messages[1])

    def test_no_observation_mutation_and_sql_error_visible(self):
        row=dict(self.row,issue_sql='broken');ep=Episode(row,FakeTools())
        ep.step({'tool':'check','sql':'broken'});ep.step({'tool':'schema'})
        obs=ep.observation();before=copy.deepcopy(obs);agent=RecoveryAgent(self.cfg)
        agent.completions=1;agent.rejection_record({'tool':'check','sql':'broken'},'invalid_query')
        payload=json.loads(agent.model_messages(obs)[1]['content'].split('TASK_DATA\n',1)[1].split('\nEND_TASK_DATA')[0])
        self.assertEqual(payload['controller_feedback']['latest_sql_error']['error'],'bad syntax')
        self.assertEqual(obs,before)

    def test_four_and_six_completions_stay_inside_eight_tools(self):
        row=dict(self.row,issue_sql='broken')
        proposals=[{'tool':'execute','sql':f'SELECT {i}'} for i in range(2,6)]+[
            {'tool':'repair_hint'},{'tool':'check','sql':'SELECT 1'}]
        for limit in [4,6]:
            with self.subTest(limit=limit):
                result,agent,_=self.run_actions(row,proposals,config=dict(self.cfg,max_completions=limit))
                self.assertEqual(agent.completions,limit)
                self.assertEqual(result['tool_calls'],8)
                self.assertEqual(result['sql'],'SELECT 1')

    def test_unavailable_tool_recovery_returns_only_official_action(self):
        result,agent,_=self.run_actions(self.row,[{'tool':'filesystem','path':'not-a-tool'},
                                                        {'tool':'submit','candidate':'c0'}])
        self.assertEqual(agent.completions,2)
        self.assertEqual([x['tool'] for x in result['trace']],['check','schema','submit'])

    def test_rejections_are_fresh_per_task(self):
        one=RecoveryAgent(self.cfg);one.rejection_record({'tool':'check','sql':'SELECT 1'},'duplicate_action')
        self.assertEqual(RecoveryAgent(self.cfg).rejections,[])


if __name__=='__main__':unittest.main()
