import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'submission')]
from mp3_sql.environment import rollout,replay
from sql_repair_agent.plan_agent import PlanAgent
from sql_repair_agent.plan_constraint_agent import PlanConstraintAgent
from sql_repair_agent.review_agent import parse_proposal
from test_agent import FakeTools


class PlanProtocolTests(unittest.TestCase):
    def test_internal_plan_is_removed_before_official_tool_replay(self):
        row={'id':'train-unit','db_id':'unit','question':'Return one number','issue_sql':'SELECT 1',
             'ordered':False,'example':{'columns':['x'],'rows':[[1]],'row_count':1,'truncated':False}}
        cfg={'max_completions':4,'max_new_tokens':512,'review_matched':'always','use_hints':True}
        for cls in [PlanAgent,PlanConstraintAgent]:
            with self.subTest(controller=cls.__name__):
                agent=cls(cfg);tools=FakeTools()
                proposals=iter(['{"tool":"check","reason_code":"probe","plan":{"project":["2"]},"sql":"SELECT 2"}',
                                '{"tool":"submit","candidate":"c0"}'])
                def generate(obs):
                    agent.completions+=1
                    return parse_proposal(next(proposals))
                with patch.object(agent,'generate_proposal',side_effect=generate):
                    result=rollout(row,tools,agent)
                self.assertEqual(agent.completions,2)
                self.assertEqual([e['tool'] for e in result['trace']],['check','schema','check','submit'])
                self.assertNotIn('plan',str(result['trace']))
                replay(row,tools,result)

if __name__=='__main__':unittest.main()
