import copy
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'submission')]
from sql_repair_agent.semantic_agent import contract_edits,SemanticAgent
from sql_repair_agent.contract_agent import ContractAgent
from sql_repair_agent.recovery_agent import RecoveryAgent
from mp3_sql.environment import Episode,rollout,replay
from test_agent import FakeTools
from unittest.mock import patch

class ContractTests(unittest.TestCase):
    def obs(self,q,columns):return {'task':{'question':q},'example':{'columns':columns}}
    def test_projection_permutation_uses_public_column_contract(self):
        variants=contract_edits(self.obs('Count for each city',['COUNT(*)','city']),'SELECT city, COUNT(*) AS n FROM t GROUP BY city')
        self.assertEqual(variants[0][1],'projection_contract');self.assertTrue(variants[0][0].startswith('SELECT COUNT(*) AS n, city'))
    def test_explicit_distinct_can_be_proposed_even_if_example_has_no_duplicates(self):
        values=contract_edits(self.obs('Show distinct venues',['venue']),'SELECT venue FROM debate')
        self.assertIn('DISTINCT',values[0][0])
    def test_count_distinct_is_not_blindly_rewritten(self):
        self.assertEqual(contract_edits(self.obs('How many distinct venues?',['COUNT(*)']),'SELECT COUNT(*) FROM debate'),[])
    def test_already_aligned_and_duplicate_labels_do_not_generate_swaps(self):
        self.assertEqual(contract_edits(self.obs('Balances',['balance','balance','name']),'SELECT a.balance,b.balance,c.name FROM a,b,c'),[])
    def test_static_check_is_real_and_preserved_on_fast_submit(self):
        row={'id':'train-unit','question':'Show distinct venues','issue_sql':'SELECT venue FROM debate',
             'example':{'columns':['venue'],'rows':[[1]],'row_count':1},'ordered':False}
        tools=FakeTools();query=tools.query
        def both_match(row,sql,check):
            result=query(row,sql,check);result['matches_example']=True;return result
        tools.query=both_match
        agent=SemanticAgent({'max_completions':4,'max_new_tokens':384,'review_matched':'risk'})
        with patch.object(agent,'generate_proposal',side_effect=AssertionError('No model required')):
            result=rollout(row,tools,agent)
        self.assertIn('DISTINCT',result['sql']);self.assertEqual(result['tool_calls'],2)
        replay(row,tools,result)

    def test_static_proposal_does_not_mutate_observation(self):
        row={'id':'train-unit','question':'Show distinct venues','issue_sql':'SELECT venue FROM debate',
             'example':{'columns':['venue'],'rows':[[1]],'row_count':1},'ordered':False}
        ep=Episode(row,FakeTools());ep.step({'tool':'check','sql':row['issue_sql']});obs=ep.observation();before=copy.deepcopy(obs)
        action=SemanticAgent({'max_completions':4,'max_new_tokens':384}).act(obs)
        self.assertEqual(action['tool'],'check');self.assertEqual(obs,before)

    def test_contract_variant_keeps_original_recovery_prompt(self):
        row={'id':'train-unit','question':'Return 1','issue_sql':'SELECT 1',
             'example':{'columns':['x'],'rows':[[1]],'row_count':1},'ordered':False}
        ep=Episode(row,FakeTools());ep.step({'tool':'check','sql':'SELECT 1'});ep.step({'tool':'schema'})
        cfg={'max_completions':4,'max_new_tokens':384}
        self.assertEqual(ContractAgent(cfg).model_messages(ep.observation()),RecoveryAgent(cfg).model_messages(ep.observation()))

    def test_unknown_column_labels_do_not_invent_a_projection(self):
        self.assertEqual(contract_edits(self.obs('Names',['x','y']),'SELECT city, COUNT(*) FROM t'),[])

if __name__=='__main__':unittest.main()
