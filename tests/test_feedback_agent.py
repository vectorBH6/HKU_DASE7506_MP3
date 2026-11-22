import copy
import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'submission')]
from sql_repair_agent.feedback_agent import FeedbackAgent,sql_diagnostics,enrich_messages
from sql_repair_agent.review_prompts import messages_for
from training.prepare_feedback_v4 import mutations
from training.finalize_feedback_v4 import diagnose

SCHEMA=[{'name':'employee','columns':[{'name':'id','type':'INTEGER','pk':1},{'name':'city','type':'TEXT','pk':0}]}]

class FeedbackTests(unittest.TestCase):
    def test_binding_understands_alias_and_detects_missing_column(self):
        good=sql_diagnostics('SELECT e.city FROM employee AS e',SCHEMA)
        bad=sql_diagnostics('SELECT e.missing FROM employee AS e',SCHEMA)
        self.assertEqual(good.get('schema_binding'),'resolved')
        self.assertIn('schema_binding_warning',bad)

    def test_sqlite_double_quoted_values_are_not_false_column_errors(self):
        result=sql_diagnostics('SELECT id FROM employee WHERE city = "Helsinki"',SCHEMA)
        self.assertNotIn('schema_binding_warning',result)
        self.assertIn('SQLite',result['schema_binding'])

    def test_diagnosis_accepts_official_structured_errors(self):
        self.assertEqual(diagnose('SELECT x','SELECT 1',{'error':{'message':'no such column: x'}}),'repair_identifier')

    def test_structural_warning_is_not_correctness(self):
        result=sql_diagnostics('SELECT COUNT(*) FROM employee e JOIN employee x ON e.id=x.id',SCHEMA)
        self.assertTrue(result['flags']);self.assertNotIn('correct',result)
        self.assertEqual(result['schema_binding'],'resolved')

    def test_enrichment_preserves_observation_and_candidate_ids(self):
        obs={'task':{'id':'train-unit','question':'How many employees?'},'issue_sql':'SELECT COUNT(*) FROM employee',
             'example':{'columns':['COUNT(*)'],'row_count':1,'rows':[[1]]},'ordered':False,'remaining':6,
             'history':[{'tool':'check','sql':'SELECT COUNT(*) FROM employee','observation':{'ok':True,'matches_example':True,'columns':['COUNT(*)'],'rows':[[1]],'row_count':1}},
                        {'tool':'schema','observation':SCHEMA}]}
        before=copy.deepcopy(obs);messages=enrich_messages(messages_for(obs,{}),obs)
        payload=json.loads(messages[1]['content'].split('TASK_DATA\n',1)[1].split('\nEND_TASK_DATA',1)[0])
        self.assertEqual(obs,before);self.assertEqual(payload['checked_candidates'][0]['id'],'c0')
        self.assertIn('c0',payload['sql_diagnostics'])

    def test_perturbations_cover_boundaries_and_column_order(self):
        values=mutations('SELECT city, COUNT(id) FROM employee WHERE id > 2 GROUP BY city ORDER BY COUNT(id) DESC LIMIT 1')
        kinds={k for k,_ in values}
        self.assertIn('predicate_boundary',kinds);self.assertIn('projection_order',kinds)
        self.assertIn('ordering',kinds);self.assertEqual(len(values),len({s for _,s in values}))

    def test_model_candidate_preference_is_explicit_and_task_local(self):
        cfg={'max_completions':4,'max_new_tokens':384,'prefer_model_candidate':True}
        a=FeedbackAgent(cfg);b=FeedbackAgent(cfg)
        a.audit({'task':{'id':'train-unit','question':''},'history':[]},None,{'tool':'check','sql':'SELECT 1'},'model_query')
        self.assertEqual(a.model_queries,['SELECT 1']);self.assertEqual(b.model_queries,[])

if __name__=='__main__':unittest.main()
