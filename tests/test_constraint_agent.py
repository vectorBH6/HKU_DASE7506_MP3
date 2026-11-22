import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'submission')]
from sql_repair_agent.constraint_agent import explicit_edits

class ConstraintTests(unittest.TestCase):
    def obs(self,question):return {'task':{'question':question},'example':{'columns':['n']}}
    def test_inclusive_lower_bound(self):
        value=explicit_edits(self.obs('At least two employees'),'SELECT COUNT(*) FROM employee HAVING COUNT(*) > 2')
        self.assertIn('>= 2',value[0][0])
    def test_upper_bound_long_phrase_is_not_read_as_more_than(self):
        value=explicit_edits(self.obs('No more than 3 employees'),'SELECT COUNT(*) FROM employee HAVING COUNT(*) < 3')
        self.assertIn('<= 3',value[0][0])
    def test_strict_boundary(self):
        value=explicit_edits(self.obs('Older than 30'),'SELECT name FROM employee WHERE age >= 30')
        self.assertIn('> 30',value[0][0])
    def test_ambiguous_comparisons_are_not_rewritten(self):
        self.assertEqual(explicit_edits(self.obs('At least two'), 'SELECT a FROM t WHERE a > 2 AND b > 2'),[])
    def test_unrelated_threshold_and_correct_predicate_are_unchanged(self):
        self.assertEqual(explicit_edits(self.obs('At least two'),'SELECT a FROM t WHERE a > 5'),[])
        self.assertEqual(explicit_edits(self.obs('At least two'),'SELECT a FROM t WHERE a >= 2'),[])
    def test_explicit_distinct_count_is_local_not_a_star_guess(self):
        self.assertIn('COUNT(DISTINCT city)',explicit_edits(self.obs('How many different cities?'),'SELECT COUNT(city) FROM employee')[0][0])
        self.assertEqual(explicit_edits(self.obs('How many different cities?'),'SELECT COUNT(*) FROM employee'),[])
        self.assertEqual(explicit_edits(self.obs('Count cities different from Rome'),'SELECT COUNT(city) FROM employee'),[])

if __name__=='__main__':unittest.main()
