"""Pool provenance, bounded generation, and SQLite semantics on hand-written data."""
import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submission'))
from sql_repair_agent.pool_edits_v8 import build_pool


class PoolEditsTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.db.executescript("""
            CREATE TABLE people(person_id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE orders(buyer INTEGER, amount INTEGER);
            INSERT INTO people VALUES (1,'A'),(2,'B'),(3,'C');
            INSERT INTO orders VALUES (1,25),(1,10),(2,30),(NULL,15);
        """)
        self.schema = [
            {'name': 'people', 'columns': [{'name': 'person_id', 'pk': True}, {'name': 'name'}]},
            {'name': 'orders', 'columns': [{'name': 'buyer'}, {'name': 'amount'}],
             'foreign_keys': [{'from': 'buyer', 'table': 'people', 'to': 'person_id'}]},
        ]

    def pool(self, sql, question, columns, hints=()):
        return build_pool({'issue_sql': sql, 'task': {'question': question},
                           'example': {'columns': columns}}, self.schema, hints)

    def results(self, pool, family):
        return [sorted(self.db.execute(row['sql']).fetchall(), key=repr)
                for row in pool if row['family'] == family]

    def test_budget_preserves_raw_sources_and_duplicate_hint_position(self):
        draft = 'select name from people'
        hints = [f'select name from people where person_id={i}' for i in range(52)]
        hints.insert(1, hints[0])
        pool = self.pool(draft, 'show identifiers', ['person_id'], hints)
        raw = [r for r in pool if r['family'] == 'raw']
        self.assertEqual(raw[0], {'sql': draft, 'source': 'draft', 'family': 'raw'})
        self.assertEqual([r['source'] for r in raw[1:]], ['hint_0', *[f'hint_{i}' for i in range(2,53)]])
        self.assertEqual(pool[:53], raw)
        self.assertEqual(len(pool) - len(raw), 48)

    def test_nullable_anti_membership_and_correlated_table_alias(self):
        sql = 'SELECT p.name FROM people p WHERE p.person_id NOT IN (SELECT o.buyer FROM orders o)'
        self.assertEqual(self.db.execute(sql).fetchall(), [])
        pool = self.pool(sql, 'names of people without orders', ['name'])
        self.assertIn([('C',)], self.results(pool, 'quantifier'))
        for draft in ['SELECT p.name FROM people p',
                      'SELECT p.name FROM people p JOIN orders o ON p.person_id=o.buyer']:
            self.assertIn([('C',)], self.results(self.pool(draft, 'people without orders', ['name']), 'relation'))

    def test_foreign_key_and_explicit_group_projection(self):
        wrong_join = 'SELECT p.name,o.amount FROM people p JOIN orders o ON p.person_id=o.amount'
        repaired = self.pool(wrong_join, 'show names and amounts', ['name','amount'])
        self.assertIn([('A',10), ('A',25), ('B',30)], self.results(repaired, 'fk_join'))
        draft = 'SELECT p.name,o.amount FROM people p JOIN orders o ON p.person_id=o.buyer'
        pool = self.pool(draft, 'total for each name', ['name','SUM(amount)'])
        self.assertIn([('A',35), ('B',30)], self.results(pool, 'grouping'))

    def test_scope_and_ordinal_references(self):
        for draft in ('SELECT name,person_id FROM people ORDER BY 1',
                      'WITH ids AS (SELECT name,person_id FROM people) SELECT name,person_id FROM ids',
                      'SELECT name,person_id FROM people UNION SELECT name,person_id FROM people'):
            self.assertEqual(len(self.pool(draft, 'show ids and names', ['person_id','name'])), 1)
