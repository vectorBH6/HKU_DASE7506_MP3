import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]

from mp3_sql.environment import replay, rollout
from mp3_sql.predict import load_factory
from sql_repair_agent.contract_agent import ContractAgent
from sql_repair_agent.sql_utils import sql_key
from test_agent import FakeTools


class ReviewPoolTests(unittest.TestCase):
    row = {"id": "unit", "question": "Return the number", "db_id": "example", "issue_sql": "SELECT 1",
           "ordered": False, "example": {"columns": ["n"], "rows": [[1]], "row_count": 1}}
    draft = {"tool": "check", "sql": "SELECT 1"}

    def run_arm(self, policy, hints, matching, *, budget=8, prefix=None, schema=None):
        prefix = [self.draft] if prefix is None else prefix
        create_agent = load_factory(ROOT / "submission/submission.py")
        agent = create_agent(ROOT / "submission/configs" / f"v8_{policy}.json")
        tools = FakeTools()
        tools.schema = lambda row: schema or []
        tools.repair_hint = lambda row: hints
        regular = tools.query
        def query(row, sql, check):
            result = regular(row, sql, check)
            if check:
                result["matches_example"] = sql_key(sql) in matching
            return result
        tools.query = query
        calls = prefix + [{"tool": "submit", "sql": self.row["issue_sql"]}]
        with patch.object(ContractAgent, "act", side_effect=calls) as prefix_act:
            result = rollout(self.row, tools, agent, budget)
        self.assertEqual(prefix_act.call_count, len(calls))
        self.assertEqual(replay(self.row, tools, result, budget), result["sql"])
        self.assertEqual([{k: v for k, v in e.items() if k != "observation"}
                          for e in result["trace"][:len(prefix)]], prefix)
        return result

    def test_first_and_choose_have_identical_calls_and_different_answers(self):
        hints = ["select 1;", "DROP TABLE junk", "SELECT 2", "select 2;", "SELECT 3"]
        matching = {"SELECT 1", "SELECT 2", "SELECT 3"}
        first = self.run_arm("first", hints, matching)
        with patch("sql_repair_agent.review_pool_v8.rank_scores", return_value=[.6, .65, .85]) as rank:
            chosen = self.run_arm("choose", hints, matching)
        self.assertEqual(first["trace"][:-1], chosen["trace"][:-1])
        self.assertEqual(first["tool_calls"], 4)
        self.assertEqual(first["sql"], "SELECT 2")
        self.assertEqual(chosen["sql"], "SELECT 3")
        _, schema, candidates, stage = rank.call_args.args
        self.assertEqual((schema, stage), ([], "choose"))
        self.assertEqual([c["source"] for c in candidates], ["draft", "hint_2", "hint_4"])

    def test_score_margin_retains_old_match_without_changing_charge(self):
        with patch("sql_repair_agent.review_pool_v8.rank_scores", return_value=[.65, .76]):
            result = self.run_arm("choose", ["SELECT 2"], {"SELECT 1", "SELECT 2"})
        self.assertEqual(result["sql"], "SELECT 1")
        self.assertEqual(result["tool_calls"], 3)

    def test_no_new_match_preserves_answer_and_no_margin_for_failed_draft(self):
        result = self.run_arm("first", ["SELECT 2"], set())
        self.assertEqual(result["sql"], "SELECT 1")
        with patch("sql_repair_agent.review_pool_v8.rank_scores", return_value=[.01]):
            result = self.run_arm("choose", ["SELECT 2"], {"SELECT 2"})
        self.assertEqual(result["sql"], "SELECT 2")

    def test_real_hint_only_requested_when_two_tools_remain(self):
        for budget in [1, 2]:
            with self.subTest(budget=budget):
                result = self.run_arm("first", ["SELECT 2"], {"SELECT 2"}, budget=budget)
                self.assertEqual(result["tool_calls"], 1)
        result = self.run_arm("first", ["SELECT 2", "SELECT 3"], {"SELECT 3"}, budget=3)
        self.assertEqual(result["sql"], "SELECT 1")
        self.assertEqual(result["trace"][-2]["sql"], "SELECT 2")
        self.assertEqual(result["tool_calls"], 3)

    def test_hints_and_schema_already_in_prefix_are_reused(self):
        prefix = [self.draft, {"tool": "schema"}, {"tool": "repair_hint"}, {"tool": "check", "sql": "SELECT 2"}]
        schema = [{"name": "things", "columns": [], "foreign_keys": []}]
        with patch("sql_repair_agent.review_pool_v8.rank_scores", return_value=[.15, .8]) as rank:
            result = self.run_arm("choose", ["select 2;", "SELECT 3"], {"SELECT 2", "SELECT 3"},
                                  prefix=prefix, schema=schema, budget=5)
        self.assertEqual(result["sql"], "SELECT 3")
        self.assertEqual(result["trace"][-2]["sql"], "SELECT 3")
        self.assertEqual(rank.call_args.args[1], schema)

    def test_struct_checks_ranked_edits_after_raw_and_selects_only_matches(self):
        pool = [{"sql": f"SELECT {number}", "family": family, "source": "draft"}
                for number, family in [(1, "raw"), (3, "projection"), (4, "relation"), (5, "aggregate")]]
        def score(obs, schema, candidates, stage):
            scores = {"check": {"SELECT 3": .4, "SELECT 4": .9, "SELECT 5": .2},
                      "choose": {"SELECT 3": .8, "SELECT 4": .6}}[stage]
            return [scores[c["sql"]] for c in candidates]
        with patch("sql_repair_agent.review_pool_v8.build_pool", return_value=pool) as build, \
                patch("sql_repair_agent.review_pool_v8.rank_scores", side_effect=score):
            result = self.run_arm("struct", ["SELECT 2"], {"SELECT 3", "SELECT 4"}, budget=5)
        self.assertEqual([e["sql"] for e in result["trace"] if e["tool"] == "check"],
                         ["SELECT 1", "SELECT 2", "SELECT 4", "SELECT 3"])
        self.assertEqual(result["sql"], "SELECT 3")
        self.assertEqual(build.call_count, 1)

    def test_struct_skips_if_original_or_new_hint_matches(self):
        with patch("sql_repair_agent.review_pool_v8.build_pool", side_effect=AssertionError("unexpected edit")):
            for matching in [{"SELECT 1"}, {"SELECT 2"}]:
                with self.subTest(matching=matching):
                    result = self.run_arm("struct", ["SELECT 2"], matching)
                    self.assertIn(result["sql"], matching)


if __name__ == "__main__":
    unittest.main()
