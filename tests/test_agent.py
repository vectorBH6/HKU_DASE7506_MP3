import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]
from mp3_sql.environment import Episode, rollout, replay
from sql_repair_agent.agent import RepairAgent

class FakeTools:
    def schema(self, row): return []
    def query(self, row, sql, check):
        return {"ok": sql != "broken", "columns": ["x"], "rows": [[1]], "row_count": 1,
                "truncated": False, "error": "bad syntax" if sql == "broken" else None,
                "matches_example": sql == "SELECT 1"}
    def repair_hint(self, row): return ["SELECT 1"] * 4
    def retrieve(self, row, query): return []

class AgentTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": "train-unit", "db_id": "unit", "question": "Return 1", "issue_sql": "broken",
                    "ordered": False, "example": {"columns": ["x"], "rows": [[1]], "row_count": 1, "truncated": False}}
        self.config = {"mode": "feedback", "use_hints": True, "use_variants": True}

    def test_feedback_changes_decision_and_trace_replays(self):
        tools = FakeTools()
        failed = rollout(self.row, tools, RepairAgent(self.config))
        self.assertEqual(failed["sql"], "SELECT 1")
        self.assertEqual([x["tool"] for x in failed["trace"]], ["check", "schema", "repair_hint", "check", "submit"])
        self.assertEqual(replay(self.row, tools, failed), "SELECT 1")
        correct = dict(self.row, issue_sql="SELECT 1")
        passed = rollout(correct, tools, RepairAgent(self.config))
        self.assertEqual(passed["tool_calls"], 1)

    def test_duplicate_or_invalid_model_respects_budget(self):
        config = dict(self.config, mode="qwen", max_completions=2, max_new_tokens=256, use_retrieval=True)
        agent = RepairAgent(config)
        def generate(observation):
            agent.completions += 1
            return "SELECT 2"
        tools = FakeTools()
        tools.repair_hint = lambda row: ["SELECT 2"] * 4
        with patch.object(agent, "generate", side_effect=generate):
            result = rollout(self.row, tools, agent)
        self.assertEqual(agent.completions, 2)
        self.assertLessEqual(result["tool_calls"], 8)
        actions = [str({k:v for k,v in e.items() if k != "observation"}) for e in result["trace"][:-1]]
        self.assertEqual(len(actions), len(set(actions)))
        replay(self.row, tools, result)

    def test_observation_is_not_mutated(self):
        obs = Episode(self.row, FakeTools()).observation()
        before = copy.deepcopy(obs)
        RepairAgent(self.config).act(obs)
        self.assertEqual(obs, before)

if __name__ == "__main__": unittest.main()
