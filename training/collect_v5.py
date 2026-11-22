"""Independent complete train episodes; no validation/test labels or gold prompts."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]
from mp3_sql.common import write_json
from mp3_sql.environment import Tools, rollout
from submission import create_agent
from training.v5_utils import TrainReward, read_lines, append_line
from sql_repair_agent.contract_agent import ContractAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "runs/v5_train_coverage")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--limit", type=int, default=480)
    ap.add_argument("--report", type=Path, default=ROOT / "reports/v5_train_coverage.json")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MP3_GENERATION_LOG"] = str(args.output / "generations.jsonl")
    reward = TrainReward()
    tools = Tools(reward.folder, reward.manifest, cache={})
    config = json.loads((ROOT / "submission/configs/contract_v4_selected.json").read_text())
    config.update({"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0})
    write_json(args.output / "collection_config.json", {"config": config, "groups": args.groups,
               "train_only": True, "limit": args.limit, "actor_receives_references": False,
               "reward": "unchanged official train replay and all-three-instance grading"})
    completed = {(r["id"], r["sample"]) for r in read_lines(args.output / "episodes.jsonl")}
    started = time.perf_counter()
    for row in reward.rows[:args.limit]:
        for sample in range(args.groups):
            if (row["id"], sample) in completed:
                continue
            agent = ContractAgent({**config, "sample_index": sample})
            prediction = rollout(row, tools, agent)
            label = reward.score(prediction)
            candidates = reward.candidates(prediction)
            item = {"id": row["id"], "db_id": row["db_id"], "sample": sample,
                    "passed": label["passed"], "prediction": prediction,
                    "checked_candidates": candidates, "completions": agent.completions}
            append_line(args.output / "episodes.jsonl", item)
        print("TASK_DONE", row["id"], "seconds", round(time.perf_counter() - started, 2), flush=True)
    records = read_lines(args.output / "episodes.jsonl")
    groups = {}
    for r in records:
        groups.setdefault(r["id"], []).append(r)
    summary = {"train_only": True, "tasks": len(groups), "episodes": len(records),
               "sampled_final_correct": sum(r["passed"] for r in records),
               "tasks_with_correct_final": sum(any(r["passed"] for r in g) for g in groups.values()),
               "tasks_with_correct_checked_candidate": sum(any(c["passed"] for r in g for c in r["checked_candidates"]) for g in groups.values()),
               "mixed_reward_groups": sum(0 < sum(r["passed"] for r in g) < len(g) for g in groups.values()),
               "all_failed_groups": [k for k, g in groups.items() if not any(r["passed"] for r in g)],
               "public_match_wrong_candidates": sum(c["matches_example"] and not c["passed"] for r in records for c in r["checked_candidates"]),
               "seconds_this_invocation": time.perf_counter() - started,
               "episodes_sha256": hashlib.sha256((args.output / "episodes.jsonl").read_bytes()).hexdigest()}
    write_json(args.report, summary)
    print("COVERAGE_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
