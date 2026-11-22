"""Build SFT states using training tasks and ONLY official public tools."""
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]
from mp3_sql.common import load_bank, read_jsonl, write_json, write_jsonl
from mp3_sql.environment import Episode, Tools
from sql_repair_agent.prompts import messages_for


def main():
    rows, manifest, folder = load_bank("train")
    memory = {r["id"]: r for r in read_jsonl(folder / "train_memory.jsonl")}
    tools = Tools(folder, manifest)
    def prepare(row):
        episode = Episode(row, tools)
        episode.step({"tool": "check", "sql": row["issue_sql"]})
        episode.step({"tool": "schema"})
        episode.step({"tool": "retrieve", "query": (row["question"] + " " + row["issue_sql"])[:2000]})
        target = memory[row["id"]]["gold_sql"]
        records = [{"id": row["id"], "state": "original_feedback", "messages": messages_for(episode.observation()), "target": target}]
        if not episode.trace[0]["observation"]["matches_example"]:
            episode.step({"tool": "repair_hint"})
            hints = episode.trace[-1]["observation"]
            if hints and hints[0] != row["issue_sql"]:
                episode.step({"tool": "check", "sql": hints[0]})
                if not episode.trace[-1]["observation"]["matches_example"]:
                    records.append({"id": row["id"], "state": "failed_hint_feedback", "messages": messages_for(episode.observation()), "target": target})
        episode.step({"tool": "submit", "sql": target})
        return records, {"id": row["id"], "trace": episode.trace}
    with ThreadPoolExecutor(max_workers=4) as pool:
        prepared = list(pool.map(prepare, rows))
    records = [record for group, trace in prepared for record in group]
    out = ROOT / "artifacts/prepared_data/sft_v001"
    write_jsonl(out / "train.jsonl", records)
    write_jsonl(out / "tool_traces.jsonl", [trace for group, trace in prepared])
    provenance = {"split": "train", "tasks": len(rows), "examples": len(records), "label_source": "train_memory.jsonl:gold_sql",
                  "feedback_source": "unchanged official Episode/Tools; public training databases only",
                  "validation_or_test_used": False,
                  "sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                             for name in ["train.jsonl", "train_memory.jsonl", "manifest.json", "repair_hints.json"]},
                  "prepared_sha256": hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()}
    write_json(out / "provenance.json", provenance)
    print(json.dumps(provenance, indent=2))

if __name__ == "__main__":
    main()
