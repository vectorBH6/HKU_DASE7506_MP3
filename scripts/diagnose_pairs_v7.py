"""Full greedy rollouts on V7 synthetic pairs and separate support/witness grading."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]
from mp3_sql.common import BUDGET, digest, read_jsonl, result_fingerprint, write_json
from mp3_sql.environment import Tools, replay, rollout
from mp3_sql.predict import load_factory
from mp3_sql.sqlite_tool import execute_sql
from sql_repair_agent.review_agent import parse_proposal
from sql_repair_agent.sql_utils import sql_key

DATA = ROOT / "artifacts/prepared_data/v7"
WORK = ROOT / "runs/v7_pair_eval"
CONFIG = ROOT / "submission/configs/contract_v4_selected.json"
PUBLIC = DATA / "holdout/public.jsonl"
PRIVATE = DATA / "holdout/private.jsonl"
ARMS = ("baseline", "ordinary", "paired")


class PublicTools(Tools):
    """Preserve official executor, schema and check semantics on synthetic fixtures."""

    def __init__(self, rows):
        self.folder = DATA
        self.isolated = False
        self.cache = {}
        self.manifest = {"databases": {}}
        self.expected = {}
        for row in rows:
            example = row["example"]
            if len(example["rows"]) != example["row_count"]:
                raise ValueError("Synthetic public result must be completely visible")
            self.expected[row["id"]] = result_fingerprint(
                {"ok": True, "columns": example["columns"], "rows": example["rows"]}, row["ordered"])
            db = {"schema": row["schema"], "public_path": row["public_path"],
                  "public_sha256": digest(DATA / row["public_path"])}
            if self.manifest["databases"].setdefault(row["db_id"], db) != db:
                raise ValueError("Fixture id collision")

    def repair_hint(self, row):
        return []

    def retrieve(self, row, query):
        return []


def population():
    rows = read_jsonl(PUBLIC)
    ids = [r["id"] for r in rows]
    holdout_schemas = set(json.loads((ROOT / "artifacts/prepared_data/v6/split.json").read_text())["holdout_db_ids"])
    if not {r["source_db_id"] for r in rows} <= holdout_schemas:
        raise ValueError("Pair evaluation includes schemas outside the predeclared incremental holdout")
    groups = defaultdict(list)
    for row in rows:
        groups[row["pair_id"]].append(row)
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Empty population or duplicate task ID")
    for pair in groups.values():
        if len(pair) != 2 or pair[0]["question"] == pair[1]["question"]:
            raise ValueError("Each pair needs two distinct questions")
        if any(pair[0][k] != pair[1][k] for k in (
                "category", "issue_sql", "example", "ordered", "public_path", "schema", "db_id")):
            raise ValueError("Members must share the draft and public evidence")
    return rows


def predict(arm, adapter, device):
    rows = population()
    tools = PublicTools(rows)
    out = WORK / arm
    out.mkdir(parents=True, exist_ok=False)
    adapter = adapter.resolve()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "MP3_BASE_MODEL": str(ROOT / "artifacts/models/Qwen3-0.6B"),
                       "MP3_ADAPTER_DIR": str(adapter), "MP3_DEVICE": device,
                       "MP3_GENERATION_LOG": str(out / "generations.jsonl")})
    factory = load_factory(ROOT / "submission/submission.py")
    started = time.monotonic()
    with (out / "predictions.jsonl").open("x") as f:
        for row in rows:
            prediction = rollout(row, tools, factory(CONFIG), budget=BUDGET)
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            f.flush()
    write_json(out / "run.json", {
        "arm": arm, "device": device, "tasks": len(rows), "seconds": time.monotonic() - started,
        "adapter": str(adapter.relative_to(ROOT)), "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
        "base": "Qwen/Qwen3-0.6B", "public_sha256": digest(PUBLIC), "config_sha256": digest(CONFIG),
        "public_fixtures": {name: d["public_sha256"] for name, d in tools.manifest["databases"].items()},
        "predictions_sha256": digest(out / "predictions.jsonl"),
        "script_sha256": digest(Path(__file__)),
        "synthetic_tools": "Official Episode, rollout, schema, execute/check on separate synthetic public fixture. Synthetic hint and retrieve return empty lists; grading is a separate command.",
    })


def subset_metrics(members):
    by_pair = defaultdict(dict)
    for r in members:
        by_pair[r["pair_id"]][r["side"]] = r["correct"]
    if any(set(c) != {"keep", "repair"} for c in by_pair.values()):
        raise ValueError("Incomplete pair")
    return {"pairs": len(by_pair), "keep": sum(c["keep"] for c in by_pair.values()),
            "repair": sum(c["repair"] for c in by_pair.values()),
            "both": sum(all(c.values()) for c in by_pair.values()),
            "public_pass": sum(r["public_pass"] for r in members),
            "no_generation": sum(r["generations"] == 0 for r in members),
            "first_model_actions": {s: dict(Counter(r["first_model_action"] for r in members
                                                   if r["side"] == s)) for s in ("keep", "repair")}}


def grade(arm):
    rows = population()
    tools = PublicTools(rows)
    labels = {r["id"]: r for r in read_jsonl(PRIVATE)}
    out = WORK / arm
    meta = json.loads((out / "run.json").read_text())
    preds = read_jsonl(out / "predictions.jsonl")
    ids = [r["id"] for r in rows]
    if (meta["public_sha256"] != digest(PUBLIC) or set(labels) != set(ids) or
            [p["id"] for p in preds] != ids or meta["predictions_sha256"] != digest(out/"predictions.jsonl")):
        raise ValueError("Run/population mismatch")
    if meta["public_fixtures"] != {k: v["public_sha256"] for k, v in tools.manifest["databases"].items()}:
        raise ValueError("Public fixture changed after inference")
    gens = defaultdict(list)
    genfile = out / "generations.jsonl"
    for g in read_jsonl(genfile) if genfile.exists() else []:
        if g["identity"]["adapter"]["sha256"] != meta["adapter_sha256"]:
            raise ValueError("Unexpected adapter in generation log")
        gens[g["task_id"]].append(g)

    members = []
    fixture_hashes = {}
    for row, p in zip(rows, preds):
        lab = labels[row["id"]]
        if (lab["pair_id"] != row["pair_id"] or lab["category"] != row["category"]
                or lab["public_path"] != row["public_path"] or not lab["witness_paths"]):
            raise ValueError("Mismatched private labels")
        final = replay(row, tools, p)
        checks = [e for e in p["trace"] if e["tool"] == "check" and e["sql"] == final]
        if not checks:
            raise ValueError("Submitted SQL lacks matching check")
        public_pass = checks[-1]["observation"]["matches_example"]
        if not tools.query(row, lab["gold_sql"], True)["matches_example"]:
            raise ValueError("Reference failed the public support")
        if not tools.query(row, row["issue_sql"], True)["matches_example"]:
            raise ValueError("Prompt pair lacks deliberate support ambiguity")
        tables = [t["name"] for t in row["schema"]]
        witness_ok, draft_ok = [], []
        for relative in lab["witness_paths"]:
            path = DATA / relative
            fixture_hashes[relative] = digest(path)
            values = {}
            for sql in (lab["gold_sql"], row["issue_sql"], final):
                if sql not in values:
                    result = execute_sql(path, sql, tables)
                    values[sql] = result_fingerprint(result, row["ordered"]) if result["ok"] else None
            expected = values[lab["gold_sql"]]
            if expected is None:
                raise RuntimeError("Gold SQL failed on witness")
            witness_ok.append(values[final] == expected)
            draft_ok.append(values[row["issue_sql"]] == expected)
        if (all(draft_ok) != (lab["side"] == "keep")) or lab["side"] not in {"keep", "repair"}:
            raise ValueError("Witness does not validate the paired keep/repair label")
        gs = gens[row["id"]]
        first = (parse_proposal(gs[0]["generated_text"]) or {}).get("tool", "invalid") if gs else "no_generation"
        members.append({"id": row["id"], "pair_id": row["pair_id"], "category": row["category"],
                        "source_db_id": row["source_db_id"], "side": lab["side"],
                        "public_pass": public_pass, "witness_pass": witness_ok,
                        "correct": bool(public_pass and all(witness_ok)),
                        "same_draft": sql_key(final) == sql_key(row["issue_sql"]),
                        "tool_calls": p["tool_calls"], "generations": len(gs), "first_model_action": first})
    if any(r["generations"] > 4 or r["tool_calls"] > BUDGET for r in members):
        raise ValueError("Budget exceeded")
    category = sorted({r["category"] for r in members})
    summary = {"arm": arm, "overall": subset_metrics(members),
               "by_category": {k: subset_metrics([r for r in members if r["category"] == k]) for k in category},
               "generations": sum(r["generations"] for r in members),
               "charged_tools": sum(r["tool_calls"] for r in members), "members": members,
               "run": meta, "private_sha256": digest(PRIVATE), "witness_fixtures": fixture_hashes}
    write_json(out / "score.json", summary)
    print(json.dumps(summary["overall"], ensure_ascii=False))


def compare():
    scores = {arm: json.loads((WORK / arm / "score.json").read_text()) for arm in ARMS}
    base = scores["baseline"]
    for score in scores.values():
        for field in ("public_sha256", "config_sha256", "public_fixtures", "device", "script_sha256"):
            if score["run"][field] != base["run"][field]:
                raise ValueError("Inconsistent evaluation settings")
        if score["private_sha256"] != base["private_sha256"] or score["witness_fixtures"] != base["witness_fixtures"]:
            raise ValueError("Witnesses changed between arms")
    def successes(score):
        p = defaultdict(list)
        for member in score["members"]:
            p[member["pair_id"]].append(member["correct"])
        return {k for k, vals in p.items() if all(vals)}
    changes = {}
    for left, right in (("baseline","ordinary"),("baseline","paired"),("ordinary","paired")):
        a, b = successes(scores[left]), successes(scores[right])
        changes[f"{right}_vs_{left}"] = {"both_gained": len(b-a), "both_lost": len(a-b)}
    path = ROOT / "reports/v7_pairs_diagnostic.json"
    write_json(path, {"population": {"pairs": base["overall"]["pairs"],
                        "schemas": len({r["source_db_id"] for r in base["members"]}),
                        "public_sha256": base["run"]["public_sha256"], "private_sha256": base["private_sha256"]},
                     "method": "Full greedy controller. Each side must match support and every purpose-built hidden synthetic witness. Both means both sides independently correct on their own intent. First action refers to the raw first model generation; no_generation denotes controller-only decisions. This diagnostic evaluates train-origin synthetic schemas held out from incremental V7 training.",
                     "arms": {arm: {k: score[k] for k in ("overall","by_category","generations","charged_tools")}
                              for arm, score in scores.items()}, "pair_changes": changes})
    print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmds = p.add_subparsers(dest="stage", required=True)
    pred = cmds.add_parser("predict")
    pred.add_argument("--arm", choices=ARMS, required=True)
    pred.add_argument("--adapter", required=True, type=Path)
    pred.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    cmds.add_parser("grade").add_argument("--arm", choices=ARMS, required=True)
    cmds.add_parser("compare")
    args = p.parse_args()
    if args.stage == "predict":
        predict(args.arm, args.adapter, args.device)
    elif args.stage == "grade":
        grade(args.arm)
    else:
        compare()


if __name__ == "__main__":
    main()
