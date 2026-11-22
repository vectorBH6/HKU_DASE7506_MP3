"""Measure immediate correct SQL proposals on the same observed train failure states."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]
from mp3_sql.common import BUDGET, digest, read_jsonl, write_json, write_jsonl
from sql_repair_agent.review_agent import parse_proposal
from sql_repair_agent.sql_utils import extract_sql, sql_key

WORK = ROOT / "runs/v6_fixed_repair_diagnostic"
SOURCE = ROOT / "artifacts/prepared_data/v6/generator/holdout.jsonl"
COVERAGE = ROOT / "runs/v5_train_coverage"
CONFIG = ROOT / "submission/configs/contract_v4_selected.json"
ADAPTERS = {"old": "artifacts/adapters/onpolicy_v001/epoch_02",
            "generator": "artifacts/adapters/v6_generator/epoch_01"}


def payload(messages):
    return json.loads(messages[1]["content"].split("TASK_DATA\n", 1)[1].split("\nEND_TASK_DATA", 1)[0])


def prepare():
    holdout = set(json.loads((SOURCE.parent.parent / "split.json").read_text())["holdout_ids"])
    rows = [r for r in read_jsonl(SOURCE) if r["state"] == "repair_failed_context"]
    rows.sort(key=lambda r: (r["id"], r["sample"], r["attempt"]))
    chosen = {}
    for row in rows:
        chosen.setdefault(row["id"], row)
    sources = set()
    with (COVERAGE / "generations.jsonl").open() as handle:
        for line in handle:
            original = json.loads(line)
            task = original["task_id"]
            if task not in chosen:
                continue
            r = chosen[task]
            seed = int(hashlib.sha256(f"42:{task}:{r['attempt']}:{r['sample']}".encode()).hexdigest()[:8], 16)
            if original["generation"]["seed"] == seed and original["attempt"] == r["attempt"]:
                if r["messages"] != original["messages"] or task in sources:
                    raise ValueError("Generation provenance mismatch")
                sources.add(task)
    if sources != chosen.keys() or not sources <= holdout:
        raise ValueError("Missing sampling provenance or held-out task")
    episode_rows = {(r["id"], r["sample"]): r for r in read_jsonl(COVERAGE / "episodes.jsonl")}
    prompts, prefixes = [], []
    for task, r in chosen.items():
        episode = episode_rows[task, r["sample"]]
        p = payload(r["messages"])
        used = BUDGET - p["remaining_tool_calls"]
        trace = episode["prediction"]["trace"][:used]
        checks = [e for e in trace if e["tool"] == "check"]
        if not checks or not 0 < used < BUDGET or any(e["tool"] == "submit" for e in trace):
            raise ValueError("Prefix does not allow a fresh check")
        got = [(sql_key(e["sql"]), e["observation"]) for e in checks]
        expected = [(sql_key(c["sql"]), c["feedback"]) for c in p["checked_candidates"]]
        if expected != got:
            raise ValueError("Prompt check list does not match the historical trace prefix")
        truth = {sql_key(c["sql"]): c["passed"] for c in episode["checked_candidates"]}
        if episode["passed"] or any(truth[sql] for sql, _ in got):
            raise ValueError("Historical state already contains a correct candidate or ended successfully")
        prefixes.append({"id": task, "sample": r["sample"], "attempt": r["attempt"], "trace": trace})
        prompts.append({"id": task, "messages": r["messages"], "attempt": r["attempt"]})
    WORK.mkdir(exist_ok=False)
    write_jsonl(WORK / "prompts.jsonl", prompts)
    write_jsonl(WORK / "prefixes.jsonl", prefixes)
    write_json(WORK / "population.json", {"source": str(SOURCE.relative_to(ROOT)), "contexts": len(rows),
               "distinct_tasks": len(prompts), "schemas": len({r["db_id"] for r in chosen.values()}),
               "rule": "All incremental train-holdout tasks with repair_failed_context. One earliest sample, "
                       "then earliest attempt, per task. No correct check yet; original episode failed.",
               "source_sha256": digest(SOURCE), "prompts_sha256": digest(WORK / "prompts.jsonl"),
               "prefixes_sha256": digest(WORK / "prefixes.jsonl")})


def infer(arm):
    from sql_repair_agent.model import Generator

    out = WORK / arm
    out.mkdir(exist_ok=True)
    log = out / "generations.jsonl"
    adapter = ROOT / ADAPTERS[arm]
    config = json.loads(CONFIG.read_text())
    if config["max_new_tokens"] != 384 or config["max_completions"] != 4 or config.get("do_sample"):
        raise ValueError("Expected the unchanged greedy configuration")
    prompts = read_jsonl(WORK / "prompts.jsonl")
    sha = digest(adapter / "adapter_model.safetensors")
    previous = read_jsonl(log) if log.exists() else []
    if len(previous) > len(prompts):
        raise ValueError("Invalid existing log length")
    for g, p in zip(previous, prompts):
        if (g["messages"] != p["messages"] or g["task_id"] != p["id"] or g["attempt"] != p["attempt"]
                or g["identity"]["adapter"]["sha256"] != sha or g["identity"]["device"] != "cuda"):
            raise ValueError("Prior generation log belongs to a different run")
    os.environ.update(MP3_BASE_MODEL=str(ROOT / "artifacts/models/Qwen3-0.6B"), MP3_DEVICE="cuda",
                      MP3_ADAPTER_DIR=str(adapter), MP3_GENERATION_LOG=str(log),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if len(previous) < len(prompts):
        model = Generator(True)
        for p in prompts[len(previous):]:
            model.generate(p["messages"], p["id"], p["attempt"], config)
    write_json(out / "identity.json", {"adapter": ADAPTERS[arm], "adapter_sha256": sha,
                                       "config_sha256": digest(CONFIG), "device": "cuda",
                                       "generations_sha256": digest(log)})


def grade_arm(arm, prompts, prefixes, reward, tools):
    from mp3_sql.environment import Episode
    from sql_repair_agent.review_prompts import messages_for

    folder = WORK / arm
    log = folder / "generations.jsonl"
    identity = json.loads((folder / "identity.json").read_text())
    if digest(log) != identity["generations_sha256"]:
        raise ValueError("Inference log identity mismatch")
    correct, action_counts, outcome_counts = set(), Counter(), Counter()
    outcomes = []
    for g, p, hist in zip(read_jsonl(log), prompts, prefixes, strict=True):
        if g["messages"] != p["messages"] or g["task_id"] != p["id"] or p["id"] != hist["id"]:
            raise ValueError("Inference response and state mismatch")
        ep = Episode(reward.by_id[p["id"]], tools)
        for old in hist["trace"]:
            ep.step({k: v for k, v in old.items() if k != "observation"})
            if ep.trace[-1] != old:
                raise ValueError("Official replay disagrees with sampled prefix")
        observed = payload(p["messages"])
        observed.pop("controller_feedback", None)
        if payload(messages_for(ep.observation(), {})) != observed:
            raise ValueError("Official prefix does not reconstruct the prompt's public state")
        proposal = parse_proposal(g["generated_text"]) or {}
        tool = proposal.get("tool", "unparsed")
        action_counts[tool] += 1
        outcome = "invalid_action" if not proposal else "non_check"
        if tool == "check":
            sql = extract_sql(proposal["sql"]) if isinstance(proposal.get("sql"), str) else None
            previous = {sql_key(event["sql"]) for event in ep.trace if event["tool"] == "check"}
            if not sql:
                outcome = "invalid_check"
            elif sql_key(sql) in previous:
                outcome = "duplicate_check"
            else:
                ep.step({"tool": "check", "sql": sql})
                ep.step({"tool": "submit", "sql": sql})
                pred = {"id": p["id"], "sql": sql, "trace": ep.trace, "tool_calls": BUDGET - ep.remaining}
                passed = reward.score(pred)["passed"]
                outcome = "new_correct" if passed else "new_incorrect"
                if passed:
                    correct.add(p["id"])
        outcome_counts[outcome] += 1
        outcomes.append({"id": p["id"], "outcome": outcome, "tool": tool})
    write_jsonl(folder / "outcomes.jsonl", outcomes)
    return {"identity": identity, "outcomes": outcome_counts, "actions": action_counts,
            "new_checks": outcome_counts["new_correct"] + outcome_counts["new_incorrect"]}, correct


def score():
    from mp3_sql.environment import Tools
    from training.v5_utils import TrainReward

    population = json.loads((WORK / "population.json").read_text())
    for name in ("prompts", "prefixes"):
        if digest(WORK / f"{name}.jsonl") != population[f"{name}_sha256"]:
            raise ValueError("Saved evaluation population changed")
    prompts, prefixes = (read_jsonl(WORK / f"{name}.jsonl") for name in ("prompts", "prefixes"))
    reward = TrainReward()
    tools = Tools(reward.folder, reward.manifest, cache={})
    old, successes = grade_arm("old", prompts, prefixes, reward, tools)
    generator, new_successes = grade_arm("generator", prompts, prefixes, reward, tools)
    report = {"dataset": population, "old": old, "generator": generator,
              "paired_correct": {"wins": len(new_successes - successes), "losses": len(successes - new_successes),
                                 "both_correct": len(successes & new_successes),
                                 "neither_correct": len(prompts) - len(successes | new_successes)},
              "method": "Actual historical prompt and matching replayed public prefix. One unconstrained greedy "
                        "model action, normalized and duplicate-checked using the deployed SQL functions. Each "
                        "valid new check is submitted immediately and independently graded on the official three "
                        "train instances; non-check actions have no subsequent rollout here. Same incremental "
                        "holdout used for both models; diagnostic not used for checkpoint selection."}
    write_json(ROOT / "reports/v6_fixed_repair_diagnostic.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("prepare")
    sub.add_parser("infer").add_argument("--arm", choices=ADAPTERS, required=True)
    sub.add_parser("score")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "infer":
        infer(args.arm)
    else:
        score()
