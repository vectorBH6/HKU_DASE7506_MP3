"""Isolated offline response diagnostic on existing train-holdout chooser prompts."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]
from mp3_sql.common import digest, read_jsonl, write_json, write_jsonl
from sql_repair_agent.review_agent import parse_proposal
from sql_repair_agent.sql_utils import sql_key

WORK = ROOT / "runs/v6_chooser_pair_diagnostic"
SOURCE = ROOT / "artifacts/prepared_data/v6/chooser/holdout.jsonl"
CONFIG = ROOT / "submission/configs/contract_v4_selected.json"


def unpack(messages):
    return json.loads(messages[1]["content"].split("TASK_DATA\n", 1)[1].split("\nEND_TASK_DATA", 1)[0])


def prepare():
    rows = [r for r in read_jsonl(SOURCE) if r["state"] == "contrast_public_match"]
    holdout = set(json.loads((SOURCE.parent.parent / "split.json").read_text())["holdout_ids"])
    contexts = defaultdict(dict)
    for r in rows:
        key = (r["pair_id"], r["sample"])
        if r["id"] not in holdout or r["order"] in contexts[key]:
            raise ValueError("Invalid or duplicate holdout context")
        contexts[key][r["order"]] = r
    if not rows or any(set(orders) != {"original", "swapped"} for orders in contexts.values()):
        raise ValueError("Every source context needs both display orders")

    prompts, labels, seen = [], [], set()
    for (pair_id, _), orders in sorted(contexts.items()):
        if pair_id in seen:
            continue
        seen.add(pair_id)
        original_payload = unpack(orders["original"]["messages"])
        swap_payload = unpack(orders["swapped"]["messages"])
        original_choices = original_payload.pop("checked_candidates")
        swap_choices = swap_payload.pop("checked_candidates")
        if original_payload != swap_payload:
            raise ValueError("Non-choice features differ between orderings")
        original = orders["original"]
        def slot(option):
            return next(i for i, c in enumerate(original_choices) if c["id"] == original[option])
        pos, neg = slot("positive_candidate"), slot("negative_candidate")
        expected = list(original_choices)
        expected[pos], expected[neg] = expected[neg], expected[pos]
        if [{**c, "id": f"c{j}"} for j, c in enumerate(expected)] != swap_choices:
            raise ValueError("Swap modified evidence beyond the candidate display order")
        for order in ("original", "swapped"):
            r = orders[order]
            choices = unpack(r["messages"])["checked_candidates"]
            positive, negative = r["positive_candidate"], r["negative_candidate"]
            mapped = {c["id"]: c for c in choices}
            if (not mapped[positive]["feedback"]["matches_example"] or
                not mapped[negative]["feedback"]["matches_example"] or
                json.loads(r["target"])["candidate"] != positive or
                positive != ("c" + str(pos if order == "original" else neg))):
                raise ValueError("Invalid candidate-pair ground truth")
            prompts.append({"id": r["id"], "messages": r["messages"]})
            labels.append({"id": r["id"], "db_id": r["db_id"], "pair": pair_id,
                           "sample": r["sample"], "order": order,
                           "positive": positive, "negative": negative})
    WORK.mkdir(parents=True, exist_ok=False)
    write_jsonl(WORK / "prompts.jsonl", prompts)
    write_jsonl(WORK / "labels.jsonl", labels)
    write_json(WORK / "population.json", {
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": digest(SOURCE),
        "prompts_sha256": digest(WORK / "prompts.jsonl"), "labels_sha256": digest(WORK / "labels.jsonl"),
        "source_contexts": len(contexts), "distinct_pairs": len(seen), "prompts": len(prompts),
        "tasks": len({r["id"] for r in labels}), "schemas": len({r["db_id"] for r in labels}),
        "sampling": "All held-out contrast pairs; one earliest sample index per task + normalized "
                    "positive/negative SQL pair, in original and swapped order. Other choices retained."
    })


def infer(name, adapter, device):
    from sql_repair_agent.model import Generator

    config = json.loads(CONFIG.read_text())
    if config["max_new_tokens"] != 384 or config["max_completions"] != 4 or config.get("do_sample"):
        raise ValueError("Expected the unchanged selected configuration")
    prompts = read_jsonl(WORK / "prompts.jsonl")
    folder = WORK / name
    folder.mkdir(exist_ok=True)
    log = folder / "generations.jsonl"
    param_hash = digest(adapter / "adapter_model.safetensors")
    completed = read_jsonl(log) if log.exists() else []
    if len(completed) > len(prompts):
        raise ValueError("Too many responses in existing log")
    for item, p in zip(completed, prompts):
        if (item["messages"] != p["messages"] or item["task_id"] != p["id"]
            or item["identity"]["adapter"]["sha256"] != param_hash or item["identity"]["device"] != device):
            raise ValueError("Existing log does not match this model or prompt list")
    os.environ.update(MP3_BASE_MODEL=str(ROOT / "artifacts/models/Qwen3-0.6B"), MP3_DEVICE=device,
                      MP3_ADAPTER_DIR=str(adapter), MP3_GENERATION_LOG=str(log),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if len(completed) < len(prompts):
        model = Generator(True)
        for p in prompts[len(completed):]:
            model.generate(p["messages"], p["id"], 1, config)
    write_json(folder / "identity.json", {"adapter": str(adapter), "adapter_sha256": param_hash,
                                          "device": device, "config_sha256": digest(CONFIG),
                                          "generations_sha256": digest(log)})


def evaluate(name, prompts, labels):
    folder = WORK / name
    identity = json.loads((folder / "identity.json").read_text())
    log = folder / "generations.jsonl"
    if digest(log) != identity["generations_sha256"]:
        raise ValueError("Generation log changed after inference")
    outcomes, actions, by_order, pairs = Counter(), Counter(), defaultdict(Counter), defaultdict(dict)
    by_position = defaultdict(Counter)
    for item, p, label in zip(read_jsonl(log), prompts, labels, strict=True):
        if item["messages"] != p["messages"] or item["task_id"] != p["id"] or p["id"] != label["id"]:
            raise ValueError("Prediction and labelled prompt are not aligned")
        proposal = parse_proposal(item["generated_text"]) or {}
        tool = proposal.get("tool", "unparsed")
        actions[tool] += 1
        choices = unpack(p["messages"])["checked_candidates"]
        outcome = "non_submit"
        if tool == "submit":
            available = {c["id"] for c in choices}
            choice = proposal.get("candidate")
            if choice not in available and isinstance(proposal.get("sql"), str):
                choice = next((c["id"] for c in choices if sql_key(c["sql"]) == sql_key(proposal["sql"])), None)
            if choice == label["positive"]:
                outcome = "selected_target"
            elif choice == label["negative"]:
                outcome = "selected_counterexample"
            else:
                outcome = "other_candidate" if choice in available else "invalid_submit"
        outcomes[outcome] += 1
        by_order[label["order"]][outcome] += 1
        earlier = int(label["positive"][1:]) < int(label["negative"][1:])
        by_position["target_earlier" if earlier else "target_later"][outcome] += 1
        pairs[label["pair"]][label["order"]] = outcome
    robust = {key for key, orders in pairs.items() if set(orders.values()) == {"selected_target"}}
    return ({"identity": identity, "outcomes": outcomes, "action_counts": actions,
             "outcomes_by_order": by_order, "outcomes_by_target_position": by_position,
             "target_selected_in_both_orders": len(robust),
             "target_selected_in_at_least_one_order": sum("selected_target" in v.values() for v in pairs.values())},
            robust, {(key, order) for key, choices in pairs.items() for order, result in choices.items()
                      if result == "selected_target"})


def score(output):
    population = json.loads((WORK / "population.json").read_text())
    for role in ("prompts", "labels"):
        if digest(WORK / f"{role}.jsonl") != population[f"{role}_sha256"]:
            raise ValueError(f"{role} file has changed")
    prompts, labels = (read_jsonl(WORK / f"{role}.jsonl") for role in ("prompts", "labels"))
    old, old_pairs, old_prompts = evaluate("old", prompts, labels)
    chooser, new_pairs, new_prompts = evaluate("chooser", prompts, labels)
    result = {"dataset": population, "old": old, "chooser": chooser,
              "paired": {"newly_correct_displays": len(new_prompts - old_prompts),
                         "regressed_displays": len(old_prompts - new_prompts),
                         "newly_correct_both_order_pairs": len(new_pairs - old_pairs),
                         "regressed_both_order_pairs": len(old_pairs - new_pairs)},
              "method": "One unconstrained, greedy first action per fixed prompt. The actor received only messages. "
                        "A valid submit can name a candidate or give equivalent normalized SQL; other actions "
                        "are counted separately, without running a controller or a database. Held out only "
                        "from V6 incremental fine-tuning. Diagnostic is independent of checkpoint selection."}
    write_json(output, result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("prepare")
    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--name", choices=["old", "chooser"], required=True)
    infer_parser.add_argument("--adapter", type=Path, required=True)
    infer_parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--output", type=Path, default=ROOT / "reports/v6_chooser_pairs_diagnostic.json")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "infer":
        infer(args.name, args.adapter, args.device)
    else:
        score(args.output)


if __name__ == "__main__":
    main()
