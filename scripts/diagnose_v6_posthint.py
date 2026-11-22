"""Exploratory follow-up on the frozen first-step holdout after both policies requested hints."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from unittest.mock import patch

from diagnose_v6_repairs import ADAPTERS, CONFIG, ROOT, WORK as INITIAL
from mp3_sql.common import BUDGET, digest, load_bank, read_jsonl, write_json, write_jsonl
from mp3_sql.environment import Episode, Tools
from sql_repair_agent.contract_agent import ContractAgent

WORK = ROOT / "runs/v6_posthint_repair_diagnostic"


class Captured(Exception):
    def __init__(self, proposal, action, reason):
        self.proposal, self.action, self.reason = proposal, action, reason


class ObservedContract(ContractAgent):
    def guard(self, observation, proposal):
        action, reason = super().guard(observation, proposal)
        if self.completions == 2:
            raise Captured(proposal, action, reason)
        return action, reason


class ReplayFirst:
    def __init__(self, initial, expected_trace, episode, model):
        self.initial, self.expected, self.episode, self.model = initial, expected_trace, episode, model

    def generate(self, messages, task_id, attempt, config):
        if attempt == 1:
            if (task_id != self.initial["task_id"] or self.initial["attempt"] != 1 or
                self.episode.trace != self.expected or messages != self.initial["messages"]):
                raise ValueError("Unable to reconstruct original state and exact first-generation prompt")
            return self.initial["generated_text"]
        if attempt != 2 or self.episode.trace[len(self.expected)]["tool"] != "repair_hint":
            raise ValueError("Unexpected next model decision or unexecuted common hint")
        return self.model.generate(messages, task_id, attempt, config)


def infer(arm):
    from sql_repair_agent.model import Generator

    population = json.loads((INITIAL / "population.json").read_text())
    if digest(INITIAL / "prefixes.jsonl") != population["prefixes_sha256"]:
        raise ValueError("First-step diagnostic population changed")
    first_log = INITIAL / arm / "generations.jsonl"
    first_identity = json.loads((INITIAL / arm / "identity.json").read_text())
    if digest(first_log) != first_identity["generations_sha256"]:
        raise ValueError("First-generation log changed")
    adapter = ROOT / ADAPTERS[arm]
    if digest(adapter / "adapter_model.safetensors") != first_identity["adapter_sha256"]:
        raise ValueError("Adapter differs from the first-step diagnostic")
    out = WORK / arm
    out.mkdir(exist_ok=False, parents=True)
    os.environ.update(MP3_BASE_MODEL=str(ROOT / "artifacts/models/Qwen3-0.6B"), MP3_DEVICE="cuda",
                      MP3_ADAPTER_DIR=str(adapter), MP3_GENERATION_LOG=str(out / "generations.jsonl"),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    config = json.loads(CONFIG.read_text())
    manifest_rows, manifest, folder = load_bank("train")
    rows = {r["id"]: r for r in manifest_rows}
    tools = Tools(folder, manifest, cache={})
    model = Generator(True)
    if model.identity["adapter"]["sha256"] != first_identity["adapter_sha256"]:
        raise ValueError("Incorrect model adapter")
    decisions = []
    for first, hist in zip(read_jsonl(first_log), read_jsonl(INITIAL / "prefixes.jsonl"), strict=True):
        tid = hist["id"]
        ep = Episode(rows[tid], tools)
        agent = ObservedContract(config)
        source = ReplayFirst(first, hist["trace"], ep, model)
        item = {"id": tid, "sample": hist["sample"]}
        with patch("sql_repair_agent.model.get_generator", return_value=source):
            try:
                while ep.submitted is None:
                    action = agent.act(ep.observation())
                    ep.step(action)
            except Captured as choice:
                item.update(proposal=choice.proposal, accepted=choice.action, guard_reason=choice.reason,
                            no_next_model_decision=False)
            else:
                item.update(proposal=None, accepted=None, guard_reason="controller_submitted",
                            no_next_model_decision=True)
        if ep.trace[len(hist["trace"])]["tool"] != "repair_hint":
            raise ValueError("First action failed to execute the common hint through official tools")
        item.update(trace=ep.trace, private_state={"completions": agent.completions,
                    "rejections": agent.rejections, "contract_checks": agent.contract_checks,
                    "model_queries": agent.model_queries})
        decisions.append(item)
    write_jsonl(out / "observed_decisions.jsonl", decisions)
    write_json(out / "identity.json", {"adapter_sha256": first_identity["adapter_sha256"],
               "first_generations_sha256": digest(first_log), "model_config_sha256": digest(CONFIG),
               "first_population_sha256": digest(INITIAL / "population.json"),
               "decisions_sha256": digest(out / "observed_decisions.jsonl"),
               "generations_sha256": digest(out / "generations.jsonl") if (out / "generations.jsonl").exists() else None})


def load(arm):
    out = WORK / arm
    identity = json.loads((out / "identity.json").read_text())
    if digest(out / "observed_decisions.jsonl") != identity["decisions_sha256"]:
        raise ValueError("Saved decisions changed")
    log = out / "generations.jsonl"
    if identity["generations_sha256"] != (digest(log) if log.exists() else None):
        raise ValueError("Saved next-action model log changed")
    logs = read_jsonl(log) if log.exists() else []
    messages = {g["task_id"]: g["messages"] for g in logs if g["attempt"] == 2}
    if len(messages) != len(logs):
        raise ValueError("Unexpected generation count or attempt")
    decisions = read_jsonl(out / "observed_decisions.jsonl")
    if set(messages) != {r["id"] for r in decisions if not r["no_next_model_decision"]}:
        raise ValueError("Missing next-decision logs")
    return decisions, messages, identity


def grade_arm(arm, decisions, reward, tools):
    totals, proposal_tools, reasons, correct = Counter(), Counter(), Counter(), set()
    details = []
    for item in decisions:
        tid, action = item["id"], item["accepted"]
        reasons[item["guard_reason"]] += 1
        proposal_tools[(item["proposal"] or {}).get("tool", "none")] += 1
        if item["no_next_model_decision"]:
            outcome = "no_next_model_decision"
        elif not action:
            outcome = "rejected_" + item["guard_reason"]
        elif action["tool"] != "check":
            outcome = "other_accepted_action"
        else:
            ep = Episode(reward.by_id[tid], tools)
            for event in item["trace"]:
                ep.step({k: v for k, v in event.items() if k != "observation"})
                if ep.trace[-1] != event:
                    raise ValueError("Train grader cannot reproduce the inference prefix")
            ep.step(action)
            ep.step({"tool": "submit", "sql": action["sql"]})
            pred = {"id": tid, "trace": ep.trace, "tool_calls": BUDGET - ep.remaining, "sql": action["sql"]}
            if reward.score(pred)["passed"]:
                outcome = "new_correct"
                correct.add(tid)
            else:
                outcome = "new_incorrect"
        totals[outcome] += 1
        details.append({"id": tid, "outcome": outcome, "guard_reason": item["guard_reason"]})
    write_jsonl(WORK / arm / "outcomes.jsonl", details)
    return {"outcomes": totals, "proposed_tools": proposal_tools, "guard_reasons": reasons}, correct


def grade():
    from training.v5_utils import TrainReward

    left, lmsg, li = load("old")
    right, rmsg, ri = load("generator")
    cohort = read_jsonl(INITIAL / "prefixes.jsonl")
    suffix_counts = Counter()
    for old, new, original in zip(left, right, cohort, strict=True):
        if (old["id"] != new["id"] or old["id"] != original["id"] or old["trace"] != new["trace"] or
            old["private_state"] != new["private_state"] or old["no_next_model_decision"] != new["no_next_model_decision"]):
            raise ValueError("Paired contexts or controller private states differ")
        suffix = old["trace"][len(original["trace"]):]
        if not suffix or suffix[0]["tool"] != "repair_hint":
            raise ValueError("Missing shared hint")
        suffix_counts["/".join(e["tool"] for e in suffix)] += 1
    if lmsg != rmsg:
        raise ValueError("Paired second-decision prompts differ")
    reward = TrainReward()
    tools = Tools(reward.folder, reward.manifest, cache={})
    old, old_ok = grade_arm("old", left, reward, tools)
    gen, gen_ok = grade_arm("generator", right, reward, tools)
    first_result = ROOT / "reports/v6_fixed_repair_diagnostic.json"
    first = json.loads(first_result.read_text())
    report = {"exploratory": True, "used_for_selection": False, "n": len(cohort),
              "original_population": first["dataset"], "original_result_sha256": digest(first_result),
              "first_action_counts": {arm: first[arm]["actions"] for arm in ADAPTERS},
              "identical_next_model_prompts": len(lmsg), "post_initial_prefix_tools": dict(suffix_counts),
              "old": {**old, "identity": li}, "generator": {**gen, "identity": ri},
              "paired_correct": {"wins": len(gen_ok - old_ok), "losses": len(old_ok - gen_ok),
                                 "both_correct": len(old_ok & gen_ok),
                                 "neither_immediate_correct_check": len(cohort) - len(old_ok | gen_ok)},
              "method": "Post-hoc continuation of all original 22 tasks. Replayed the deployed controller from "
                        "the start using original greedy action text for its first generation, then official "
                        "public tools (including the common cached hint), preserving all automatic steps and "
                        "controller memory. Next normal proposal classified by the production guard before retry "
                        "or fallback. Grader independently replays each accepted new SQL through all three train "
                        "instances. An additional accepted action other than check does not have its own rollout."}
    write_json(ROOT / "reports/v6_posthint_repair_diagnostic.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("infer").add_argument("--arm", choices=ADAPTERS, required=True)
    sub.add_parser("grade")
    args = parser.parse_args()
    if args.stage == "infer":
        infer(args.arm)
    else:
        grade()
