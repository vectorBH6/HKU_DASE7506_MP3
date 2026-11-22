"""Train-only terminal reward, kept completely outside the exported actor."""
import copy
import json
from pathlib import Path

from mp3_sql.common import BUDGET, PROTOCOL, digest, load_bank
from mp3_sql.score import load_cases, score_document


class TrainReward:
    def __init__(self):
        self.rows, self.manifest, self.folder = load_bank("train")
        self.by_id = {r["id"]: r for r in self.rows}
        self.cases, self.case_root = load_cases("train", self.folder)
        self.cache = {}
        self.results = {}
        self.header = {"protocol": PROTOCOL, "split": "train", "budget": BUDGET,
                       "bank_sha256": digest(self.folder / "train.jsonl"),
                       "manifest_sha256": digest(self.folder / "manifest.json")}

    def score(self, prediction):
        row = self.by_id[prediction["id"]]
        # No executor reimplementation: official replay + unchanged official grader.
        # Only the training batch's ID list is sliced; tasks/cases remain original.
        subset = {**self.cases, "ids": [row["id"]]}
        result = score_document({**self.header, "predictions": [prediction]},
                                [row], self.manifest, self.folder, subset,
                                self.case_root, workers=1, cache=self.cache)
        return result["per_question"][0]

    def candidates(self, prediction):
        labelled = []
        for index, event in enumerate(prediction["trace"]):
            if event["tool"] != "check":
                continue
            prefix = copy.deepcopy(prediction["trace"][:index + 1])
            prefix.append({"tool": "submit", "sql": event["sql"]})
            item = {"id": prediction["id"], "sql": event["sql"], "trace": prefix,
                    "tool_calls": len(prefix) - 1}
            result = self.score(item)
            labelled.append({"sql": event["sql"], "passed": result["passed"],
                             "matches_example": event["observation"]["matches_example"]})
        return labelled


def read_lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if Path(path).exists() else []


def append_line(path, item):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
