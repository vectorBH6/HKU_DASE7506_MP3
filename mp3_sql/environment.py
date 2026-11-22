"""Real SQL tools and a causal, independently replayable agent protocol."""
from collections import Counter
import copy
import hashlib
import json
import math
import re

from mp3_sql.common import BUDGET, MAX_SQL_BYTES, result_fingerprint, read_jsonl
from mp3_sql.sqlite_tool import execute_sql


def words(text):
    return re.findall(r"[a-z][a-z0-9_]*|\d+", text.lower())


class Tools:
    def __init__(self, folder, manifest, isolated=False, cache=None):
        self.folder = folder
        self.manifest = manifest
        self.isolated = isolated
        self.cache = cache
        self.expected = json.loads((folder / "public_results.json").read_text())
        self.hints = (json.loads((folder / "repair_hints.json").read_text())
                      if "repair_hints.json" in manifest.get("support_files", {}) else {})
        self.library = read_jsonl(folder / "train_memory.jsonl")
        self.bags = [Counter(words(item["question"] + " " + item["issue_sql"]))
                     for item in self.library]
        frequency = Counter(word for bag in self.bags for word in bag)
        self.idf = {word: math.log(1 + (len(self.bags) - count + .5) / (count + .5))
                    for word, count in frequency.items()}

    def schema(self, row):
        return copy.deepcopy(self.manifest["databases"][row["db_id"]]["schema"])

    def query(self, row, sql, check):
        metadata = self.manifest["databases"][row["db_id"]]
        identity = (metadata["public_sha256"], sql, self.isolated)
        result = self.cache.get(identity) if self.cache is not None else None
        if result is None:
            result = execute_sql(self.folder / metadata["public_path"], sql,
                                 [table["name"] for table in metadata["schema"]], self.isolated)
            if self.cache is not None and len(self.cache) < 20000:
                self.cache[identity] = result
        feedback = {"ok": result["ok"], "columns": result["columns"],
                    "rows": result["rows"][:8], "row_count": len(result["rows"]),
                    "truncated": len(result["rows"]) > 8, "error": result["error"]}
        if check:
            feedback["matches_example"] = bool(result["ok"] and
                result_fingerprint(result, row["ordered"]) == self.expected[row["id"]])
        return feedback

    def retrieve(self, row, query):
        tokens = set(words(query))
        scores = []
        for index, (item, bag) in enumerate(zip(self.library, self.bags)):
            if item["id"] == row["id"]:
                continue
            score = sum(self.idf.get(word, 0) * bag[word] / (bag[word] + 1.2)
                        for word in tokens)
            scores.append((score, -index, item))
        return [copy.deepcopy(item[2]) for item in sorted(
            scores, key=lambda item: item[:2], reverse=True)[:3]]

    def repair_hint(self, row):
        if row["id"] not in self.hints:
            raise ValueError("This bank has no prepared repair hints for the task")
        record = self.hints[row["id"]]
        if record["issue_sha256"] != hashlib.sha256(row["issue_sql"].encode()).hexdigest():
            raise ValueError("Cached repair hint does not match the original draft")
        return copy.deepcopy(record["suggestions"])


class Episode:
    def __init__(self, row, tools, budget=BUDGET):
        if type(budget) is not int or not 0 <= budget <= BUDGET:
            raise ValueError("Invalid episode budget")
        self.row, self.tools = row, tools
        self.remaining = budget
        self.trace = []
        self.submitted = None
        self.used = set()

    def observation(self):
        return {"task": {"id": self.row["id"], "question": self.row["question"]},
                "issue_sql": self.row["issue_sql"], "example": copy.deepcopy(self.row["example"]),
                "ordered": self.row["ordered"], "remaining": self.remaining,
                "history": copy.deepcopy(self.trace)}

    def step(self, action):
        if self.submitted is not None:
            raise ValueError("Episode already submitted")
        if not isinstance(action, dict) or "tool" not in action:
            raise ValueError("Action must be a tool object")
        tool = action["tool"]
        fields = {"schema": {"tool"}, "execute": {"tool", "sql"}, "check": {"tool", "sql"},
                  "retrieve": {"tool", "query"}, "repair_hint": {"tool"},
                  "submit": {"tool", "sql"}}
        if tool not in fields or set(action) != fields[tool]:
            raise ValueError("Unknown tool or incorrect action fields")
        if tool in ("execute", "check", "submit"):
            if not isinstance(action["sql"], str) or len(action["sql"].encode()) > MAX_SQL_BYTES:
                raise ValueError("SQL must be a string of at most 16000 bytes")
        if tool == "retrieve":
            if not isinstance(action["query"], str) or len(action["query"]) > 2000:
                raise ValueError("Retrieval query must be a string of at most 2000 characters")
        if tool == "submit":
            self.submitted = action["sql"]
            self.trace.append(copy.deepcopy(action))
            return
        if not self.remaining:
            raise ValueError("Tool budget exceeded")
        identity = json.dumps(action, sort_keys=True)
        if identity in self.used:
            raise ValueError("Identical tool calls cannot be repeated")
        if tool == "repair_hint" and not any(
                event["tool"] in ("execute", "check") and event["sql"] == self.row["issue_sql"]
                for event in self.trace):
            raise ValueError("Execute or check the original draft before requesting repair hints")
        if tool == "schema":
            feedback = self.tools.schema(self.row)
        elif tool == "retrieve":
            feedback = self.tools.retrieve(self.row, action["query"])
        elif tool == "repair_hint":
            feedback = self.tools.repair_hint(self.row)
        else:
            feedback = self.tools.query(self.row, action["sql"], tool == "check")
        self.used.add(identity)
        self.remaining -= 1
        self.trace.append({**copy.deepcopy(action), "observation": feedback})


def rollout(row, tools, agent, budget=BUDGET):
    episode = Episode(row, tools, budget)
    for _ in range(budget + 1):
        episode.step(agent.act(episode.observation()))
        if episode.submitted is not None:
            return {"id": row["id"], "sql": episode.submitted, "trace": episode.trace,
                    "tool_calls": budget - episode.remaining}
    raise ValueError("Agent failed to submit")


def replay(row, tools, prediction, budget=BUDGET):
    if prediction.get("id") != row["id"] or not isinstance(prediction.get("trace"), list):
        raise ValueError("Prediction ID or trajectory mismatch")
    if len(prediction["trace"]) > budget + 1:
        raise ValueError("Overlong trajectory")
    episode = Episode(row, tools, budget)
    for event in prediction["trace"]:
        if not isinstance(event, dict):
            raise ValueError("Malformed trajectory event")
        episode.step({key: value for key, value in event.items() if key != "observation"})
        if episode.trace[-1] != event:
            raise ValueError("Forged tool observation")
    if episode.submitted is None or episode.submitted != prediction.get("sql"):
        raise ValueError("Missing or inconsistent submission")
    if (type(prediction.get("tool_calls")) is not int
            or prediction["tool_calls"] != budget - episode.remaining):
        raise ValueError("Incorrect tool call count")
    return episode.submitted
