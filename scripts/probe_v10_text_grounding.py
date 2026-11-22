"""Evaluate text-predicate edits on frozen public tool histories with a train-selected budgeted policy."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import re
import sys

from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]
from mp3_sql.common import BUDGET, PROTOCOL, digest, load_bank, read_jsonl
from mp3_sql.environment import Episode, Tools
from mp3_sql.score import load_cases, score_document
from sql_repair_agent.sql_utils import sql_key


def name_columns(schema):
    for tab in schema:
        cols = {re.sub("[^a-z0-9]", "", c["name"].lower()): c["name"] for c in tab["columns"]}
        first = [cols[n] for n in ("fname", "firstname", "givenname", "forename") if n in cols]
        last = [cols[n] for n in ("lname", "lastname", "surname", "familyname") if n in cols]
        if len(first) == len(last) == 1:
            yield tab["name"].casefold(), first[0], last[0]


def text_edits(sql, question, schema):
    try:
        tree = parse_one(sql, read="sqlite")
    except (ValueError, SqlglotError, RecursionError):
        return
    if tree is None:
        return
    table_names = {t.alias_or_name.casefold(): t.name.casefold() for t in tree.find_all(exp.Table)}
    for index, eq in enumerate(tree.find_all(exp.EQ)):
        column, literal = eq.this, eq.expression
        if not isinstance(column, exp.Column) or not isinstance(literal, exp.Literal) or not literal.is_string:
            continue
        v = literal.this.strip()
        if not v or "%" in v or "_" in v:
            continue
        folded = exp.Lower(this=column.copy())
        replacements = [
            ("nocase", exp.EQ(this=folded.copy(), expression=exp.Literal.string(v.lower()))),
            ("contains", exp.Like(this=folded.copy(), expression=exp.Literal.string("%" + v.lower() + "%"))),
        ]
        for m in re.finditer(r"(?<!\w)([\w]+)\s+" + re.escape(v) + r"(?!\w)", question, re.I):
            if m[0][0].isupper():
                replacements.append(("prefix", exp.EQ(this=folded.copy(), expression=exp.Literal.string(m[0].lower()))))
        words = v.split()
        if len(words) == 2 and all(w[:1].isupper() and w.replace("-", "").isalpha() for w in words):
            for tab, first, last in name_columns(schema):
                if column.name.casefold() not in (first.casefold(), last.casefold()) or tab not in table_names.values():
                    continue
                if column.table and table_names.get(column.table.casefold()) != tab:
                    continue
                def part(name, value):
                    c = column.copy()
                    c.set("this", exp.to_identifier(name, quoted=True))
                    return exp.EQ(this=c, expression=exp.Literal.string(value))
                replacements.append(("full_name", exp.and_(part(first, words[0]), part(last, words[1]))))
        for family, edit in replacements:
            changed = tree.copy()
            list(changed.find_all(exp.EQ))[index].replace(edit)
            query = changed.sql(dialect="sqlite")
            if len(query.encode()) <= 16000:
                yield family, query


def candidates(row, trace, previous):
    schema = next((e["observation"] for e in trace if e["tool"] == "schema"), [])
    checked = [e["sql"] for e in trace if e["tool"] == "check"]
    seen = {sql_key(s) for s in checked}
    for sql in dict.fromkeys([previous, *reversed(checked)]):
        for family, query in text_edits(sql, row["question"], schema):
            key = sql_key(query)
            if key not in seen:
                seen.add(key)
                yield family, query


def load_predictions(split, name):
    if name == "train":
        # Pick one archived rollout per train question independently of its grade.
        return {r["id"]: r["prediction"] for r in read_jsonl(ROOT / "runs/v5_train_coverage/episodes.jsonl")
                if r["sample"] == 0}
    if name == "v4b_holdout":
        return {r["id"]:r for r in read_jsonl(ROOT / "runs/v6_holdout/baseline/predictions.jsonl")}
    exp_name = {"v8": "e82_v8_struct", "v10": "e100_v10_semantic_struct"}[name]
    path = ROOT / "runs" / exp_name / split / "r001/predictions.json"
    return {r["id"]: r for r in json.loads(path.read_text())["predictions"]}


class PublicBank:
    def __init__(self, split):
        self.split = split
        self.rows, self.manifest, self.folder = load_bank(split)
        self.tools = Tools(self.folder, self.manifest, cache={})

    def terminal(self, row, old):
        ep = Episode(row, self.tools)
        for e in old["trace"][:-1]:
            ep.step({k:v for k,v in e.items() if k != "observation"})
            if ep.trace[-1] != e:
                raise ValueError("Historical prefix disagrees with real public observation")
        return ep

    def options(self, row, old):
        ep = self.terminal(row, old)
        match = any(e["observation"].get("matches_example") for e in ep.trace if e["tool"] == "check")
        options = []
        if ep.remaining:
            for family, sql in candidates(row, ep.trace, old["sql"]):
                # Independent public observations are memoized only for offline
                # assessment of multiple orderings. A policy sees its own checks.
                options.append((family, sql, self.tools.query(row, sql, True)["matches_example"]))
        return {"row":row, "old":old, "opts":options, "matched":match}


FAMILIES = [
    ("full_name", "prefix", "nocase", "contains"),
    ("full_name", "prefix", "contains", "nocase"),
    ("full_name", "nocase", "contains"),
    ("full_name", "nocase"),
    ("full_name",),
]


def choose(case, policy):
    old = case["old"]
    if policy["gate"] == "off":
        return old["sql"], []
    if case["matched"] and (policy["gate"] == "failed" or
                            policy["gate"] == "failed_or_empty" and case["row"]["example"]["row_count"] > 0):
        return old["sql"], []
    budget = min(policy["checks"], BUDGET - old["tool_calls"])
    candidates_by_family = [option for f in policy["families"] for option in case["opts"] if option[0] == f]
    checks = []
    for _, sql, publicly_correct in candidates_by_family[:budget]:
        checks.append(sql)
        if publicly_correct:
            return sql, checks
    return old["sql"], checks


class IndependentGrader:
    def __init__(self, bank):
        self.bank = bank
        self.cases, self.root = load_cases(bank.split, bank.folder)
        self.cache = {}
        self.labels = {}
        self.base = {"protocol":PROTOCOL, "split":bank.split, "budget":BUDGET,
                     "bank_sha256":digest(bank.folder / (bank.split + ".jsonl")),
                     "manifest_sha256":digest(bank.folder / "manifest.json")}

    def grade(self, row, pred):
        one = dict(self.cases, ids=[row["id"]])
        doc = dict(self.base, predictions=[pred])
        return score_document(doc, [row], self.bank.manifest, self.bank.folder, one, self.root,
                              workers=1, cache=self.cache)["per_question"][0]["passed"]

    def grade_option(self, c, candidate):
        key = c["row"]["id"], candidate
        if key not in self.labels:
            p = c["old"]
            if candidate != p["sql"]:
                ep = self.bank.terminal(c["row"], p)
                ep.step({"tool":"check", "sql":candidate})
                ep.step({"tool":"submit", "sql":candidate})
                p = {"id":c["row"]["id"], "sql":candidate, "trace":ep.trace, "tool_calls":BUDGET-ep.remaining}
            self.labels[key] = self.grade(c["row"], p)
        return self.labels[key]

    def evaluate(self, cases, policy, full=False):
        result = Counter(n=len(cases), baseline=0, correct=0, wins=0, losses=0, changed=0,
                         charged_extra=0, max_tools=0)
        for c in cases:
            old = c["old"]
            sql, actions = choose(c, policy)
            before = self.grade_option(c, old["sql"])
            after = self.grade_option(c, sql)
            cost = old["tool_calls"] + len(actions)
            if full and actions:
                ep = self.bank.terminal(c["row"], old)
                for check in actions:
                    ep.step({"tool":"check","sql":check})
                ep.step({"tool":"submit","sql":sql})
                current={"id":old["id"], "sql":sql, "trace":ep.trace, "tool_calls":BUDGET-ep.remaining}
                if self.grade(c["row"],current) != after:
                    raise ValueError("Scored full policy differs from independently labeled SQL")
            result.update(baseline=int(before), correct=int(after), wins=int(after and not before),
                          losses=int(before and not after), changed=int(sql != old["sql"]), charged_extra=len(actions))
            result["max_tools"] = max(result["max_tools"], cost)
        return dict(result)

    def coverage(self, cases):
        c = Counter(wrong=0,wrong_with_candidates=0, wrong_with_public=0,wrong_with_three=0)
        by_family = Counter()
        for row in cases:
            if self.grade_option(row, row["old"]["sql"]):
                continue
            c["wrong"] += 1
            c["wrong_with_candidates"] += bool(row["opts"])
            c["wrong_with_public"] += any(o[2] for o in row["opts"])
            families = {f for f, sql, m in row["opts"] if m and self.grade_option(row, sql)}
            c["wrong_with_three"] += bool(families)
            by_family.update(families)
        return {**dict(c),"tasks_per_correct_family":dict(by_family)}


def prepare(bank, predictions, ids):
    return [bank.options(row, predictions[row["id"]]) for row in bank.rows if row["id"] in ids]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10", action="store_true", help="Evaluate the previously fixed training policy on V10 validation")
    args=parser.parse_args()
    out=ROOT/"runs/v10_text_grounding"
    out.mkdir(exist_ok=True)
    lock=out/"training_selection.json"
    dateset=json.loads((ROOT/"artifacts/prepared_data/v6/split.json").read_text())
    if not lock.exists():
        bank=PublicBank("train")
        tg=IndependentGrader(bank)
        archive=load_predictions("train","train")
        fit=prepare(bank,archive,set(dateset["fit_ids"]))
        options=[{"gate":"off","checks":0,"families":[]}]
        options.extend({"gate":g,"checks":n,"families":list(f)} for g,n,f in product(
            ("failed","failed_or_empty","all"),(1,3,8),FAMILIES))
        scores=[tg.evaluate(fit,p) for p in options]
        best=min(range(len(scores)),key=lambda i:(-scores[i]["correct"],scores[i]["charged_extra"],scores[i]["losses"],i))
        policy=options[best]
        outrow={"fit_scope":"382 train tasks, archived V5 first rollout; 24 database schemas excluded",
                "selection":"maximize fit correct, then minimize tool calls, losses, fixed enumeration order",
                "score_source":"official score_document after committed independent public trajectories",
                "policy":policy,"fit":tg.evaluate(fit,policy,True),"fit_coverage":tg.coverage(fit),
                "searched":len(options),"script_sha256":digest(Path(__file__))}
        with lock.open("x") as fd:json.dump(outrow,fd,indent=2)
        print("Frozen on train:",outrow,flush=True)
        for arch in ("train","v4b_holdout"):
            options=prepare(bank,load_predictions("train",arch),set(dateset["holdout_ids"]))
            result={"population":"98 held-out train tasks","archive":arch,
                    "policy":tg.evaluate(options,policy,True),"coverage":tg.coverage(options)}
            (out/(arch+"_holdout.json")).write_text(json.dumps(result,indent=2)+"\n")
            print(arch,result,flush=True)
    else:
        data=json.loads(lock.read_text())
        if data["script_sha256"]!=digest(Path(__file__)):
            raise ValueError("Probe changed since training selection")
        policy=data["policy"]
    b=PublicBank("validation"); g=IndependentGrader(b)
    for name in (["v10"] if args.v10 else ["v8"]):
        preds=load_predictions("validation",name); options=prepare(b,preds,set(preds))
        result={"population":"120 validation tasks; approach motivated by previous V8 validation errors",
                "archive":name,"scope":"policy uses only observed public feedback; no additional model generations",
                "policy":g.evaluate(options,policy,True),"coverage":g.coverage(options)}
        (out/(name+"_validation.json")).write_text(json.dumps(result,indent=2)+"\n")
        print(name,result,flush=True)

if __name__=="__main__":
    main()
