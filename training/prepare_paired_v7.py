"""Build intent-paired and ordinary SQL repair data on projections of official TRAIN schemas."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]
from mp3_sql.common import digest, load_bank, read_jsonl, result_fingerprint, write_json, write_jsonl
from sql_repair_agent.review_prompts import messages_for

SPLIT = ROOT / "artifacts/prepared_data/v6/split.json"
EPISODES = ROOT / "runs/v5_train_coverage/episodes.jsonl"
REPLAY = ROOT / "artifacts/prepared_data/v6/generator/train.jsonl"
CATEGORIES = ("entity_count", "join_multiplicity", "comparison_boundary")
NUMERIC = ("INT", "REAL", "FLOAT", "DOUBLE", "NUM", "DEC")


def stable(*values):
    raw = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def words(name):
    return re.sub(r"\s+", " ", re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ")).strip().lower()


def contains(sql, name):
    return bool(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", sql, re.I))


def mechanism_hint(sql, category):
    patterns = {"entity_count": r"\bcount\s*\(|\bdistinct\b",
                "join_multiplicity": r"\bjoin\b",
                "comparison_boundary": r"[<>]"}
    return bool(re.search(patterns[category], sql, re.I))


def is_numeric(column):
    return any(t in column["type"].upper() for t in NUMERIC)


def motifs(schema, category):
    tables = {t["name"].lower(): t for t in schema}
    if category == "comparison_boundary":
        for table in schema:
            primary = [c for c in table["columns"] if c["pk"]]
            if len(primary) != 1:
                continue
            foreign = {f["from"].lower() for f in table["foreign_keys"]}
            for col in table["columns"]:
                if not col["pk"] and is_numeric(col) and col["name"].lower() not in foreign:
                    yield (table["name"], primary[0]["name"], col["name"])
        return
    for child in schema:
        child_pk = [c for c in child["columns"] if c["pk"]]
        if len(child_pk) != 1:
            continue
        for fk in child["foreign_keys"]:
            parent = tables.get(fk["table"].lower())
            if not parent or child is parent or not fk["to"]:
                continue
            parent_pk = [c for c in parent["columns"] if c["pk"]]
            if len(parent_pk) != 1 or parent_pk[0]["name"].lower() != fk["to"].lower():
                continue
            col = next((c for c in child["columns"] if c["name"].lower() == fk["from"].lower()), None)
            if not col or col["pk"] or is_numeric(col) != is_numeric(parent_pk[0]):
                continue
            yield (parent["name"], parent_pk[0]["name"], child["name"], child_pk[0]["name"], col["name"])


def anchors(rows):
    gold = {r["id"]: r["gold_sql"] for r in read_jsonl(ROOT / "mp3_sql/data/train_memory.jsonl")}
    hard = defaultdict(set)
    for episode in read_jsonl(EPISODES):
        for item in episode["checked_candidates"]:
            if item["matches_example"] and not item["passed"]:
                hard[episode["id"]].add(item["sql"].strip())
    by_db = defaultdict(list)
    for row in rows:
        tid = row["id"]
        by_db[row["db_id"]].append((tid, gold[tid], None))
        for sql in sorted(hard[tid]):
            by_db[row["db_id"]].append((tid, gold[tid], sql))
    return by_db


def choose(db_ids, metadata, sources, category, count):
    # One motif per schema/category. Prefer actual error anchors sharing table names;
    # operator overlap is a relevance proxy and does not diagnose the original failure.
    per_schema = []
    for db in db_ids:
        best = None
        for key in motifs(metadata[db]["schema"], category):
            tables = [key[0]] if len(key) == 3 else [key[0], key[2]]
            cols = [key[-1]] if len(key) == 3 else [key[1], key[-1]]
            for tid, gold, wrong in sources[db]:
                sql = wrong or gold
                table_hits = sum(contains(sql, t) for t in tables)
                if not table_hits:
                    continue
                rank = (bool(wrong), table_hits / len(tables), mechanism_hint(sql, category),
                        sum(contains(sql, c) for c in cols), stable("anchor", category, db, key, tid, sql))
                if best is None or rank > best[0]:
                    best = (rank, key, tid, gold, wrong)
        if best:
            per_schema.append((db, best))
    per_schema.sort(key=lambda x: (not bool(x[1][-1]), stable("choose", category, x[0])))
    if len(per_schema) < count:
        raise ValueError(f"Insufficient eligible train schemas in {category}: {len(per_schema)} < {count}")
    return per_schema[:count]


def projected_schema(schema, key):
    indexed = {t["name"]: t for t in schema}
    if len(key) == 3:
        specs = [(key[0], key[1:], [])]
    else:
        parent, pk, child, ck, fk = key
        specs = [(parent, [pk], []), (child, [ck, fk], [{"table": parent, "from": fk, "to": pk}])]
    result = []
    for t, names, fks in specs:
        table = indexed[t]
        extra = next((c["name"] for c in table["columns"] if c["name"] not in names
                      and "TEXT" in c["type"].upper() and not c["pk"]), None)
        selected = set(names) | ({extra} if extra else set())
        result.append({"name": t, "columns": [dict(c) for c in table["columns"] if c["name"] in selected],
                       "foreign_keys": fks})
    return result


def query_pair(category, key, variant, threshold, high):
    # Preserve comparable public column names across the two legitimate intents.
    if category == "comparison_boundary":
        t, _, field = key
        noun, field_n = words(t), words(field)
        op1, op2 = (">", ">=") if high else ("<", "<=")
        strict, incl = ("greater than", "at least") if high else ("less than", "at most")
        alt, eq = ("above", "or above") if high else ("below", "or below")
        q0 = [f"How many records in {noun} have {field_n} {strict} {threshold}?",
              f"Count {noun} entries with {field_n} strictly {alt} {threshold}.",
              f"Among the {noun} records, how many have {field_n} {alt} {threshold}, excluding {threshold} itself?"]
        q1 = [f"How many records in {noun} have {field_n} {incl} {threshold}?",
              f"Count {noun} entries with {field_n} {threshold} {eq}.",
              f"Among the {noun} records, how many have {field_n} {threshold} {eq}, including equality?"]
        sql = [f"SELECT COUNT(*) AS n FROM {quote(t)} WHERE {quote(field)} {op} {threshold}" for op in (op1,op2)]
    else:
        parent, pk, child, _, fk = key
        p, c, k = words(parent), words(child), words(fk)
        if category == "entity_count":
            q0 = [f"How many records are in {c}?", f"Count all entries in {c}, including repeated {k} values.",
                  f"What is the total number of rows recorded in {c}?"]
            q1 = [f"How many different {k} values are recorded in {c}?", f"Count the distinct {p} represented by {k} in {c}.",
                  f"In {c}, how many separate {k} values occur, counting repeated values just once?"]
            sql = [f"SELECT COUNT(*) AS n FROM {quote(child)}",
                   f"SELECT COUNT(DISTINCT {quote(fk)}) AS n FROM {quote(child)}"]
        else:
            q0 = [f"How many {p} records are there in total, regardless of entries in {c}?",
                  f"Count each record in {p} once, including records with no related {c}.",
                  f"What is the size of the {p} table, including any records with no matching {c} entry?"]
            q1 = [f"How many matching pairs of {p} and {c} records are linked by {c}.{k}?",
                  f"Count every matching {p} and {c} combination connected through {k}, counting each match.",
                  f"How many times do records from {p} match an entry in {c} via {k}, counting all occurrences?"]
            sql = [f"SELECT COUNT(*) AS n FROM {quote(parent)}",
                   f"SELECT COUNT(*) AS n FROM {quote(parent)} AS p JOIN {quote(child)} AS c ON p.{quote(pk)} = c.{quote(fk)}"]
    return [q0[variant], q1[variant]], sql


def create_fixture(path, schema, key, index, base_n, threshold):
    # The miniature is a projection of a train schema. It contains invented rows only.
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        for t in schema:
            defs = []
            for col in t["columns"]:
                affinity = "INTEGER" if is_numeric(col) else "TEXT"
                defs.append(quote(col["name"])+" "+affinity+(" PRIMARY KEY" if col["pk"] else "")
                            +(" NOT NULL" if col["not_null"] and not col["pk"] else ""))
            for fk in t["foreign_keys"]:
                defs.append(f"FOREIGN KEY ({quote(fk['from'])}) REFERENCES {quote(fk['table'])}({quote(fk['to'])})")
            conn.execute(f"CREATE TABLE {quote(t['name'])} ({', '.join(defs)})")
        def insert(t, values):
            columns = t["columns"]
            complete = [values.get(col["name"], f"label_{values[next(iter(values))]}") for col in columns]
            conn.execute(f"INSERT INTO {quote(t['name'])} VALUES ({','.join('?' for _ in complete)})", complete)
        def ids(t, number):
            pk = next(c for c in t["columns"] if c["pk"])
            return list(range(1, number+1)) if is_numeric(pk) else [f"K{i}" for i in range(1,number+1)]
        if len(key) == 3:
            t, pk, col = key
            offsets = [-4, -2, 1, 5][:base_n+1] if index == 0 else ([-3,0,3] if index == 1 else [-1,0,0,0,6])
            for val,id_ in zip(offsets, ids(schema[0],len(offsets))):
                insert(schema[0], {pk:id_, col:threshold+val})
        else:
            p, pk, c, ck, fk = key
            n = base_n + (0 if index == 0 else (1 if index == 1 else 3))
            parent_ids = ids(schema[0],n)
            for id_ in parent_ids:
                insert(schema[0], {pk:id_})
            refs = parent_ids if index == 0 else (parent_ids+[parent_ids[1],parent_ids[1]]
                    if index == 1 else [parent_ids[0]]+parent_ids[:base_n])
            for cid, ref in zip(ids(schema[1],len(refs)), refs):
                insert(schema[1], {ck:cid, fk:ref})
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
        conn.commit()
    finally:
        conn.close()


def result(path, sql):
    conn = sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)
    try:
        cursor = conn.execute(sql)
        rows = [list(r) for r in cursor.fetchall()]
        return {"ok": True, "columns":[d[0] for d in cursor.description], "rows":rows,
                "row_count":len(rows), "truncated":False, "error":None}
    finally:
        conn.close()


def example(r):
    return {k:r[k] for k in ("columns","rows","row_count","truncated")}


def sft_row(sample_id, info, question, draft, gold_index, draft_index, results):
    exp, observed = results[gold_index], results[draft_index]
    feedback = {**observed, "matches_example": result_fingerprint(observed) == result_fingerprint(exp)}
    obs = {"task":{"id":sample_id, "question":question}, "issue_sql":draft, "ordered":False,
           "example":example(exp),"remaining":6,"history":[{"tool":"check","sql":draft,"observation":feedback},
                                                          {"tool":"schema","observation":info["schema"]}]}
    same = gold_index == draft_index
    target = ({"tool":"submit","candidate":"c0","reason_code":"consistent"} if same else
              {"tool":"check","sql":info["sql"][gold_index],"reason_code":"semantic_repair"})
    return {"id":sample_id,"db_id":info["source_db_id"],"source_train_id":info["source_train_id"],
            "pair_id":info["pair_id"],"state":"synthetic_keep" if same else "synthetic_repair",
            "category":info["category"],"messages":messages_for(obs,{}),
            "target":json.dumps(target,separators=(",",":")), "draft_public_match":feedback["matches_example"]}


def generate(out):
    rows, manifest, _ = load_bank("train")
    split = json.loads(SPLIT.read_text()); sources=anchors(rows)
    metadata=manifest["databases"]; all_ids={r["id"] for r in rows}
    assert set(split["fit_ids"]).isdisjoint(split["holdout_ids"])
    assert set(split["fit_ids"]) | set(split["holdout_ids"]) == all_ids
    assert set(split["fit_db_ids"]).isdisjoint(split["holdout_db_ids"])
    replay=[r for r in read_jsonl(REPLAY) if r["state"].startswith("ordinary_")]
    assert len(replay)==132 and all(r["db_id"] in split["fit_db_ids"] for r in replay)
    arms={"ordinary":[], "paired":[]}
    actor, private, provenance, fixtures=[],[],[],{}
    (out/"fixtures").mkdir(parents=True)
    for partition, quota in (("fit",32),("holdout",8)):
        for category in CATEGORIES:
            selected=choose(split[partition+"_db_ids"],metadata,sources,category,quota)
            for i,(db, (rank, key, tid, gold_source, hard_sql)) in enumerate(selected):
                assert tid in split[partition+"_ids"]
                pair_id="p"+stable("paired-v7",db,key,category)[:16]
                rng=random.Random(pair_id)
                schema=projected_schema(metadata[db]["schema"],key)
                # Holdout uses one reserved NL formulation per category plus familiar formulations on other pairs.
                form=i%2 if partition=="fit" else i%3
                threshold=rng.choice([10,15,20,30,40,50,60,75,90,100])
                questions,sql=query_pair(category,key,form,threshold,i%4<2)
                paths=[out/"fixtures"/f"{pair_id}-{j}.sqlite" for j in range(3)]
                base_n=rng.randint(2,4)
                results=[]
                for j,path in enumerate(paths):
                    create_fixture(path,schema,key,j,base_n,threshold)
                    res=[result(path,q) for q in sql]
                    if (result_fingerprint(res[0])==result_fingerprint(res[1]))!=(j==0):
                        raise ValueError(f"Invalid witness: {pair_id} {j}")
                    results.append(res)
                    fixtures[str(path.relative_to(out))]=digest(path)
                keep=i%2; repair=1-keep
                info={"pair_id":pair_id,"source_db_id":db,"source_train_id":tid,"schema":schema,
                      "category":category,"sql":sql}
                provenance.append({"pair_id":pair_id,"partition":partition,"category":category,"source_db_id":db,
                                   "source_train_id":tid,"schema_motif":key,"source_gold":gold_source,
                                   "source_public_match_wrong_sql":hard_sql,"source_query_operator_overlap":rank[2],
                                   "nl_form":form,"shared_draft":keep, "gold_alternatives":sql,
                                   "questions":questions,"results":[[r["rows"] for r in res] for res in results]})
                if partition=="fit":
                    # Paired arm: the same public fixture and draft is right or wrong solely as a function of intent.
                    # Ordinary arm: one intent, original or conventionally corrupted draft, visibly different answers.
                    for j,(target,draft) in enumerate(((keep,keep),(repair,keep))):
                        sid="v7-"+stable(pair_id,"training",j)[:22]
                        arms["paired"].append(sft_row(sid,info,questions[target],sql[draft],target,draft,results[0]))
                        ordinary_draft=repair if j==0 else keep
                        arms["ordinary"].append(sft_row(sid,info,questions[repair],sql[ordinary_draft],
                                                         repair,ordinary_draft,results[1]))
                else:
                    for target in (keep,repair):
                        sid="s"+stable(pair_id,questions[target],"case")[:22]
                        actor.append({"id":sid,"pair_id":pair_id,"db_id":pair_id,"source_db_id":db,"category":category,
                                      "question":questions[target],"issue_sql":sql[keep],"ordered":False,
                                      "example":example(results[0][target]),"schema":schema,
                                      "public_path":str(paths[0].relative_to(out))})
                        private.append({"id":sid,"pair_id":pair_id,"category":category,
                                        "side":"keep" if target==keep else "repair","gold_sql":sql[target],
                                        "public_path":str(paths[0].relative_to(out)),
                                        "witness_paths":[str(p.relative_to(out)) for p in paths[1:]]})
    sources_hash={str(p.relative_to(ROOT)):digest(p) for p in [SPLIT,EPISODES,REPLAY,ROOT/"mp3_sql/data/train.jsonl",
                        ROOT/"mp3_sql/data/train_memory.jsonl",ROOT/"mp3_sql/data/manifest.json",Path(__file__).resolve()]}
    audit={"train_only":True,"source_sha256":sources_hash,
           "train_schema_holdout_scope":"Incremental LoRA update only; historical initialization used official train.",
           "design":"Each mechanism has both draft directions equally often. Arms share one repair target per motif and all ordinary replay. Paired changes intent on the same ambiguous draft/instance; conventional supervision sees one intent twice, with correct and corrupted drafts on discriminating data.",
           "fixtures":"Three invented finite SQLite instances per pair; projection of train schema names, keys and columns. Real benchmark rows never copied. The public outcomes coincide; each witness gives distinct answers. They test these realizations, not universal SQL equivalence.",
           "source_relation":"Synthetic questions are new templates grounded in train SQL-referenced tables. Some motifs anchor on hard-negative queries with shared tables; this is schema or operator evidence, not an assertion that the error mechanism is identical.",
           "counts":{},"hard_sql_anchored_pairs":{},"fixture_sha256":fixtures, "arms":{}}
    for part in ("fit","holdout"):
        pop=[p for p in provenance if p["partition"]==part]
        audit["counts"][part]={"pairs":len(pop),"schemas":len({p["source_db_id"] for p in pop}),
                                "source_tasks":len({p["source_train_id"] for p in pop}),
                                "by_category":dict(Counter(p["category"] for p in pop)),
                                "draft_directions":dict(Counter(p["shared_draft"] for p in pop))}
        audit["hard_sql_anchored_pairs"][part]=sum(bool(p["source_public_match_wrong_sql"]) for p in pop)
    for arm,records in arms.items():
        synthetic=list(records)
        records.extend(replay)
        # Use identical record ID order across arms to keep subsequent trainer shuffling aligned.
        records.sort(key=lambda x:(stable("order",x["id"],x["state"]),x.get("attempt",0),x.get("sample",0)))
        din=out/arm; din.mkdir()
        write_jsonl(din/"train.jsonl",records)
        summary={"examples":len(records),"synthetic":len(synthetic),"real_replay":len(replay),
                 "fit_schemas_including_replay":len({r["db_id"] for r in records}),
                 "target_actions":dict(Counter(json.loads(r["target"])["tool"] for r in records)),
                 "synthetic_public_match":dict(Counter(str(r["draft_public_match"]) for r in synthetic)),
                 "synthetic_unique_questions":len({json.loads(r["messages"][1]["content"].split("TASK_DATA\n",1)[1].split("\nEND_TASK_DATA")[0])["question"] for r in synthetic}),
                 "unique_synthetic_repair_sql":len({json.loads(r["target"])["sql"] for r in synthetic if r["state"]=="synthetic_repair"}),
                 "training_sha256":digest(din/"train.jsonl")}
        audit["arms"][arm]=summary
        write_json(din/"provenance.json",{"arm":arm,"source":"official train schema/SQL-derived synthetic fixtures and train-only successful policy replay",
                   "train_only":True,"sources_sha256":sources_hash,"design":audit["design"],"counts":summary})
    assert [r["id"] for r in arms["ordinary"]] == [r["id"] for r in arms["paired"]]
    audit["different_message_target_records"]=sum((a["messages"],a["target"])!=(b["messages"],b["target"])
                                                  for a,b in zip(arms["paired"],arms["ordinary"]))
    for name,pop in (("public",actor),("private",private)):
        write_jsonl(out/"holdout"/(name+".jsonl"),sorted(pop,key=lambda r:r["id"]))
    write_jsonl(out/"motifs.jsonl",provenance)
    write_json(out/"split.json",split)
    write_json(out/"audit.json",audit)
    return {k:v for k,v in audit.items() if k!="fixture_sha256"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/prepared_data/v7")
    target=parser.parse_args().output.resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        out=Path(tmp)/"v7"
        audit=generate(out)
        out.rename(target)
    print(json.dumps(audit,indent=2))
