"""Probe cross-table column binding and one- or two-edge foreign-key path repairs."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

from sqlglot import exp

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"submission")]
from mp3_sql.common import BUDGET, PROTOCOL, digest, load_bank, read_jsonl, write_json
from mp3_sql.environment import Episode, Tools
from mp3_sql.score import load_cases, score_document
from sql_repair_agent.pool_edits_v8 import (_bindings, _column, _fk_edges, _parse_query, _project,
                                            _resolve, _tokens, build_pool)
from sql_repair_agent.pool_rank_v8 import value
from sql_repair_agent.sql_utils import sql_key
from training.train_pool_v8 import setup, prediction

MAX_NEW=12


def graph(schema):
    edges=defaultdict(list)
    cat={t["name"].casefold():t for t in schema}
    for left in schema:
        for right in schema:
            if left is right:
                continue
            fields=sorted(set(_fk_edges(left,right)),key=lambda x:(x[0].casefold(),x[1].casefold()))
            edges[left["name"].casefold()].extend((l,right["name"].casefold(),r) for l,r in fields)
    return cat,edges


def paths(bind, edges):
    present={t["name"].casefold() for _,t in bind.values()}
    for alias,table in bind.values():
        start=table["name"].casefold()
        for a,m,b in edges[start]:
            path=[(alias,a,m,b)]
            yield path
            if m in present or m in bind:
                continue
            for c,t,d in edges[m]:
                if t not in (start,m):
                    yield [*path,(m,c,t,d)]


def local_columns(node, select):
    return [c for c in node.find_all(exp.Column) if c.find_ancestor(exp.Select) is select]


def qualify_known(select, bind):
    for c in local_columns(select,select):
        if c.table:
            continue
        out=_resolve(c,bind)
        if out is not None:
            c.set("table",out.args["table"].copy())


def owned(table,name):
    return next((c["name"] for c in table["columns"] if c["name"].casefold()==name.casefold()),None)


def fk_join(edge,cat,alias_override=None):
    src,left,dest,right=edge
    name=cat[dest]["name"]
    alias=alias_override or name
    target=exp.Table(this=exp.to_identifier(name,quoted=True))
    return exp.Join(this=target,on=exp.EQ(this=_column(left,src),expression=_column(right,alias)))


def extensions(select,bind,cat,edges,question,labels):
    here={tab["name"].casefold() for _,tab in bind.values()}
    columns=[]
    for node in [*select.expressions, select.args.get("where")]:
        if node is not None:
            columns+=local_columns(node,select)
    table_names={name:{c["name"].casefold() for c in table["columns"]} for name,table in cat.items()}
    owners=defaultdict(set)
    for path in paths(bind,edges):
        dest=path[-1][2]
        if dest not in here:
            for n in table_names[dest]:
                owners[n].add(dest)
    for path in paths(bind,edges):
        if any(dest in bind or dest in here for _,_,dest,_ in path):
            continue
        dest=path[-1][2]
        tab=cat[dest]
        signs=_tokens(tab["name"])
        mentioned=bool(signs) and signs<=_tokens(question)
        remap={}
        reasons=set()
        for c in columns:
            n=c.name.casefold()
            if n not in table_names[dest]:
                continue
            qualified_target=c.table.casefold()==dest
            unique=owners[n]=={dest}
            if not qualified_target and not unique:
                continue
            old=_resolve(c,bind)
            if old is None:
                if not c.table or c.table.casefold() in bind or qualified_target:
                    remap[sql_key(c.sql(dialect="sqlite"))]=owned(tab,c.name)
                    reasons.add("unresolved")
            elif mentioned and n not in {"id","name"} and _tokens(c.name)&_tokens(question):
                remap[sql_key(c.sql(dialect="sqlite"))]=owned(tab,c.name)
                reasons.add("mentioned")
        header_target=any(str(label).casefold() in table_names[dest] and
                          owners[str(label).casefold()]=={dest} and
                          not any(owned(table,str(label)) for _,table in bind.values()) for label in labels)
        if not remap and not header_target:
            continue
        changed=select.copy()
        for node in [*changed.expressions,changed.args.get("where")]:
            if node is None:continue
            for c in local_columns(node,changed):
                field=remap.get(sql_key(c.sql(dialect="sqlite")))
                if field:
                    c.set("this",exp.to_identifier(field,quoted=True))
                    c.set("table",exp.to_identifier(tab["name"],quoted=True))
        qualify_known(changed,bind)
        for edge in path:
            changed.append("joins",fk_join(edge,cat))
        if header_target:
            out=_project(changed,labels,_bindings(changed,cat))
            if out is not None:
                changed=out; reasons.add("header")
        if not reasons:
            continue
        category="bind_"+("_".join(sorted(reasons)))
        priority=0 if "unresolved" in reasons else 1 if "header" in reasons else 3
        yield (priority,len(path)),category,changed


def bridges(select,bind,cat,edges):
    joins=select.args.get("joins",[])
    if not joins:
        return
    aliases=list(bind)
    for ix,j in enumerate(joins):
        old=j.args.get("on")
        if j.args.get("side") or j.args.get("using") or not isinstance(j.this,exp.Table) or not isinstance(old,exp.EQ):
            continue
        if not all(isinstance(col,exp.Column) and col.table for col in (old.this,old.expression)):
            continue
        target=j.this.alias_or_name.casefold()
        if target not in (old.this.table.casefold(),old.expression.table.casefold()):
            continue
        peer_col,other=(old.this,old.expression) if old.expression.table.casefold()==target else (old.expression,old.this)
        source=peer_col.table.casefold()
        if source not in aliases[:ix+1]:
            continue
        target_name=bind[target][1]["name"].casefold()
        source_name=bind[source][1]["name"].casefold()
        direct={(a.casefold(),b.casefold()) for a,t,b in edges[source_name] if t==target_name}
        if (peer_col.name.casefold(),other.name.casefold()) in direct:
            continue
        for path in paths({source:bind[source]},edges):
            if len(path)!=2 or path[-1][2]!=target_name:
                continue
            middle=path[0][2]
            if middle in bind or any(tab["name"].casefold()==middle for _,tab in bind.values()):
                continue
            changed=select.copy()
            qualify_known(changed,bind)
            js=changed.args["joins"]
            replacement=js[ix].copy()
            replacement.set("on",fk_join(path[1],cat,bind[target][0]).args["on"])
            changed.set("joins",[*js[:ix],fk_join(path[0],cat),replacement,*js[ix+1:]])
            yield (1,2),"bridge",changed


def candidates(row, raw, schema, baseline):
    catalog,edges=graph(schema)
    seen={sql_key(s) for s in baseline}
    items=[]
    for order,source in enumerate(dict.fromkeys(raw)):
        seed=_parse_query(source)
        if seed is None:continue
        signature=Counter((x.this,x.is_string) for x in seed.find_all(exp.Literal))
        for idx,select in enumerate(seed.find_all(exp.Select)):
            bind=_bindings(select,catalog)
            if not bind:continue
            labels=row["example"]["columns"] if idx==0 and isinstance(seed,exp.Select) else []
            proposals=[*bridges(select,bind,catalog,edges),
                       *extensions(select,bind,catalog,edges,row["question"],labels)]
            for rank,family,change in proposals:
                if select is seed:
                    whole=change
                else:
                    whole=seed.copy()
                    list(whole.find_all(exp.Select))[idx].replace(change)
                if signature!=Counter((x.this,x.is_string) for x in whole.find_all(exp.Literal)):
                    continue
                sql=whole.sql(dialect="sqlite")
                if len(sql.encode())<=16000:
                    items.append((rank,order,idx,family,sql))
    result=[]
    for rank,order,idx,family,sql in sorted(items):
        key=sql_key(sql)
        if key not in seen:
            result.append((family,sql));seen.add(key)
        if len(result)==MAX_NEW:break
    return result


def main():
    out=ROOT/"runs/v13_fk_paths"
    out.mkdir(exist_ok=True)
    splits=json.loads((ROOT/"artifacts/prepared_data/v6/split.json").read_text())
    prior=read_jsonl(ROOT/"runs/v8_pool_train/train_pool.jsonl")
    pool={p["id"]:p["candidates"] for p in prior}
    weights=json.loads((ROOT/"submission/configs/pool_v8_weights.json").read_text())
    rules={"limit":MAX_NEW,"path_lengths":[1,2],"baseline_sha256":digest(ROOT/"runs/v8_pool_train/train_pool.jsonl"),
           "scope":"Official train only; compare V8 full bounded pool and native raw-then-ranked 5 checks after schema, draft, hint. Offline oracle separately checks every novel option. Deployable policy uses only unused budget if no old checked candidate passed public; first new public match wins.",
           "script_sha256":digest(Path(__file__)),"novelty":"Normalized SQL differs from every member of the task's complete old pool"}
    rows,manifest,folder=load_bank("train"); tools=Tools(folder,manifest,cache={})
    cache={}; ref,case_root=load_cases("train",folder)
    hdr={"protocol":PROTOCOL,"split":"train","budget":BUDGET,"bank_sha256":digest(folder/"train.jsonl"),
         "manifest_sha256":digest(folder/"manifest.json")}
    def grade(row,pred):
        doc={**hdr,"predictions":[pred]}
        return score_document(doc,[row],manifest,folder,{**ref,"ids":[row["id"]]},case_root,workers=1,cache=cache)["per_question"][0]["passed"]
    for group in ("fit","holdout"):
        ids=set(splits[group+"_ids"])
        stat=Counter()
        edits=Counter(); new_correct=Counter()
        for row in rows:
            i=row["id"]
            if i not in ids:continue
            stat["n"]+=1
            ep=setup(row,tools); obs=ep.observation(); schema=ep.trace[0]["observation"]
            raw=[c["sql"] for c in build_pool(obs,schema,ep.trace[2]["observation"]) if c["family"]=="raw"]
            origs=pool[i]
            assert {sql_key(x) for x in raw}=={sql_key(c["sql"]) for c in origs if c["family"]=="raw"}
            new=candidates(row,raw,schema,[c["sql"] for c in origs])
            full=any(c["correct"] for c in origs)
            stat["original_full_pool_correct"]+=full
            stat["new_candidates"]+=len(new)
            stat["tasks_with_new_candidates"]+=bool(new)
            labels={}
            for j,(family,sql) in enumerate(new):
                if not j%5:branch=setup(row,tools)
                branch.step({"tool":"check","sql":sql})
                is_public=branch.trace[-1]["observation"]["matches_example"]
                ok=grade(row,prediction(branch,sql)) if is_public else False
                labels[sql]=(is_public,ok)
                edits[family]+=1
                if ok:new_correct[family]+=1
            any_new=any(p for _,p in labels.values())
            stat["correct_new_candidates"]+=sum(p for _,p in labels.values())
            stat["new_candidates_public"]+=sum(p for p,_ in labels.values())
            stat["tasks_with_correct_new_candidate"]+=any_new
            stat["newly_covered_over_old_full_pool"]+=any_new and not full

            rest=origs[1:]
            static=[c for c in rest if c["family"]=="raw"]
            ranked=sorted([c for c in rest if c["family"]!="raw"],reverse=True,
                          key=lambda c:value(c["features"],weights["check"]))
            checked=[origs[0],*(static+ranked)[:5]]
            for c in checked[1:]:
                ep.step({"tool":"check","sql":c["sql"]})
                assert ep.trace[-1]["observation"]["matches_example"]==c["public"]
            good=[c for c in checked if c["public"]]
            best=max(good,key=lambda c:value(c["features"],weights["choose"])) if good else origs[0]
            before=best["correct"]; after=before; final=best["sql"]
            charged=0
            if not good:
                stat["failed_prefix_has_budget"]+=bool(ep.remaining)
                for _,sql in new[:ep.remaining]:
                    ep.step({"tool":"check","sql":sql});charged+=1
                    if ep.trace[-1]["observation"]["matches_example"]:
                        final=sql;after=labels[sql][1]
                        break
            if charged:
                assert after==grade(row,prediction(ep,final))
            stat["original_budgeted_correct"]+=before
            stat["new_budgeted_correct"]+=after
            stat["wins"]+=after and not before
            stat["losses"]+=before and not after
            stat["extra_checks"]+=charged
            stat["changed_answers"]+=final!=best["sql"]
            stat["max_tools"]=max(stat["max_tools"],BUDGET-ep.remaining)
        rules[group]={**dict(stat),"novel_families":dict(edits),"correct_families":dict(new_correct)}
        print(group,rules[group],flush=True)
    write_json(out/"train_holdout.json",rules)

if __name__=="__main__":
    main()
