"""Test whether checking actual public column values makes terminal literal repairs reliable."""
from collections import Counter
import json
from pathlib import Path
import re
import sys
from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"submission"),str(ROOT/"scripts")]
from mp3_sql.common import BUDGET, digest
from probe_v10_text_grounding import IndependentGrader, PublicBank, load_predictions, name_columns
from sql_repair_agent.sql_utils import sql_key


def quote(name):
    return exp.to_identifier(name,quoted=True).sql(dialect="sqlite")


def resolve_column(column, schema):
    select = column.find_ancestor(exp.Select)
    if select is None:
        return None
    fr = select.args.get("from_") or select.args.get("from")
    if not fr:
        return None
    tables = [fr.this]+[j.this for j in select.args.get("joins",[])]
    found = []
    for ref in tables:
        if not isinstance(ref,exp.Table):
            continue
        if column.table and column.table.casefold() not in (ref.alias_or_name.casefold(),ref.name.casefold()):
            continue
        tab = schema.get(ref.name.casefold())
        if tab is not None:
            col = next((c for c in tab["columns"] if c["name"].casefold()==column.name.casefold()), None)
            if col is not None:
                found.append((tab,col["name"]))
    return found[0] if len(found)==1 else None


def lookups(source, rowschema):
    schema={t["name"].casefold():t for t in rowschema}
    names={t:(f,l) for t,f,l in name_columns(rowschema)}
    try:
        tree=parse_one(source,read="sqlite")
    except (SqlglotError,RecursionError,ValueError):
        return
    if tree is None:
        return
    for ix,cmp in enumerate(tree.find_all(exp.EQ)):
        col,val=cmp.this,cmp.expression
        if not isinstance(col,exp.Column) or not isinstance(val,exp.Literal) or not val.is_string:
            continue
        literal=val.this.strip()
        if not literal or len(literal)>160:
            continue
        slot=resolve_column(col,schema)
        if slot is None:
            continue
        table,colname=slot
        qtab,qcol=quote(table["name"]),quote(colname)
        value=exp.Literal.string(literal).sql(dialect="sqlite")
        parts=literal.split()
        pair=names.get(table["name"].casefold(),())
        if len(parts)==2 and all(w[:1].isupper() and w.replace("-","").isalpha() for w in parts) and colname in pair:
            f,l=map(quote,pair)
            fv,lv=[exp.Literal.string(w).sql(dialect="sqlite") for w in parts]
            probe=(f"SELECT 1 FROM {qtab} WHERE LOWER({f})=LOWER({fv}) "
                   f"AND LOWER({l})=LOWER({lv}) LIMIT 1")
            yield (source,ix,col,literal,"name",pair),probe
        if len(literal)>1:
            # See all case/substring matches up to the ambiguity guard. The
            # source value and tables come from this task's public observation.
            probe=(f"SELECT DISTINCT CAST({qcol} AS TEXT) AS value FROM {qtab} "
                   f"WHERE INSTR(LOWER(CAST({qcol} AS TEXT)), LOWER({value})) > 0 "
                   "ORDER BY LENGTH(value), value LIMIT 5")
            yield (source,ix,col,literal,"text",()),probe


def proposal(spec, observation, question):
    if not observation.get("ok") or not observation["rows"]:
        return
    source,ix,col,literal,kind,pair=spec
    low=exp.Lower(this=col.copy())
    if kind=="name":
        def grounded(column,part):
            c=col.copy()
            c.set("this",exp.to_identifier(column,quoted=True))
            return exp.EQ(this=exp.Lower(this=c),expression=exp.Literal.string(part.lower()))
        first,last=literal.split()
        edit=exp.and_(grounded(pair[0],first),grounded(pair[1],last))
    else:
        vals=[r[0] for r in observation["rows"] if r and isinstance(r[0],str)]
        if not vals or literal in vals:
            return
        eq=[v for v in vals if v.casefold()==literal.casefold()]
        if eq:
            canonical=literal
        else:
            if len(vals)!=1 or len(vals[0])>4*len(literal)+12:
                return
            canonical=vals[0]
            if not re.search(r"(?<!\w)"+re.escape(literal)+r"(?!\w)",canonical,re.I):
                return
            if not re.search(r"(?<!\w)"+re.escape(canonical)+r"(?!\w)",question,re.I):
                return
        edit=exp.EQ(this=low,expression=exp.Literal.string(canonical.lower()))
    tree=parse_one(source,read="sqlite")
    list(tree.find_all(exp.EQ))[ix].replace(edit)
    new=tree.sql(dialect="sqlite")
    return new if len(new.encode())<=16000 else None


def make_index(bank, pred, ids):
    records=[]
    for row in bank.rows:
        if row["id"] not in ids:
            continue
        old=pred[row["id"]]
        ep=bank.terminal(row,old)
        hist=ep.trace; prev=[e for e in hist if e["tool"]=="check"]
        matches=any(e["observation"]["matches_example"] for e in prev)
        all_sources=list(dict.fromkeys([old["sql"],*(e["sql"] for e in reversed(prev))]))
        schema=next((e["observation"] for e in hist if e["tool"]=="schema"),[])
        probes=[(spec,q) for sql in all_sources for spec,q in lookups(sql,schema)] if ep.remaining>=2 else []
        records.append((row,old,matches,probes))
    return records


def run_case(bank,case,gate,only_incumbent):
    row,old,matches,look=case
    if gate=="off" or not look or matches and (gate=="failed" or gate=="failed_or_empty" and row["example"]["row_count"]):
        return old,[]
    ep=bank.terminal(row,old)
    checks={sql_key(e["sql"]) for e in ep.trace if e["tool"]=="check"}
    feedback={e["sql"]:e["observation"] for e in ep.trace if e["tool"]=="execute"}
    changed=old["sql"]
    for spec,q in look:
        if only_incumbent and spec[0]!=old["sql"]:
            continue
        if q not in feedback:
            if ep.remaining<2:
                continue
            ep.step({"tool":"execute","sql":q})
            feedback[q]=ep.trace[-1]["observation"]
        sql=proposal(spec,feedback[q],row["question"])
        if not sql or sql_key(sql) in checks or not ep.remaining:
            continue
        checks.add(sql_key(sql))
        ep.step({"tool":"check","sql":sql})
        if ep.trace[-1]["observation"]["matches_example"]:
            changed=sql
            break
    extra=ep.trace[len(old["trace"])-1:]
    ep.step({"tool":"submit","sql":changed})
    return {"id":row["id"],"sql":changed,"trace":ep.trace,"tool_calls":BUDGET-ep.remaining},extra


def evaluate(bank,grader,cases,setting,full=False):
    s=Counter(n=len(cases),baseline=0,correct=0,changed=0,wins=0,losses=0,extra_checks=0,
              extra_executions=0,extra_tools=0,max_tools=0,touched=0)
    for c in cases:
        row,old,_,_=c
        after,extra=run_case(bank,c,*setting)
        key=(row["id"],after["sql"])
        oldg=grader.grade_option({"row":row,"old":old},old["sql"])
        if key not in grader.labels or full and extra:
            grader.labels[key]=grader.grade(row,after)
        good=grader.labels[key]
        s.update(correct=int(good),baseline=int(oldg),wins=int(good and not oldg),losses=int(oldg and not good),
                 changed=int(after["sql"]!=old["sql"]),extra_tools=len(extra),touched=int(bool(extra)),
                 extra_executions=sum(e["tool"]=="execute" for e in extra),extra_checks=sum(e["tool"]=="check" for e in extra))
        s["max_tools"]=max(s["max_tools"],after["tool_calls"])
    return dict(s)


def main():
    path=ROOT/"runs/v10_text_grounding"
    selected=path/"public_value_training_selection.json"
    split=json.loads((ROOT/"artifacts/prepared_data/v6/split.json").read_text())
    bank=PublicBank("train"); grade=IndependentGrader(bank)
    archive=load_predictions("train","train")
    if selected.exists():
        locked=json.loads(selected.read_text())
        if locked["script_sha256"]!=digest(Path(__file__)):
            raise ValueError("Frozen policy code has changed")
        policy=locked["policy"]
    else:
        fit=make_index(bank,archive,set(split["fit_ids"]))
        policies=[("off",True)]+[(g,inc) for g in ("failed","failed_or_empty","all") for inc in (True,False)]
        comparisons=[evaluate(bank,grade,fit,p) for p in policies]
        best=min(range(len(policies)),key=lambda i:(-comparisons[i]["correct"],comparisons[i]["extra_tools"],
                                                     comparisons[i]["losses"],i))
        policy=policies[best]
        locked={"fit":"382 train questions; V5 first existing rollout",
                "rank":"maximize three-instance training accuracy; then minimize tool calls, losses, enumeration",
                "policies":policies,"comparisons":comparisons,"policy":policy,"script_sha256":digest(Path(__file__)),
                "fit_result":evaluate(bank,grade,fit,policy,True)}
        with selected.open("x") as f:
            json.dump(locked,f,indent=2)
        print("fit",policy,comparisons,flush=True)
    for name in ("train","v4b_holdout"):
        cases=make_index(bank,load_predictions("train",name),set(split["holdout_ids"]))
        result=evaluate(bank,grade,cases,policy,True)
        (path/(name+"_value_holdout.json")).write_text(json.dumps(result,indent=2)+"\n")
        print(name,result,flush=True)
    vb=PublicBank("validation"); vg=IndependentGrader(vb)
    for name in ("v8","v10"):
        if name=="v8":
            preds=load_predictions("validation",name)
        else:
            p=ROOT/"runs/v10_prefix_probe/validation/predictions.json"
            preds={r["id"]:r for r in json.loads(p.read_text())["predictions"]}
        cases=make_index(vb,preds,set(preds))
        result=evaluate(vb,vg,cases,policy,True)
        (path/(name+"_value_validation.json")).write_text(json.dumps(result,indent=2)+"\n")
        print(name,result,flush=True)

if __name__=="__main__":
    main()
