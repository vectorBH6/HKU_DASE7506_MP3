"""Reconstruct the fixed, ordered logical calls for a memoized GPU router screening arm."""
from collections import Counter
from pathlib import Path
import sys
from unittest.mock import patch

from probe_same_adapter_router import OUT, OLD_V4, ARMS, ROOT, genkey, read
from mp3_sql.common import load_bank, read_jsonl, write_jsonl
from mp3_sql.environment import Tools, rollout
from mp3_sql.predict import load_factory


def main(adapter):
    out=OUT/adapter
    cfg=read(out/"config.json")
    archive=[]
    if adapter=="v4":
        archive.extend(ROOT/"runs"/run/"validation/r001/generations.jsonl" for run in OLD_V4)
    if (out/"fresh_generations.jsonl").is_file():
        archive.append(out/"fresh_generations.jsonl")
    sources={}
    for path in archive:
        for event in read_jsonl(path):
            assert event["identity"]["adapter"]["sha256"]==cfg["adapter_sha256"] and event["identity"]["device"]=="cuda"
            k=genkey(event["messages"],event["task_id"],event["attempt"],cfg)
            if k in sources:
                assert sources[k]["generated_text"]==event["generated_text"]
            sources[k]=event
    messages=[]
    class ArchivedGenerator:
        def generate(self, prompt, task, attempt, config):
            event=sources[genkey(prompt,task,attempt,config)]
            assert event["messages"]==prompt
            messages.append(event)
            return event["generated_text"]
    rows,m,f=load_bank("validation")
    factory=load_factory(ROOT/"submission"/(ARMS["router"]+".py"))
    tool=Tools(f,m,cache={})
    with patch("sql_repair_agent.model.get_generator",return_value=ArchivedGenerator()):
        reconstructed=[rollout(row,tool,factory(out/"config.json")) for row in rows]
    assert reconstructed==read(out/"router_predictions.json")["predictions"]
    assert len(messages)==read(out/"worker.json")["branches"]["router"]["model_calls"]
    write_jsonl(out/"router_generations.jsonl",messages)
    counts=Counter(g["task_id"] for g in messages)
    assert max(counts.values())<=4
    print(adapter,len(messages),len(counts),max(counts.values()),flush=True)


if __name__=="__main__":
    main(sys.argv[1])
