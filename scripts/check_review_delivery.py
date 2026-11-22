"""Inventory future source assets and losslessly compressed logs; no archive/upload."""
import gzip
import hashlib
import json
from pathlib import Path
from analyze_review_runs import EXPERIMENTS, latest

ROOT=Path(__file__).resolve().parents[1]


def main():
    candidates=set()
    log_dir=ROOT/"reports/reproduction_logs"
    log_dir.mkdir(exist_ok=True)
    for folder in ["submission","scripts","training","tests","reports"]:
        for p in (ROOT/folder).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and "reproduction_logs" not in p.parts and not p.name.endswith("delivery_manifest.json") and p.suffix in {".py",".md",".txt",".json",".csv",".sh"}:
                candidates.add(p)
    for name in ["NOTICE","requirements.txt","requirements-model.txt","artifacts/models/base_manifest.json","artifacts/environment-freeze.txt","artifacts/prepared_data/onpolicy_v001/provenance.json"]:
        candidates.add(ROOT/name)
    # Include inspection logs without discarding prompts or tokens. Raw logs stay
    # in runs; gzip -dc reconstructs the exact bytes in a prospective export.
    names=EXPERIMENTS+["e00_draft","e01_feedback_cpu","e02_qwen3_base","a01_no_hints","a02_no_variants","a03_no_retrieval","e10_onpolicy_collection"]
    log_index=[]
    for experiment in dict.fromkeys(names):
        for split in ["train","validation","test"]:
            found=latest(experiment,split)
            if not found: continue
            run,meta=found
            for name in ["run_metadata.json","score.json","prediction_resources.txt","scoring_resources.txt","affinity_changes.json"]:
                if (run/name).exists(): candidates.add(run/name)
            for name in ["generations.jsonl","decisions.jsonl"]:
                p=run/name
                if not p.exists(): continue
                raw=p.read_bytes()
                target=log_dir/f"{experiment}_{split}_{run.name}_{name}.gz"
                target.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
                assert gzip.decompress(target.read_bytes())==raw
                candidates.add(target)
                log_index.append({"source":str(p.relative_to(ROOT)),"compressed":str(target.relative_to(ROOT)),"raw_sha256":hashlib.sha256(raw).hexdigest()})
    index_path=log_dir/"index.json"
    index_path.write_text(json.dumps(log_index,ensure_ascii=False,indent=2)+"\n")
    candidates.add(index_path)
    for version in ["sft_v002","onpolicy_v001"]:
        for p in (ROOT/"runs/training"/version).iterdir():
            if p.is_file(): candidates.add(p)
    files={str(p.relative_to(ROOT)):{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(candidates)}
    total=sum(x["bytes"] for x in files.values())
    assert total<=16*1024**2, (total,"Prune optional duplicates before export")
    result={"created_archive":False,"source_bytes":total,"source_limit_bytes":16*1024**2,
            "note":"Future export inventory only; raw logs remain intact. gzip -dc restores inspection logs. Base, datasets, venv and adapters excluded; adapter exported separately.","lossless_logs":log_index,"files":files}
    (ROOT/"reports/v2_delivery_manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(f"Future source assets: {total/1024**2:.3f} MiB, {len(files)} files; no archive or dist created")


if __name__=="__main__": main()
