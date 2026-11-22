"""Prepare a size-bounded future source inventory with lossless inspection logs."""
import gzip,hashlib,json
from pathlib import Path
from analyze_review_runs import latest
from analyze_feedback_v4 import CPU
ROOT=Path(__file__).resolve().parents[1]


def main():
    previous=json.loads((ROOT/'reports/v3_delivery_manifest.json').read_text());candidates={ROOT/name for name in previous['files']}
    for folder in ['submission','scripts','training','tests','reports']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('delivery_manifest.json') and p.suffix in ['.py','.md','.txt','.json','.csv','.sh','.gz']:candidates.add(p)
    directory=ROOT/'reports/v4_reproduction_logs';directory.mkdir(exist_ok=True);index=[]
    def compress(p):
        raw=p.read_bytes();target=directory/((str(p.relative_to(ROOT)).replace('/','__'))+'.gz')
        target.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0));assert gzip.decompress(target.read_bytes())==raw
        candidates.add(target);index.append({'source':str(p.relative_to(ROOT)),'compressed':str(target.relative_to(ROOT)),'raw_sha256':hashlib.sha256(raw).hexdigest()})
    exps=CPU+['e30_feedback_collection']+[p.name for p in (ROOT/'runs').glob('dev_v4_*')]
    for exp in dict.fromkeys(exps):
        for split in ['train','validation','test']:
            found=latest(exp,split)
            if not found:continue
            run,_=found
            for name in ['run_metadata.json','score.json','prediction_resources.txt','scoring_resources.txt','affinity_changes.json']:
                if (run/name).exists():candidates.add(run/name)
            for name in ['generations.jsonl','decisions.jsonl','predictions.json']:
                if (run/name).exists():compress(run/name)
    # Exact duplicate historical outputs/traces need not consume the 16 MiB source budget.
    for p in list(candidates):
        if p.exists() and p.suffix=='.json' and p.stat().st_size>300000 and p!=ROOT/'submission/outputs/test.json':
            compress(p);candidates.remove(p)
    for version in ['feedback_v4','feedback_v4_extra']:
        folder=ROOT/'runs/training'/version
        if folder.exists():candidates.update(p for p in folder.iterdir() if p.is_file())
    for name in ['artifacts/prepared_data/feedback_v4/provenance.json','artifacts/prepared_data/synthesis_v4/provenance.json','artifacts/v4-tests.log','artifacts/v4-contract-tests.log','artifacts/v4-constraint-tests.log','artifacts/v4-final-tests.log']:
        if (ROOT/name).exists():candidates.add(ROOT/name)
    index_path=directory/'index.json';index_path.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n');candidates.add(index_path)
    candidates={p for p in candidates if p.exists() and not p.name.endswith('delivery_manifest.json')}
    files={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(candidates)}
    total=sum(x['bytes'] for x in files.values());assert total<=16*1024**2,total
    result={'created_archive':False,'source_bytes':total,'source_limit_bytes':16*1024**2,'files':files,'lossless_logs':index,
            'note':'Future inventory only. Canonical outputs/test.json remains full JSON; raw local files are not deleted. gzip restores exact historical/dev prompts, tool traces and outputs using index source paths. Base, datasets, venv, prepared data and adapters excluded; selected adapter exported separately.'}
    (ROOT/'reports/v4_delivery_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'Future source assets {total/1024**2:.3f} MiB, {len(files)} files; no dist or archive created')

if __name__=='__main__':main()
