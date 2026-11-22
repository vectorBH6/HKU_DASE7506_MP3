"""Replay each checked train candidate with the unchanged official scorer."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "runs/e10_onpolicy_collection/train/r001/predictions.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/prepared_data/onpolicy_v001/candidate_scores")
    args = parser.parse_args()
    document = json.loads(args.source.read_text())
    assert document["split"] == "train", "Candidate labels must use train only"
    assert max(sum(e["tool"] == "check" for e in p["trace"]) for p in document["predictions"]) == 5
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(5):
        rows = []
        for original in document["predictions"]:
            checks = [i for i,e in enumerate(original["trace"]) if e["tool"] == "check"]
            end = checks[min(index, len(checks)-1)]
            prefix = copy.deepcopy(original["trace"][:end+1])
            sql = prefix[-1]["sql"]
            prefix.append({"tool": "submit", "sql": sql})
            rows.append({"id": original["id"], "sql": sql, "trace": prefix, "tool_calls": len(prefix)-1})
        predictions = args.output_dir / f"candidate_{index:02d}.predictions.json"
        score = args.output_dir / f"candidate_{index:02d}.score.json"
        predictions.write_text(json.dumps({**document,"predictions":rows}, ensure_ascii=False, indent=2)+"\n")
        with score.with_suffix(".log").open("w") as log:
            subprocess.run([sys.executable,"-m","mp3_sql.score","--split","train","--workers","4",
                            "--predictions",str(predictions),"--output",str(score)],
                           cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads(score.read_text())
        print(index,result["correct"],result["n"],flush=True)


if __name__ == "__main__":
    main()
