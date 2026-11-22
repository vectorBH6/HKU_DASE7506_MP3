"""Inventory a future source export without creating dist or an archive."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    candidates = set()
    for folder in ["submission", "scripts", "training", "tests", "reports"]:
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.name != "delivery_manifest.json" and path.suffix in {".py", ".md", ".txt", ".json", ".csv", ".sh"}:
                candidates.add(path)
    candidates.add(ROOT / "NOTICE")
    candidates.add(ROOT / "artifacts/models/base_manifest.json")
    candidates.add(ROOT / "artifacts/environment-freeze.txt")
    for path in (ROOT / "runs/training/sft_v002").iterdir():
        if path.is_file():
            candidates.add(path)
    for exp in ["e00_draft", "e01_feedback_cpu", "e02_qwen3_base", "e03_qwen3_lora", "a01_no_hints", "a02_no_variants", "a03_no_retrieval"]:
        for run in (ROOT / "runs" / exp).glob("*/r001"):
            for name in ["generations.jsonl", "score.json", "run_metadata.json", "prediction_resources.txt", "scoring_resources.txt"]:
                if (run / name).exists():
                    candidates.add(run / name)
    assert (ROOT / "submission/outputs/test.json").exists(), "Final predictions must exist before inventory"
    files = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(candidates)}
    total = sum(item["bytes"] for item in files.values())
    assert total <= 16 * 1024 ** 2
    record = {"created_archive": False, "source_bytes": total, "source_limit_bytes": 16 * 1024 ** 2,
              "note": "Prospective source export inventory only; no dist directory or upload. Adapter is separate.", "files": files}
    (ROOT / "reports/delivery_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Prospective source assets: {total / 1024 ** 2:.3f} MiB; {len(files)} files; no archive created")

if __name__ == "__main__": main()
