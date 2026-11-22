"""Read-only verification of official files, fixed base and LoRA-only artifacts."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="sft_v002")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/artifact_verification.json")
    args = parser.parse_args()
    official = json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text())["files"]
    # Delegate all database-file validation to the supplied official checker.
    subprocess.run([sys.executable, "-m", "mp3_sql.check_data"], cwd=ROOT, check=True)
    changed = [name for name, meta in official.items() if not name.endswith(".sqlite") and sha(ROOT / name) != meta["sha256"]]
    assert not changed, changed
    base = json.loads((ROOT / "artifacts/models/base_manifest.json").read_text())
    altered = [name for name, meta in base["files"].items() if sha(ROOT / "artifacts/models/Qwen3-0.6B" / name) != meta["sha256"]]
    assert not altered, altered
    adapters = []
    for directory in sorted((ROOT / "artifacts/adapters" / args.run_name).glob("epoch_*")):
        assert {p.name for p in directory.iterdir()} == {"adapter_model.safetensors", "adapter_config.json"}
        config = json.loads((directory / "adapter_config.json").read_text())
        assert config["bias"] == "none" and not config.get("modules_to_save") and config["peft_type"] == "LORA"
        with safe_open(directory / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            assert keys and all(".lora_A." in key or ".lora_B." in key for key in keys)
            parameters = sum(math.prod(handle.get_slice(key).get_shape()) for key in keys)
        size = sum(p.stat().st_size for p in directory.iterdir())
        assert size <= 512 * 1024 ** 2
        adapters.append({"path": str(directory.relative_to(ROOT)), "tensor_count": len(keys),
                         "parameters": parameters, "bytes": size, "sha256": sha(directory / "adapter_model.safetensors")})
    meta = json.loads((ROOT / "runs/training" / args.run_name / "training_metadata.json").read_text())
    assert meta["status"] == "completed"
    assert len(adapters) == meta["config"]["epochs"]
    assert all(a["parameters"] == meta["trainable_parameters"] for a in adapters)
    assert all(".lora_A." in name or ".lora_B." in name for name in meta["trainable_names"])
    assert meta["total_parameters"] <= 1000000000
    report = {"official_files_checked": len(official), "official_files_modified": changed, "database_validation": "unchanged mp3_sql.check_data",
              "base_and_tokenizer_files_checked": len(base["files"]), "base_files_modified": altered,
              "only_lora_A_B_trainable": True, "adapters": adapters,
              "base_parameters": meta["base_parameters"], "total_parameters": meta["total_parameters"]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
