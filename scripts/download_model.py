"""Provision the exact public base once; evaluation never downloads anything."""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
if __name__ == "__main__":
    target = ROOT / "artifacts/models/Qwen3-0.6B"
    snapshot_download("Qwen/Qwen3-0.6B", revision=REVISION, local_dir=target,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md", "LICENSE"],
                      max_workers=2)
    manifest = {"repository": "Qwen/Qwen3-0.6B", "revision": REVISION,
                "files": {str(p.relative_to(target)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in sorted(target.rglob("*")) if p.is_file() and ".cache" not in p.parts}}
    (ROOT / "artifacts/models/base_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
