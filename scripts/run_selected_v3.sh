#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH
cd "$(dirname "$0")/.."
export MP3_BASE_MODEL="${MP3_BASE_MODEL:-$PWD/artifacts/models/Qwen3-0.6B}"
export MP3_ADAPTER_DIR="${MP3_ADAPTER_DIR:-$PWD/artifacts/adapters/onpolicy_v001/epoch_02}"
.venv/bin/python scripts/run_experiment.py --experiment reproduction_selected \
  --split "${1:-test}" --config submission/configs/selected.json
