#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH
cd "$(dirname "$0")/.."
.venv/bin/python -m mp3_sql.check_data
for split in validation test; do
  .venv/bin/python scripts/run_experiment.py --experiment e00_draft --split "$split"
done
