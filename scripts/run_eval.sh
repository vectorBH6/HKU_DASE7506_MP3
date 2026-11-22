#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH
cd "$(dirname "$0")/.."
.venv/bin/python scripts/run_experiment.py "$@"
