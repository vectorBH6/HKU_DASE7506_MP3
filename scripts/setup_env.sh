#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
case "${1:---cpu}" in
  --cuda)
    .venv/bin/python -m pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    ;;
  --cpu)
    .venv/bin/python -m pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
    ;;
  *) echo "Usage: bash scripts/setup_env.sh [--cpu|--cuda]" >&2; exit 2 ;;
esac
.venv/bin/python -m pip install -r requirements-model.txt
.venv/bin/python -m pip install -r submission/requirements.txt
.venv/bin/python -m pip check
