#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
export WN_DATA_DIR="${WN_DATA_DIR:-$REPO_ROOT/.runtime/wordnet}"
mkdir -p "$WN_DATA_DIR"
"$PYTHON_BIN" -m pip install --no-cache-dir "wn==1.1.0"
"$PYTHON_BIN" - <<'PY'
import wn
try:
    wn.Wordnet("oewn:2025", expand="")
except Exception:
    wn.download("oewn:2025")
print("Open English WordNet oewn:2025 is ready")
PY
