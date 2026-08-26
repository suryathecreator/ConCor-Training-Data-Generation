#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HF_CLI="${HF_CLI:-$REPO_ROOT/.venv/bin/hf}"
MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/models}"
mkdir -p "$MODEL_ROOT"

"$HF_CLI" download Qwen/Qwen3.8-27B --local-dir "$MODEL_ROOT/Qwen3.8-27B"
"$HF_CLI" download facebook/sam3 sam3.pt --local-dir "$MODEL_ROOT"
echo "QWEN_SOURCE=$MODEL_ROOT/Qwen3.8-27B"
echo "SAM3_CHECKPOINT_SOURCE=$MODEL_ROOT/sam3.pt"
