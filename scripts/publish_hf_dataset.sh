#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
HF_REPO_ID="${HF_REPO_ID:-suryadv/gpic-bcc-sam3-qwen38-27b}"
VLLM_PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
HF_CLI="${HF_CLI:-$(dirname "$VLLM_PYTHON")/hf}"
TOKEN_FILE="${TOKEN_FILE:-}"
EXPORT_DIR="${EXPORT_DIR:-$CAMPAIGN_ROOT/hf_publish}"
SCAFFOLD="$REPO_ROOT/hf_dataset"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "$TOKEN_FILE" ] && [ -s "$TOKEN_FILE" ]; then
  HF_TOKEN="$(<"$TOKEN_FILE")"
  export HF_TOKEN
fi
mkdir -p "$EXPORT_DIR"
rsync -a "$SCAFFOLD/" "$EXPORT_DIR/"
"$VLLM_PYTHON" -m sam3_mask_captioning.cli campaign-export-hf \
  "$CAMPAIGN_ROOT" "$EXPORT_DIR" --shard-size 100 --no-image-bytes \
  --checkpoint-units "${EXPORT_CHECKPOINT_UNITS:-100}"
# The public audit Parquet already contains portable per-image dispositions.
# Never upload internal campaign registries/reports: they may contain cluster
# paths and scheduler metadata that are irrelevant to dataset consumers.
"$HF_CLI" repos create "$HF_REPO_ID" --repo-type dataset --exist-ok
"$HF_CLI" upload-large-folder "$HF_REPO_ID" "$EXPORT_DIR" \
  --repo-type dataset --num-workers "${HF_UPLOAD_WORKERS:-8}"
