#!/usr/bin/env bash
#SBATCH --job-name=bcc-q38-materialize
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
: "${REPO_ROOT:?REPO_ROOT is required}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
VLLM_PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
TARGET_TOTAL="${TARGET_TOTAL:-20000}"
MATERIALIZE_CHUNK_SIZE="${MATERIALIZE_CHUNK_SIZE:-5000}"
UNIT_SIZE="${UNIT_SIZE:-8}"
SEED="${SEED:-20260825}"
TOKEN_FILE="${TOKEN_FILE:-}"
SOURCE="${SOURCE:-gpic}"
DATASET_NAME="${DATASET_NAME:-stanford-vision-lab/gpic}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p slurm/logs "$CAMPAIGN_ROOT"
if ! [[ "$MATERIALIZE_CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "MATERIALIZE_CHUNK_SIZE must be a positive integer" >&2
  exit 2
fi
"$VLLM_PYTHON" -m sam3_mask_captioning.cli campaign-init "$CAMPAIGN_ROOT" \
  --unit-size "$UNIT_SIZE" --seed "$SEED" --dataset "$DATASET_NAME" --split "$SOURCE_SPLIT" \
  --terminal-stage bcc --preview-pairs 10
while :; do
  current_total="$("$VLLM_PYTHON" - "$CAMPAIGN_ROOT/campaign_registry.json" <<'PY'
import json
import sys
from pathlib import Path

print(int(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("source_count") or 0))
PY
)"
  if [ "$current_total" -ge "$TARGET_TOTAL" ]; then
    break
  fi
  remaining="$((TARGET_TOTAL - current_total))"
  add_count="$MATERIALIZE_CHUNK_SIZE"
  if [ "$remaining" -lt "$add_count" ]; then
    add_count="$remaining"
  fi
  echo "[materialize] current=$current_total add=$add_count target=$TARGET_TOTAL"
  extend_args=(--source "$SOURCE" --add-images "$add_count" --seed "$SEED")
  if [ "$SOURCE" = "manifest" ]; then
    : "${MANIFEST:?MANIFEST is required when SOURCE=manifest}"
    extend_args+=(--manifest "$MANIFEST")
    [ -n "${IMAGE_ROOT:-}" ] && extend_args+=(--image-root "$IMAGE_ROOT")
  else
    extend_args+=(--cache-dir "$HF_HOME" --split "$SOURCE_SPLIT")
  fi
  if [ -n "$TOKEN_FILE" ] && [ -s "$TOKEN_FILE" ]; then
    extend_args+=(--token-file "$TOKEN_FILE")
  fi
  if [ -n "${EXCLUDE_CSV:-}" ]; then
    extend_args+=(--exclude-csv "$EXCLUDE_CSV")
  fi
  "$VLLM_PYTHON" -m sam3_mask_captioning.cli campaign-extend "$CAMPAIGN_ROOT" \
    "${extend_args[@]}"
done
