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
UNIT_SIZE="${UNIT_SIZE:-8}"
SEED="${SEED:-20260825}"
TOKEN_FILE="${TOKEN_FILE:-}"
SOURCE="${SOURCE:-gpic}"
DATASET_NAME="${DATASET_NAME:-stanford-vision-lab/gpic}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"

cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
mkdir -p slurm/logs "$CAMPAIGN_ROOT"
"$VLLM_PYTHON" -m sam3_mask_captioning.cli campaign-init "$CAMPAIGN_ROOT" \
  --unit-size "$UNIT_SIZE" --seed "$SEED" --dataset "$DATASET_NAME" --split "$SOURCE_SPLIT" \
  --terminal-stage bcc --preview-pairs 10
extend_args=(--source "$SOURCE" --target-total "$TARGET_TOTAL" --seed "$SEED")
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
