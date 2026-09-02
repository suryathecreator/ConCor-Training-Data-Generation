#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VLLM_PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
SAM3_PYTHON="${SAM3_PYTHON:-$REPO_ROOT/.venv-sam3/bin/python}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/qwen38_27b.yaml}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$REPO_ROOT/outputs/campaigns/local}"
SOURCE="${SOURCE:-manifest}"
TARGET_TOTAL="${TARGET_TOTAL:-10}"
UNIT_SIZE="${UNIT_SIZE:-8}"
SEED="${SEED:-20260826}"
MANIFEST="${MANIFEST:-$REPO_ROOT/examples/images.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
EXCLUDE_CSV="${EXCLUDE_CSV:-}"
DATASET_NAME="${DATASET_NAME:-$([ "$SOURCE" = "gpic" ] && printf '%s' 'stanford-vision-lab/gpic' || printf '%s' 'local-images')}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"
HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"

cd "$REPO_ROOT"
export HF_HOME PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$VLLM_PYTHON" -m sam3_mask_captioning.cli campaign-init "$CAMPAIGN_ROOT" \
  --unit-size "$UNIT_SIZE" --seed "$SEED" --dataset "$DATASET_NAME" --split "$SOURCE_SPLIT" \
  --terminal-stage bcc --preview-pairs 10

extend=(--source "$SOURCE" --target-total "$TARGET_TOTAL" --seed "$SEED")
if [ "$SOURCE" = "manifest" ]; then
  extend+=(--manifest "$MANIFEST")
  [ -n "$IMAGE_ROOT" ] && extend+=(--image-root "$IMAGE_ROOT")
else
  extend+=(--cache-dir "$HF_HOME")
fi
[ -n "$EXCLUDE_CSV" ] && extend+=(--exclude-csv "$EXCLUDE_CSV")
"$VLLM_PYTHON" -m sam3_mask_captioning.cli campaign-extend "$CAMPAIGN_ROOT" "${extend[@]}"

for stage in image-review sam3 mask-caption-qa consistency bcc; do
  stage_python="$VLLM_PYTHON"
  case "$stage" in sam3|consistency) stage_python="$SAM3_PYTHON" ;; esac
  "$stage_python" -m sam3_mask_captioning.cli --config "$CONFIG" campaign-worker \
    "$CAMPAIGN_ROOT" --stage "$stage" --worker-index 0
  "$VLLM_PYTHON" -m sam3_mask_captioning.cli campaign-merge "$CAMPAIGN_ROOT" --stage "$stage"
done

"$VLLM_PYTHON" -m sam3_mask_captioning.cli campaign-publish "$CAMPAIGN_ROOT"
"$VLLM_PYTHON" -m sam3_mask_captioning.cli campaign-export-hf \
  "$CAMPAIGN_ROOT" "$CAMPAIGN_ROOT/hf_export" --shard-size 100 --no-image-bytes
echo "Complete: $CAMPAIGN_ROOT"
