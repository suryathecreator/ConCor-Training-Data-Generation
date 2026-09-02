#!/usr/bin/env bash
#SBATCH --job-name=bcc-q38-export
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
: "${REPO_ROOT:?REPO_ROOT is required}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
VLLM_PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
EXPORT_DIR="${EXPORT_DIR:-$CAMPAIGN_ROOT/hf_export}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
if [ "${PUBLISH_HF:-0}" = "1" ]; then
  CAMPAIGN_ROOT="$CAMPAIGN_ROOT" EXPORT_DIR="$EXPORT_DIR" \
    "$REPO_ROOT/scripts/publish_hf_dataset.sh"
else
  mkdir -p "$EXPORT_DIR"
  rsync -a "$REPO_ROOT/hf_dataset/" "$EXPORT_DIR/"
  "$VLLM_PYTHON" -m sam3_mask_captioning.cli campaign-export-hf \
    "$CAMPAIGN_ROOT" "$EXPORT_DIR" --shard-size 100 --no-image-bytes
fi
