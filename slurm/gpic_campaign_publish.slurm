#!/usr/bin/env bash
#SBATCH --job-name=bcc10k-publish
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10080
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
PYTHON_BIN="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
unset VLLM_PYTHON
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
"$PYTHON_BIN" -m sam3_mask_captioning.cli campaign-publish "$CAMPAIGN_ROOT" --daemon --poll-seconds 30
