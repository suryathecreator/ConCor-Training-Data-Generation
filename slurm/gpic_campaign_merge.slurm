#!/usr/bin/env bash
#SBATCH --job-name=bcc10k-merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10080
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
STAGE="${STAGE:?STAGE is required}"
PYTHON_BIN="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
unset VLLM_PYTHON
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
"$PYTHON_BIN" -m sam3_mask_captioning.cli campaign-merge "$CAMPAIGN_ROOT" \
  --stage "$STAGE" --wait --poll-seconds "${MERGE_POLL_SECONDS:-30}"

# A faster cohort may have completed all units while another cohort remained
# queued. Those exact same-stage arrays are now redundant and safe to cancel.
if [ -n "${WORKER_JOB_IDS:-}" ]; then
  IFS=: read -r -a worker_jobs <<< "$WORKER_JOB_IDS"
  scancel "${worker_jobs[@]}" 2>/dev/null || true
fi
