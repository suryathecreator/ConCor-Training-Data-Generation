#!/usr/bin/env bash
#SBATCH --job-name=bcc10k-worker
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
CONFIG="${CONFIG:?CONFIG is required}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
STAGE="${STAGE:?STAGE is required}"
SAM3_REPO_ROOT="${SAM3_REPO_ROOT:-$REPO_ROOT/third_party/sam3}"
VLLM_PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
SAM3_PYTHON="${SAM3_PYTHON:-$VLLM_PYTHON}"
HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"

case "$STAGE" in
  image-review|mask-caption|mask-qa|mask-caption-qa|bcc-draft|bcc-rewrite|bcc) PYTHON_BIN="$VLLM_PYTHON" ;;
  sam3|consistency) PYTHON_BIN="$SAM3_PYTHON" ;;
  *) echo "Unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
unset VLLM_PYTHON
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

cd "$REPO_ROOT"
mkdir -p slurm/logs "$CAMPAIGN_ROOT"
export SAM3_REPO_ROOT HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="$REPO_ROOT/src:$SAM3_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WN_DATA_DIR="${WN_DATA_DIR:-$REPO_ROOT/.runtime/wordnet}"
# Packed inputs are expanded below node-local storage and exposed through
# short-lived directory symlinks. This avoids creating hundreds of small files
# per unit on shared GPFS, including after a preemption or continuation.
export BCC_NODE_HYDRATION_ROOT="${SLURM_TMPDIR:-/tmp}/${USER:-user}-bcc-hydration/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$BCC_NODE_HYDRATION_ROOT"

# Successor arrays use afterany for preemption safety. A merge watcher can be
# one polling interval behind the last unit, so a successor may briefly start
# after the stage is already complete. Avoid a pointless 55.6-GB model copy and
# cache restore in that race by checking only the tiny durable success markers.
registry="$CAMPAIGN_ROOT/campaign_registry.json"
if command -v jq >/dev/null 2>&1 && [ -s "$registry" ]; then
  unit_count="$(jq -r '.unit_count // 0' "$registry")"
  if [[ "$unit_count" =~ ^[0-9]+$ ]] && [ "$unit_count" -gt 0 ]; then
    stage_complete=1
    for ((unit_id = 0; unit_id < unit_count; unit_id++)); do
      printf -v unit_name '%06d' "$unit_id"
      if [ ! -s "$CAMPAIGN_ROOT/units/$unit_name/stages/$STAGE/_SUCCESS.json" ]; then
        stage_complete=0
        break
      fi
    done
    if [ "$stage_complete" = "1" ]; then
      echo "[work-preflight] stage=$STAGE already complete; skipping runtime/model staging"
      exit 0
    fi
  fi
fi

case "$STAGE" in
  sam3|consistency)
    source "$REPO_ROOT/scripts/stage_sam3_runtime.sh"
    ;;
esac

QWEN_WORKER=0
case "$STAGE" in
  image-review)
    QWEN_WORKER=1
    export BCC_CACHE_PROFILE="${BCC_CACHE_PROFILE:-review}"
    export BCC_CACHE_ENGINE_PROFILES="16384x32@32768"
    source "$REPO_ROOT/scripts/stage_qwen_runtime.sh"
    ;;
  mask-caption|mask-qa|mask-caption-qa)
    QWEN_WORKER=1
    export BCC_CACHE_PROFILE="${BCC_CACHE_PROFILE:-mask}"
    export BCC_CACHE_ENGINE_PROFILES="16384x32@65536"
    source "$REPO_ROOT/scripts/stage_qwen_runtime.sh"
    ;;
  bcc-draft|bcc-rewrite|bcc)
    QWEN_WORKER=1
    export BCC_CACHE_PROFILE="${BCC_CACHE_PROFILE:-bcc}"
    export BCC_CACHE_ENGINE_PROFILES="49152x2@49152"
    source "$REPO_ROOT/scripts/stage_qwen_runtime.sh"
    ;;
esac

# Give the four eight-task cohorts disjoint rotations through the unit list.
# Claims are already atomic, but this avoids making four workers race for each
# of the same first eight units at every stage.
case "${GPU_COHORT:-gpu}" in
  h200) COHORT_OFFSET=0 ;;
  a100) COHORT_OFFSET=8 ;;
  l40s) COHORT_OFFSET=16 ;;
  a40) COHORT_OFFSET=8 ;;
  *) COHORT_OFFSET=0 ;;
esac
WORKER_INDEX="$((COHORT_OFFSET + ${SLURM_ARRAY_TASK_ID:-0}))"

# Stop before the scheduler's actual deadline. With eight images per unit, the
# active work is bounded and a successor can deterministically resume it.
WORKER_WALL_SECONDS="${WORKER_WALL_SECONDS:-86400}"
DRAIN_SECONDS="${DRAIN_SECONDS:-600}"
worker_now="$(date +%s)"
configured_end="$((worker_now + WORKER_WALL_SECONDS))"
effective_end="$configured_end"
drain_deadline_source="configured_fallback"

# QOS/partition policy may silently cap the requested #SBATCH walltime. The
# SLURM_JOB_END_TIME environment variable can retain an obsolete launch value,
# so query the live array task and choose the earlier trustworthy end.
live_job_id="${SLURM_JOB_ID:-}"
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  live_job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
if [ -n "$live_job_id" ] && command -v squeue >/dev/null 2>&1; then
  scheduler_end_iso="$(squeue -h -j "$live_job_id" -o '%e' 2>/dev/null | head -n 1 || true)"
  if [ -n "$scheduler_end_iso" ] && [ "$scheduler_end_iso" != "N/A" ] && \
     [ "$scheduler_end_iso" != "Unknown" ]; then
    scheduler_end_epoch="$(date -d "$scheduler_end_iso" +%s 2>/dev/null || true)"
    if [ -n "$scheduler_end_epoch" ] && \
       [ "$scheduler_end_epoch" -gt "$worker_now" ] && \
       [ "$scheduler_end_epoch" -lt "$effective_end" ]; then
      effective_end="$scheduler_end_epoch"
      drain_deadline_source="live_slurm_end_time"
    fi
  fi
fi
STOP_CLAIMING_AT_EPOCH="$((effective_end - DRAIN_SECONDS))"
if [ "$STOP_CLAIMING_AT_EPOCH" -lt "$worker_now" ]; then
  STOP_CLAIMING_AT_EPOCH="$worker_now"
fi
echo "[drain] source=$drain_deadline_source effective_end=$effective_end stop_claiming=$STOP_CLAIMING_AT_EPOCH drain_seconds=$DRAIN_SECONDS"
worker_args=(
  --stage "$STAGE"
  --worker-index "$WORKER_INDEX"
  --stop-claiming-at-epoch "$STOP_CLAIMING_AT_EPOCH"
)
if [ -n "${MAX_UNITS:-}" ]; then
  worker_args+=(--max-units "$MAX_UNITS")
fi

set +e
"$PYTHON_BIN" -m sam3_mask_captioning.cli \
  --config "$CONFIG" \
  campaign-worker "$CAMPAIGN_ROOT" \
  "${worker_args[@]}"
worker_status=$?
set -e
if [ "$QWEN_WORKER" = "1" ]; then
  bcc_pack_runtime_cache
fi
exit "$worker_status"
