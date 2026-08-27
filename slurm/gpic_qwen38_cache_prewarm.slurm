#!/usr/bin/env bash
#SBATCH --job-name=bcc-q38-prewarm
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1440
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
: "${REPO_ROOT:?REPO_ROOT is required}"
: "${CONFIG:?CONFIG is required}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${GPU_COHORT:?GPU_COHORT is required}"
VLLM_PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
PREWARM_PYTHON="$VLLM_PYTHON"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
PREWARM_SECTION="${PREWARM_SECTION:-all}"
case "$PREWARM_SECTION" in
  image_review)
    export BCC_CACHE_PROFILE="${BCC_CACHE_PROFILE:-review}"
    export BCC_CACHE_ENGINE_PROFILES="16384x32@32768"
    ;;
  caption)
    export BCC_CACHE_PROFILE="${BCC_CACHE_PROFILE:-mask}"
    export BCC_CACHE_ENGINE_PROFILES="16384x32@65536"
    ;;
  image_caption)
    export BCC_CACHE_PROFILE="${BCC_CACHE_PROFILE:-bcc}"
    export BCC_CACHE_ENGINE_PROFILES="49152x2@49152"
    ;;
  all)
    export BCC_CACHE_PROFILE="${BCC_CACHE_PROFILE:-all}"
    export BCC_CACHE_ENGINE_PROFILES="16384x32@32768,16384x32@65536,49152x2@49152"
    ;;
  *) echo "Unknown PREWARM_SECTION=$PREWARM_SECTION" >&2; exit 2 ;;
esac
source "$REPO_ROOT/scripts/stage_qwen_runtime.sh"
unset VLLM_PYTHON
WARMUP_IMAGE="$(find "$CAMPAIGN_ROOT/units" -path '*/source_images/*' -type f -print | sort | sed -n '1p')"
if [ -z "$WARMUP_IMAGE" ]; then
  # Completed upstream stages compact loose sources into per-unit tar files.
  # Recover one actual selected input into node-local scratch; never synthesize
  # a stand-in and never write this discarded warmup into campaign outputs.
  SOURCE_TAR="$(find "$CAMPAIGN_ROOT/units" -path '*/artifacts/source.tar' -type f -print | sort | sed -n '1p')"
  if [ -n "$SOURCE_TAR" ]; then
    mapfile -t SOURCE_MEMBERS < <(tar -tf "$SOURCE_TAR")
    WARMUP_MEMBER=""
    for candidate in "${SOURCE_MEMBERS[@]}"; do
      case "$candidate" in
        source_images/*) WARMUP_MEMBER="$candidate"; break ;;
      esac
    done
    if [ -n "$WARMUP_MEMBER" ]; then
      WARMUP_EXTRACT_ROOT="$JOB_CACHE_ROOT/real-image-warmup"
      mkdir -p "$WARMUP_EXTRACT_ROOT"
      tar -xf "$SOURCE_TAR" -C "$WARMUP_EXTRACT_ROOT" "$WARMUP_MEMBER"
      WARMUP_IMAGE="$WARMUP_EXTRACT_ROOT/$WARMUP_MEMBER"
    fi
  fi
fi
if [ -z "$WARMUP_IMAGE" ]; then
  echo "No materialized source image is available for visual prewarming" >&2
  exit 2
fi
# A normal submission runs one job for each engine profile so unrelated profiles
# compile concurrently and each stage waits only for its own archive. `all` is
# retained for compatibility with already-submitted jobs and older archives.
if [ "$PREWARM_SECTION" = "all" ] || [ "$PREWARM_SECTION" = "image_review" ]; then
  BCC_VLLM_MAX_MODEL_LEN=16384 BCC_VLLM_MAX_NUM_SEQS=32 \
    "$PREWARM_PYTHON" "$REPO_ROOT/scripts/prewarm_qwen_engine.py" \
    --config "$CONFIG" --section image_review \
    --warmup-image "$WARMUP_IMAGE" --warmup-image-count 1
fi
if [ "$PREWARM_SECTION" = "all" ] || [ "$PREWARM_SECTION" = "caption" ]; then
  BCC_VLLM_MAX_MODEL_LEN=16384 BCC_VLLM_MAX_NUM_SEQS=32 \
    "$PREWARM_PYTHON" "$REPO_ROOT/scripts/prewarm_qwen_engine.py" \
    --config "$CONFIG" --section caption \
    --warmup-image "$WARMUP_IMAGE" --warmup-image-count 1
fi
if [ "$PREWARM_SECTION" = "all" ] || [ "$PREWARM_SECTION" = "image_caption" ]; then
  BCC_VLLM_MAX_MODEL_LEN=49152 BCC_VLLM_MAX_NUM_SEQS=2 \
    "$PREWARM_PYTHON" "$REPO_ROOT/scripts/prewarm_qwen_engine.py" \
    --config "$CONFIG" --section image_caption \
    --warmup-image "$WARMUP_IMAGE" --warmup-image-count 4
fi
export BCC_REFRESH_COMPILE_CACHE=1
bcc_pack_runtime_cache
