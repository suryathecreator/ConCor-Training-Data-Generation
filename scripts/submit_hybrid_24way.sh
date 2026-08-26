#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/qwen38_27b.yaml}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$REPO_ROOT/outputs/campaigns/gpic_20k_qwen38_27b}"
TARGET_TOTAL="${TARGET_TOTAL:-20000}"
UNIT_SIZE="${UNIT_SIZE:-8}"
SEED="${SEED:-20260826}"
MODE="${MODE:-production}"
START_STAGE="${START_STAGE:-image-review}"
CONTINUATIONS="${CONTINUATIONS:-}"
CACHE_ARCHIVES_READY="${CACHE_ARCHIVES_READY:-0}"
VLLM_PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
SAM3_PYTHON="${SAM3_PYTHON:-$REPO_ROOT/.venv-sam3/bin/python}"
SAM3_REPO_ROOT="${SAM3_REPO_ROOT:-$REPO_ROOT/third_party/sam3}"
HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
SAM3_REVISION="${SAM3_REVISION:-local}"
SAM3_CHECKPOINT_SOURCE="${SAM3_CHECKPOINT_SOURCE:-$REPO_ROOT/models/sam3.pt}"
QWEN_REVISION="${QWEN_REVISION:-local}"
QWEN_SOURCE="${QWEN_SOURCE:-$REPO_ROOT/models/Qwen3.8-27B}"
TOKEN_FILE="${TOKEN_FILE:-}"
EXCLUDE_CSV="${EXCLUDE_CSV:-}"
SOURCE="${SOURCE:-gpic}"
MANIFEST="${MANIFEST:-}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
DATASET_NAME="${DATASET_NAME:-$([ "$SOURCE" = "gpic" ] && printf '%s' 'stanford-vision-lab/gpic' || printf '%s' 'local-images')}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"
PYTHON_RUNTIME_KEY="${BCC_PYTHON_RUNTIME_KEY:-py311-torch211-vllm026-import-heavy-smallfiles-v2}"
PYTHON_RUNTIME_ARCHIVE="${BCC_PYTHON_RUNTIME_ARCHIVE:-$REPO_ROOT/.runtime/qwen38_python_runtime/$PYTHON_RUNTIME_KEY.tar.zst}"
PYTHON_RUNTIME_METADATA="${PYTHON_RUNTIME_ARCHIVE%.tar.zst}.metadata"
WORDNET_DIR="${WN_DATA_DIR:-$REPO_ROOT/.runtime/wordnet}"

CPU_PARTITION="${CPU_PARTITION:-cpu}"
CPU_ACCOUNT="${CPU_ACCOUNT:-}"
A40_PARTITION="${A40_PARTITION:-gpu-a40}"
H200_PARTITION="${H200_PARTITION:-gpu-h200}"
GPU_ACCOUNT="${GPU_ACCOUNT:-}"
A40_TP="${A40_TP:-2}"
H200_TP="${H200_TP:-1}"
A40_GRES_QWEN="${A40_GRES_QWEN:-gpu:a40:2}"
A40_GRES_SAM3="${A40_GRES_SAM3:-gpu:a40:1}"
H200_GRES_QWEN="${H200_GRES_QWEN:-gpu:h200:1}"
H200_GRES_SAM3="${H200_GRES_SAM3:-gpu:h200:1}"
CPU_ACCOUNT_ARGS=()
GPU_ACCOUNT_ARGS=()
[ -n "$CPU_ACCOUNT" ] && CPU_ACCOUNT_ARGS=(--account="$CPU_ACCOUNT")
[ -n "$GPU_ACCOUNT" ] && GPU_ACCOUNT_ARGS=(--account="$GPU_ACCOUNT")

if [ "$MODE" = "canary" ]; then
  A40_WORKERS="${A40_WORKERS:-1}"
  H200_WORKERS="${H200_WORKERS:-1}"
  CONTINUATIONS="${CONTINUATIONS:-2}"
else
  A40_WORKERS="${A40_WORKERS:-16}"
  H200_WORKERS="${H200_WORKERS:-8}"
  CONTINUATIONS="${CONTINUATIONS:-4}"
fi

if [ "$A40_WORKERS" -lt 1 ] && [ "$H200_WORKERS" -lt 1 ]; then
  echo "At least one GPU worker is required" >&2
  exit 2
fi
case "$START_STAGE" in
  image-review|sam3|mask-caption-qa|consistency|bcc) ;;
  *)
    echo "START_STAGE must be image-review, sam3, mask-caption-qa, consistency, or bcc" >&2
    exit 2
    ;;
esac
if [ ! -s "$QWEN_SOURCE/model.safetensors.index.json" ]; then
  echo "Missing Qwen3.8-27B checkpoint at $QWEN_SOURCE" >&2
  exit 2
fi
if [ ! -s "$SAM3_CHECKPOINT_SOURCE" ]; then
  echo "Missing pinned SAM3 checkpoint at $SAM3_CHECKPOINT_SOURCE" >&2
  exit 2
fi
if [ ! -d "$WORDNET_DIR/oewn-2025" ] && \
   ! WN_DATA_DIR="$WORDNET_DIR" "$VLLM_PYTHON" -c 'import wn; wn.Wordnet("oewn:2025", expand="")' >/dev/null 2>&1; then
  echo "Pinned oewn:2025 is missing; run scripts/install_wordnet.sh" >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p slurm/logs "$CAMPAIGN_ROOT/submissions" "$CAMPAIGN_ROOT/runtime_cache_archives"

TOTAL_UNITS="$(( (TARGET_TOTAL + UNIT_SIZE - 1) / UNIT_SIZE ))"
WEIGHT_DENOMINATOR="$(( A40_WORKERS + 2 * H200_WORKERS ))"
BASE_MAX_UNITS="$(( (TOTAL_UNITS + WEIGHT_DENOMINATOR - 1) / WEIGHT_DENOMINATOR ))"
if [ "$MODE" = "canary" ]; then
  # A canary is too small to strand work behind a queued GPU cohort and repay
  # another model startup. Atomic claims still prevent duplicate units.
  A40_MAX_UNITS="${A40_MAX_UNITS:-$TOTAL_UNITS}"
  H200_MAX_UNITS="${H200_MAX_UNITS:-$TOTAL_UNITS}"
else
  A40_MAX_UNITS="${A40_MAX_UNITS:-$BASE_MAX_UNITS}"
  H200_MAX_UNITS="${H200_MAX_UNITS:-$((2 * BASE_MAX_UNITS))}"
fi

common_export="REPO_ROOT=$REPO_ROOT,CONFIG=$CONFIG,CAMPAIGN_ROOT=$CAMPAIGN_ROOT,VLLM_PYTHON=$VLLM_PYTHON,SAM3_PYTHON=$SAM3_PYTHON,SAM3_REPO_ROOT=$SAM3_REPO_ROOT,SAM3_REVISION=$SAM3_REVISION,SAM3_CHECKPOINT_SOURCE=$SAM3_CHECKPOINT_SOURCE,HF_HOME=$HF_HOME,QWEN_SOURCE=$QWEN_SOURCE,QWEN_REVISION=$QWEN_REVISION,TOKEN_FILE=$TOKEN_FILE,EXCLUDE_CSV=$EXCLUDE_CSV,SOURCE=$SOURCE,MANIFEST=$MANIFEST,IMAGE_ROOT=$IMAGE_ROOT,DATASET_NAME=$DATASET_NAME,SOURCE_SPLIT=$SOURCE_SPLIT,WN_DATA_DIR=$WORDNET_DIR,BCC_PYTHON_RUNTIME_KEY=$PYTHON_RUNTIME_KEY,BCC_PYTHON_RUNTIME_ARCHIVE=$PYTHON_RUNTIME_ARCHIVE,WORKER_WALL_SECONDS=86400,DRAIN_SECONDS=600"

parsed_job_id() {
  local submission="$1" label="$2" job_id="${1%%;*}"
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "Invalid or empty Slurm job ID for $label: $submission" >&2
    return 2
  fi
  echo "$job_id"
}

materialize_job="${MATERIALIZE_JOB_ID:-}"
if [ -z "$materialize_job" ]; then
  if ! materialize_submission="$(sbatch --parsable \
    --partition="$CPU_PARTITION" "${CPU_ACCOUNT_ARGS[@]}" \
    --export="ALL,$common_export,TARGET_TOTAL=$TARGET_TOTAL,UNIT_SIZE=$UNIT_SIZE,SEED=$SEED" \
    "$REPO_ROOT/slurm/gpic_qwen38_materialize.slurm")"; then
    echo "Failed to submit campaign materializer" >&2
    exit 1
  fi
  materialize_job="$(parsed_job_id "$materialize_submission" materializer)"
fi

# Build the immutable small-file runtime once. Prewarms wait for this job, so
# every later Qwen allocation sees the same node-local Python overlay without
# requiring a separate operator command.
runtime_pack_job="${PYTHON_RUNTIME_PACK_JOB_ID:-}"
if [ -z "$runtime_pack_job" ] && \
   { [ ! -s "$PYTHON_RUNTIME_ARCHIVE" ] || [ ! -s "$PYTHON_RUNTIME_METADATA" ]; }; then
  if ! runtime_pack_submission="$(sbatch --parsable \
    --partition="$CPU_PARTITION" "${CPU_ACCOUNT_ARGS[@]}" \
    --export="ALL,REPO_ROOT=$REPO_ROOT,VLLM_PYTHON=$VLLM_PYTHON,BCC_PYTHON_RUNTIME_KEY=$PYTHON_RUNTIME_KEY" \
    "$REPO_ROOT/slurm/gpic_qwen_python_runtime_pack.slurm")"; then
    echo "Failed to submit node-local Python runtime pack" >&2
    exit 1
  fi
  runtime_pack_job="$(parsed_job_id "$runtime_pack_submission" "Python runtime pack")"
fi

submit_prewarm() {
  local cohort="$1" partition="$2" gres="$3" tp="$4" profile="$5" section="$6" max_len="$7" max_seqs="$8"
  local result
  local dependencies="afterok:$materialize_job"
  if [ -n "$runtime_pack_job" ]; then
    dependencies="$dependencies:$runtime_pack_job"
  fi
  if ! result="$(sbatch --parsable \
    --job-name="bcc-q38-prewarm-$cohort-$profile" \
    --partition="$partition" "${GPU_ACCOUNT_ARGS[@]}" --gres="$gres" \
    --dependency="$dependencies" \
    --export="ALL,$common_export,GPU_COHORT=$cohort,BCC_CACHE_PROFILE=$profile,PREWARM_SECTION=$section,BCC_VLLM_TENSOR_PARALLEL_SIZE=$tp,BCC_VLLM_MAX_MODEL_LEN=$max_len,BCC_VLLM_MAX_NUM_SEQS=$max_seqs" \
    "$REPO_ROOT/slurm/gpic_qwen38_cache_prewarm.slurm")"; then
    echo "Failed to submit $cohort/$profile prewarm" >&2
    return 1
  fi
  parsed_job_id "$result" "$cohort/$profile prewarm"
}

need_review=0
need_mask=0
case "$START_STAGE" in
  image-review) need_review=1; need_mask=1 ;;
  sam3|mask-caption-qa) need_mask=1 ;;
esac

a40_review_prewarm=""
a40_mask_prewarm=""
a40_bcc_prewarm=""
h200_review_prewarm=""
h200_mask_prewarm=""
h200_bcc_prewarm=""
if [ "$A40_WORKERS" -gt 0 ]; then
  a40_legacy="${A40_PREWARM_JOB_ID:-}"
  if [ "$need_review" = "1" ]; then
    a40_review_prewarm="${A40_REVIEW_PREWARM_JOB_ID:-$a40_legacy}"
    if [ -z "$a40_review_prewarm" ]; then
      a40_review_prewarm="$(submit_prewarm a40 "$A40_PARTITION" "$A40_GRES_QWEN" "$A40_TP" review image_review 16384 32)"
    fi
  fi
  if [ "$need_mask" = "1" ]; then
    a40_mask_prewarm="${A40_MASK_PREWARM_JOB_ID:-$a40_legacy}"
    if [ -z "$a40_mask_prewarm" ]; then
      a40_mask_prewarm="$(submit_prewarm a40 "$A40_PARTITION" "$A40_GRES_QWEN" "$A40_TP" mask caption 16384 32)"
    fi
  fi
  a40_bcc_prewarm="${A40_BCC_PREWARM_JOB_ID:-$a40_legacy}"
  if [ -z "$a40_bcc_prewarm" ]; then
    a40_bcc_prewarm="$(submit_prewarm a40 "$A40_PARTITION" "$A40_GRES_QWEN" "$A40_TP" bcc image_caption 49152 2)"
  fi
fi
if [ "$H200_WORKERS" -gt 0 ]; then
  h200_legacy="${H200_PREWARM_JOB_ID:-}"
  if [ "$need_review" = "1" ]; then
    h200_review_prewarm="${H200_REVIEW_PREWARM_JOB_ID:-$h200_legacy}"
    if [ -z "$h200_review_prewarm" ]; then
      h200_review_prewarm="$(submit_prewarm h200 "$H200_PARTITION" "$H200_GRES_QWEN" "$H200_TP" review image_review 16384 32)"
    fi
  fi
  if [ "$need_mask" = "1" ]; then
    h200_mask_prewarm="${H200_MASK_PREWARM_JOB_ID:-$h200_legacy}"
    if [ -z "$h200_mask_prewarm" ]; then
      h200_mask_prewarm="$(submit_prewarm h200 "$H200_PARTITION" "$H200_GRES_QWEN" "$H200_TP" mask caption 16384 32)"
    fi
  fi
  h200_bcc_prewarm="${H200_BCC_PREWARM_JOB_ID:-$h200_legacy}"
  if [ -z "$h200_bcc_prewarm" ]; then
    h200_bcc_prewarm="$(submit_prewarm h200 "$H200_PARTITION" "$H200_GRES_QWEN" "$H200_TP" bcc image_caption 49152 2)"
  fi
fi

submit_worker_chain() {
  local stage="$1" cohort="$2" workers="$3" partition="$4" gres="$5" tp="$6" max_units="$7" prerequisite="$8" prewarm="$9" cache_profile="${10}"
  if [ "$workers" -lt 1 ]; then
    return
  fi
  local qwen=0
  case "$stage" in image-review|mask-caption-qa|bcc) qwen=1 ;; esac
  local first_dependency="afterok:$prerequisite"
  if [ "$qwen" = "1" ] && [ -n "$prewarm" ] && [ "$CACHE_ARCHIVES_READY" != "1" ]; then
    first_dependency="$first_dependency:$prewarm"
  fi
  local max_model_len=16384 max_num_seqs=32 memory=0.82
  if [ "$stage" = "bcc" ]; then
    max_model_len=49152
    max_num_seqs=2
    memory=0.80
  fi
  local prior="" all_ids=""
  for generation in $(seq 0 "$CONTINUATIONS"); do
    local dependency="$first_dependency"
    if [ "$generation" -gt 0 ]; then
      dependency="afterany:$prior"
    fi
    local result
    if ! result="$(sbatch --parsable \
      --job-name="bcc-q38-$stage-$cohort-r$generation" \
      --partition="$partition" "${GPU_ACCOUNT_ARGS[@]}" --gres="$gres" \
      --array="0-$((workers - 1))%$workers" --dependency="$dependency" \
      --export="ALL,$common_export,STAGE=$stage,GPU_COHORT=$cohort,MAX_UNITS=$max_units,BCC_CACHE_PROFILE=$cache_profile,BCC_VLLM_TENSOR_PARALLEL_SIZE=$tp,BCC_VLLM_MAX_MODEL_LEN=$max_model_len,BCC_VLLM_MAX_NUM_SEQS=$max_num_seqs,BCC_VLLM_GPU_MEMORY_UTILIZATION=$memory" \
      "$REPO_ROOT/slurm/gpic_campaign_worker.slurm")"; then
      echo "Failed to submit $stage/$cohort continuation $generation" >&2
      return 1
    fi
    prior="$(parsed_job_id "$result" "$stage/$cohort continuation $generation")"
    all_ids="${all_ids:+$all_ids:}$prior"
  done
  echo "$all_ids"
}

submit_merge() {
  local stage="$1" prerequisite="$2" worker_jobs="$3"
  local result
  if ! result="$(sbatch --parsable \
    --job-name="bcc-q38-$stage-merge" \
    --partition="$CPU_PARTITION" "${CPU_ACCOUNT_ARGS[@]}" \
    --dependency="afterok:$prerequisite" \
    --export="ALL,$common_export,STAGE=$stage,WORKER_JOB_IDS=$worker_jobs" \
    "$REPO_ROOT/slurm/gpic_campaign_merge.slurm")"; then
    echo "Failed to submit $stage merge" >&2
    return 1
  fi
  parsed_job_id "$result" "$stage merge"
}

previous="$materialize_job"
submission_lines="materialize=$materialize_job\npython_runtime_pack=$runtime_pack_job\na40_review_prewarm=$a40_review_prewarm\na40_mask_prewarm=$a40_mask_prewarm\na40_bcc_prewarm=$a40_bcc_prewarm\nh200_review_prewarm=$h200_review_prewarm\nh200_mask_prewarm=$h200_mask_prewarm\nh200_bcc_prewarm=$h200_bcc_prewarm\n"
stage_enabled=0
for stage in image-review sam3 mask-caption-qa consistency bcc; do
  if [ "$stage" = "$START_STAGE" ]; then
    stage_enabled=1
  fi
  if [ "$stage_enabled" = "0" ]; then
    continue
  fi
  a40_gres="$A40_GRES_SAM3"
  h200_gres="$H200_GRES_SAM3"
  if [ "$stage" = "image-review" ] || [ "$stage" = "mask-caption-qa" ] || [ "$stage" = "bcc" ]; then
    a40_gres="$A40_GRES_QWEN"
    h200_gres="$H200_GRES_QWEN"
  fi
  cache_profile="none"
  a40_stage_prewarm=""
  h200_stage_prewarm=""
  case "$stage" in
    image-review)
      cache_profile="review"
      a40_stage_prewarm="$a40_review_prewarm"
      h200_stage_prewarm="$h200_review_prewarm"
      ;;
    mask-caption-qa)
      cache_profile="mask"
      a40_stage_prewarm="$a40_mask_prewarm"
      h200_stage_prewarm="$h200_mask_prewarm"
      ;;
    bcc)
      cache_profile="bcc"
      a40_stage_prewarm="$a40_bcc_prewarm"
      h200_stage_prewarm="$h200_bcc_prewarm"
      ;;
  esac
  a40_jobs="$(submit_worker_chain "$stage" a40 "$A40_WORKERS" "$A40_PARTITION" "$a40_gres" "$A40_TP" "$A40_MAX_UNITS" "$previous" "$a40_stage_prewarm" "$cache_profile")"
  h200_jobs="$(submit_worker_chain "$stage" h200 "$H200_WORKERS" "$H200_PARTITION" "$h200_gres" "$H200_TP" "$H200_MAX_UNITS" "$previous" "$h200_stage_prewarm" "$cache_profile")"
  worker_jobs="${a40_jobs}${a40_jobs:+${h200_jobs:+:}}${h200_jobs}"
  merged="$(submit_merge "$stage" "$previous" "$worker_jobs")"
  submission_lines+="$stage=$worker_jobs merge=$merged\n"
  echo "stage=$stage arrays=$worker_jobs merge=$merged"
  previous="$merged"
done

publisher_job="${PUBLISHER_JOB_ID:-}"
if [ -z "$publisher_job" ]; then
  if ! publisher_submission="$(sbatch --parsable \
    --partition="$CPU_PARTITION" "${CPU_ACCOUNT_ARGS[@]}" \
    --dependency="afterok:$materialize_job" \
    --export="ALL,$common_export" \
    "$REPO_ROOT/slurm/gpic_campaign_publish.slurm")"; then
    echo "Failed to submit campaign publisher" >&2
    exit 1
  fi
  publisher_job="$(parsed_job_id "$publisher_submission" publisher)"
fi
site_job="${SITE_JOB_ID:-}"
if [ -z "$site_job" ]; then
  if ! site_submission="$(sbatch --parsable \
    --partition="$CPU_PARTITION" "${CPU_ACCOUNT_ARGS[@]}" \
    --dependency="afterok:$materialize_job" \
    --export="ALL,$common_export,SITE_PORT=${SITE_PORT:-8765}" \
    "$REPO_ROOT/slurm/gpic_campaign_site.slurm")"; then
    echo "Failed to submit campaign site" >&2
    exit 1
  fi
  site_job="$(parsed_job_id "$site_submission" site)"
fi
export_job="${EXPORT_JOB_ID:-}"
if [ -z "$export_job" ]; then
  if ! export_submission="$(sbatch --parsable \
    --partition="$CPU_PARTITION" "${CPU_ACCOUNT_ARGS[@]}" \
    --dependency="afterok:$previous" \
    --export="ALL,$common_export,PUBLISH_HF=${PUBLISH_HF:-0}" \
    "$REPO_ROOT/slurm/gpic_campaign_export_hf.slurm")"; then
    echo "Failed to submit Hugging Face export" >&2
    exit 1
  fi
  export_job="$(parsed_job_id "$export_submission" export)"
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
ledger="$CAMPAIGN_ROOT/submissions/$timestamp.txt"
printf '%bpublisher=%s\nsite=%s\nexport=%s\ntarget_total=%s\nunit_size=%s\na40_max_units=%s\nh200_max_units=%s\nmode=%s\nstart_stage=%s\n' \
  "$submission_lines" "$publisher_job" "$site_job" "$export_job" "$TARGET_TOTAL" "$UNIT_SIZE" "$A40_MAX_UNITS" "$H200_MAX_UNITS" "$MODE" "$START_STAGE" > "$ledger"
echo "materialize_job=$materialize_job publisher_job=$publisher_job site_job=$site_job export_job=$export_job"
echo "submission_ledger=$ledger"
