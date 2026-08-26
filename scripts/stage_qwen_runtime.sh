#!/usr/bin/env bash
# Source this file from a Slurm worker. It stages the immutable Qwen checkpoint
# once per compute node and restores an architecture-specific compiled cache.

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${GPU_COHORT:?GPU_COHORT is required}"
: "${QWEN_SOURCE:?QWEN_SOURCE is required}"

QWEN_REVISION="${QWEN_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
TENSOR_PARALLEL_SIZE="${BCC_VLLM_TENSOR_PARALLEL_SIZE:-1}"
CACHE_PROFILE="${BCC_CACHE_PROFILE:-}"
if [ -z "$CACHE_PROFILE" ]; then
  # Jobs submitted before the profile split do not carry BCC_CACHE_PROFILE.
  # Infer it from their durable stage name so a review-only `-all` archive can
  # never shadow the mask/BCC graphs in the older combined archive.
  case "${STAGE:-}" in
    image-review) CACHE_PROFILE="review" ;;
    mask-caption|mask-qa|mask-caption-qa) CACHE_PROFILE="mask" ;;
    bcc-draft|bcc-rewrite|bcc) CACHE_PROFILE="bcc" ;;
    *) CACHE_PROFILE="all" ;;
  esac
fi
NODE_MODEL_BASE="${BCC_NODE_MODEL_BASE:-/tmp/${USER:-user}-bcc-models}"
NODE_MODEL_DIR="$NODE_MODEL_BASE/Qwen3.8-27B-$QWEN_REVISION"
mkdir -p "$NODE_MODEL_BASE"

# Prefer an explicitly configured toolkit, then a toolkit bundled with the
# Python environment, then the node's nvcc. This keeps cache keys portable
# across clusters while preserving reproducible architecture-specific caches.
QWEN_PYTHON_BIN="${PYTHON_BIN:-${VLLM_PYTHON:-}}"
if [ -z "$QWEN_PYTHON_BIN" ] || [ ! -x "$QWEN_PYTHON_BIN" ]; then
  echo "A vLLM Python executable is required to locate the CUDA toolkit" >&2
  return 2 2>/dev/null || exit 2
fi
ORIGINAL_SITE_PACKAGES="$($QWEN_PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CUDA_TOOLKIT_ROOT="${BCC_CUDA_HOME:-${CUDA_HOME:-}}"
if [ -z "$CUDA_TOOLKIT_ROOT" ]; then
  for candidate in "$ORIGINAL_SITE_PACKAGES/nvidia/cu13" "$ORIGINAL_SITE_PACKAGES/nvidia/cu12"; do
    if [ -x "$candidate/bin/nvcc" ]; then
      CUDA_TOOLKIT_ROOT="$candidate"
      break
    fi
  done
fi
if [ -z "$CUDA_TOOLKIT_ROOT" ] && command -v nvcc >/dev/null 2>&1; then
  CUDA_TOOLKIT_ROOT="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi
if [ -z "$CUDA_TOOLKIT_ROOT" ] || [ ! -x "$CUDA_TOOLKIT_ROOT/bin/nvcc" ]; then
  echo "No executable nvcc found; set BCC_CUDA_HOME to the matching CUDA toolkit" >&2
  return 2 2>/dev/null || exit 2
fi
export CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export CUDA_PATH="$CUDA_TOOLKIT_ROOT"
export PATH="$CUDA_TOOLKIT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_ROOT/lib:$CUDA_TOOLKIT_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
NVCC_RELEASE="$($CUDA_TOOLKIT_ROOT/bin/nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p' | head -n 1)"
if [ -z "$NVCC_RELEASE" ]; then
  echo "Unable to determine CUDA compiler release" >&2
  return 2 2>/dev/null || exit 2
fi

# Spawned tensor-parallel ranks otherwise perform thousands of small random
# reads from the shared Python environment. A pinned archive turns those GPFS
# reads into one sequential transfer per node, then all imports come from local
# XFS. The full package directories include their extension modules, avoiding
# the broken-package behavior of a Python-source-only overlay.
PYTHON_RUNTIME_KEY="${BCC_PYTHON_RUNTIME_KEY:-py311-torch211-vllm026-import-heavy-smallfiles-v2}"
PYTHON_RUNTIME_ARCHIVE="${BCC_PYTHON_RUNTIME_ARCHIVE:-$REPO_ROOT/.runtime/qwen38_python_runtime/$PYTHON_RUNTIME_KEY.tar.zst}"
NODE_PYTHON_BASE="${BCC_NODE_PYTHON_BASE:-/tmp/${USER:-user}-bcc-python-runtime}"
NODE_PYTHON_DIR="$NODE_PYTHON_BASE/$PYTHON_RUNTIME_KEY"
if [ -s "$PYTHON_RUNTIME_ARCHIVE" ]; then
  mkdir -p "$NODE_PYTHON_BASE"
  exec 7>"$NODE_PYTHON_BASE/$PYTHON_RUNTIME_KEY.lock"
  flock 7
  if [ ! -s "$NODE_PYTHON_DIR/.bcc-runtime-ready" ]; then
    python_temp="$(mktemp -d "$NODE_PYTHON_BASE/.$PYTHON_RUNTIME_KEY.XXXXXX")"
    zstd -t "$PYTHON_RUNTIME_ARCHIVE" >/dev/null
    tar -I zstd -xf "$PYTHON_RUNTIME_ARCHIVE" -C "$python_temp"
    test -s "$python_temp/torch/__init__.py"
    test -s "$python_temp/vllm/__init__.py"
    test -s "$python_temp/triton/__init__.py"
    printf '%s\n' "$PYTHON_RUNTIME_KEY" > "$python_temp/.bcc-runtime-ready"
    if [ -e "$NODE_PYTHON_DIR" ]; then
      mv "$NODE_PYTHON_DIR" "$NODE_PYTHON_BASE/incomplete-$PYTHON_RUNTIME_KEY-${SLURM_JOB_ID:-local}-$$"
    fi
    mv "$python_temp" "$NODE_PYTHON_DIR"
  fi
  flock -u 7
  export PYTHONPATH="$NODE_PYTHON_DIR${PYTHONPATH:+:$PYTHONPATH}"
  echo "[python-runtime] node_local=$NODE_PYTHON_DIR archive=$PYTHON_RUNTIME_ARCHIVE"
else
  echo "[python-runtime] archive unavailable; using shared environment: $PYTHON_RUNTIME_ARCHIVE" >&2
fi

# flock makes simultaneous array tasks on one node share one 55.6-GB copy.
exec 8>"$NODE_MODEL_BASE/Qwen3.8-27B-$QWEN_REVISION.lock"
flock 8
if [ ! -s "$NODE_MODEL_DIR/.bcc-stage-ready" ]; then
  model_temp="$(mktemp -d "$NODE_MODEL_BASE/.Qwen3.8-27B.XXXXXX")"
  rsync -aL --delete "$QWEN_SOURCE/" "$model_temp/"
  test -s "$model_temp/model.safetensors.index.json"
  printf '%s\n' "$QWEN_REVISION" > "$model_temp/.bcc-stage-ready"
  if [ -e "$NODE_MODEL_DIR" ]; then
    # Preserve an interrupted copy for diagnosis; never delete checkpoint data.
    mv "$NODE_MODEL_DIR" "$NODE_MODEL_BASE/incomplete-$QWEN_REVISION-${SLURM_JOB_ID:-local}-$$"
  fi
  mv "$model_temp" "$NODE_MODEL_DIR"
fi
flock -u 8
export BCC_QWEN_MODEL_PATH="$NODE_MODEL_DIR"

JOB_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/${USER:-user}-bcc-runtime-cache"
PERSISTENT_CACHE_DIR="${BCC_PERSISTENT_CACHE_DIR:-$REPO_ROOT/.runtime/qwen38_compile_cache_archives}"
CAMPAIGN_CACHE_DIR="$CAMPAIGN_ROOT/runtime_cache_archives"
BASE_CACHE_KEY="qwen38-27b-${GPU_COHORT}-tp${TENSOR_PARALLEL_SIZE}-vllm026-nvcc${NVCC_RELEASE}"
CACHE_KEY="$BASE_CACHE_KEY-$CACHE_PROFILE"
CACHE_ARCHIVE="$PERSISTENT_CACHE_DIR/$CACHE_KEY.tar.zst"
LEGACY_CACHE_ARCHIVE="$PERSISTENT_CACHE_DIR/$BASE_CACHE_KEY.tar.zst"
CAMPAIGN_CACHE_ARCHIVE="$CAMPAIGN_CACHE_DIR/$CACHE_KEY.tar.zst"
CAMPAIGN_LEGACY_CACHE_ARCHIVE="$CAMPAIGN_CACHE_DIR/$BASE_CACHE_KEY.tar.zst"
mkdir -p "$JOB_CACHE_ROOT" "$PERSISTENT_CACHE_DIR" "$CAMPAIGN_CACHE_DIR"

RESTORE_CACHE_ARCHIVE="$CACHE_ARCHIVE"
if [ ! -s "$RESTORE_CACHE_ARCHIVE" ]; then
  for cache_candidate in \
    "$CAMPAIGN_CACHE_ARCHIVE" \
    "$LEGACY_CACHE_ARCHIVE" \
    "$CAMPAIGN_LEGACY_CACHE_ARCHIVE"; do
    if [ -s "$cache_candidate" ]; then
      RESTORE_CACHE_ARCHIVE="$cache_candidate"
      break
    fi
  done
fi
if [ -s "$RESTORE_CACHE_ARCHIVE" ]; then
  zstd -t "$RESTORE_CACHE_ARCHIVE" >/dev/null
  # GNU tar 1.30 supports -I but not the newer --zstd convenience option.
  tar -I zstd -xf "$RESTORE_CACHE_ARCHIVE" -C "$JOB_CACHE_ROOT"
fi

export VLLM_CACHE_ROOT="$JOB_CACHE_ROOT/vllm"
export TRITON_CACHE_DIR="$JOB_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE_ROOT/torchinductor"
export CUDA_CACHE_PATH="$JOB_CACHE_ROOT/cuda"
export TORCH_EXTENSIONS_DIR="$JOB_CACHE_ROOT/torch-extensions"
export XDG_CACHE_HOME="$JOB_CACHE_ROOT/xdg"
export FLASHINFER_WORKSPACE_BASE="$JOB_CACHE_ROOT/flashinfer"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" \
  "$FLASHINFER_WORKSPACE_BASE"

bcc_pack_runtime_cache() {
  mkdir -p "$PERSISTENT_CACHE_DIR"
  exec 9>"$PERSISTENT_CACHE_DIR/$CACHE_KEY.lock"
  flock 9
  if [ -s "$CACHE_ARCHIVE" ] && [ "${BCC_REFRESH_COMPILE_CACHE:-0}" != "1" ]; then
    flock -u 9
    return
  fi
  cache_temp="$PERSISTENT_CACHE_DIR/.$CACHE_KEY.${SLURM_JOB_ID:-local}.$$.tar.zst"
  tar -I zstd -cf "$cache_temp" -C "$JOB_CACHE_ROOT" .
  mv "$cache_temp" "$CACHE_ARCHIVE"
  printf '%s\n' \
    "cohort=$GPU_COHORT" \
    "tensor_parallel_size=$TENSOR_PARALLEL_SIZE" \
    "qwen_revision=$QWEN_REVISION" \
    "vllm=0.26.0" \
    "nvcc_release=$NVCC_RELEASE" \
    "cache_profile=$CACHE_PROFILE" \
    "engine_profiles=${BCC_CACHE_ENGINE_PROFILES:-legacy-or-unspecified}" \
    "created_by_job=${SLURM_JOB_ID:-local}" \
    > "$PERSISTENT_CACHE_DIR/$CACHE_KEY.metadata"
  flock -u 9
}
