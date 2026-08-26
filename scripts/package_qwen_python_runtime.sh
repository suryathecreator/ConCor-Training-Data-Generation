#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VLLM_PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
RUNTIME_KEY="${BCC_PYTHON_RUNTIME_KEY:-py311-torch211-vllm026-import-heavy-smallfiles-v2}"
OUTPUT_DIR="${BCC_PYTHON_RUNTIME_ARCHIVE_DIR:-$REPO_ROOT/.runtime/qwen38_python_runtime}"
ARCHIVE="$OUTPUT_DIR/$RUNTIME_KEY.tar.zst"
METADATA="$OUTPUT_DIR/$RUNTIME_KEY.metadata"
LARGE_FILE_BYTES="${BCC_PYTHON_RUNTIME_LARGE_FILE_BYTES:-1048576}"

SITE_PACKAGES="$($VLLM_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PACKAGES=(
  torch torchgen functorch torchvision
  vllm triton transformers flashinfer xgrammar
  tokenizers safetensors compressed_tensors
)
for package in "${PACKAGES[@]}"; do
  if [ ! -e "$SITE_PACKAGES/$package" ]; then
    echo "Missing import-heavy runtime package: $SITE_PACKAGES/$package" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/$RUNTIME_KEY.lock"
flock 9
if [ -s "$ARCHIVE" ] && [ -s "$METADATA" ]; then
  zstd -t "$ARCHIVE" >/dev/null
  echo "$ARCHIVE"
  exit 0
fi

temporary="$(mktemp "$OUTPUT_DIR/.$RUNTIME_KEY.XXXXXX.tar.zst")"
staging_base="${SLURM_TMPDIR:-/tmp}/${USER:-user}-bcc-python-runtime-pack"
mkdir -p "$staging_base"
staging="$(mktemp -d "$staging_base/$RUNTIME_KEY.XXXXXX")"
cleanup() {
  if [ -d "$staging" ]; then
    rm -rf -- "$staging"
  fi
  if [ -e "$temporary" ]; then
    # Preserve an interrupted package for diagnosis rather than silently
    # treating it as a valid immutable runtime.
    mv "$temporary" "$temporary.interrupted" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Imports are dominated by tens of thousands of small Python/data files. The
# handful of large extension binaries are already read sequentially and remain
# pinned in the scrubbed environment. Copy small files into the immutable
# archive, but represent each large regular file as an absolute symlink to that
# pinned environment. This avoids duplicating gigabytes while ensuring package
# __path__ resolution stays entirely inside the complete local package tree.
for package in "${PACKAGES[@]}"; do
  rsync -a --max-size="$LARGE_FILE_BYTES" \
    "$SITE_PACKAGES/$package/" "$staging/$package/"
  while IFS= read -r -d '' source_file; do
    relative="${source_file#"$SITE_PACKAGES/"}"
    target="$staging/$relative"
    mkdir -p "$(dirname "$target")"
    ln -s "$source_file" "$target"
  done < <(find "$SITE_PACKAGES/$package" -type f \
    -size +"$((LARGE_FILE_BYTES - 1))"c -print0)
done

ZSTD_CLEVEL=1 ZSTD_NBTHREADS="${SLURM_CPUS_PER_TASK:-4}" \
  tar -I zstd -cf "$temporary" -C "$staging" .
zstd -t "$temporary" >/dev/null
mv "$temporary" "$ARCHIVE"
rm -rf -- "$staging"
trap - EXIT

"$VLLM_PYTHON" - "$METADATA" "$RUNTIME_KEY" "$SITE_PACKAGES" \
  "$LARGE_FILE_BYTES" <<'PY'
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

path, key, site_packages, large_file_bytes = sys.argv[1:]
payload = {
    "created_at": time.time(),
    "key": key,
    "python": platform.python_version(),
    "site_packages": site_packages,
    "large_file_bytes": int(large_file_bytes),
    "large_file_strategy": "absolute_symlink_to_pinned_scrubbed_environment",
    "versions": {
        name: importlib.metadata.version(name)
        for name in ("torch", "vllm", "triton", "transformers")
    },
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
echo "$ARCHIVE"
