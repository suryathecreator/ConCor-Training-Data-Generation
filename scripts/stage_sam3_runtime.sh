#!/usr/bin/env bash
# Source from a SAM3 Slurm worker after the stage-complete preflight.

set -euo pipefail

: "${SAM3_CHECKPOINT_SOURCE:?SAM3_CHECKPOINT_SOURCE is required}"
SAM3_REVISION="${SAM3_REVISION:-3c879f39826c281e95690f02c7821c4de09afae7}"
NODE_MODEL_BASE="${BCC_NODE_MODEL_BASE:-/tmp/${USER:-user}-bcc-models}"
NODE_SAM3_DIR="$NODE_MODEL_BASE/SAM3-$SAM3_REVISION"
NODE_SAM3_PARTIAL="$NODE_MODEL_BASE/.SAM3-$SAM3_REVISION.partial"
mkdir -p "$NODE_MODEL_BASE"

# Multiple tasks may share one physical node. The fixed partial path lets a
# same-node requeue resume an interrupted sequential copy instead of starting
# another 3.45-GB transfer; the final directory appears only after validation.
exec 6>"$NODE_MODEL_BASE/SAM3-$SAM3_REVISION.lock"
flock 6
source_size="$(stat -Lc '%s' "$SAM3_CHECKPOINT_SOURCE")"
if [ ! -s "$NODE_SAM3_DIR/.bcc-stage-ready" ] || \
   [ "$(stat -Lc '%s' "$NODE_SAM3_DIR/sam3.pt" 2>/dev/null || echo 0)" != "$source_size" ]; then
  if [ -e "$NODE_SAM3_DIR" ]; then
    mv "$NODE_SAM3_DIR" \
      "$NODE_MODEL_BASE/incomplete-SAM3-$SAM3_REVISION-${SLURM_JOB_ID:-local}-$$"
  fi
  mkdir -p "$NODE_SAM3_PARTIAL"
  rsync -aL --partial --inplace \
    "$SAM3_CHECKPOINT_SOURCE" "$NODE_SAM3_PARTIAL/sam3.pt"
  test "$(stat -Lc '%s' "$NODE_SAM3_PARTIAL/sam3.pt")" = "$source_size"
  printf 'revision=%s\nsource=%s\nsize=%s\n' \
    "$SAM3_REVISION" "$SAM3_CHECKPOINT_SOURCE" "$source_size" \
    > "$NODE_SAM3_PARTIAL/.bcc-stage-ready"
  mv "$NODE_SAM3_PARTIAL" "$NODE_SAM3_DIR"
fi
flock -u 6

export BCC_SAM3_CHECKPOINT_PATH="$NODE_SAM3_DIR/sam3.pt"
echo "[sam3-runtime] node_local=$BCC_SAM3_CHECKPOINT_PATH bytes=$source_size"
