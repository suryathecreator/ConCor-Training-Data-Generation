from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .campaign_manifest import load_registry
from .io_utils import sha256_file, write_json


STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "image-review": (
        "image_reviews.jsonl",
        "initial_rejected_images.jsonl",
        "image_review_errors.jsonl",
        "image_review_raw.jsonl",
    ),
    "sam3": (
        "sam3_completed_images.jsonl",
        "sam3_failed_images.jsonl",
        "sam3_masks.jsonl",
        "sam3_rejected_masks.jsonl",
        "sam3_errors.jsonl",
        "sam3_prompt_batch_metrics.jsonl",
        "sam3_raw.jsonl",
    ),
    "mask-caption": ("caption_candidates.jsonl", "caption_errors.jsonl", "caption_raw.jsonl"),
    "mask-qa": (
        "mask_quality_reviews.jsonl",
        "captions.jsonl",
        "rejected_captions.jsonl",
        "mask_review_errors.jsonl",
        "mask_quality_raw.jsonl",
    ),
    "mask-caption-qa": (
        "caption_candidates.jsonl",
        "caption_rejected_masks.jsonl",
        "mask_quality_reviews.jsonl",
        "captions.jsonl",
        "rejected_captions.jsonl",
        "caption_errors.jsonl",
        "mask_review_errors.jsonl",
        "caption_raw.jsonl",
        "mask_quality_raw.jsonl",
    ),
    "consistency": (
        "consistent_captions.jsonl",
        "consistency_rejected_captions.jsonl",
        "sam3_consistency_reviews.jsonl",
        "sam3_consistency_errors.jsonl",
        "sam3_consistency_batch_metrics.jsonl",
    ),
    "bcc-draft": (
        "bcc_canonical_captions.jsonl",
        "bcc_duplicate_masks.jsonl",
        "image_caption_candidates.jsonl",
        "image_caption_raw.jsonl",
    ),
    "bcc-rewrite": (
        "image_text_pairs.jsonl",
        "bcc_validation_audit.jsonl",
        "bcc_exclusions.jsonl",
        "bcc_visual_audits.jsonl",
        "bcc_visual_audit_raw.jsonl",
        "image_caption_qa_raw.jsonl",
        "mask_rle.jsonl",
    ),
    "bcc": (
        "bcc_canonical_captions.jsonl",
        "bcc_duplicate_masks.jsonl",
        "image_caption_candidates.jsonl",
        "image_text_pairs.jsonl",
        "bcc_validation_audit.jsonl",
        "bcc_exclusions.jsonl",
        "bcc_visual_audits.jsonl",
        "image_caption_raw.jsonl",
        "image_caption_qa_raw.jsonl",
        "bcc_visual_audit_raw.jsonl",
        "mask_rle.jsonl",
    ),
}


def _remove_previous(manifest_path: Path) -> None:
    if not manifest_path.exists():
        return
    try:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return
    for stream in prior.get("streams") or []:
        for shard in stream.get("shards") or []:
            path = manifest_path.parent / str(shard.get("file") or "")
            if path.is_file() and path.parent == manifest_path.parent:
                path.unlink()


def merge_stage_jsonl(
    campaign_root: str | Path,
    stage: str,
    *,
    max_shard_bytes: int | None = None,
) -> dict[str, Any]:
    """Concatenate unit outputs, rolling to numbered shards only when needed."""
    root = Path(campaign_root).expanduser().resolve()
    unit_count = int(load_registry(root).get("unit_count") or 0)
    limit = int(max_shard_bytes or os.environ.get("BCC_MERGED_SHARD_BYTES", 512 * 1024 * 1024))
    output = root / "merged" / stage
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    _remove_previous(manifest_path)
    streams: list[dict[str, Any]] = []

    for filename in STAGE_OUTPUTS.get(stage, ()):
        stem = Path(filename).stem
        shards: list[dict[str, Any]] = []
        handle = None
        temporary: Path | None = None
        shard_index = 0
        shard_rows = 0
        shard_bytes = 0
        total_rows = 0

        def open_shard() -> None:
            nonlocal handle, temporary, shard_rows, shard_bytes
            final_name = f"{stem}.jsonl" if shard_index == 0 else f"{stem}-{shard_index:05d}.jsonl"
            temporary = output / f".{final_name}.tmp"
            handle = temporary.open("wb")
            shard_rows = 0
            shard_bytes = 0

        def close_shard() -> None:
            nonlocal handle, temporary
            if handle is None or temporary is None:
                return
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            if shard_index == 0 and not shards:
                final_name = f"{stem}.jsonl"
            else:
                final_name = f"{stem}-{shard_index:05d}.jsonl"
                # The first file must become numbered if a rollover occurs.
                if shard_index == 1 and shards and shards[0]["file"] == f"{stem}.jsonl":
                    old = output / shards[0]["file"]
                    numbered = output / f"{stem}-00000.jsonl"
                    old.replace(numbered)
                    shards[0]["file"] = numbered.name
            final = output / final_name
            temporary.replace(final)
            shards.append(
                {
                    "file": final.name,
                    "rows": shard_rows,
                    "bytes": final.stat().st_size,
                    "sha256": sha256_file(final),
                }
            )
            handle = None
            temporary = None

        open_shard()
        for unit_id in range(unit_count):
            source = root / "units" / f"{unit_id:06d}" / filename
            if not source.exists():
                continue
            with source.open("rb") as reader:
                for line in reader:
                    if not line.strip():
                        continue
                    if shard_rows and shard_bytes + len(line) > limit:
                        close_shard()
                        shard_index += 1
                        open_shard()
                    assert handle is not None
                    handle.write(line if line.endswith(b"\n") else line + b"\n")
                    shard_rows += 1
                    shard_bytes += len(line)
                    total_rows += 1
        close_shard()
        streams.append({"name": filename, "rows": total_rows, "shards": shards})

    manifest = {
        "stage": stage,
        "unit_count": unit_count,
        "max_shard_bytes": limit,
        "streams": streams,
    }
    write_json(manifest, manifest_path)
    return manifest
