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


MERGE_STATE_VERSION = "campaign-jsonl-merge-v2-resumable"


def _new_stream_state(filename: str) -> dict[str, Any]:
    return {
        "name": filename,
        "next_unit_id": 0,
        "shard_index": 0,
        "active_rows": 0,
        "active_bytes": 0,
        "total_rows": 0,
        "shards": [],
        "complete": False,
    }


def _valid_committed_shards(output: Path, stream: dict[str, Any]) -> bool:
    return all(
        (output / str(shard.get("file") or "")).is_file()
        and (output / str(shard.get("file") or "")).stat().st_size
        == int(shard.get("bytes") or -1)
        for shard in stream.get("shards") or []
    )


def _reset_stream(output: Path, stream: dict[str, Any]) -> dict[str, Any]:
    filename = str(stream.get("name") or "")
    stem = Path(filename).stem
    for shard in stream.get("shards") or []:
        path = output / str(shard.get("file") or "")
        if path.is_file() and path.parent == output:
            path.unlink()
    partial = output / f".{stem}-{int(stream.get('shard_index') or 0):05d}.jsonl.partial"
    if partial.is_file():
        partial.unlink()
    return _new_stream_state(filename)


def _finish_active_shard(
    output: Path,
    stream: dict[str, Any],
    handle: Any,
    partial: Path,
) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    handle.close()
    index = int(stream["shard_index"])
    final = output / f"{Path(str(stream['name'])).stem}-{index:05d}.jsonl"
    partial.replace(final)
    stream["shards"].append(
        {
            "file": final.name,
            "rows": int(stream["active_rows"]),
            "bytes": final.stat().st_size,
            "sha256": sha256_file(final),
        }
    )
    stream["shard_index"] = index + 1
    stream["active_rows"] = 0
    stream["active_bytes"] = 0


def merge_stage_jsonl(
    campaign_root: str | Path,
    stage: str,
    *,
    max_shard_bytes: int | None = None,
) -> dict[str, Any]:
    """Concatenate unit outputs with an atomic, per-unit restart cursor.

    A preemption can lose at most the unit currently being copied. Completed
    shards and the active partial shard are reused after a restart.
    """
    root = Path(campaign_root).expanduser().resolve()
    unit_count = int(load_registry(root).get("unit_count") or 0)
    limit = int(max_shard_bytes or os.environ.get("BCC_MERGED_SHARD_BYTES", 512 * 1024 * 1024))
    checkpoint_units = max(1, int(os.environ.get("BCC_MERGE_CHECKPOINT_UNITS", "32")))
    output = root / "merged" / stage
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    state_path = output / ".merge_state.json"
    signature = {
        "version": MERGE_STATE_VERSION,
        "stage": stage,
        "unit_count": unit_count,
        "max_shard_bytes": limit,
        "checkpoint_units": checkpoint_units,
        "stream_names": list(STAGE_OUTPUTS.get(stage, ())),
    }
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    if state.get("signature") != signature:
        state = {"signature": signature, "streams": {}, "complete": False}
        write_json(state, state_path)

    stream_states = state.setdefault("streams", {})
    for filename in STAGE_OUTPUTS.get(stage, ()):
        stream = stream_states.setdefault(filename, _new_stream_state(filename))
        if str(stream.get("name") or "") != filename or not _valid_committed_shards(output, stream):
            stream = _reset_stream(output, stream)
            stream_states[filename] = stream
            write_json(state, state_path)
        if stream.get("complete"):
            continue

        stem = Path(filename).stem
        partial = output / f".{stem}-{int(stream['shard_index']):05d}.jsonl.partial"
        expected_bytes = int(stream.get("active_bytes") or 0)
        if expected_bytes and not partial.exists():
            stream = _reset_stream(output, stream)
            stream_states[filename] = stream
            partial = output / f".{stem}-00000.jsonl.partial"
            expected_bytes = 0
            write_json(state, state_path)
        handle = partial.open("r+b" if partial.exists() else "w+b")
        handle.truncate(expected_bytes)
        handle.seek(expected_bytes)
        try:
            for unit_id in range(int(stream.get("next_unit_id") or 0), unit_count):
                source = root / "units" / f"{unit_id:06d}" / filename
                if source.exists():
                    with source.open("rb") as reader:
                        for line in reader:
                            if not line.strip():
                                continue
                            payload = line if line.endswith(b"\n") else line + b"\n"
                            handle.write(payload)
                            stream["active_rows"] = int(stream["active_rows"]) + 1
                            stream["active_bytes"] = int(stream["active_bytes"]) + len(payload)
                            stream["total_rows"] = int(stream["total_rows"]) + 1

                stream["next_unit_id"] = unit_id + 1
                checkpoint = (
                    (unit_id + 1) % checkpoint_units == 0
                    or unit_id + 1 == unit_count
                    or int(stream["active_bytes"]) >= limit
                )
                if checkpoint:
                    handle.flush()
                    os.fsync(handle.fileno())
                if checkpoint and int(stream["active_bytes"]) >= limit:
                    _finish_active_shard(output, stream, handle, partial)
                    write_json(state, state_path)
                    partial = output / f".{stem}-{int(stream['shard_index']):05d}.jsonl.partial"
                    handle = partial.open("w+b")
                elif checkpoint:
                    write_json(state, state_path)

            if int(stream["active_rows"]) or not stream["shards"]:
                _finish_active_shard(output, stream, handle, partial)
            else:
                handle.close()
                if partial.exists():
                    partial.unlink()
            stream["complete"] = True
            write_json(state, state_path)
        except BaseException:
            if not handle.closed:
                handle.close()
            raise

    streams: list[dict[str, Any]] = []
    for filename in STAGE_OUTPUTS.get(stage, ()):
        stream = stream_states[filename]
        shards = list(stream.get("shards") or [])
        if len(shards) == 1 and shards[0]["file"].endswith("-00000.jsonl"):
            old = output / shards[0]["file"]
            final = output / f"{Path(filename).stem}.jsonl"
            if old.exists():
                old.replace(final)
            shards[0]["file"] = final.name
            stream["shards"] = shards
            write_json(state, state_path)
        streams.append(
            {"name": filename, "rows": int(stream.get("total_rows") or 0), "shards": shards}
        )

    manifest = {
        "version": MERGE_STATE_VERSION,
        "stage": stage,
        "unit_count": unit_count,
        "max_shard_bytes": limit,
        "checkpoint_units": checkpoint_units,
        "streams": streams,
    }
    write_json(manifest, manifest_path)
    state["complete"] = True
    write_json(state, state_path)
    return manifest
