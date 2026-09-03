from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .campaign_manifest import load_registry
from .io_utils import write_json


EXPORT_STATE_VERSION = "concor-hf-export-v3-resumable"
DATASET_EXPORT_VERSION = "concor-bcc-parquet-v3-resumable-concor1-compatible"
CONCOR_CAPTION_FORMAT_VERSION = "concor-1-caption-schema-v1"


def _unit_dir(root: Path, unit_id: int) -> Path:
    return root / "units" / f"{unit_id:06d}"


def _success_path(unit: Path, stage: str) -> Path:
    return unit / "stages" / stage / "_SUCCESS.json"


def _source_signature(
    root: Path,
    registry: dict[str, Any],
    *,
    include_image_bytes: bool,
    shard_size: int,
    checkpoint_units: int,
) -> str:
    """Fingerprint immutable unit inputs without rereading their large payloads."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "version": EXPORT_STATE_VERSION,
                "unit_count": int(registry.get("unit_count") or 0),
                "source_count": int(registry.get("source_count") or 0),
                "terminal_stage": str(registry.get("terminal_stage") or "bcc"),
                "include_image_bytes": include_image_bytes,
                "shard_size": shard_size,
                "checkpoint_units": checkpoint_units,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    terminal_stage = str(registry.get("terminal_stage") or "bcc")
    tracked = (
        "selected_images.jsonl",
        "image_reviews.jsonl",
        "sam3_masks.jsonl",
        "consistent_captions.jsonl",
        "mask_rle.jsonl",
        "bcc_validation_audit.jsonl",
        "bcc_exclusions.jsonl",
    )
    for unit_id in range(int(registry.get("unit_count") or 0)):
        unit = _unit_dir(root, unit_id)
        paths = [unit / name for name in tracked]
        paths.append(_success_path(unit, terminal_stage))
        for path in paths:
            try:
                stat = path.stat()
                marker = f"{unit_id}:{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n"
            except FileNotFoundError:
                marker = f"{unit_id}:{path.name}:missing\n"
            digest.update(marker.encode("utf-8"))
    return digest.hexdigest()


def _parquet_valid(path: Path, expected_rows: int, expected_bytes: int | None = None) -> bool:
    if not path.is_file():
        return False
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        return False
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows) == int(expected_rows)
    except Exception:
        return False


def _write_table_atomic(records: list[dict[str, Any]], path: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(records), temporary, compression="zstd")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path.stat().st_size


def _unit_records(
    root: Path,
    registry: dict[str, Any],
    unit_id: int,
    *,
    include_image_bytes: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Import at runtime so dataset_export can expose this implementation while
    # retaining its stable helper API without a module-import cycle.
    from .dataset_export import (
        _counts,
        _current,
        _parseable,
        _portable_public_json,
        _rows,
        _source_bytes,
        to_concor_caption_record,
    )

    unit = _unit_dir(root, unit_id)
    terminal_stage = str(registry.get("terminal_stage") or "bcc")
    terminal_complete = _success_path(unit, terminal_stage).exists()
    sources = _rows(unit / "selected_images.jsonl")
    reviews = _current(unit / "image_reviews.jsonl")
    sam3_counts = _counts(_rows(unit / "sam3_masks.jsonl"))
    consistent_rows = _rows(unit / "consistent_captions.jsonl")
    consistent_counts = _counts(consistent_rows)
    consistent_ids: dict[str, list[str]] = defaultdict(list)
    for row in consistent_rows:
        consistent_ids[str(row.get("image_id") or "")].append(str(row.get("mask_id") or ""))
    rles = {
        str(row.get("mask_id") or ""): row.get("rle")
        for row in _rows(unit / "mask_rle.jsonl")
    }
    audits = _current(unit / "bcc_validation_audit.jsonl")
    exclusions = _current(unit / "bcc_exclusions.jsonl")
    records: list[dict[str, Any]] = []
    standard: list[dict[str, Any]] = []
    for source in sources:
        image_id = str(source.get("image_id") or "")
        audit = audits.get(image_id)
        exclusion = exclusions.get(image_id)
        parseable = _parseable(audit)
        groups = list((audit or {}).get("groups") or []) if parseable else []
        linked_mask_ids = {
            str(mask_id)
            for group in groups
            for mask_id in (group.get("instance_ids") or [group.get("mask_id")])
            if mask_id
        }
        final_count = len(linked_mask_ids)
        review = reviews.get(image_id) or {}
        if parseable:
            reason = "parseable_min_10" if final_count >= 10 else "parseable_1_to_9"
        else:
            reason = str(
                (audit or {}).get("reason_code")
                or (audit or {}).get("exclusion_reason")
                or (exclusion or {}).get("reason_code")
                or ("pipeline_incomplete" if not terminal_complete else "did_not_reach_bcc")
            )
        payload: bytes | None = None
        filename = Path(str((source.get("source_context") or {}).get("source_member") or "")).name
        if include_image_bytes and parseable and final_count >= 1:
            payload, filename = _source_bytes(source)
        record = {
            "dataset_export_version": DATASET_EXPORT_VERSION,
            "image_id": image_id,
            "source_manifest_index": int(source.get("source_manifest_index") or 0),
            "source_dataset": str((source.get("source_context") or {}).get("source_dataset") or registry.get("dataset") or ""),
            "source_split": str((source.get("source_context") or {}).get("split") or registry.get("split") or ""),
            "source_pair_key": str((source.get("source_context") or {}).get("pair_key") or ""),
            "image_filename": filename,
            "image_mime_type": mimetypes.guess_type(filename)[0] or "image/jpeg",
            "image_bytes": payload,
            "image_review_accepted": review.get("accepted"),
            "sam3_proposal_count": int(sam3_counts.get(image_id, 0)),
            "post_consistency_mask_count": int(consistent_counts.get(image_id, 0)),
            "final_linked_mask_count": final_count,
            "bcc_parseable": parseable,
            "included_min_10": bool(parseable and final_count >= 10),
            "disposition": reason,
            "caption": str((audit or {}).get("caption") or "") if parseable else "",
            "correspondence_groups_json": json.dumps(_portable_public_json(groups), ensure_ascii=False, sort_keys=True),
            "accepted_mask_rles_json": json.dumps(
                {mask_id: rles[mask_id] for mask_id in sorted(linked_mask_ids) if mask_id in rles},
                ensure_ascii=False,
                sort_keys=True,
            ),
            "post_consistency_mask_rles_json": json.dumps(
                {mask_id: rles[mask_id] for mask_id in consistent_ids.get(image_id, []) if mask_id in rles},
                ensure_ascii=False,
                sort_keys=True,
            ),
            "bcc_record_json": json.dumps(
                _portable_public_json(audit or exclusion or {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        records.append(record)
        if parseable and final_count >= 1:
            standard.append(to_concor_caption_record(record))
    return records, standard


def _load_state(path: Path, signature: str) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    if state.get("signature") != signature:
        state = {
            "version": EXPORT_STATE_VERSION,
            "signature": signature,
            "chunks": {},
            "output_shards": {},
            "complete": False,
        }
        write_json(state, path)
    return state


def _write_output_shards(
    records: list[dict[str, Any]],
    output: Path,
    shard_size: int,
    *,
    manifest_root: Path,
    file_prefix: str,
    state: dict[str, Any],
    state_key: str,
    save_state: Callable[[], None],
) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    saved = state.setdefault("output_shards", {}).setdefault(state_key, {})
    manifest: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    expected_keys: set[str] = set()
    for start in range(0, len(records), shard_size):
        index = start // shard_size
        key = str(index)
        expected_keys.add(key)
        shard_records = records[start : start + shard_size]
        path = output / f"{file_prefix}-{index:05d}.parquet"
        expected_paths.add(path)
        prior = saved.get(key) or {}
        reusable = (
            prior.get("path") == path.relative_to(manifest_root).as_posix()
            and int(prior.get("rows") or -1) == len(shard_records)
            and int(prior.get("start", -1)) == start
            and _parquet_valid(path, len(shard_records), int(prior.get("bytes") or -1))
        )
        if not reusable:
            size = _write_table_atomic(shard_records, path)
            prior = {
                "path": path.relative_to(manifest_root).as_posix(),
                "start": start,
                "end": start + len(shard_records) - 1,
                "rows": len(shard_records),
                "bytes": size,
            }
            saved[key] = prior
            save_state()
        manifest.append({name: prior[name] for name in ("path", "start", "end", "rows")})
    for key in list(saved):
        if key not in expected_keys:
            saved.pop(key, None)
    for old in output.glob(f"{file_prefix}-*.parquet"):
        if old not in expected_paths:
            old.unlink()
    save_state()
    return manifest


def _histogram(values: list[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def export_hf_dataset(
    campaign_root: str | Path,
    output_dir: str | Path,
    *,
    shard_size: int = 100,
    include_image_bytes: bool = True,
    checkpoint_units: int = 100,
) -> dict[str, Any]:
    """Resume-safe export from unit outputs; merged JSONL is never required."""
    from .dataset_export import _histogram_svg

    import pyarrow.parquet as pq

    root = Path(campaign_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    shard_size = max(1, int(shard_size))
    checkpoint_units = max(1, int(checkpoint_units))
    registry = load_registry(root)
    signature = _source_signature(
        root,
        registry,
        include_image_bytes=include_image_bytes,
        shard_size=shard_size,
        checkpoint_units=checkpoint_units,
    )
    checkpoint_root = output.parent / f".{output.name}.checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_root / "state.json"
    state = _load_state(state_path, signature)

    def save_state() -> None:
        write_json(state, state_path)

    unit_count = int(registry.get("unit_count") or 0)
    chunks = state.setdefault("chunks", {})
    for unit_start in range(0, unit_count, checkpoint_units):
        index = unit_start // checkpoint_units
        key = str(index)
        unit_end = min(unit_count, unit_start + checkpoint_units)
        records_path = checkpoint_root / "chunks" / f"chunk-{index:05d}-records.parquet"
        standard_path = checkpoint_root / "chunks" / f"chunk-{index:05d}-standard.parquet"
        prior = chunks.get(key) or {}
        records_ok = _parquet_valid(
            records_path,
            int(prior.get("record_rows") or -1),
            int(prior.get("record_bytes") or -1),
        )
        standard_rows = int(prior.get("standard_rows") or 0)
        standard_ok = standard_rows == 0 or _parquet_valid(
            standard_path,
            standard_rows,
            int(prior.get("standard_bytes") or -1),
        )
        if not (
            int(prior.get("unit_start", -1)) == unit_start
            and int(prior.get("unit_end") or -1) == unit_end
            and records_ok
            and standard_ok
        ):
            chunk_records: list[dict[str, Any]] = []
            chunk_standard: list[dict[str, Any]] = []
            for unit_id in range(unit_start, unit_end):
                rich, standard = _unit_records(
                    root,
                    registry,
                    unit_id,
                    include_image_bytes=include_image_bytes,
                )
                chunk_records.extend(rich)
                chunk_standard.extend(standard)
            record_bytes = _write_table_atomic(chunk_records, records_path)
            standard_bytes = 0
            if chunk_standard:
                standard_bytes = _write_table_atomic(chunk_standard, standard_path)
            elif standard_path.exists():
                standard_path.unlink()
            prior = {
                "unit_start": unit_start,
                "unit_end": unit_end,
                "record_rows": len(chunk_records),
                "record_bytes": record_bytes,
                "standard_rows": len(chunk_standard),
                "standard_bytes": standard_bytes,
            }
            chunks[key] = prior
            save_state()
            print(
                f"[export-checkpoint] units={unit_start}:{unit_end} "
                f"records={len(chunk_records)} standard={len(chunk_standard)}",
                flush=True,
            )

    all_processed: list[dict[str, Any]] = []
    standard_by_image: dict[str, dict[str, Any]] = {}
    for index in range((unit_count + checkpoint_units - 1) // checkpoint_units):
        records_path = checkpoint_root / "chunks" / f"chunk-{index:05d}-records.parquet"
        all_processed.extend(pq.read_table(records_path).to_pylist())
        if int(chunks[str(index)].get("standard_rows") or 0):
            standard_path = checkpoint_root / "chunks" / f"chunk-{index:05d}-standard.parquet"
            for row in pq.read_table(standard_path).to_pylist():
                standard_by_image[str(row["image_id"])] = row

    parseable_1_plus = [row for row in all_processed if row["bcc_parseable"] and row["final_linked_mask_count"] >= 1]
    min_10 = [row for row in parseable_1_plus if row["final_linked_mask_count"] >= 10]
    masks_1_to_9 = [row for row in parseable_1_plus if 1 <= row["final_linked_mask_count"] < 10]
    views = {
        "min_10_masks": min_10,
        "masks_1_to_9": masks_1_to_9,
        "parseable_1_plus": parseable_1_plus,
        "audit_all_processed": all_processed,
    }
    standard_views = {
        "gpic_min_10": [standard_by_image[str(row["image_id"])] for row in min_10],
        "gpic_1_to_9": [standard_by_image[str(row["image_id"])] for row in masks_1_to_9],
        "gpic_parseable_1_plus": [standard_by_image[str(row["image_id"])] for row in parseable_1_plus],
    }
    shard_manifests = {
        name: _write_output_shards(
            rows,
            output / "data" / name,
            shard_size,
            manifest_root=output,
            file_prefix="train",
            state=state,
            state_key=f"data/{name}",
            save_state=save_state,
        )
        for name, rows in views.items()
    }
    standard_shards = {
        name: _write_output_shards(
            rows,
            output / "train",
            shard_size,
            manifest_root=output,
            file_prefix=name,
            state=state,
            state_key=f"train/{name}",
            save_state=save_state,
        )
        for name, rows in standard_views.items()
    }
    histograms = {
        "all_sam3_proposal_counts": _histogram([int(row["sam3_proposal_count"]) for row in all_processed]),
        "all_post_consistency_mask_counts": _histogram([int(row["post_consistency_mask_count"]) for row in all_processed]),
        "parseable_1_plus_final_linked_masks": _histogram([int(row["final_linked_mask_count"]) for row in parseable_1_plus]),
        "min_10_final_linked_masks": _histogram([int(row["final_linked_mask_count"]) for row in min_10]),
        "masks_1_to_9_final_linked_masks": _histogram([int(row["final_linked_mask_count"]) for row in masks_1_to_9]),
    }
    stats = {
        "dataset_export_version": DATASET_EXPORT_VERSION,
        "concor_caption_format_version": CONCOR_CAPTION_FORMAT_VERSION,
        "source_signature": signature,
        "processed_raw_images": len(all_processed),
        "parseable_1_plus": len(parseable_1_plus),
        "min_10_masks": len(min_10),
        "masks_1_to_9": len(masks_1_to_9),
        "audit_only_zero_or_unparseable": len(all_processed) - len(parseable_1_plus),
        "histograms": histograms,
        "shards": shard_manifests,
        "concor_standard_shards": standard_shards,
    }
    write_json(stats, output / "stats" / "summary.json")
    _histogram_svg(histograms["all_sam3_proposal_counts"], "All processed images: SAM3 proposals", output / "stats" / "all_sam3_proposals.svg")
    _histogram_svg(histograms["all_post_consistency_mask_counts"], "All processed images: post-consistency masks", output / "stats" / "all_post_consistency_masks.svg")
    _histogram_svg(histograms["parseable_1_plus_final_linked_masks"], "Parseable BCC examples (>=1): final linked masks", output / "stats" / "parseable_1_plus.svg")
    _histogram_svg(histograms["min_10_final_linked_masks"], "Parseable BCC examples (>=10): final linked masks", output / "stats" / "min_10_masks.svg")
    _histogram_svg(histograms["masks_1_to_9_final_linked_masks"], "Parseable BCC examples (1-9): final linked masks", output / "stats" / "masks_1_to_9.svg")
    state["complete"] = True
    state["stats"] = {
        key: stats[key]
        for key in ("processed_raw_images", "parseable_1_plus", "min_10_masks", "masks_1_to_9")
    }
    save_state()
    return stats
