from __future__ import annotations

import json
import mimetypes
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .artifact_store import read_tar_member
from .campaign_manifest import load_registry
from .io_utils import read_jsonl, write_json


DATASET_EXPORT_VERSION = "concor-bcc-parquet-v1"
TRAINING_VIEWS = ("min_10_masks", "masks_1_to_9", "parseable_1_plus")
ALL_VIEWS = (*TRAINING_VIEWS, "audit_all_processed")


def _unit_dir(root: Path, unit_id: int) -> Path:
    return root / "units" / f"{unit_id:06d}"


def _success_path(unit: Path, stage: str) -> Path:
    return unit / "stages" / stage / "_SUCCESS.json"


def _rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _current(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("image_id") or ""): row
        for row in _rows(path)
        if row.get("image_id")
    }


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("image_id") or "")] += 1
    return dict(counts)


def _histogram(values: list[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def _histogram_svg(histogram: dict[str, int], title: str, path: Path) -> None:
    width, height, margin = 1100, 430, 58
    items = [(int(key), int(value)) for key, value in histogram.items()]
    max_value = max((value for _, value in items), default=1)
    bar_width = max(2.0, (width - 2 * margin) / max(1, len(items)))
    bars: list[str] = []
    for index, (mask_count, frequency) in enumerate(items):
        bar_height = (height - 2 * margin) * frequency / max_value
        x = margin + index * bar_width
        y = height - margin - bar_height
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(1.0, bar_width - 1):.2f}" '
            f'height="{bar_height:.2f}" fill="#377d68"><title>{mask_count} masks: {frequency} images</title></rect>'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="#f7f3e8"/><text x="{margin}" y="32" font-family="system-ui" font-size="20">{title}</text>'
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#24352f"/>'
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="system-ui">mask count</text>'
        + "".join(bars)
        + "</svg>\n",
        encoding="utf-8",
    )


def _source_bytes(source: dict[str, Any]) -> tuple[bytes, str]:
    context = source.get("source_context") or {}
    member = str(context.get("source_member") or "")
    archive = str(context.get("source_archive") or "")
    if archive and member:
        return read_tar_member(archive, member), Path(member).name
    path = Path(str(source.get("image_path") or ""))
    return path.read_bytes(), path.name


def _parseable(audit: dict[str, Any] | None) -> bool:
    if not audit or not str(audit.get("caption") or "").strip():
        return False
    groups = list(audit.get("groups") or [])
    if not groups:
        return False
    after = ((audit.get("validation") or {}).get("after_rewrite") or {})
    if after.get("parseable") is False:
        return False
    return str(audit.get("reason_code") or "") != "final_rewrite_unparseable"


def _write_parquet_shards(
    records: list[dict[str, Any]], output: Path, shard_size: int
) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("train-*.parquet"):
        old.unlink()
    manifest: list[dict[str, Any]] = []
    for start in range(0, len(records), shard_size):
        shard_records = records[start : start + shard_size]
        path = output / f"train-{start // shard_size:05d}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(shard_records), temporary, compression="zstd")
        temporary.replace(path)
        manifest.append(
            {
                "path": path.relative_to(output.parent.parent).as_posix(),
                "start": start,
                "end": start + len(shard_records) - 1,
                "rows": len(shard_records),
            }
        )
    return manifest


def export_hf_dataset(
    campaign_root: str | Path,
    output_dir: str | Path,
    *,
    shard_size: int = 100,
    include_image_bytes: bool = True,
) -> dict[str, Any]:
    """Export count-based training views and an audit view of every source row.

    Acceptance flags never choose between training views. A parseable BCC
    record is routed solely by final linked-mask count: >=10, 1--9, or the
    convenience union >=1. Zero-mask, unparseable, and upstream-rejected rows
    appear only in ``audit_all_processed``.
    """
    root = Path(campaign_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    registry = load_registry(root)
    terminal_stage = str(registry.get("terminal_stage") or "bcc")
    all_processed: list[dict[str, Any]] = []

    for unit_id in range(int(registry.get("unit_count") or 0)):
        unit = _unit_dir(root, unit_id)
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

        for source in sources:
            image_id = str(source.get("image_id") or "")
            audit = audits.get(image_id)
            exclusion = exclusions.get(image_id)
            parseable = _parseable(audit)
            groups = list((audit or {}).get("groups") or []) if parseable else []
            linked_mask_ids = {
                str(group.get("mask_id") or "") for group in groups if group.get("mask_id")
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
                "correspondence_groups_json": json.dumps(groups, ensure_ascii=False, sort_keys=True),
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
                "bcc_record_json": json.dumps(audit or exclusion or {}, ensure_ascii=False, sort_keys=True),
            }
            all_processed.append(record)

    parseable_1_plus = [row for row in all_processed if row["bcc_parseable"] and row["final_linked_mask_count"] >= 1]
    min_10 = [row for row in parseable_1_plus if row["final_linked_mask_count"] >= 10]
    masks_1_to_9 = [row for row in parseable_1_plus if 1 <= row["final_linked_mask_count"] < 10]
    views = {
        "min_10_masks": min_10,
        "masks_1_to_9": masks_1_to_9,
        "parseable_1_plus": parseable_1_plus,
        "audit_all_processed": all_processed,
    }
    shard_manifests = {
        name: _write_parquet_shards(rows, output / "data" / name, max(1, shard_size))
        for name, rows in views.items()
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
        "processed_raw_images": len(all_processed),
        "parseable_1_plus": len(parseable_1_plus),
        "min_10_masks": len(min_10),
        "masks_1_to_9": len(masks_1_to_9),
        "audit_only_zero_or_unparseable": len(all_processed) - len(parseable_1_plus),
        "histograms": histograms,
        "shards": shard_manifests,
    }
    write_json(stats, output / "stats" / "summary.json")
    _histogram_svg(histograms["all_sam3_proposal_counts"], "All processed images: SAM3 proposals", output / "stats" / "all_sam3_proposals.svg")
    _histogram_svg(histograms["all_post_consistency_mask_counts"], "All processed images: post-consistency masks", output / "stats" / "all_post_consistency_masks.svg")
    _histogram_svg(histograms["parseable_1_plus_final_linked_masks"], "Parseable BCC examples (>=1): final linked masks", output / "stats" / "parseable_1_plus.svg")
    _histogram_svg(histograms["min_10_final_linked_masks"], "Parseable BCC examples (>=10): final linked masks", output / "stats" / "min_10_masks.svg")
    _histogram_svg(histograms["masks_1_to_9_final_linked_masks"], "Parseable BCC examples (1-9): final linked masks", output / "stats" / "masks_1_to_9.svg")
    return stats
