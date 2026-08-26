from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .campaign_manifest import load_registry
from .io_utils import read_jsonl, write_json, write_jsonl


LEDGER_FIELDS = (
    "source_manifest_index",
    "campaign_unit",
    "image_id",
    "source_dataset",
    "source_split",
    "source_pair_key",
    "source_file",
    "image_review_status",
    "image_review_reason",
    "sam3_proposal_count",
    "sam3_rejected_count",
    "mask_caption_count",
    "mask_qa_kept_count",
    "consistency_passed_count",
    "consistency_rejected_count",
    "bcc_status",
    "bcc_parseable",
    "final_linked_mask_count",
    "bcc_reason",
    "audit_issue_count",
    "last_completed_stage",
)
STAGE_ORDER = (
    "image-review",
    "sam3",
    "mask-caption",
    "mask-qa",
    "mask-caption-qa",
    "consistency",
    "bcc-draft",
    "bcc-rewrite",
    "bcc",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _by_image(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("image_id") or ""): row for row in _rows(path) if row.get("image_id")}


def _counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in _rows(path):
        counts[str(row.get("image_id") or "")] += 1
    return dict(counts)


def _last_completed_stage(unit: Path, stages: tuple[str, ...]) -> str:
    completed = [stage for stage in stages if (unit / "stages" / stage / "_SUCCESS.json").exists()]
    return completed[-1] if completed else "materialized"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_run_ledger(campaign_root: str | Path) -> dict[str, Any]:
    """Write one durable outcome row for every materialized source image."""
    root = Path(campaign_root).expanduser().resolve()
    registry = load_registry(root)
    ledger: list[dict[str, Any]] = []
    for unit_id in range(int(registry.get("unit_count") or 0)):
        unit = root / "units" / f"{unit_id:06d}"
        reviews = _by_image(unit / "image_reviews.jsonl")
        audits = _by_image(unit / "bcc_validation_audit.jsonl")
        exclusions = _by_image(unit / "bcc_exclusions.jsonl")
        proposal_counts = _counts(unit / "sam3_masks.jsonl")
        sam3_rejected = _counts(unit / "sam3_rejected_masks.jsonl")
        caption_counts = _counts(unit / "caption_candidates.jsonl")
        qa_counts = _counts(unit / "captions.jsonl")
        consistency_counts = _counts(unit / "consistent_captions.jsonl")
        consistency_rejected = _counts(unit / "consistency_rejected_captions.jsonl")
        last_stage = _last_completed_stage(unit, STAGE_ORDER)
        for source in _rows(unit / "selected_images.jsonl"):
            image_id = str(source.get("image_id") or "")
            context = source.get("source_context") or {}
            review = reviews.get(image_id) or {}
            audit = audits.get(image_id) or {}
            exclusion = exclusions.get(image_id) or {}
            groups = list(audit.get("groups") or [])
            after = ((audit.get("validation") or {}).get("after_rewrite") or {})
            parseable = bool(audit and audit.get("caption") and groups and after.get("parseable") is not False)
            issues = list(after.get("issues") or audit.get("issues") or [])
            if audit:
                bcc_status = "parseable" if parseable else "unparseable"
            elif exclusion:
                bcc_status = "excluded"
            else:
                bcc_status = "not_reached"
            ledger.append(
                {
                    "source_manifest_index": int(source.get("source_manifest_index") or 0),
                    "campaign_unit": unit_id,
                    "image_id": image_id,
                    "source_dataset": str(context.get("source_dataset") or registry.get("dataset") or ""),
                    "source_split": str(context.get("split") or registry.get("split") or ""),
                    "source_pair_key": str(context.get("pair_key") or ""),
                    "source_file": Path(str(context.get("source_member") or source.get("image_path") or "")).name,
                    "image_review_status": "accepted" if review.get("accepted") else ("rejected" if review else "not_run"),
                    "image_review_reason": str(review.get("reject_reason") or review.get("rationale") or ""),
                    "sam3_proposal_count": int(proposal_counts.get(image_id, 0)),
                    "sam3_rejected_count": int(sam3_rejected.get(image_id, 0)),
                    "mask_caption_count": int(caption_counts.get(image_id, 0)),
                    "mask_qa_kept_count": int(qa_counts.get(image_id, 0)),
                    "consistency_passed_count": int(consistency_counts.get(image_id, 0)),
                    "consistency_rejected_count": int(consistency_rejected.get(image_id, 0)),
                    "bcc_status": bcc_status,
                    "bcc_parseable": parseable,
                    "final_linked_mask_count": len({str(group.get("mask_id") or "") for group in groups if group.get("mask_id")}) if parseable else 0,
                    "bcc_reason": str(audit.get("reason_code") or exclusion.get("reason_code") or ""),
                    "audit_issue_count": len(issues),
                    "last_completed_stage": last_stage,
                }
            )
    ledger.sort(key=lambda row: int(row["source_manifest_index"]))
    reports = root / "reports"
    write_jsonl(ledger, reports / "run_ledger.jsonl")
    _write_csv(ledger, reports / "run_ledger.csv")
    summary = {
        "source_count": len(ledger),
        "last_completed_stage": dict(Counter(str(row["last_completed_stage"]) for row in ledger)),
        "image_review_status": dict(Counter(str(row["image_review_status"]) for row in ledger)),
        "bcc_status": dict(Counter(str(row["bcc_status"]) for row in ledger)),
        "min_10_parseable": sum(bool(row["bcc_parseable"] and int(row["final_linked_mask_count"]) >= 10) for row in ledger),
        "parseable_1_plus": sum(bool(row["bcc_parseable"] and int(row["final_linked_mask_count"]) >= 1) for row in ledger),
    }
    write_json(summary, reports / "run_ledger_summary.json")
    return summary
