from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, write_json
from .validate import validate_caption_rows


def _count(path: Path) -> int:
    return len(read_jsonl(path)) if path.exists() else 0


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    captions = run_dir / "captions.jsonl"
    caption_rows = read_jsonl(captions) if captions.exists() else []
    image_ids = {row["image_id"] for row in caption_rows if row.get("image_id")}
    summary = {
        "run_dir": str(run_dir),
        "selected_images": _count(run_dir / "selected_images.jsonl"),
        "image_reviews": _count(run_dir / "image_reviews.jsonl"),
        "initial_rejected_images": _count(run_dir / "initial_rejected_images.jsonl"),
        "sam3_masks": _count(run_dir / "sam3_masks.jsonl"),
        "sam3_rejected_masks": _count(run_dir / "sam3_rejected_masks.jsonl"),
        "caption_candidates": _count(run_dir / "caption_candidates.jsonl"),
        "caption_rejected_masks": _count(run_dir / "caption_rejected_masks.jsonl"),
        "mask_quality_reviews": _count(run_dir / "mask_quality_reviews.jsonl"),
        "second_pass_rejected_captions": _count(run_dir / "rejected_captions.jsonl"),
        "captions_written": len(caption_rows),
        "captioned_images": len(image_ids),
        "uncertain_captions": sum(1 for row in caption_rows if row.get("uncertain")),
        "sam3_errors": _count(run_dir / "sam3_errors.jsonl"),
        "caption_errors": _count(run_dir / "caption_errors.jsonl"),
        "image_review_errors": _count(run_dir / "image_review_errors.jsonl"),
        "mask_review_errors": _count(run_dir / "mask_review_errors.jsonl"),
        "validation": validate_caption_rows(captions, check_paths=True) if captions.exists() else None,
    }
    categories = read_jsonl(run_dir / "image_categories.jsonl") if (run_dir / "image_categories.jsonl").exists() else []
    if categories:
        summary["image_categories"] = {
            category: sum(1 for row in categories if row.get("category") == category)
            for category in sorted({row.get("category") for row in categories})
        }
    write_json(summary, run_dir / "summary.json")
    return summary
