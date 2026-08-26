from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_jsonl

REQUIRED_KEYS = {
    "image_id",
    "mask_id",
    "bbox",
    "area",
    "object",
    "caption",
    "attributes",
    "uncertain",
    "sam3_score",
    "model",
    "source_image_path",
    "mask_path",
    "full_overlay_path",
    "crop_overlay_path",
}


def validate_caption_rows(path: str | Path, check_paths: bool = True) -> dict[str, Any]:
    rows = read_jsonl(path)
    errors = []
    for line_no, row in enumerate(rows, start=1):
        missing = sorted(REQUIRED_KEYS.difference(row))
        if missing:
            errors.append({"line": line_no, "error": f"missing keys: {missing}"})
            continue
        bbox = row["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append({"line": line_no, "error": "bbox must be a 4-item list"})
        if check_paths:
            for key in ("source_image_path", "mask_path", "full_overlay_path", "crop_overlay_path", "crop_image_path", "inverse_crop_path"):
                if key not in row or not row[key]:
                    continue
                if not Path(row[key]).exists():
                    errors.append({"line": line_no, "error": f"missing path for {key}: {row[key]}"})
    return {"rows": len(rows), "valid": not errors, "errors": errors}
