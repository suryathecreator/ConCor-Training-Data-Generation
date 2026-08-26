from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .io_utils import read_jsonl, write_jsonl
from .mask_utils import multi_overlay_image


def _load_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def finalize_run(config: dict[str, Any], run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    selected = read_jsonl(run_dir / "selected_images.jsonl") if (run_dir / "selected_images.jsonl").exists() else []
    reviews = {
        row["image_id"]: row
        for row in read_jsonl(run_dir / "image_reviews.jsonl")
    } if (run_dir / "image_reviews.jsonl").exists() else {}
    captions = read_jsonl(run_dir / "captions.jsonl") if (run_dir / "captions.jsonl").exists() else []
    caption_rejected = read_jsonl(run_dir / "caption_rejected_masks.jsonl") if (run_dir / "caption_rejected_masks.jsonl").exists() else []
    rejected = read_jsonl(run_dir / "rejected_captions.jsonl") if (run_dir / "rejected_captions.jsonl").exists() else []
    accepted_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    caption_rejected_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in captions:
        accepted_by_image[row["image_id"]].append(row)
    for row in caption_rejected:
        caption_rejected_by_image[row["image_id"]].append(row)
    for row in rejected:
        rejected_by_image[row["image_id"]].append(row)

    min_final_masks = int(config.get("quality_filter", {}).get("min_final_masks_for_accept", 10))
    overlay_dir = run_dir / "final_accepted_overlays"
    categories = []
    for record in selected:
        image_id = record["image_id"]
        review = reviews.get(image_id)
        accepted_rows = accepted_by_image.get(image_id, [])
        caption_rejected_rows = caption_rejected_by_image.get(image_id, [])
        rejected_rows = rejected_by_image.get(image_id, [])
        final_overlay_path = ""
        if accepted_rows:
            masks = [_load_mask(row["mask_path"]) for row in accepted_rows]
            source_image = accepted_rows[0]["source_image_path"]
            out_path = overlay_dir / f"{image_id}.png"
            multi_overlay_image(source_image, masks, out_path)
            final_overlay_path = str(out_path)
        if review and not review.get("accepted"):
            category = "initial_rejected"
        elif len(accepted_rows) >= min_final_masks:
            category = "accepted_both"
        else:
            category = "second_pass_rejected"
        categories.append(
            {
                "image_id": image_id,
                "source_image_path": record["image_path"],
                "category": category,
                "initial_review": review,
                "accepted_mask_count": len(accepted_rows),
                "caption_rejected_mask_count": len(caption_rejected_rows),
                "rejected_second_pass_mask_count": len(rejected_rows),
                "final_overlay_path": final_overlay_path,
            }
        )
    out = run_dir / "image_categories.jsonl"
    write_jsonl(categories, out)
    return out
