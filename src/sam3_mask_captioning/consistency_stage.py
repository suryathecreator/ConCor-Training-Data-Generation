from __future__ import annotations

import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .caption_cleanup import extract_main_candidate
from .io_utils import append_jsonl, read_jsonl, read_jsonl_indexed, write_jsonl
from .mask_utils import bbox_area, bbox_intersection_area, mask_bbox, mask_iou
from .sam3_stage import (
    _as_numpy_boxes,
    _as_numpy_masks,
    _as_numpy_scores,
    _sam3_autocast_factory,
    _xyxy_to_xywh,
    run_sam3_text_prompts,
)


def _load_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def _existing_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl_indexed(path):
            mask_id = str(row.get("mask_id") or "")
            if mask_id:
                ids.add(mask_id)
    return ids


def _same_region(
    source_bbox: list[int],
    candidate_bbox: list[int],
    min_overlap: float,
) -> bool:
    intersection = bbox_intersection_area(source_bbox, candidate_bbox)
    smaller = min(bbox_area(source_bbox), bbox_area(candidate_bbox))
    return bool(smaller and intersection / smaller >= min_overlap)


def run_sam3_consistency(
    config: dict[str, Any],
    run_dir: str | Path,
    *,
    rows: list[dict[str, Any]] | None = None,
    processor: Any | None = None,
    mock: bool = False,
) -> Path:
    """Re-query SAM3 with spaCy-derived concepts and validate with mask IoU."""
    run_dir = Path(run_dir)
    input_path = run_dir / "captions.jsonl"
    input_rows = list(rows) if rows is not None else (read_jsonl(input_path) if input_path.exists() else [])
    passed_path = run_dir / "consistent_captions.jsonl"
    rejected_path = run_dir / "consistency_rejected_captions.jsonl"
    reviews_path = run_dir / "sam3_consistency_reviews.jsonl"
    errors_path = run_dir / "sam3_consistency_errors.jsonl"
    stage_config = config.get("consistency_filter", {})
    resume = bool(config.get("resume", False) or stage_config.get("resume", False))
    if resume:
        completed = _existing_ids([passed_path, rejected_path])
        input_rows = [row for row in input_rows if str(row.get("mask_id") or "") not in completed]
    else:
        for stale in (passed_path, rejected_path, reviews_path, errors_path):
            if stale.exists():
                stale.unlink()
    if not input_rows:
        if not passed_path.exists():
            write_jsonl([], passed_path)
        if not rejected_path.exists():
            write_jsonl([], rejected_path)
        return passed_path

    threshold = float(stage_config.get("mask_iou_threshold", 0.50))
    same_region_min_overlap = float(stage_config.get("same_region_min_bbox_overlap", 0.10))
    derived: dict[str, dict[str, str]] = {}
    rows_by_query: dict[str, list[dict[str, Any]]] = {}
    for row in input_rows:
        extraction = extract_main_candidate(
            object_text=str(row.get("object") or ""),
            caption=str(row.get("caption") or ""),
            source_prompt=str(row.get("source_prompt") or ""),
        )
        mask_id = str(row["mask_id"])
        derived[mask_id] = extraction
        rows_by_query.setdefault(extraction["candidate"], []).append(row)

    source_paths = {str(row.get("source_image_path") or "") for row in input_rows}
    if len(source_paths) != 1:
        raise ValueError("SAM3 consistency batches must contain exactly one source image")
    source_path = next(iter(source_paths))
    query_outputs: dict[str, list[dict[str, Any]]] = {}
    autocast_context = nullcontext if mock else _sam3_autocast_factory(config)

    if mock:
        for query, query_rows in rows_by_query.items():
            query_outputs[query] = [
                {
                    "mask": _load_mask(row["mask_path"]),
                    "bbox": list(row["bbox"]),
                    "score": 1.0,
                }
                for row in query_rows
            ]
    else:
        if processor is None:
            from .sam3_stage import _load_processor

            processor = _load_processor(config)
        with Image.open(source_path) as handle:
            image = handle.convert("RGB")
        with autocast_context():
            state = processor.set_image(image)
        queries = list(rows_by_query)
        query_regions = [
            [list(row["bbox"]) for row in rows_by_query[query]]
            for query in queries
        ]
        try:
            outputs, batch_metrics = run_sam3_text_prompts(
                processor,
                state,
                queries,
                batch_size=int(stage_config.get("prompt_batch_size", 8)),
                autocast_context=autocast_context,
                region_bboxes_by_prompt=query_regions,
                min_region_overlap=same_region_min_overlap,
            )
        except Exception as exc:
            for query in queries:
                query_outputs[query] = []
            append_jsonl(
                {
                    "image_id": input_rows[0].get("image_id"),
                    "queries": queries,
                    "stage": "sam3_consistency",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                errors_path,
            )
            if not config.get("continue_on_error", True):
                raise
        else:
            append_jsonl(
                {
                    "image_id": input_rows[0].get("image_id"),
                    "stage": "sam3_consistency",
                    **batch_metrics,
                },
                run_dir / "sam3_consistency_batch_metrics.jsonl",
            )
            for query, output in zip(queries, outputs):
                masks = _as_numpy_masks(output.get("masks"))
                boxes = _as_numpy_boxes(output.get("boxes"))
                scores = _as_numpy_scores(output.get("scores"))
                query_outputs[query] = []
                for index, mask in enumerate(masks):
                    box = boxes[index] if index < len(boxes) else []
                    query_outputs[query].append(
                        {
                            "mask": mask,
                            "bbox": _xyxy_to_xywh(box, mask),
                            "score": scores[index] if index < len(scores) else 0.0,
                        }
                    )

    for row in input_rows:
        mask_id = str(row["mask_id"])
        extraction = derived[mask_id]
        query = extraction["candidate"]
        original_mask = _load_mask(row["mask_path"])
        source_bbox = list(row["bbox"])
        candidates = query_outputs.get(query, [])
        local_candidates = [
            candidate
            for candidate in candidates
            if _same_region(source_bbox, candidate["bbox"], same_region_min_overlap)
        ]
        best_iou = 0.0
        best_index = -1
        best_score = 0.0
        for index, candidate in enumerate(local_candidates):
            similarity = mask_iou(original_mask, candidate["mask"])
            if similarity > best_iou:
                best_iou = similarity
                best_index = index
                best_score = float(candidate.get("score", 0.0))
        passed = best_iou >= threshold
        review = {
            "image_id": row.get("image_id"),
            "mask_id": mask_id,
            "query": query,
            "query_extraction": extraction,
            "metric": "mask_iou",
            "threshold": threshold,
            "best_iou": best_iou,
            "best_local_candidate_index": best_index,
            "best_sam3_score": best_score,
            "candidate_count": len(candidates),
            "same_region_candidate_count": len(local_candidates),
            "same_region_min_bbox_overlap": same_region_min_overlap,
            "passed": passed,
            "reason": "sam3_requery_iou_pass" if passed else "sam3_requery_iou_below_threshold",
        }
        append_jsonl(review, reviews_path)
        output = dict(row)
        output["sam3_consistency"] = review
        output["main_candidate"] = query
        if passed:
            append_jsonl(output, passed_path)
        else:
            append_jsonl(output, rejected_path)
    return passed_path
