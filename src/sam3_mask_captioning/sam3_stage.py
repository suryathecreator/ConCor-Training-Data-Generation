from __future__ import annotations

import json
import os
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from tqdm import tqdm

from .io_utils import append_jsonl, read_jsonl, write_jsonl
from .mask_utils import (
    bbox_area,
    bbox_intersection_area,
    crop_image,
    crop_overlay,
    filter_candidates,
    inverse_crop_image,
    mask_area,
    mask_bbox,
    multi_overlay_image,
    overlay_image,
    sanitize_id,
    save_mask,
)


def _accepted_reviews(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "image_reviews.jsonl"
    if not path.exists():
        return []
    return [row for row in read_jsonl(path) if row.get("accepted")]


def _as_numpy_masks(value: Any) -> list[np.ndarray]:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.to(torch.float32)
        value = value.numpy()
    array = np.asarray(value)
    if array.size == 0:
        return []
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 2:
        array = array[None, ...]
    return [np.asarray(item, dtype=bool) for item in array]


def _as_numpy_boxes(value: Any) -> list[list[float]]:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.to(torch.float32)
        value = value.numpy()
    array = np.asarray(value)
    if array.size == 0:
        return []
    if array.ndim == 1:
        array = array[None, :]
    return [[float(v) for v in row[:4]] for row in array]


def _as_numpy_scores(value: Any) -> list[float]:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.to(torch.float32)
        value = value.numpy()
    array = np.asarray(value).reshape(-1)
    return [float(v) for v in array]


def _box_xyxy_to_xywh(box: list[float]) -> list[int] | None:
    if len(box) >= 4 and any(abs(v) > 0 for v in box):
        x0, y0, x1, y1 = box[:4]
        x0 = int(max(0, round(x0)))
        y0 = int(max(0, round(y0)))
        x1 = int(max(x0, round(x1)))
        y1 = int(max(y0, round(y1)))
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        return [x0, y0, width, height]
    return None


def _xyxy_to_xywh(box: list[float], mask: np.ndarray) -> list[int]:
    converted = _box_xyxy_to_xywh(box)
    if converted is not None:
        return converted
    return mask_bbox(mask)


def _xyxy_overlaps_regions(
    box: list[float],
    regions: list[list[int]],
    min_overlap: float,
) -> bool:
    """Apply the consistency stage's same-region predicate before upsampling."""
    candidate = _box_xyxy_to_xywh(box)
    if candidate is None:
        # A missing box cannot be safely ruled out before its mask exists.
        return True
    for region in regions:
        intersection = bbox_intersection_area(region, candidate)
        smaller = min(bbox_area(region), bbox_area(candidate))
        if smaller and intersection / smaller >= min_overlap:
            return True
    return False


def _component_count(mask: np.ndarray, cap: int) -> int:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    ys, xs = np.where(mask)
    for start_y, start_x in zip(ys, xs):
        if seen[start_y, start_x]:
            continue
        count += 1
        if count > cap:
            return count
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
    return count


def _sparse_component_rejects(
    candidates: list[dict[str, Any]],
    *,
    max_components: int,
    sparse_fill_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        components = _component_count(candidate["mask"], max_components + 1)
        candidate["component_count"] = components
        if components > max_components and float(candidate.get("bbox_fill", 0.0)) < sparse_fill_threshold:
            row = dict(candidate)
            row["reject_reason"] = "sparse_tangled_components"
            row["reject_detail"] = {
                "component_count": components,
                "max_components": max_components,
                "sparse_fill_threshold": sparse_fill_threshold,
            }
            rejected.append(row)
        else:
            kept.append(candidate)
    return kept, rejected


def _dedupe_completed_images(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = read_jsonl(path)
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        image_id = str(row.get("image_id") or "").strip()
        if not image_id:
            continue
        if image_id not in latest:
            order.append(image_id)
        latest[image_id] = row
    if len(rows) != len(latest):
        write_jsonl((latest[image_id] for image_id in order), path)
    return set(latest)


def _purge_incomplete_image(run_dir: Path, image_id: str) -> None:
    """Remove non-committed SAM3 rows/files before an image-level retry."""
    canonical_streams = (
        "sam3_masks.jsonl",
        "sam3_rejected_masks.jsonl",
        "sam3_raw.jsonl",
        "sam3_prompt_batch_metrics.jsonl",
    )
    for filename in canonical_streams:
        path = run_dir / filename
        if not path.exists():
            continue
        rows = read_jsonl(path)
        kept = [row for row in rows if str(row.get("image_id") or "") != image_id]
        if len(kept) != len(rows):
            write_jsonl(kept, path)
    prefix = f"{sanitize_id(image_id)}_"
    for directory in (
        "masks",
        "inverse_crops",
        "overlays",
        "crop_overlays",
        "crop_images",
    ):
        root = run_dir / directory
        if not root.is_dir():
            continue
        for path in root.glob(f"{prefix}*"):
            if path.is_file():
                path.unlink()
    overview = run_dir / "sam3_all_masks" / f"{sanitize_id(image_id)}.jpg"
    try:
        overview.unlink()
    except FileNotFoundError:
        pass


def _dedupe_mask_manifest(path: Path) -> None:
    if not path.exists():
        return
    rows = read_jsonl(path)
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        mask_id = str(row.get("mask_id") or "").strip()
        if not mask_id:
            passthrough.append(row)
            continue
        if mask_id not in latest:
            order.append(mask_id)
        latest[mask_id] = row
    canonical = [latest[mask_id] for mask_id in order] + passthrough
    if len(canonical) != len(rows):
        write_jsonl(canonical, path)


def _load_processor(config: dict[str, Any]):
    sam3_config = config.get("sam3", {})
    project_root = Path(config.get("project_root", ".")).resolve()
    repo_value = str(os.environ.get("SAM3_REPO_ROOT") or sam3_config.get("repo_root") or "").strip()
    if repo_value:
        repo_root = Path(repo_value).expanduser()
        if not repo_root.is_absolute():
            repo_root = project_root / repo_root
        if not repo_root.is_dir():
            raise FileNotFoundError(f"SAM3_REPO_ROOT does not exist: {repo_root}")
        sys.path.insert(0, str(repo_root))
    deps_value = str(os.environ.get("SAM3_PYTHON_DEPS_ROOT") or "").strip()
    if deps_value:
        deps_root = Path(deps_value).expanduser()
        if not deps_root.is_absolute():
            deps_root = project_root / deps_root
        if deps_root.is_dir():
            sys.path.append(str(deps_root))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = str(sam3_config.get("device", "cuda"))
    checkpoint_path = (
        os.environ.get("BCC_SAM3_CHECKPOINT_PATH")
        or sam3_config.get("checkpoint_path")
        or None
    )
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=bool(sam3_config.get("load_from_hf", True))
        and checkpoint_path is None,
        compile=bool(sam3_config.get("compile", False)),
    )
    return Sam3Processor(
        model,
        resolution=int(sam3_config.get("resolution", 1008)),
        device=device,
        confidence_threshold=float(sam3_config.get("confidence_threshold", 0.45)),
    )


def _sam3_autocast_factory(config: dict[str, Any]) -> Callable[[], Any]:
    sam3_config = config.get("sam3", {})
    device = str(sam3_config.get("device", "cuda")).lower()
    enabled = bool(sam3_config.get("autocast", device.startswith("cuda")))
    dtype_name = str(sam3_config.get("autocast_dtype", "bfloat16")).lower()
    if not enabled or not device.startswith("cuda"):
        return nullcontext

    import torch

    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    dtype = dtype_map.get(dtype_name)
    if dtype is None:
        return nullcontext

    return lambda: torch.autocast(device_type="cuda", dtype=dtype)


def _sam3_text_prompt_batch(
    processor: Any,
    state: dict[str, Any],
    prompts: list[str],
    *,
    region_bboxes_by_prompt: list[list[list[int]]] | None = None,
    min_region_overlap: float = 0.0,
) -> list[dict[str, Any]]:
    """Ground several texts against one cached image backbone in one forward pass."""
    import torch
    from sam3.model import box_ops
    from sam3.model.data_misc import FindStage, interpolate

    if not prompts:
        return []
    model = processor.model
    device = processor.device
    count = len(prompts)
    backbone_out = dict(state["backbone_out"])
    backbone_out.update(model.backbone.forward_text(prompts, device=device))
    find_input = FindStage(
        img_ids=torch.zeros(count, device=device, dtype=torch.long),
        text_ids=torch.arange(count, device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    outputs = model.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_input,
        geometric_prompt=model._get_dummy_prompt(num_prompts=count),
        find_target=None,
    )
    probabilities = outputs["pred_logits"].sigmoid()
    presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
    probabilities = (probabilities * presence).squeeze(-1)
    results: list[dict[str, Any]] = []
    height = int(state["original_height"])
    width = int(state["original_width"])
    scale = torch.tensor([width, height, width, height], device=device)
    for index in range(count):
        keep = probabilities[index] > float(processor.confidence_threshold)
        selected = torch.nonzero(keep, as_tuple=False).flatten()
        unfiltered_count = int(selected.numel())
        boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][index][selected])
        boxes = boxes * scale[None, :]
        regions = (
            region_bboxes_by_prompt[index]
            if region_bboxes_by_prompt is not None
            and index < len(region_bboxes_by_prompt)
            else []
        )
        if regions and len(boxes):
            box_rows = boxes.detach().cpu().tolist()
            local_keep = torch.tensor(
                [
                    _xyxy_overlaps_regions(row, regions, min_region_overlap)
                    for row in box_rows
                ],
                dtype=torch.bool,
                device=selected.device,
            )
            selected = selected[local_keep]
            boxes = boxes[local_keep]
        scores = probabilities[index][selected]
        masks = outputs["pred_masks"][index][selected]
        if not len(masks):
            results.append(
                {
                    "masks": np.empty((0, height, width), dtype=bool),
                    "boxes": boxes.detach().cpu(),
                    "scores": scores.detach().cpu(),
                    "unfiltered_candidate_count": unfiltered_count,
                    "region_filtered_candidate_count": 0,
                }
            )
            continue
        masks = interpolate(
            masks.unsqueeze(1),
            (height, width),
            mode="bilinear",
            align_corners=False,
        ).sigmoid() > 0.5
        results.append(
            {
                "masks": masks.squeeze(1).detach().cpu(),
                "boxes": boxes.detach().cpu(),
                "scores": scores.detach().cpu(),
                "unfiltered_candidate_count": unfiltered_count,
                "region_filtered_candidate_count": int(len(masks)),
            }
        )
    return results


def run_sam3_text_prompts(
    processor: Any,
    state: dict[str, Any],
    prompts: list[str],
    *,
    batch_size: int,
    autocast_context: Callable[[], Any] = nullcontext,
    region_bboxes_by_prompt: list[list[list[int]]] | None = None,
    min_region_overlap: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adaptive same-image SAM3 text batching with a measured serial fallback."""
    import time
    import torch

    started = time.perf_counter()
    configured = max(1, int(batch_size))
    requested = min(
        configured, int(getattr(processor, "_safe_text_prompt_batch_size", configured))
    )
    if region_bboxes_by_prompt is not None and len(region_bboxes_by_prompt) != len(prompts):
        raise ValueError("region_bboxes_by_prompt must align one-to-one with prompts")
    outputs: list[dict[str, Any]] = []
    actual_batches: list[int] = []
    fallbacks: list[str] = []

    def serial(
        chunk: list[str], chunk_regions: list[list[list[int]]] | None
    ) -> list[dict[str, Any]]:
        serial_outputs: list[dict[str, Any]] = []
        for chunk_index, prompt in enumerate(chunk):
            with autocast_context():
                output = processor.set_text_prompt(prompt=prompt, state=state)
            masks = _as_numpy_masks(output.get("masks"))
            boxes = _as_numpy_boxes(output.get("boxes"))
            scores = _as_numpy_scores(output.get("scores"))
            unfiltered_count = len(masks)
            regions = (
                chunk_regions[chunk_index]
                if chunk_regions is not None and chunk_index < len(chunk_regions)
                else []
            )
            if regions:
                local_indices = [
                    index
                    for index in range(len(masks))
                    if index >= len(boxes)
                    or _xyxy_overlaps_regions(
                        boxes[index], regions, min_region_overlap
                    )
                ]
                masks = [masks[index] for index in local_indices]
                boxes = [boxes[index] for index in local_indices if index < len(boxes)]
                scores = [scores[index] for index in local_indices if index < len(scores)]
            serial_outputs.append(
                {
                    "masks": masks,
                    "boxes": boxes,
                    "scores": scores,
                    "unfiltered_candidate_count": unfiltered_count,
                    "region_filtered_candidate_count": len(masks),
                }
            )
        return serial_outputs

    for start in range(0, len(prompts), requested):
        initial_regions = (
            region_bboxes_by_prompt[start : start + requested]
            if region_bboxes_by_prompt is not None
            else None
        )
        queue = [(prompts[start : start + requested], initial_regions)]
        while queue:
            chunk, chunk_regions = queue.pop(0)
            try:
                with autocast_context():
                    chunk_outputs = _sam3_text_prompt_batch(
                        processor,
                        state,
                        chunk,
                        region_bboxes_by_prompt=chunk_regions,
                        min_region_overlap=min_region_overlap,
                    )
            except RuntimeError as exc:
                message = str(exc).casefold()
                is_oom = "out of memory" in message or "cublas_status_alloc_failed" in message
                if is_oom and len(chunk) > 1:
                    torch.cuda.empty_cache()
                    midpoint = max(1, len(chunk) // 2)
                    processor._safe_text_prompt_batch_size = midpoint
                    left_regions = chunk_regions[:midpoint] if chunk_regions is not None else None
                    right_regions = chunk_regions[midpoint:] if chunk_regions is not None else None
                    queue = [
                        (chunk[:midpoint], left_regions),
                        (chunk[midpoint:], right_regions),
                    ] + queue
                    continue
                fallbacks.append(repr(exc))
                chunk_outputs = serial(chunk, chunk_regions)
            except Exception as exc:
                fallbacks.append(repr(exc))
                chunk_outputs = serial(chunk, chunk_regions)
            outputs.extend(chunk_outputs)
            actual_batches.append(len(chunk))
    metrics = {
        "query_count": len(prompts),
        "configured_batch_size": configured,
        "requested_batch_size": requested,
        "actual_batch_sizes": actual_batches,
        "serial_fallback_count": len(fallbacks),
        "fallback_errors": fallbacks[:8],
        "region_prefilter_enabled": region_bboxes_by_prompt is not None,
        "unfiltered_candidate_count": sum(
            int(output.get("unfiltered_candidate_count") or 0) for output in outputs
        ),
        "region_filtered_candidate_count": sum(
            int(output.get("region_filtered_candidate_count") or 0) for output in outputs
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return outputs, metrics


def run_sam3(
    config: dict[str, Any],
    run_dir: Path,
    limit: int | None = None,
    mock: bool = False,
    reviews_override: list[dict[str, Any]] | None = None,
    processor_override: Any | None = None,
) -> Path:
    reviews = list(reviews_override) if reviews_override is not None else _accepted_reviews(run_dir)
    if limit is not None:
        reviews = reviews[: int(limit)]
    masks_path = run_dir / "sam3_masks.jsonl"
    rejected_path = run_dir / "sam3_rejected_masks.jsonl"
    raw_path = run_dir / "sam3_raw.jsonl"
    completed_path = run_dir / "sam3_completed_images.jsonl"
    failed_path = run_dir / "sam3_failed_images.jsonl"
    errors_path = run_dir / "sam3_errors.jsonl"

    sam3_config = config.get("sam3", {})
    filter_config = config.get("filter", {})
    resume = bool(config.get("resume", False) or sam3_config.get("resume", False))
    if resume:
        completed_ids = _dedupe_completed_images(completed_path)
        reviews = [row for row in reviews if str(row.get("image_id") or "") not in completed_ids]
        for review in reviews:
            _purge_incomplete_image(run_dir, str(review.get("image_id") or ""))
        _dedupe_mask_manifest(masks_path)
    else:
        for stale in (masks_path, rejected_path, raw_path, completed_path, failed_path, errors_path):
            if stale.exists():
                stale.unlink()
    if not reviews:
        if not masks_path.exists():
            write_jsonl([], masks_path)
        if not rejected_path.exists():
            write_jsonl([], rejected_path)
        return masks_path

    processor = None if mock else (processor_override or _load_processor(config))
    autocast_context = nullcontext if mock else _sam3_autocast_factory(config)
    padding = int(sam3_config.get("crop_padding_px", 32))
    emit_debug_derivatives = bool(sam3_config.get("emit_debug_derivatives", False))
    emit_overview = bool(sam3_config.get("emit_overview", False))
    max_masks = int(filter_config.get("max_masks_per_image", 120))
    max_components = int(filter_config.get("max_components", 12))
    sparse_fill_threshold = float(filter_config.get("sparse_component_bbox_fill", 0.18))
    attempted_count = 0
    failed_count = 0

    for review in tqdm(reviews, desc="sam3"):
        attempted_count += 1
        image_id = str(review["image_id"])
        source_path = str(review["source_image_path"])
        prompts = [str(item).strip() for item in review.get("sam3_prompts") or [] if str(item).strip()]
        try:
            with Image.open(source_path) as image_handle:
                image = image_handle.convert("RGB")
                width, height = image.size
            image_area = width * height
            candidates: list[dict[str, Any]] = []
            if mock:
                blank = np.zeros((height, width), dtype=bool)
                blank[height // 4 : height // 2, width // 4 : width // 2] = True
                candidates.append(
                    {
                        "raw_index": 0,
                        "prompt_index": 0,
                        "source_prompt": prompts[0] if prompts else "mock object",
                        "mask": blank,
                        "bbox": mask_bbox(blank),
                        "area": mask_area(blank),
                        "score": 0.99,
                    }
                )
            else:
                with autocast_context():
                    state = processor.set_image(image)
                prompt_outputs, batch_metrics = run_sam3_text_prompts(
                    processor,
                    state,
                    prompts,
                    batch_size=int(sam3_config.get("text_prompt_batch_size", 8)),
                    autocast_context=autocast_context,
                )
                append_jsonl(
                    {
                        "image_id": image_id,
                        "stage": "initial_sam3_prompts",
                        **batch_metrics,
                    },
                    run_dir / "sam3_prompt_batch_metrics.jsonl",
                )
                raw_index = 0
                for prompt_index, (prompt, output) in enumerate(zip(prompts, prompt_outputs)):
                    masks = _as_numpy_masks(output.get("masks"))
                    boxes = _as_numpy_boxes(output.get("boxes"))
                    scores = _as_numpy_scores(output.get("scores"))
                    append_jsonl(
                        {
                            "image_id": image_id,
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "mask_count": len(masks),
                            "scores": scores,
                        },
                        raw_path,
                    )
                    for mask_index, mask in enumerate(masks):
                        score = scores[mask_index] if mask_index < len(scores) else 0.0
                        box = boxes[mask_index] if mask_index < len(boxes) else []
                        candidates.append(
                            {
                                "raw_index": raw_index,
                                "prompt_index": prompt_index,
                                "prompt_mask_index": mask_index,
                                "source_prompt": prompt,
                                "mask": mask,
                                "bbox": _xyxy_to_xywh(box, mask),
                                "area": mask_area(mask),
                                "score": score,
                            }
                        )
                        raw_index += 1

            kept, rejected = filter_candidates(
                candidates,
                image_area=image_area,
                min_area=int(filter_config.get("min_mask_area", 192)),
                min_area_fraction=float(filter_config.get("min_mask_area_fraction", 0.0005)),
                dedupe_iou=float(filter_config.get("dedupe_iou", 0.98)),
                min_bbox_fill=float(filter_config.get("min_bbox_fill", 0.08)),
                max_mask_area_fraction=float(filter_config.get("max_mask_area_fraction", 0.70)),
                max_bbox_area_fraction=float(filter_config.get("max_bbox_area_fraction", 0.90)),
                containment_threshold=float(filter_config.get("containment_threshold", 1.01)),
                bbox_containment_threshold=float(filter_config.get("bbox_containment_threshold", 1.01)),
                contained_area_ratio=float(filter_config.get("contained_area_ratio", 0.0)),
                containment_score_margin=float(filter_config.get("containment_score_margin", 0.0)),
                disable_containment=True,
                disable_dedupe_iou=bool(filter_config.get("disable_dedupe_iou", False)),
            )
            kept, sparse_rejected = _sparse_component_rejects(
                kept,
                max_components=max_components,
                sparse_fill_threshold=sparse_fill_threshold,
            )
            rejected.extend(sparse_rejected)
            kept = sorted(kept, key=lambda item: float(item.get("score", 0.0)), reverse=True)
            if max_masks > 0:
                kept = kept[:max_masks]

            masks_for_overview = [item["mask"] for item in kept]
            if masks_for_overview and emit_overview:
                multi_overlay = run_dir / "sam3_all_masks" / f"{sanitize_id(image_id)}.jpg"
                multi_overlay_image(image, masks_for_overview, multi_overlay)

            for kept_index, candidate in enumerate(kept):
                mask_id = f"{sanitize_id(image_id)}_p{int(candidate.get('prompt_index', 0)):03d}_m{kept_index:04d}"
                mask_path = run_dir / "masks" / f"{mask_id}.png"
                overlay_path = run_dir / "overlays" / f"{mask_id}.jpg"
                crop_overlay_path = run_dir / "crop_overlays" / f"{mask_id}.jpg"
                crop_image_path = run_dir / "crop_images" / f"{mask_id}.jpg"
                inverse_crop_path = run_dir / "inverse_crops" / f"{mask_id}.png"
                save_mask(candidate["mask"], mask_path)
                if emit_debug_derivatives:
                    overlay_image(image, candidate["mask"], overlay_path)
                    crop_overlay(image, candidate["mask"], candidate["bbox"], crop_overlay_path, padding)
                    crop_image(image, candidate["bbox"], crop_image_path, padding)
                inverse_background = inverse_crop_image(
                    image,
                    candidate["mask"],
                    candidate["bbox"],
                    inverse_crop_path,
                    padding,
                )
                out = {
                    "image_id": image_id,
                    "source_image_path": source_path,
                    "mask_id": mask_id,
                    "mask_path": str(mask_path),
                    "full_overlay_path": str(overlay_path) if emit_debug_derivatives else None,
                    "crop_overlay_path": str(crop_overlay_path) if emit_debug_derivatives else None,
                    "crop_image_path": str(crop_image_path) if emit_debug_derivatives else None,
                    "inverse_crop_path": str(inverse_crop_path),
                    "inverse_background_rgb": inverse_background["rgb"],
                    "inverse_background_selection": inverse_background,
                    "bbox": candidate["bbox"],
                    "area": candidate["area"],
                    "bbox_area": candidate.get("bbox_area"),
                    "mask_area_fraction": candidate.get("mask_area_fraction"),
                    "bbox_area_fraction": candidate.get("bbox_area_fraction"),
                    "bbox_fill": candidate.get("bbox_fill"),
                    "component_count": candidate.get("component_count"),
                    "sam3_score": candidate.get("score"),
                    "entityseg_score": candidate.get("score"),
                    "source_prompt": candidate.get("source_prompt", ""),
                    "prompt_index": candidate.get("prompt_index"),
                }
                append_jsonl(out, masks_path)

            for candidate in rejected:
                row = {
                    "image_id": image_id,
                    "source_image_path": source_path,
                    "source_prompt": candidate.get("source_prompt", ""),
                    "raw_index": candidate.get("raw_index"),
                    "bbox": candidate.get("bbox"),
                    "area": candidate.get("area"),
                    "bbox_area": candidate.get("bbox_area"),
                    "mask_area_fraction": candidate.get("mask_area_fraction"),
                    "bbox_area_fraction": candidate.get("bbox_area_fraction"),
                    "bbox_fill": candidate.get("bbox_fill"),
                    "component_count": candidate.get("component_count"),
                    "sam3_score": candidate.get("score"),
                    "reject_reason": candidate.get("reject_reason", "unknown"),
                    "reject_detail": candidate.get("reject_detail", {}),
                }
                append_jsonl(row, rejected_path)
            append_jsonl(
                {"image_id": image_id, "kept": len(kept), "rejected": len(rejected)},
                completed_path,
                durable=True,
            )
        except Exception as exc:
            failed_count += 1
            _purge_incomplete_image(run_dir, image_id)
            append_jsonl(
                {"image_id": image_id, "error": repr(exc)},
                failed_path,
                durable=True,
            )
            append_jsonl(
                {
                    "image_id": image_id,
                    "stage": "sam3",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                errors_path,
                durable=True,
            )
            if not config.get("continue_on_error", True):
                raise
    if (
        attempted_count
        and failed_count == attempted_count
        and bool(sam3_config.get("fail_if_all_attempts_error", True))
    ):
        raise RuntimeError(f"SAM3 failed for all {attempted_count} attempted images; see {errors_path}")
    _dedupe_mask_manifest(masks_path)
    return masks_path
