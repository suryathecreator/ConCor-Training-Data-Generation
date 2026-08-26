from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Any

import numpy as np


def sanitize_id(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def mask_area(mask: np.ndarray) -> int:
    return int(np.asarray(mask, dtype=bool).sum())


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def bbox_area(bbox: list[int]) -> int:
    return max(0, int(bbox[2])) * max(0, int(bbox[3]))


def bbox_intersection_area(left: list[int], right: list[int]) -> int:
    lx, ly, lw, lh = [int(item) for item in left]
    rx, ry, rw, rh = [int(item) for item in right]
    x0 = max(lx, rx)
    y0 = max(ly, ry)
    x1 = min(lx + lw, rx + rw)
    y1 = min(ly + lh, ry + rh)
    return max(0, x1 - x0) * max(0, y1 - y0)


def bbox_containment(inner: list[int], outer: list[int]) -> float:
    area = bbox_area(inner)
    return float(bbox_intersection_area(inner, outer) / area) if area else 0.0


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    inter = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(inter / union) if union else 0.0


def mask_containment(inner: np.ndarray, outer: np.ndarray) -> float:
    inner = np.asarray(inner, dtype=bool)
    outer = np.asarray(outer, dtype=bool)
    area = inner.sum()
    if not area:
        return 0.0
    return float(np.logical_and(inner, outer).sum() / area)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8), mode="L").save(path)


def _rgb_image(image_or_path: Any):
    from PIL import Image

    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")
    with Image.open(image_or_path) as handle:
        return handle.convert("RGB")


def color_for_index(index: int) -> tuple[int, int, int]:
    hue = (index * 0.61803398875) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return tuple(int(channel * 255) for channel in rgb)


def select_inverse_background_rgb(
    foreground_rgb: np.ndarray,
) -> dict[str, Any]:
    """Choose a synthetic fill color far from the foreground RGB distribution.

    The score emphasizes the lower tail of foreground-to-fill distance so the
    chosen color remains distinct from even less common object colors, rather
    than merely contrasting with the mean color.
    """
    pixels = np.asarray(foreground_rgb, dtype=np.float32).reshape(-1, 3)
    if not len(pixels):
        return {
            "rgb": [0, 0, 0],
            "score": 0.0,
            "foreground_median_rgb": [0, 0, 0],
            "foreground_mean_rgb": [0, 0, 0],
            "foreground_sample_count": 0,
        }
    max_samples = 32768
    if len(pixels) > max_samples:
        indices = np.linspace(0, len(pixels) - 1, max_samples, dtype=np.int64)
        pixels = pixels[indices]
    median = np.median(pixels, axis=0)
    mean = np.mean(pixels, axis=0)
    candidates = [
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 96, 0),
        (96, 0, 255),
        tuple(np.clip(255.0 - median, 0, 255).round().astype(int)),
        tuple(np.clip(255.0 - mean, 0, 255).round().astype(int)),
    ]
    candidates = list(dict.fromkeys(candidates))
    foreground_luma = pixels @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    median_luma = float(np.median(foreground_luma))
    scored: list[tuple[float, tuple[int, int, int], dict[str, float]]] = []
    for candidate in candidates:
        rgb = np.asarray(candidate, dtype=np.float32)
        distances = np.linalg.norm(pixels - rgb, axis=1)
        q10 = float(np.quantile(distances, 0.10))
        q25 = float(np.quantile(distances, 0.25))
        average = float(np.mean(distances))
        candidate_luma = float(rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32))
        luma_contrast = abs(candidate_luma - median_luma)
        near_fraction = float(np.mean(distances < 48.0))
        score = 0.50 * q10 + 0.22 * q25 + 0.18 * average + 0.10 * luma_contrast - 80.0 * near_fraction
        scored.append(
            (
                score,
                candidate,
                {
                    "distance_q10": q10,
                    "distance_q25": q25,
                    "distance_mean": average,
                    "luma_contrast": luma_contrast,
                    "near_fraction": near_fraction,
                },
            )
        )
    score, chosen, detail = max(scored, key=lambda item: (item[0], item[1]))
    return {
        "rgb": [int(value) for value in chosen],
        "score": float(score),
        "foreground_median_rgb": [int(value) for value in np.clip(median, 0, 255).round()],
        "foreground_mean_rgb": [int(value) for value in np.clip(mean, 0, 255).round()],
        "foreground_sample_count": int(len(pixels)),
        **detail,
    }


def overlay_image(
    image_path: str | Path,
    mask: np.ndarray,
    out_path: str | Path,
    color: tuple[int, int, int] = (255, 40, 40),
    alpha: float = 0.45,
) -> None:
    from PIL import Image

    image = _rgb_image(image_path)
    base = np.asarray(image).astype(np.float32)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[:2] != base.shape[:2]:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask = np.asarray(mask_img.resize(image.size, Image.NEAREST)) > 0
    rgb = np.asarray(color, dtype=np.float32)
    base[mask] = (1.0 - alpha) * base[mask] + alpha * rgb
    out = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def inverse_mask_image(
    image_path: str | Path,
    mask: np.ndarray,
    out_path: str | Path,
    background_rgb: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    from PIL import Image

    image = _rgb_image(image_path)
    base = np.asarray(image).copy()
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[:2] != base.shape[:2]:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask = np.asarray(mask_img.resize(image.size, Image.NEAREST)) > 0
    selection = select_inverse_background_rgb(base[mask])
    fill = tuple(background_rgb) if background_rgb is not None else tuple(selection["rgb"])
    base[~mask] = np.asarray(fill, dtype=np.uint8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(base).save(out_path)
    selection["rgb"] = [int(value) for value in fill]
    return selection


def inverse_crop_image(
    image_path: str | Path,
    mask: np.ndarray,
    bbox: list[int],
    out_path: str | Path,
    padding: int = 24,
    background_rgb: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    from PIL import Image

    image = _rgb_image(image_path)
    x0, y0, x1, y1 = crop_bounds(image.size, bbox, padding)
    cropped = image.crop((x0, y0, x1, y1))
    local_mask = np.asarray(mask, dtype=bool)[y0:y1, x0:x1]
    base = np.asarray(cropped).copy()
    selection = select_inverse_background_rgb(base[local_mask])
    fill = tuple(background_rgb) if background_rgb is not None else tuple(selection["rgb"])
    base[~local_mask] = np.asarray(fill, dtype=np.uint8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(base).save(out_path)
    selection["rgb"] = [int(value) for value in fill]
    selection["crop_bounds_xyxy"] = [int(x0), int(y0), int(x1), int(y1)]
    return selection


def multi_overlay_image(
    image_path: str | Path,
    masks: list[np.ndarray],
    out_path: str | Path,
    alpha: float = 0.45,
) -> None:
    from PIL import Image

    image = _rgb_image(image_path)
    base = np.asarray(image).astype(np.float32)
    for index, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[:2] != base.shape[:2]:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            mask = np.asarray(mask_img.resize(image.size, Image.NEAREST)) > 0
        rgb = np.asarray(color_for_index(index), dtype=np.float32)
        base[mask] = (1.0 - alpha) * base[mask] + alpha * rgb
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(out_path)


def crop_overlay(
    image_path: str | Path,
    mask: np.ndarray,
    bbox: list[int],
    out_path: str | Path,
    padding: int = 24,
) -> None:
    from PIL import Image

    image = _rgb_image(image_path)
    x0, y0, x1, y1 = crop_bounds(image.size, bbox, padding)
    cropped = image.crop((x0, y0, x1, y1))
    local_mask = np.asarray(mask, dtype=bool)[y0:y1, x0:x1]
    base = np.asarray(cropped).astype(np.float32)
    base[local_mask] = 0.55 * base[local_mask] + 0.45 * np.asarray((255, 40, 40))
    out = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def crop_bounds(image_size: tuple[int, int], bbox: list[int], padding: int = 24) -> tuple[int, int, int, int]:
    width, height = image_size
    x, y, w, h = [int(item) for item in bbox]
    x0 = max(0, x - int(padding))
    y0 = max(0, y - int(padding))
    x1 = min(width, x + w + int(padding))
    y1 = min(height, y + h + int(padding))
    return x0, y0, x1, y1


def crop_image(
    image_path: str | Path,
    bbox: list[int],
    out_path: str | Path,
    padding: int = 24,
) -> None:
    from PIL import Image

    image = _rgb_image(image_path)
    x0, y0, x1, y1 = crop_bounds(image.size, bbox, padding)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop((x0, y0, x1, y1)).save(out_path)


def dedupe_candidates(candidates: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        if all(mask_iou(candidate["mask"], item["mask"]) < threshold for item in kept):
            kept.append(candidate)
    return kept


def filter_candidates(
    candidates: list[dict[str, Any]],
    *,
    image_area: int,
    min_area: int,
    min_area_fraction: float,
    dedupe_iou: float,
    min_bbox_fill: float,
    max_mask_area_fraction: float,
    max_bbox_area_fraction: float,
    containment_threshold: float,
    bbox_containment_threshold: float,
    contained_area_ratio: float,
    containment_score_margin: float,
    disable_containment: bool = False,
    disable_dedupe_iou: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    dynamic_min_area = max(int(min_area), int(round(float(image_area) * float(min_area_fraction))))

    def reject(candidate: dict[str, Any], reason: str, detail: dict[str, Any] | None = None) -> None:
        rejected_row = dict(candidate)
        rejected_row["reject_reason"] = reason
        rejected_row["reject_detail"] = detail or {}
        rejected.append(rejected_row)

    for candidate in candidates:
        b_area = bbox_area(candidate["bbox"])
        candidate["bbox_area"] = b_area
        candidate["mask_area_fraction"] = float(candidate["area"] / image_area) if image_area else 0.0
        candidate["bbox_area_fraction"] = float(b_area / image_area) if image_area else 0.0
        candidate["bbox_fill"] = float(candidate["area"] / b_area) if b_area else 0.0
        if candidate["area"] < dynamic_min_area:
            reject(candidate, "small_area", {"min_area": dynamic_min_area})
        elif candidate["bbox_fill"] < min_bbox_fill:
            reject(candidate, "sparse_fragment", {"min_bbox_fill": min_bbox_fill})
        elif candidate["mask_area_fraction"] > max_mask_area_fraction:
            reject(candidate, "giant_mask", {"max_mask_area_fraction": max_mask_area_fraction})
        elif candidate["bbox_area_fraction"] > max_bbox_area_fraction:
            reject(candidate, "giant_bbox", {"max_bbox_area_fraction": max_bbox_area_fraction})
        else:
            active.append(candidate)

    if not disable_containment:
        contained_ids: set[int] = set()
        active_by_area = sorted(active, key=lambda item: int(item["area"]), reverse=True)
        for small in reversed(active_by_area):
            if id(small) in contained_ids:
                continue
            for large in active_by_area:
                if large is small or int(large["area"]) <= int(small["area"]):
                    continue
                area_ratio = float(small["area"] / large["area"]) if large["area"] else 0.0
                if area_ratio > contained_area_ratio:
                    continue
                if float(large.get("score", 0.0)) < float(small.get("score", 0.0)) - containment_score_margin:
                    continue
                mask_cover = mask_containment(small["mask"], large["mask"])
                box_cover = bbox_containment(small["bbox"], large["bbox"])
                if mask_cover >= containment_threshold and box_cover >= bbox_containment_threshold:
                    contained_ids.add(id(small))
                    reject(
                        small,
                        "contained_partial",
                        {
                            "larger_raw_index": large.get("raw_index"),
                            "larger_area": large.get("area"),
                            "larger_score": large.get("score"),
                            "mask_containment": mask_cover,
                            "bbox_containment": box_cover,
                            "area_ratio": area_ratio,
                        },
                    )
                    break

        active = [item for item in active if id(item) not in contained_ids]
    kept: list[dict[str, Any]] = []
    for candidate in sorted(active, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        duplicate = None
        duplicate_iou = 0.0
        if not disable_dedupe_iou:
            for kept_candidate in kept:
                iou = mask_iou(candidate["mask"], kept_candidate["mask"])
                if iou >= dedupe_iou:
                    duplicate = kept_candidate
                    duplicate_iou = iou
                    break
        if duplicate is not None:
            reject(
                candidate,
                "duplicate_iou",
                {
                    "kept_raw_index": duplicate.get("raw_index"),
                    "kept_score": duplicate.get("score"),
                    "iou": duplicate_iou,
                    "dedupe_iou": dedupe_iou,
                },
            )
        else:
            kept.append(candidate)
    return kept, rejected
