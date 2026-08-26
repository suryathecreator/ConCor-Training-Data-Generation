from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


_DESCRIPTION_STOP_WORDS = {
    "a", "an", "and", "are", "at", "in", "is", "of", "on", "section",
    "shown", "showing", "the", "to", "visible", "with",
}


def _load_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def _significant_component_count(mask: np.ndarray) -> int:
    labels, component_count = ndimage.label(mask)
    if component_count <= 1:
        return int(component_count)
    counts = np.bincount(labels.ravel())[1:]
    minimum_area = max(8, int(mask.sum()) // 100)
    return int(np.count_nonzero(counts >= minimum_area))


def _description_tokens(row: dict[str, Any]) -> set[str]:
    text = " ".join(str(row.get(key) or "") for key in ("object", "caption"))
    return {
        token
        for token in re.findall(r"[a-z]+", text.casefold())
        if token not in _DESCRIPTION_STOP_WORDS
    }


def _overlap_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(left, right).sum())
    if intersection <= 0:
        return 0.0, 0.0
    left_area = int(left.sum())
    right_area = int(right.sum())
    union = left_area + right_area - intersection
    return (
        intersection / max(1, union),
        intersection / max(1, min(left_area, right_area)),
    )


def _quality_key(row: dict[str, Any]) -> tuple[float, float, int]:
    consistency = row.get("sam3_consistency") or {}
    return (
        float(consistency.get("best_iou") or 0.0),
        float(row.get("sam3_score", row.get("entityseg_score")) or 0.0),
        int(row.get("area") or 0),
    )


def _annotate_composite_masks(
    rows: list[dict[str, Any]],
    masks: list[np.ndarray],
    kept_indexes: list[int],
    subjects: list[str],
    stage_config: dict[str, Any],
) -> None:
    """Mark a retained union mask that collectively covers distinct child masks."""
    child_containment_threshold = float(
        stage_config.get("composite_child_containment_threshold", 0.90)
    )
    union_iou_threshold = float(
        stage_config.get("composite_union_iou_threshold", 0.85)
    )
    child_max_iou = float(stage_config.get("composite_child_max_iou", 0.20))
    areas = [int(mask.sum()) for mask in masks]

    for container_index in kept_indexes:
        container_area = areas[container_index]
        if container_area <= 0 or not subjects[container_index]:
            continue
        candidates: list[int] = []
        for child_index in kept_indexes:
            if child_index == container_index:
                continue
            if subjects[child_index] != subjects[container_index]:
                continue
            child_area = areas[child_index]
            if child_area <= 0 or child_area >= 0.90 * container_area:
                continue
            intersection = int(
                np.logical_and(masks[container_index], masks[child_index]).sum()
            )
            if intersection / child_area >= child_containment_threshold:
                candidates.append(child_index)
        if len(candidates) < 2:
            continue

        distinct_children: list[int] = []
        for child_index in sorted(candidates, key=lambda index: (-areas[index], index)):
            if all(
                _overlap_metrics(masks[child_index], masks[other_index])[0]
                <= child_max_iou
                for other_index in distinct_children
            ):
                distinct_children.append(child_index)
        if len(distinct_children) < 2:
            continue

        child_union = np.logical_or.reduce(
            [masks[index] for index in distinct_children]
        )
        intersection = int(
            np.logical_and(child_union, masks[container_index]).sum()
        )
        union = int(np.logical_or(child_union, masks[container_index]).sum())
        union_iou = intersection / max(1, union)
        composite_coverage = intersection / max(1, container_area)
        if (
            union_iou < union_iou_threshold
            or composite_coverage < union_iou_threshold
        ):
            continue
        rows[container_index]["bcc_composite_mask_children"] = [
            {
                "mask_id": rows[index].get("mask_id"),
                "contained_fraction": int(
                    np.logical_and(masks[container_index], masks[index]).sum()
                ) / max(1, areas[index]),
                "iou_with_composite": _overlap_metrics(
                    masks[container_index], masks[index]
                )[0],
            }
            for index in sorted(distinct_children)
        ]
        rows[container_index]["bcc_composite_union_iou"] = union_iou
        rows[container_index]["bcc_composite_coverage"] = composite_coverage


def canonicalize_bcc_rows(
    rows: list[dict[str, Any]],
    stage_config: dict[str, Any],
    *,
    mask_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove only same-subject near-duplicate masks from the BCC input packet.

    All mask-stage checkpoints remain unchanged. Each removed row is attached to
    its selected canonical row and returned as explicit provenance.
    """
    copied = [dict(row) for row in rows]
    if len(copied) < 2 or not bool(stage_config.get("canonicalize_duplicate_masks", True)):
        return copied, []

    containment_threshold = float(
        stage_config.get("duplicate_containment_threshold", 0.97)
    )
    iou_threshold = float(stage_config.get("duplicate_iou_threshold", 0.55))
    near_exact_iou = float(stage_config.get("duplicate_near_exact_iou", 0.85))
    description_threshold = float(
        stage_config.get("duplicate_description_jaccard", 0.55)
    )

    masks: list[np.ndarray] = []
    for row in copied:
        mask_id = str(row.get("mask_id") or "")
        cached = (mask_arrays or {}).get(mask_id)
        masks.append(cached if cached is not None else _load_mask(row["mask_path"]))
    subjects = [
        str(
            row.get("main_candidate")
            or row.get("object")
            or row.get("source_prompt")
            or ""
        ).strip().casefold()
        for row in copied
    ]
    tokens = [_description_tokens(row) for row in copied]
    parent = list(range(len(copied)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(copied)):
        for right in range(left + 1, len(copied)):
            if not subjects[left] or subjects[left] != subjects[right]:
                continue
            iou, containment = _overlap_metrics(masks[left], masks[right])
            if containment < containment_threshold or iou < iou_threshold:
                continue
            combined = tokens[left] | tokens[right]
            description_jaccard = (
                len(tokens[left] & tokens[right]) / max(1, len(combined))
            )
            if iou < near_exact_iou and description_jaccard < description_threshold:
                continue
            union(left, right)

    clusters: dict[int, list[int]] = {}
    for index in range(len(copied)):
        clusters.setdefault(find(index), []).append(index)

    removed_indexes: set[int] = set()
    dropped: list[dict[str, Any]] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        canonical_index = max(members, key=lambda index: _quality_key(copied[index]))
        aliases: list[dict[str, Any]] = []
        for index in members:
            if index == canonical_index:
                continue
            removed_indexes.add(index)
            iou, containment = _overlap_metrics(masks[canonical_index], masks[index])
            combined = tokens[canonical_index] | tokens[index]
            description_jaccard = (
                len(tokens[canonical_index] & tokens[index]) / max(1, len(combined))
            )
            record = {
                "image_id": copied[index].get("image_id"),
                "canonical_mask_id": copied[canonical_index].get("mask_id"),
                "dropped_mask_id": copied[index].get("mask_id"),
                "dropped_mask_path": copied[index].get("mask_path"),
                "dropped_inverse_crop_path": copied[index].get("inverse_crop_path"),
                "main_candidate": subjects[index],
                "mask_iou": iou,
                "smaller_mask_containment": containment,
                "description_jaccard": description_jaccard,
                "reason": "same_subject_high_containment_duplicate",
            }
            aliases.append(record)
            dropped.append(record)
        copied[canonical_index]["bcc_duplicate_mask_aliases"] = aliases

    kept_indexes = [
        index for index in range(len(copied)) if index not in removed_indexes
    ]
    for index in kept_indexes:
        copied[index]["bcc_significant_component_count"] = (
            _significant_component_count(masks[index])
        )
    _annotate_composite_masks(
        copied, masks, kept_indexes, subjects, stage_config
    )
    kept = [copied[index] for index in kept_indexes]
    dropped.sort(key=lambda row: str(row.get("dropped_mask_id") or ""))
    return kept, dropped
