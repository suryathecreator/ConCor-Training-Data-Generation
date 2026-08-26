from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from .caption_stage import (
    QwenCaptioner,
    _bucketed_indexed_rows,
    _generation_metrics,
    qwen_model_config,
)
from .dataset import load_records
from .io_utils import append_jsonl, read_jsonl, write_jsonl
from .json_utils import extract_json


def _mock_image_review(record: dict[str, Any]) -> dict[str, Any]:
    parsed = {
        "worth_segmenting": True,
        "estimated_maskable_entities": 24,
        "image_type": "natural_photo",
        "rationale": "Mock review accepts the image with many plausible maskable entities.",
        "reject_reason": "",
        "sam3_prompts": ["people", "clothing", "bags", "chairs", "cups", "signs"],
    }
    return {"raw": json.dumps(parsed), "parsed": parsed}


def _image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _heuristic_reject(record: dict[str, Any], config: dict[str, Any]) -> str:
    review_config = config.get("image_review", {})
    width, height = _image_size(record["image_path"])
    min_side = int(review_config.get("min_side_px", 256))
    min_pixels = int(review_config.get("min_total_pixels", 150000))
    if min(width, height) < min_side:
        return f"image_min_side_below_{min_side}px"
    if width * height < min_pixels:
        return f"image_pixels_below_{min_pixels}"
    return ""


def _normalize(parsed: dict[str, Any], config: dict[str, Any], heuristic_reason: str) -> dict[str, Any]:
    review_config = config.get("image_review", {})
    image_type = str(parsed.get("image_type") or "").strip().lower()
    reject_types = {str(item).strip().lower() for item in review_config.get("reject_image_types", [])}
    rationale_reject_terms = [
        str(item).strip().lower()
        for item in review_config.get("rationale_reject_terms", [])
        if str(item).strip()
    ]
    try:
        estimated = int(parsed.get("estimated_maskable_entities") or parsed.get("estimated_distinct_objects") or 0)
    except Exception:
        estimated = 0
    worth = bool(parsed.get("worth_segmenting", False))
    min_objects = int(review_config.get("min_distinct_objects", 10))
    borderline_max = int(review_config.get("borderline_max_distinct_objects", min_objects + 2))
    reject_reason = str(parsed.get("reject_reason") or "").strip()
    rationale = str(parsed.get("rationale") or "").strip()
    rationale_lower = rationale.lower()
    raw_prompts = parsed.get("sam3_prompts") or []
    prompts: list[str] = []
    if isinstance(raw_prompts, str):
        raw_prompts = [raw_prompts]
    for item in raw_prompts:
        if isinstance(item, dict):
            text = str(item.get("prompt") or item.get("text") or "").strip()
        else:
            text = str(item).strip()
        if text and text not in prompts:
            prompts.append(text[:120])
    if heuristic_reason:
        accepted = False
        reject_reason = heuristic_reason
    elif image_type in reject_types or any(reject_type and reject_type in image_type for reject_type in reject_types):
        accepted = False
        reject_reason = reject_reason or f"reject_image_type:{image_type}"
    elif not worth:
        accepted = False
        reject_reason = reject_reason or "qwen_not_worth_segmenting"
    elif estimated < min_objects:
        accepted = False
        reject_reason = reject_reason or f"estimated_objects_below_{min_objects}"
    elif not prompts:
        accepted = False
        reject_reason = reject_reason or "missing_sam3_prompts"
    elif estimated <= borderline_max and any(term in rationale_lower for term in rationale_reject_terms):
        accepted = False
        reject_reason = reject_reason or "borderline_indistinct_or_background"
    else:
        accepted = True
    return {
        "accepted": accepted,
        "worth_segmenting": worth,
        "estimated_distinct_objects": estimated,
        "estimated_maskable_entities": estimated,
        "image_type": image_type,
        "rationale": rationale,
        "reject_reason": reject_reason,
        "sam3_prompts": prompts if accepted else [],
    }


def _record_seed_index(record: dict[str, Any], fallback: int) -> int:
    try:
        return int((record.get("raw_record") or {}).get("selected_index"))
    except Exception:
        return fallback


def _dedupe_jsonl_by_image_id(path: Path) -> None:
    if not path.exists():
        return
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        image_id = str(row.get("image_id") or "")
        if image_id:
            by_id[image_id] = row
    if len(by_id) != len(rows):
        write_jsonl(by_id.values(), path)


def finalize_image_review_outputs(run_dir: Path) -> dict[str, int]:
    """Atomically enforce one final review row per selected image on resume."""
    selected = read_jsonl(run_dir / "selected_images.jsonl")
    selected_ids = [str(row.get("image_id") or "") for row in selected]
    if not all(selected_ids) or len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError("selected_images.jsonl must contain unique non-empty image IDs")
    selected_set = set(selected_ids)

    def last_by_id(path: Path) -> dict[str, dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return by_id
        for row in read_jsonl(path):
            image_id = str(row.get("image_id") or "")
            if image_id:
                by_id[image_id] = row
        extra = set(by_id) - selected_set
        if extra:
            raise RuntimeError(f"{path.name} contains IDs outside the unit manifest: {sorted(extra)[:5]}")
        return by_id

    reviews_path = run_dir / "image_reviews.jsonl"
    reviews = last_by_id(reviews_path)
    missing = selected_set - set(reviews)
    if missing:
        raise RuntimeError(f"image review is missing {len(missing)} selected image(s): {sorted(missing)[:5]}")
    ordered_reviews = [reviews[image_id] for image_id in selected_ids]
    write_jsonl(ordered_reviews, reviews_path)
    write_jsonl(
        [row for row in ordered_reviews if not bool(row.get("accepted"))],
        run_dir / "initial_rejected_images.jsonl",
    )

    raw_path = run_dir / "image_review_raw.jsonl"
    raw = last_by_id(raw_path)
    if raw_path.exists():
        write_jsonl([raw[image_id] for image_id in selected_ids if image_id in raw], raw_path)
    errors_path = run_dir / "image_review_errors.jsonl"
    errors = last_by_id(errors_path)
    if errors_path.exists():
        write_jsonl(
            [errors[image_id] for image_id in selected_ids if image_id in errors],
            errors_path,
        )
    return {
        "review_count": len(ordered_reviews),
        "accepted_count": sum(bool(row.get("accepted")) for row in ordered_reviews),
        "rejected_count": sum(not bool(row.get("accepted")) for row in ordered_reviews),
        "raw_provenance_count": len(raw),
        "error_count": len(errors),
    }


def run_image_review(
    config: dict[str, Any],
    run_dir: Path,
    mock: bool = False,
    limit: int | None = None,
    captioner_override: QwenCaptioner | None = None,
) -> Path:
    records = load_records(config, limit=limit)
    write_jsonl(records, run_dir / "selected_images.jsonl")
    reviews_path = run_dir / "image_reviews.jsonl"
    raw_path = run_dir / "image_review_raw.jsonl"
    rejected_path = run_dir / "initial_rejected_images.jsonl"
    errors_path = run_dir / "image_review_errors.jsonl"
    review_config = config.get("image_review", {})
    resume = bool(config.get("resume", False) or review_config.get("resume", False))
    if resume:
        _dedupe_jsonl_by_image_id(reviews_path)
        _dedupe_jsonl_by_image_id(rejected_path)
        reviewed_ids = {
            str(row.get("image_id"))
            for row in read_jsonl(reviews_path)
        } if reviews_path.exists() else set()
        records = [record for record in records if str(record["image_id"]) not in reviewed_ids]
    else:
        for stale in (reviews_path, raw_path, rejected_path, errors_path):
            if stale.exists():
                stale.unlink()
    if not records:
        finalize_image_review_outputs(run_dir)
        return reviews_path
    captioner = None if mock else (captioner_override or QwenCaptioner(config, config_section="image_review"))
    seed_base = int(config.get("random_seed", 17)) + int(config.get("image_review", {}).get("seed_offset", 100000))
    prompt = review_config.get("prompt", "")
    batch_size = int(review_config.get("batch_size", 1) or 1)

    def write_result(record: dict[str, Any], result: dict[str, Any], heuristic_reason: str) -> None:
        append_jsonl(
            {
                "image_id": record["image_id"],
                "raw": result["raw"],
                **_generation_metrics(result),
            },
            raw_path,
        )
        parsed_obj = result.get("parsed") or extract_json(result["raw"])
        parsed = _normalize(parsed_obj, config, heuristic_reason)
        out = {
            "image_id": record["image_id"],
            "source_image_path": record["image_path"],
            **parsed,
        }
        append_jsonl(out, reviews_path)
        if not out["accepted"]:
            append_jsonl(out, rejected_path)

    def write_error(record: dict[str, Any], exc: BaseException) -> None:
        out = {
            "image_id": record.get("image_id"),
            "source_image_path": record.get("image_path"),
            "accepted": False,
            "worth_segmenting": False,
            "estimated_distinct_objects": 0,
            "image_type": "review_error",
            "rationale": "",
            "reject_reason": f"image_review_error: {exc!r}",
        }
        append_jsonl(out, reviews_path)
        append_jsonl(out, rejected_path)
        append_jsonl(
            {
                "image_id": record.get("image_id"),
                "stage": "image_review",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
            errors_path,
        )

    if not mock and batch_size > 1:
        bucketed_records = _bucketed_indexed_rows(records, ["image_path"])
        for batch_start in tqdm(range(0, len(bucketed_records), batch_size), desc="image-review"):
            batch_records = bucketed_records[batch_start : batch_start + batch_size]
            pending: list[tuple[int, dict[str, Any]]] = []
            image_sets: list[list[str]] = []
            prompts: list[str] = []
            seeds: list[int] = []
            for idx, record in batch_records:
                try:
                    heuristic_reason = _heuristic_reject(record, config)
                    if heuristic_reason:
                        result = {
                            "raw": json.dumps(
                                {
                                    "worth_segmenting": False,
                                    "estimated_distinct_objects": 0,
                                    "image_type": "heuristic_reject",
                                    "rationale": heuristic_reason,
                                    "reject_reason": heuristic_reason,
                                }
                            )
                        }
                        write_result(record, result, heuristic_reason)
                    else:
                        pending.append((idx, record))
                        image_sets.append([record["image_path"]])
                        prompts.append(prompt)
                        seeds.append(seed_base + _record_seed_index(record, idx))
                except Exception as exc:
                    write_error(record, exc)
                    if not config.get("continue_on_error", True):
                        raise
            if not pending:
                continue
            try:
                results = captioner.generate_many(
                    image_sets,
                    prompts,
                    seeds,
                    batch_size=batch_size,
                    generation_config=qwen_model_config(config, "image_review"),
                )
                for (_, record), result in zip(pending, results):
                    write_result(record, result, "")
            except Exception:
                for idx, record in pending:
                    try:
                        result = captioner.generate(
                            [record["image_path"]],
                            prompt,
                            seed_base + _record_seed_index(record, idx),
                            generation_config=qwen_model_config(config, "image_review"),
                        )
                        write_result(record, result, "")
                    except Exception as exc:
                        write_error(record, exc)
                        if not config.get("continue_on_error", True):
                            raise
        finalize_image_review_outputs(run_dir)
        return reviews_path

    for idx, record in enumerate(tqdm(records, desc="image-review")):
        try:
            heuristic_reason = _heuristic_reject(record, config)
            if heuristic_reason:
                result = {
                    "raw": json.dumps(
                        {
                            "worth_segmenting": False,
                            "estimated_distinct_objects": 0,
                            "image_type": "heuristic_reject",
                            "rationale": heuristic_reason,
                            "reject_reason": heuristic_reason,
                        }
                    )
                }
            else:
                result = _mock_image_review(record) if mock else captioner.generate(
                    [record["image_path"]],
                    prompt,
                    seed_base + _record_seed_index(record, idx),
                    generation_config=qwen_model_config(config, "image_review"),
                )
            write_result(record, result, heuristic_reason)
        except Exception as exc:
            write_error(record, exc)
            if not config.get("continue_on_error", True):
                raise
    finalize_image_review_outputs(run_dir)
    return reviews_path
