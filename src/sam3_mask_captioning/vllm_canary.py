from __future__ import annotations

import json
import platform
import re
import time
from pathlib import Path
from typing import Any

from .caption_stage import _fill_mask_prompt, qwen_model_config
from .correspondence_stage import (
    bcc_generation_config,
    build_caption_image_packet,
    build_caption_prompt,
    normalize_correspondence,
)
from .io_utils import read_jsonl, write_json
from .json_utils import extract_json
from .vllm_backend import VLLMCaptioner, _STAGE_SCHEMAS


def _group_by_image(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("image_id") or ""), []).append(row)
    return list(grouped.values())


def _not_gibberish(raw: str) -> bool:
    if not raw.strip() or len(raw) < 8:
        return False
    printable = sum(character.isprintable() for character in raw) / len(raw)
    tokens = re.findall(r"\w+", raw.casefold())
    repeated = max((tokens.count(token) for token in set(tokens)), default=0)
    return printable >= 0.98 and repeated < max(20, len(tokens) * 0.55)


def _request(
    captioner: VLLMCaptioner,
    *,
    name: str,
    images: list[str],
    prompt: str,
    runtime: dict[str, Any],
    seed: int,
    validator: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = captioner.generate(
        images,
        prompt,
        seed,
        generation_config=runtime,
    )
    raw = str(result.get("raw") or "")
    parsed = extract_json(raw)
    validator(parsed)
    if not _not_gibberish(raw):
        raise RuntimeError(f"{name} produced empty, repetitive, or non-printable output")
    return {
        "name": name,
        "passed": True,
        "elapsed_seconds": time.perf_counter() - started,
        "raw": raw,
        "parsed": parsed,
        "metrics": {key: value for key, value in result.items() if key != "raw"},
    }


def _require_keys(*keys: str):
    def validate(parsed: dict[str, Any]) -> None:
        missing = [key for key in keys if key not in parsed]
        if missing:
            raise ValueError(f"Missing required canary keys: {missing}")

    return validate


def run_vllm_canary(
    config: dict[str, Any],
    fixture_run: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    fixture_run = Path(fixture_run).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    selected = read_jsonl(fixture_run / "selected_images.jsonl")
    masks = read_jsonl(fixture_run / "sam3_masks.jsonl")
    captions = read_jsonl(fixture_run / "captions.jsonl")
    consistent = read_jsonl(fixture_run / "consistent_captions.jsonl")
    if not selected or not masks or not captions or not consistent:
        raise RuntimeError(f"Canary fixture is incomplete: {fixture_run}")
    source_by_id = {
        str(row.get("image_id") or ""): str(row.get("image_path") or "")
        for row in selected
    }
    if not source_by_id[next(iter(source_by_id))]:
        reviews = read_jsonl(fixture_run / "image_reviews.jsonl")
        source_by_id.update(
            {
                str(row.get("image_id") or ""): str(row.get("source_image_path") or "")
                for row in reviews
            }
        )
    caption_config = config.get("image_caption", {})
    minimum = int(
        caption_config.get("min_input_masks", caption_config.get("min_groups", 10))
    )
    raw_groups = [rows for rows in _group_by_image(consistent) if len(rows) >= minimum]
    raw_groups.sort(key=len)
    if not raw_groups:
        raise RuntimeError("Canary fixture has no BCC-ready image")

    middle = len(raw_groups) // 2
    # This canary gates the inference engine, multimodal packet, schema, and
    # normalization—not canonicalization, which has dedicated tests. Avoid a
    # shared-filesystem scan of thousands of unrelated legacy mask PNGs.
    standard = raw_groups[middle]
    bcc_groups = [standard]
    standard_id = str(standard[0].get("image_id") or "")
    for largest in reversed(raw_groups):
        if str(largest[0].get("image_id") or "") != standard_id:
            bcc_groups.append(largest)
            break

    captioner = VLLMCaptioner(config, config_section="image_caption")
    checks: list[dict[str, Any]] = []
    seed = int(config.get("random_seed", 17)) + 900000

    review_row = selected[0]
    review_image = source_by_id[str(review_row.get("image_id") or "")]
    review_runtime = qwen_model_config(config, "image_review")
    review_runtime["json_schema"] = _STAGE_SCHEMAS["image_review"]
    checks.append(
        _request(
            captioner,
            name="image_review",
            images=[review_image],
            prompt=str(config.get("image_review", {}).get("prompt") or ""),
            runtime=review_runtime,
            seed=seed,
            validator=_require_keys("worth_segmenting", "sam3_prompts"),
        )
    )

    mask_rows = [row for row in masks if Path(str(row.get("inverse_crop_path") or "")).exists()][:8]
    caption_runtime = qwen_model_config(config, "caption")
    caption_runtime["json_schema"] = _STAGE_SCHEMAS["caption"]
    caption_results = captioner.generate_many(
        [[str(row["inverse_crop_path"])] for row in mask_rows],
        [_fill_mask_prompt(str(config["caption"]["prompt"]), row) for row in mask_rows],
        [seed + 1 + index for index in range(len(mask_rows))],
        batch_size=len(mask_rows),
        generation_config=caption_runtime,
    )
    for result in caption_results:
        parsed = extract_json(str(result.get("raw") or ""))
        _require_keys("reject", "object", "caption", "attributes")(parsed)
        if not _not_gibberish(str(result.get("raw") or "")):
            raise RuntimeError("mask_caption batch produced gibberish")
    checks.append(
        {
            "name": "mask_caption_batch",
            "passed": True,
            "batch_size": len(caption_results),
            "metrics": [{key: value for key, value in row.items() if key != "raw"} for row in caption_results],
        }
    )

    qa_row = captions[0]
    qa_runtime = qwen_model_config(config, "quality_filter")
    qa_runtime["json_schema"] = _STAGE_SCHEMAS["quality_filter"]
    checks.append(
        _request(
            captioner,
            name="mask_description_qa",
            images=[str(qa_row["inverse_crop_path"])],
            prompt=_fill_mask_prompt(str(config["quality_filter"]["mask_review_prompt"]), qa_row),
            runtime=qa_runtime,
            seed=seed + 20,
            validator=_require_keys("keep", "reason", "failure_modes"),
        )
    )

    for index, rows in enumerate(bcc_groups):
        image_id = str(rows[0]["image_id"])
        overlay = fixture_run / "correspondence_overlays" / f"{image_id}.png"
        if not overlay.exists():
            from .correspondence_stage import write_correspondence_overlay

            overlay = write_correspondence_overlay(
                rows[0]["source_image_path"], rows, output.parent / f"canary-{image_id}-overlay.png"
            )
        packet, manifest = build_caption_image_packet(rows, overlay)
        # Exercise the exact production output budget. A dense fixture can
        # require several thousand tokens for balanced inline tags; the base
        # 1,024-token default is only the floor and would truncate otherwise
        # valid structured JSON in the stress case.
        runtime = bcc_generation_config(config, "image_caption", len(rows))
        runtime["json_schema"] = _STAGE_SCHEMAS["image_caption"]

        def validate_bcc(parsed: dict[str, Any], bcc_rows: list[dict[str, Any]] = rows) -> None:
            _require_keys("reject", "tagged_caption")(parsed)
            normalized, _ = normalize_correspondence(
                parsed, bcc_rows, min_groups=1, require_all_masks=False
            )
            if not normalized.get("caption"):
                raise ValueError("BCC canary returned no usable caption")

        checks.append(
            _request(
                captioner,
                name="bcc_standard" if index == 0 else "bcc_largest_fixture",
                images=packet,
                prompt=build_caption_prompt(rows, manifest),
                runtime=runtime,
                seed=seed + 30 + index,
                validator=validate_bcc,
            )
        )

    import torch
    import vllm

    report = {
        "passed": True,
        "fixture_run": str(fixture_run),
        "model": str(config.get("image_caption", {}).get("model_name")),
        "vllm_version": str(vllm.__version__),
        "required_vllm_version": str((config.get("inference") or {}).get("vllm_version")),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "checks": checks,
        "completed_at": time.time(),
    }
    if report["vllm_version"] != report["required_vllm_version"]:
        raise RuntimeError(
            f"vLLM version mismatch: {report['vllm_version']} != {report['required_vllm_version']}"
        )
    write_json(report, output)
    return report
