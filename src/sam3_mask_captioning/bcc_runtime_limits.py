from __future__ import annotations

import re
from typing import Any


BCC_INPUT_TOO_LARGE_REASON = "bcc_input_too_large"

_INPUT_LIMIT_PATTERNS = (
    re.compile(r"at most\s+\d+\s+image\(s\) may be provided", re.IGNORECASE),
    re.compile(r"maximum context length", re.IGNORECASE),
    re.compile(r"longer than the maximum model length", re.IGNORECASE),
)


def bcc_input_limit_error(exc: BaseException) -> str | None:
    """Return a stable diagnostic for per-prompt multimodal size failures.

    Only request-size validation errors are classified here. OOMs, model-load
    failures, and other runtime errors must still fail loudly so infrastructure
    problems are never mislabeled as bad examples.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    messages: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.extend((str(current), repr(current)))
        current = current.__cause__ or current.__context__
    combined = "\n".join(messages)
    if any(pattern.search(combined) for pattern in _INPUT_LIMIT_PATTERNS):
        return str(exc)
    return None


def generate_many_bcc_with_input_isolation(
    captioner: Any,
    items: list[dict[str, Any]],
    *,
    generation_config: dict[str, Any],
    max_images_per_prompt: int = 128,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Generate a batch while isolating only oversized individual prompts.

    vLLM validates a batch before generation, so one oversized packet used to
    quarantine the entire campaign unit. Obvious image-count violations are
    skipped up front. If context validation rejects a mixed batch, each item is
    retried alone: valid neighbors complete, while only the offending prompt is
    returned in ``skipped``.
    """
    results: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    pending: list[dict[str, Any]] = []
    limit = max(1, int(max_images_per_prompt or 128))
    for item in items:
        image_id = str(item["image_id"])
        image_count = len(item.get("packet") or [])
        if image_count > limit:
            skipped[image_id] = (
                f"visual packet has {image_count} images; configured maximum is {limit}"
            )
        else:
            pending.append(item)
    if not pending:
        return results, skipped

    def generate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return captioner.generate_many_bcc(
            [item["packet"] for item in batch],
            [item["prompt"] for item in batch],
            [item["seed"] for item in batch],
            batch_size=len(batch),
            generation_config=generation_config,
        )

    try:
        generated = generate(pending)
    except Exception as batch_exc:
        if bcc_input_limit_error(batch_exc) is None:
            raise
        for item in pending:
            image_id = str(item["image_id"])
            try:
                value = generate([item])
            except Exception as item_exc:
                diagnostic = bcc_input_limit_error(item_exc)
                if diagnostic is None:
                    raise
                skipped[image_id] = diagnostic
            else:
                if len(value) != 1:
                    raise RuntimeError(
                        f"BCC backend returned {len(value)} outputs for one prompt"
                    )
                results[image_id] = value[0]
        return results, skipped

    if len(generated) != len(pending):
        raise RuntimeError(
            f"BCC backend returned {len(generated)} outputs for {len(pending)} prompts"
        )
    results.update(
        (str(item["image_id"]), result)
        for item, result in zip(pending, generated)
    )
    return results, skipped


def bcc_input_skip_record(
    item: dict[str, Any],
    *,
    stage: str,
    diagnostic: str,
    contract_version: str,
) -> dict[str, Any]:
    return {
        "image_id": str(item.get("image_id") or ""),
        "stage": stage,
        "reason_code": BCC_INPUT_TOO_LARGE_REASON,
        "included": False,
        "mask_count": len(item.get("rows") or []),
        "visual_image_count": len(item.get("packet") or []),
        "estimated_visual_tokens": int(item.get("visual_tokens") or 0),
        "diagnostic": diagnostic,
        "contract_version": contract_version,
    }
