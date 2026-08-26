from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from .bcc_contract import ONE_REWRITE_CONTRACT_VERSION
from .bcc_audit import (
    CHECKER_AUDIT_VERSION,
    compare_issue_sets,
    composite_statistics,
    issue_records,
    quality_tier,
)
from .caption_stage import QwenCaptioner, _generation_metrics, create_captioner
from .correspondence_stage import (
    BCC_PROMPT_VERSION,
    CORRESPONDENCE_SCHEMA_VERSION,
    PIPELINE_STAGE_VERSION,
    _bcc_packet_batches,
    _decision_errors,
    _enrich_groups,
    _estimate_bcc_visual_tokens,
    _is_current_correspondence_record,
    _mock_record,
    _compact_input_manifest,
    _model_mask_context,
    bcc_authoritative_rulebook,
    bcc_generation_config,
    build_caption_image_packet,
    build_caption_prompt,
    normalize_correspondence,
    write_correspondence_overlay,
)
from .io_utils import append_jsonl, read_jsonl_indexed
from .json_utils import extract_json
from .visual_audit_stage import simple_rewrite_findings


_MALFORMED_INLINE_TAG_ERRORS = (
    "has an unclosed opening tag",
    "has a closing tag without an opening tag",
    "inline tags are crossed",
    "has nested self-overlap",
    "has an empty inline mention",
)

FINAL_NORMALIZATION_VERSION = "bcc-final-normalization-v5-wordnet-min10-2026-08-25"


def _malformed_inline_tag_errors(errors: list[str]) -> list[str]:
    """Return structural tag failures that cannot be used as BCC supervision."""
    return [
        error
        for error in errors
        if any(fragment in error for fragment in _MALFORMED_INLINE_TAG_ERRORS)
    ]


def _check_raw(
    raw: str,
    rows: list[dict[str, Any]],
    *,
    draft: bool,
) -> dict[str, Any]:
    try:
        parsed = extract_json(raw)
    except Exception as exc:
        return {
            "parseable": False,
            "parse_error": repr(exc),
            "parsed": None,
            "normalized": None,
            "errors": [f"JSON parse error: {exc!r}"],
            "issues": [],
        }
    normalized, errors = normalize_correspondence(
        parsed,
        rows,
        min_groups=1,
        require_all_masks=False,
        retain_semantically_invalid_groups=True,
    )
    # Normalization deliberately drops unknown IDs so they can never become
    # training correspondences. Surface that repair as a fatal audit finding
    # for this permissive one-rewrite contract instead of silently losing the
    # evidence that the model hallucinated a reference.
    for repair in normalized.get("link_repairs") or []:
        if repair.get("reason") == "unknown_extra_link_dropped":
            errors.append(
                "unknown link/reference dropped from model output: "
                f"{repair.get('provided_reference')!r}"
            )
    if draft:
        errors = _decision_errors(parsed, qa=False) + errors
    elif parsed.get("reject") is not False:
        errors = ["model decision reject must be false after the one rewrite"] + errors
    issues = issue_records(errors, rows)
    malformed_tags = _malformed_inline_tag_errors(errors)
    if malformed_tags:
        return {
            "parseable": False,
            "parse_error": "malformed inline correspondence tags: "
            + "; ".join(malformed_tags),
            "parsed": parsed,
            "normalized": normalized,
            "errors": errors,
            "issues": issues,
        }
    return {
        "parseable": True,
        "parse_error": "",
        "parsed": parsed,
        "normalized": normalized,
        "errors": errors,
        "issues": issues,
    }


def _draft_record(
    item: dict[str, Any],
    raw: str,
    checked: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "image_id": item["image_id"],
        "source_image_path": item["source_path"],
        "correspondence_overlay_path": item["overlay_path"],
        "bcc_input_manifest": item["manifest"],
        "prompt_version": BCC_PROMPT_VERSION,
        "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
        "stage_version": PIPELINE_STAGE_VERSION,
        "contract_version": ONE_REWRITE_CONTRACT_VERSION,
        "model": item["model"],
        "pass": 1,
        "draft_raw": raw,
        "draft_parseable": bool(checked["parseable"]),
        "draft_parse_error": checked["parse_error"],
        "draft_validation_errors": list(checked["errors"]),
        "validation": {
            "before_rewrite": {
                "parseable": bool(checked["parseable"]),
                "parse_error": checked["parse_error"],
                "issues": checked["issues"],
                "checker_version": CHECKER_AUDIT_VERSION,
            }
        },
        "backend_provenance": {
            "backend": generation.get("backend", "transformers"),
            "model": item["model"],
            "prompt_version": BCC_PROMPT_VERSION,
            "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
            "checker_version": CHECKER_AUDIT_VERSION,
        },
    }
    if checked["normalized"] is not None:
        base.update(checked["normalized"])
        return _enrich_groups(base, item["rows"])
    base.update({"caption": "", "groups": []})
    return base


def refinalize_bcc_draft_record(
    candidate: dict[str, Any],
    raw_record: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Re-normalize one durable draft response without loading model or images."""
    raw = str(raw_record.get("raw") or candidate.get("draft_raw") or "")
    item = {
        "image_id": str(candidate.get("image_id") or rows[0].get("image_id") or ""),
        "source_path": str(
            candidate.get("source_image_path")
            or rows[0].get("source_image_path")
            or ""
        ),
        "overlay_path": str(candidate.get("correspondence_overlay_path") or ""),
        "manifest": list(candidate.get("bcc_input_manifest") or []),
        "model": str(candidate.get("model") or "Qwen/Qwen3.8-27B"),
        "rows": rows,
    }
    return _draft_record(
        item,
        raw,
        _check_raw(raw, rows, draft=True),
        raw_record,
    )


def run_bcc_draft_batch(
    config: dict[str, Any],
    run_dir: str | Path,
    row_groups: list[list[dict[str, Any]]],
    *,
    captioner: QwenCaptioner | None = None,
    mock: bool = False,
) -> Path:
    """Generate one selective-link visual BCC draft for every eligible image."""
    run_dir = Path(run_dir)
    candidate_path = run_dir / "image_caption_candidates.jsonl"
    raw_path = run_dir / "image_caption_raw.jsonl"
    excluded_path = run_dir / "bcc_exclusions.jsonl"
    stage = config.get("image_caption", {})
    min_input_masks = int(stage.get("min_input_masks", stage.get("min_groups", 10)))
    completed = {
        str(row.get("image_id") or "")
        for row in (read_jsonl_indexed(candidate_path) if candidate_path.exists() else [])
        if _is_current_correspondence_record(row)
        and row.get("contract_version") == ONE_REWRITE_CONTRACT_VERSION
    }
    persisted_drafts = {
        str(row.get("image_id") or ""): row
        for row in (read_jsonl_indexed(raw_path) if raw_path.exists() else [])
        if row.get("contract_version") == ONE_REWRITE_CONTRACT_VERSION
        and int(row.get("attempt") or 0) == 1
        and row.get("image_id")
    }
    prepared: list[dict[str, Any]] = []
    seed_base = int(config.get("random_seed", 17)) + int(stage.get("seed_offset", 300000))
    for position, rows in enumerate(row_groups):
        if not rows:
            continue
        image_id = str(rows[0].get("image_id") or "")
        if image_id in completed:
            continue
        if len(rows) < min_input_masks:
            append_jsonl(
                {
                    "image_id": image_id,
                    "stage": "bcc_draft",
                    "reason_code": "below_minimum_masks_before_bcc",
                    "mask_count": len(rows),
                    "minimum": min_input_masks,
                    "included": False,
                    "contract_version": ONE_REWRITE_CONTRACT_VERSION,
                },
                excluded_path,
            )
            continue
        source_path = str(rows[0]["source_image_path"])
        overlay_path = write_correspondence_overlay(
            source_path,
            rows,
            run_dir / "correspondence_overlays" / f"{image_id}.png",
        )
        packet, manifest = build_caption_image_packet(rows, overlay_path)
        prepared.append(
            {
                "image_id": image_id,
                "rows": rows,
                "source_path": source_path,
                "overlay_path": str(overlay_path),
                "packet": packet,
                "manifest": manifest,
                "prompt": build_caption_prompt(rows, manifest),
                "seed": seed_base + position,
                "visual_tokens": _estimate_bcc_visual_tokens(packet),
                "model": str(stage.get("model_name", "Qwen/Qwen3.8-27B")),
                "persisted_draft": persisted_drafts.get(image_id),
            }
        )
    if not prepared:
        return candidate_path
    if not mock and captioner is None:
        captioner = create_captioner(config, "image_caption")
    for batch in _bcc_packet_batches(prepared, stage):
        pending = [item for item in batch if item["persisted_draft"] is None]
        if mock:
            generated = [
                {
                    "raw": json.dumps(_mock_record(item["rows"], min_input_masks)),
                    "backend": "mock",
                }
                for item in pending
            ]
        elif pending:
            generation_config = bcc_generation_config(
                config, "image_caption", max(len(item["rows"]) for item in pending)
            )
            generated = captioner.generate_many_bcc(
                [item["packet"] for item in pending],
                [item["prompt"] for item in pending],
                [item["seed"] for item in pending],
                batch_size=len(pending),
                generation_config=generation_config,
            )
        else:
            generated = []
        generated_by_image = {
            item["image_id"]: result for item, result in zip(pending, generated)
        }
        for item in batch:
            persisted = item["persisted_draft"]
            result = persisted or generated_by_image[item["image_id"]]
            raw = str(result.get("raw") or "")
            if persisted is None:
                append_jsonl(
                    {
                        "image_id": item["image_id"],
                        "attempt": 1,
                        "raw": raw,
                        **_generation_metrics(result),
                        "backend": result.get("backend", "transformers"),
                        "batched": len(pending) > 1,
                        "estimated_visual_tokens": item["visual_tokens"],
                        "prompt_version": BCC_PROMPT_VERSION,
                        "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                        "stage_version": PIPELINE_STAGE_VERSION,
                        "contract_version": ONE_REWRITE_CONTRACT_VERSION,
                        "bcc_input_manifest": item["manifest"],
                    },
                    raw_path,
                    durable=True,
                )
            checked = _check_raw(raw, item["rows"], draft=True)
            append_jsonl(
                _draft_record(item, raw, checked, result),
                candidate_path,
                durable=True,
            )
    return candidate_path


def _rewrite_prompt(
    candidate: dict[str, Any],
    visual_audit: dict[str, Any],
    rows: list[dict[str, Any]],
    input_manifest: list[dict[str, Any]],
) -> str:
    audit_payload = {
        key: visual_audit.get(key)
        for key in (
            "audit_parseable",
            "audit_pass",
            "task_accuracy_percent",
            "summary",
            "issues",
            "mask_decisions",
            "rewrite_plan",
        )
    }
    return (
        """# Task: final full-visual BCC rewrite

This is the single rewrite after an initial caption and an independent model audit. Inspect the complete visual
packet again: IMAGE 1 is authoritative; IMAGE 2 maps one-based IDs to accepted masks; IMAGE 3 onward isolates each
mask on a synthetic exterior that is never scene content. The masks are imperfect proposals, so select the subset
that produces the most correct, informative, and natural caption; try to use more accepted masks when they fit fluent
relations, collective spans, coordination, or coreference, but drop a dubious or awkward mask instead of forcing it.

Use the MODEL_VISUAL_AUDIT as evidence and an actionable editing plan, not as unquestionable truth. Apply the
SIMPLE_DETERMINISTIC_FINDINGS, which are limited to parse/tag structure, direct occurrence coverage, obvious
possessive/punctuation faults, and likely unmasked nouns. Do not import subjective findings from the historical
heuristic scorer. Independently verify every change against the images and rulebook.

The largest error is leaving a concrete noun phrase or referring expression in the prose without a correct mask
link. For each such phrase, either wrap the complete noun phrase in the compatible supplied ID tag(s), or remove the
phrase and any relation that depends on it. Never attach an unsupported jacket/object/part to a nearby person or body
part just to preserve wording; for example, remove `a black jacket` when no jacket mask exists. Every later pronoun,
possessive, repeated noun phrase, or reflexive for a used entity
must be linked to the same group.

Rewrite freely enough to make the tag-stripped caption read like polished, connected prose. Remove ordinal mask
catalogs, forced per-mask sentences, fragments, fused words, repeated possessives, and stray periods/commas. Normal
sentence punctuation and spaces must remain when tags are removed. Shared plural spans can link several masks with
perfectly nested same-boundary tags. Do not target a numeric link quota: use only the accurately grounded masks that
fit truthful, connected, natural prose.

Return only one compact JSON object in exactly this shape:
{"reject":false,"tagged_caption":"[1]A person[/1] ..."}

"""
        + bcc_authoritative_rulebook()
        + "\nORDERED_INPUT_IMAGES:\n"
        + json.dumps(
            _compact_input_manifest(input_manifest),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\nACCEPTED_MASK_CONTEXT:\n"
        + json.dumps(
            _model_mask_context(rows), ensure_ascii=False, separators=(",", ":")
        )
        + "\nINITIAL_MODEL_ANSWER:\n"
        + str(candidate.get("draft_raw") or "")[:24000]
        + "\nMODEL_VISUAL_AUDIT:\n"
        + json.dumps(audit_payload, ensure_ascii=False, separators=(",", ":"))
        + "\nSIMPLE_DETERMINISTIC_FINDINGS:\n"
        + json.dumps(
            simple_rewrite_findings(candidate, rows),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\nFINAL_CHECK: remove tags mentally; the result must be natural prose with ordinary punctuation, and every concrete reference that remains must be fully linked to its true supplied mask ID."
    )


def _omitted_masks(
    rows: list[dict[str, Any]], final_groups: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    mentioned = {str(group.get("mask_id") or "") for group in final_groups}
    return [
        {
            "mask_id": str(row.get("mask_id") or ""),
            "overlay_number": index + 1,
            "reason": "not_naturally_incorporated_by_model",
            "fatal": False,
            "status": "available_but_unlinked",
            "main_candidate": str(row.get("main_candidate") or ""),
            "object": str(row.get("object") or ""),
            "source_sam3_prompt": str(row.get("source_prompt") or ""),
            "mask_caption": str(row.get("caption") or ""),
            "mask_attributes": list(row.get("attributes") or []),
            "inverse_background_rgb": row.get("inverse_background_rgb"),
            "mask_path": row.get("mask_path"),
            "inverse_crop_path": row.get("inverse_crop_path"),
        }
        for index, row in enumerate(rows)
        if str(row.get("mask_id") or "") not in mentioned
    ]


def run_bcc_one_rewrite_batch(
    config: dict[str, Any],
    run_dir: str | Path,
    row_groups: list[list[dict[str, Any]]],
    *,
    captioner: QwenCaptioner | None = None,
    mock: bool = False,
) -> Path:
    """Rewrite every visually audited draft once, then enforce hard output gates."""
    run_dir = Path(run_dir)
    candidate_path = run_dir / "image_caption_candidates.jsonl"
    visual_audit_path = run_dir / "bcc_visual_audits.jsonl"
    raw_path = run_dir / "image_caption_qa_raw.jsonl"
    final_path = run_dir / "image_text_pairs.jsonl"
    audit_path = run_dir / "bcc_validation_audit.jsonl"
    excluded_path = run_dir / "bcc_exclusions.jsonl"
    candidates = {
        str(row.get("image_id") or ""): row
        for row in (read_jsonl_indexed(candidate_path) if candidate_path.exists() else [])
        if _is_current_correspondence_record(row)
        and row.get("contract_version") == ONE_REWRITE_CONTRACT_VERSION
    }
    visual_audits = {
        str(row.get("image_id") or ""): row
        for row in (
            read_jsonl_indexed(visual_audit_path)
            if visual_audit_path.exists()
            else []
        )
        if row.get("contract_version") == ONE_REWRITE_CONTRACT_VERSION
    }
    require_visual_audit = bool(
        config.get("image_caption_audit", {}).get("enabled", False)
    )
    persisted_rewrites = {
        str(row.get("image_id") or ""): row
        for row in (read_jsonl_indexed(raw_path) if raw_path.exists() else [])
        if row.get("contract_version") == ONE_REWRITE_CONTRACT_VERSION
        and int(row.get("rewrite_attempt") or 0) == 1
        and row.get("image_id")
    }
    final_ids = {
        str(row.get("image_id") or "")
        for row in (read_jsonl_indexed(final_path) if final_path.exists() else [])
        if row.get("contract_version") == ONE_REWRITE_CONTRACT_VERSION
    }
    audit_by_image = {
        str(row.get("image_id") or ""): row
        for row in (read_jsonl_indexed(audit_path) if audit_path.exists() else [])
        if row.get("contract_version") == ONE_REWRITE_CONTRACT_VERSION
        and row.get("image_id")
    }
    exclusion_ids = {
        str(row.get("image_id") or "")
        for row in (read_jsonl_indexed(excluded_path) if excluded_path.exists() else [])
        if row.get("contract_version") == ONE_REWRITE_CONTRACT_VERSION
    }
    # An audit is the commit record for a completed rewrite. If a worker was
    # preempted between writing it and its paired included/excluded stream,
    # reconstruct the missing side deterministically without another model
    # call. This also repairs an interrupted prior resume.
    for image_id, audit in audit_by_image.items():
        if audit.get("included"):
            if image_id not in final_ids:
                append_jsonl(audit, final_path, durable=True)
                final_ids.add(image_id)
        elif image_id not in exclusion_ids:
            reason = str(
                audit.get("reason_code")
                or audit.get("exclusion_reason")
                or "final_rewrite_excluded"
            )
            append_jsonl({**audit, "reason_code": reason}, excluded_path, durable=True)
            exclusion_ids.add(image_id)
    completed = set(audit_by_image)
    stage = config.get("image_caption_qa", {})
    min_input_masks = int(
        config.get("image_caption", {}).get(
            "min_input_masks", config.get("image_caption", {}).get("min_groups", 10)
        )
    )
    min_linked_masks = max(
        10, int(stage.get("min_linked_masks_after_caption", 10))
    )
    seed_base = int(config.get("random_seed", 17)) + int(stage.get("seed_offset", 400000))
    prepared: list[dict[str, Any]] = []
    for position, rows in enumerate(row_groups):
        if not rows:
            continue
        image_id = str(rows[0].get("image_id") or "")
        candidate = candidates.get(image_id)
        visual_audit = visual_audits.get(image_id)
        if visual_audit is None and not require_visual_audit:
            visual_audit = {
                "audit_version": "legacy-direct-rewrite-without-middle-audit",
                "audit_parseable": True,
                "audit_parse_error": "",
                "audit_pass": False,
                "task_accuracy_percent": 0.0,
                "summary": "Legacy direct-call compatibility path.",
                "issues": [],
                "mask_decisions": [],
                "rewrite_plan": "Independently verify and rewrite the draft.",
            }
        if candidate is None or visual_audit is None or image_id in completed:
            continue
        persisted_rewrite = persisted_rewrites.get(image_id)
        if persisted_rewrite is None:
            packet, manifest = build_caption_image_packet(
                rows, str(candidate.get("correspondence_overlay_path") or "")
            )
            visual_tokens = _estimate_bcc_visual_tokens(packet)
        else:
            # A durable raw response needs no second model call or rehydrated
            # visuals; it is finalized deterministically on resume.
            packet = []
            manifest = list(candidate.get("bcc_input_manifest") or [])
            visual_tokens = 0
        prepared.append(
            {
                "image_id": image_id,
                "rows": rows,
                "candidate": candidate,
                "visual_audit": visual_audit,
                "packet": packet,
                "manifest": manifest,
                "visual_tokens": visual_tokens,
                "prompt": _rewrite_prompt(
                    candidate, visual_audit, rows, manifest
                ),
                "seed": seed_base + position,
                # If a worker died after fsyncing the model response but before
                # writing the audit, finish deterministically from that response.
                # Never spend a second rewrite call for the same contract.
                "persisted_rewrite": persisted_rewrite,
            }
        )
    if not prepared:
        return final_path
    # A durable raw rewrite is sufficient for deterministic re-finalization.
    # Do not load a 27B model merely to normalize/check already generated
    # responses (for example after a postprocessor bug fix).
    if (
        not mock
        and captioner is None
        and any(item["persisted_rewrite"] is None for item in prepared)
    ):
        captioner = create_captioner(config, "image_caption_qa")
    for batch in _bcc_packet_batches(prepared, stage):
        pending = [item for item in batch if item["persisted_rewrite"] is None]
        if mock:
            generated = [
                {
                    "raw": json.dumps(_mock_record(item["rows"], min_input_masks)),
                    "backend": "mock",
                }
                for item in pending
            ]
        elif pending:
            runtime = bcc_generation_config(
                config,
                "image_caption_qa",
                max(len(item["rows"]) for item in pending),
                text_only_repair=False,
            )
            generated = captioner.generate_many_bcc(
                [item["packet"] for item in pending],
                [item["prompt"] for item in pending],
                [item["seed"] for item in pending],
                batch_size=len(pending),
                generation_config=runtime,
            )
        else:
            generated = []
        generated_by_image = {
            item["image_id"]: result for item, result in zip(pending, generated)
        }
        for item in batch:
            persisted = item["persisted_rewrite"]
            result = persisted or generated_by_image[item["image_id"]]
            raw = str(result.get("raw") or "")
            if persisted is None:
                append_jsonl(
                    {
                        "image_id": item["image_id"],
                        "rewrite_attempt": 1,
                        "raw": raw,
                        **_generation_metrics(result),
                        "backend": result.get("backend", "transformers"),
                        "text_only_rewrite": False,
                        "visual_rewrite": True,
                        "batched": len(pending) > 1,
                        "estimated_visual_tokens": item["visual_tokens"],
                        "prompt_version": BCC_PROMPT_VERSION,
                        "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                        "stage_version": PIPELINE_STAGE_VERSION,
                        "contract_version": ONE_REWRITE_CONTRACT_VERSION,
                        "bcc_input_manifest": item["manifest"],
                    },
                    raw_path,
                    durable=True,
                )
            candidate = item["candidate"]
            before_validation = (candidate.get("validation") or {}).get("before_rewrite") or {}
            before_issues = list(before_validation.get("issues") or [])
            checked = _check_raw(raw, item["rows"], draft=False)
            metrics = compare_issue_sets(
                before_issues,
                checked["issues"],
                before_parseable=bool(before_validation.get("parseable")),
                after_parseable=bool(checked["parseable"]),
            )
            audit_base = {
                "image_id": item["image_id"],
                "final_normalization_version": FINAL_NORMALIZATION_VERSION,
                "contract_version": ONE_REWRITE_CONTRACT_VERSION,
                "prompt_version": BCC_PROMPT_VERSION,
                "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                "stage_version": PIPELINE_STAGE_VERSION,
                "validation": {
                    "before_rewrite": before_validation,
                    "after_rewrite": {
                        "parseable": bool(checked["parseable"]),
                        "parse_error": checked["parse_error"],
                        "issues": checked["issues"],
                        "checker_version": CHECKER_AUDIT_VERSION,
                    },
                },
                "rewrite_metrics": metrics,
                "model_visual_audit": {
                    key: item["visual_audit"].get(key)
                    for key in (
                        "audit_version",
                        "audit_parseable",
                        "audit_parse_error",
                        "audit_pass",
                        "task_accuracy_percent",
                        "summary",
                        "issues",
                        "mask_decisions",
                        "rewrite_plan",
                    )
                },
                "simple_deterministic_findings": simple_rewrite_findings(
                    candidate, item["rows"]
                ),
            }
            if not checked["parseable"] or checked["normalized"] is None:
                exclusion = {
                    **audit_base,
                    "included": False,
                    "reason_code": "final_rewrite_unparseable",
                    "raw": raw,
                }
                append_jsonl(exclusion, audit_path, durable=True)
                append_jsonl(exclusion, excluded_path, durable=True)
                audit_by_image[item["image_id"]] = exclusion
                exclusion_ids.add(item["image_id"])
                continue
            normalized = checked["normalized"]
            enriched = _enrich_groups(normalized, item["rows"])
            final_groups = list(enriched.get("groups") or [])
            omitted = _omitted_masks(item["rows"], final_groups)
            first_pass_omitted = _omitted_masks(
                item["rows"], list(candidate.get("groups") or [])
            )
            tier = quality_tier(checked["issues"])
            composite = composite_statistics(item["rows"], final_groups)
            included = len(final_groups) >= min_linked_masks
            record = {
                "image_id": item["image_id"],
                "source_image_path": candidate.get("source_image_path"),
                "correspondence_overlay_path": candidate.get("correspondence_overlay_path"),
                "bcc_input_manifest": item["manifest"],
                "prompt_version": BCC_PROMPT_VERSION,
                "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                "stage_version": PIPELINE_STAGE_VERSION,
                "contract_version": ONE_REWRITE_CONTRACT_VERSION,
                "model": str(stage.get("model_name", "Qwen/Qwen3.8-27B")),
                "pass": 2,
                "rewrite_count": 1,
                "first_pass_raw": candidate.get("draft_raw", ""),
                "rewrite_raw": raw,
                "first_pass_caption": candidate.get("caption", ""),
                "first_pass_groups": candidate.get("groups", []),
                "first_pass_omitted_masks": first_pass_omitted,
                "quality_tier": tier,
                "included": included,
                "omitted_masks": omitted,
                "coverage_policy": {
                    "minimum_masks_before_caption": min_input_masks,
                    "minimum_linked_masks_after_caption": min_linked_masks,
                    "all_input_masks_required_after_caption": False,
                    "collective_span_may_link_multiple_masks": True,
                },
                "composite_statistics": composite,
                "backend_provenance": {
                    **(candidate.get("backend_provenance") or {}),
                    "rewrite_backend": result.get("backend", "transformers"),
                    "rewrite_model": str(stage.get("model_name", "Qwen/Qwen3.8-27B")),
                    "audit_model": str(
                        item["visual_audit"].get("model") or "Qwen/Qwen3.8-27B"
                    ),
                    "audit_visual_context": "original+numbered_overlay+all_inverse_crops",
                    "rewrite_visual_context": "original+numbered_overlay+all_inverse_crops",
                    "contract_version": ONE_REWRITE_CONTRACT_VERSION,
                },
                **audit_base,
                **enriched,
            }
            if not included:
                record["exclusion_reason"] = "below_minimum_linked_masks_after_rewrite"
            append_jsonl(record, audit_path, durable=True)
            audit_by_image[item["image_id"]] = record
            if included:
                if item["image_id"] not in final_ids:
                    append_jsonl(record, final_path, durable=True)
                    final_ids.add(item["image_id"])
            else:
                if item["image_id"] not in exclusion_ids:
                    append_jsonl(
                        {
                            **record,
                            "reason_code": record["exclusion_reason"],
                            "minimum": min_linked_masks,
                            "retained_mask_count": len(final_groups),
                        },
                        excluded_path,
                        durable=True,
                    )
                    exclusion_ids.add(item["image_id"])
    return final_path
