from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bcc_contract import (
    ONE_REWRITE_CONTRACT_VERSION,
    SIMPLE_REWRITE_CHECK_VERSION,
    VISUAL_AUDIT_VERSION,
)
from .caption_stage import QwenCaptioner, _generation_metrics, create_captioner
from .correspondence_stage import (
    BCC_PROMPT_VERSION,
    CORRESPONDENCE_SCHEMA_VERSION,
    PIPELINE_STAGE_VERSION,
    _bcc_packet_batches,
    _candidate_links,
    _compact_input_manifest,
    _draft_mention_audit,
    _estimate_bcc_visual_tokens,
    _is_current_correspondence_record,
    _model_mask_context,
    bcc_authoritative_rulebook,
    bcc_generation_config,
    build_caption_image_packet,
)
from .io_utils import append_jsonl, read_jsonl_indexed
from .json_utils import extract_json


BCC_VISUAL_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "audit_pass": {"type": "boolean"},
        "task_accuracy_percent": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
        },
        "summary": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["fatal", "rewrite", "style"],
                    },
                    "caption_evidence": {"type": "string"},
                    "mask_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "explanation": {"type": "string"},
                    "rewrite_instruction": {"type": "string"},
                },
                "required": [
                    "code",
                    "severity",
                    "caption_evidence",
                    "mask_ids",
                    "explanation",
                    "rewrite_instruction",
                ],
                "additionalProperties": False,
            },
        },
        "mask_decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "decision": {
                        "type": "string",
                        "enum": ["keep", "drop", "add_if_natural"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["id", "decision", "reason"],
                "additionalProperties": False,
            },
        },
        "rewrite_plan": {"type": "string"},
    },
    "required": [
        "audit_pass",
        "task_accuracy_percent",
        "summary",
        "issues",
        "mask_decisions",
        "rewrite_plan",
    ],
    "additionalProperties": False,
}


_SIMPLE_ERROR_MARKERS = (
    "json parse error",
    "has an unclosed opening tag",
    "has a closing tag without an opening tag",
    "inline tags are crossed",
    "has nested self-overlap",
    "has an empty inline mention",
    "unknown link",
    "unknown id",
    "caption mention coverage:",
    "malformed punctuation",
    "fused or repeated possessive",
    "model decision reject",
)


def simple_rewrite_findings(
    candidate: dict[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return only objective or deliberately narrow rewrite hints.

    These findings drive the rewrite. The larger historical checker remains a
    diagnostic scorer after generation, but its geometry/style/semantic
    judgments are intentionally excluded here.
    """
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(code: str, evidence: str, instruction: str) -> None:
        key = (code, evidence)
        if key in seen:
            return
        seen.add(key)
        findings.append(
            {
                "code": code,
                "evidence": evidence,
                "instruction": instruction,
                "checker_version": SIMPLE_REWRITE_CHECK_VERSION,
            }
        )

    for raw_error in candidate.get("draft_validation_errors") or []:
        message = str(raw_error).strip()
        lowered = message.casefold()
        if not any(marker in lowered for marker in _SIMPLE_ERROR_MARKERS):
            continue
        if "caption mention coverage:" in lowered:
            add(
                "unlinked_or_unmasked_noun_phrase",
                message,
                "Visually verify the phrase, then link the full noun phrase to a compatible mask or remove it and its dependent relation.",
            )
        elif "punctuation" in lowered:
            add(
                "malformed_punctuation",
                message,
                "Restore ordinary sentence punctuation and smooth grammatical flow.",
            )
        elif "possessive" in lowered:
            add(
                "malformed_possessive",
                message,
                "Restore spaces and use overlapping owner/owned tags where appropriate.",
            )
        elif "unknown" in lowered:
            add(
                "unknown_mask_id",
                message,
                "Use only one-based IDs present in ACCEPTED_MASK_CONTEXT.",
            )
        else:
            add(
                "schema_or_tag_structure",
                message,
                "Return the required JSON with balanced, properly nested inline tags.",
            )

    caption = str(candidate.get("caption") or "")
    for mention in _draft_mention_audit(caption, rows):
        if mention.get("instruction") != "remove_unmasked_concrete_noun_phrase":
            continue
        phrase = str(mention.get("text") or "")
        add(
            "unmasked_noun_phrase_candidate",
            phrase,
            "This spaCy noun phrase has no compatible supplied identity; remove it unless visual inspection identifies an exact compatible mask ID.",
        )
    return findings


def build_visual_audit_prompt(
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    input_manifest: list[dict[str, Any]],
) -> str:
    mask_context = _model_mask_context(rows)
    draft_record = {
        "tagged_caption_raw": str(candidate.get("draft_raw") or ""),
        "plain_caption": str(candidate.get("caption") or ""),
        "normalized_links": _candidate_links(candidate, rows),
        "omitted_ids": [
            index + 1
            for index, row in enumerate(rows)
            if str(row.get("mask_id") or "")
            not in {
                str(group.get("mask_id") or "")
                for group in candidate.get("groups") or []
            }
        ],
    }
    return (
        """# Task: independent visual and annotation audit before the single rewrite

You are the middle stage of a three-call BCC caption pipeline. Do NOT rewrite the caption yet. Inspect the original
image, numbered accepted-mask overlay, every inverse crop, the proposed tagged caption, normalized links, weak mask
context, and the authoritative rules below. The masks are useful but imperfect proposals: a mask may be partial,
redundant, mislabeled, or simply impossible to mention naturally. Decide which masks and links support the most
truthful, natural final caption.

`task_accuracy_percent` is your estimated percentage of the complete BCC task that the draft satisfies, not your
confidence in this audit. Check visual truth, identity-to-mask alignment, whether every concrete noun phrase is
linked, whether all repeated/coreferential mentions are grouped, span boundaries, collective/shared spans,
possessive overlap, grammatical number, unsupported relations, ordinary sentence flow, and punctuation. A concrete
noun phrase with no compatible mask is a serious violation. A difficult supplied mask that the caption omits is not
itself an error. `audit_pass` is true only when the draft could be used unchanged; otherwise list actionable issues.

The SIMPLE_DETERMINISTIC_FINDINGS are intentionally narrow hints, not ground truth. Verify noun-phrase hints against
the visuals. Do not inherit the historical checker's subjective style, geometry, or semantic conclusions. For each
issue, quote concise caption evidence, list relevant one-based mask IDs, explain the rule, and give a direct rewrite
instruction. Include a short mask decision for each ID that matters; you need not mechanically catalog every ID.

Return exactly one JSON object matching this logical shape and nothing else:
{"audit_pass":false,"task_accuracy_percent":82,"summary":"...","issues":[{"code":"unlinked_concrete_entity","severity":"fatal","caption_evidence":"a jacket","mask_ids":[],"explanation":"...","rewrite_instruction":"remove the phrase and dependent relation"}],"mask_decisions":[{"id":4,"decision":"drop","reason":"ambiguous partial mask"}],"rewrite_plan":"..."}

"""
        + bcc_authoritative_rulebook()
        + "\nORDERED_INPUT_IMAGES:\n"
        + json.dumps(
            _compact_input_manifest(input_manifest),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\nACCEPTED_MASK_CONTEXT:\n"
        + json.dumps(mask_context, ensure_ascii=False, separators=(",", ":"))
        + "\nDRAFT_RECORD:\n"
        + json.dumps(draft_record, ensure_ascii=False, separators=(",", ":"))
        + "\nSIMPLE_DETERMINISTIC_FINDINGS:\n"
        + json.dumps(
            simple_rewrite_findings(candidate, rows),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + f"\nAudit version: {VISUAL_AUDIT_VERSION}"
    )


def normalize_visual_audit(
    raw: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Parse an audit without making it a hard gate for the later rewrite."""
    try:
        parsed = extract_json(raw)
    except Exception as exc:
        return {
            "audit_parseable": False,
            "audit_parse_error": repr(exc),
            "audit_pass": False,
            "task_accuracy_percent": 0.0,
            "summary": "The audit response was unparseable; the rewrite must independently recheck the draft.",
            "issues": [
                {
                    "code": "audit_unparseable",
                    "severity": "rewrite",
                    "caption_evidence": "",
                    "mask_ids": [],
                    "explanation": repr(exc),
                    "rewrite_instruction": "Recheck the full visual packet and rulebook independently.",
                }
            ],
            "mask_decisions": [],
            "rewrite_plan": "Independently rebuild a valid natural tagged caption.",
        }
    if not isinstance(parsed, dict):
        return {
            "audit_parseable": False,
            "audit_parse_error": "visual audit JSON root is not an object",
            "audit_pass": False,
            "task_accuracy_percent": 0.0,
            "summary": "The audit response had the wrong JSON shape; the rewrite must independently recheck the draft.",
            "issues": [
                {
                    "code": "audit_wrong_json_shape",
                    "severity": "rewrite",
                    "caption_evidence": "",
                    "mask_ids": [],
                    "explanation": "The visual audit JSON root was not an object.",
                    "rewrite_instruction": "Recheck the full visual packet and rulebook independently.",
                }
            ],
            "mask_decisions": [],
            "rewrite_plan": "Independently rebuild a valid natural tagged caption.",
        }
    available_ids = set(range(1, len(rows) + 1))
    issues: list[dict[str, Any]] = []
    for item in parsed.get("issues") or []:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or "rewrite")
        if severity not in {"fatal", "rewrite", "style"}:
            severity = "rewrite"
        issues.append(
            {
                "code": str(item.get("code") or "model_audit_finding"),
                "severity": severity,
                "caption_evidence": str(item.get("caption_evidence") or ""),
                "mask_ids": [
                    int(value)
                    for value in item.get("mask_ids") or []
                    if isinstance(value, (int, float)) and int(value) in available_ids
                ],
                "explanation": str(item.get("explanation") or ""),
                "rewrite_instruction": str(item.get("rewrite_instruction") or ""),
            }
        )
    decisions: list[dict[str, Any]] = []
    for item in parsed.get("mask_decisions") or []:
        if not isinstance(item, dict):
            continue
        try:
            link_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if link_id not in available_ids:
            continue
        decision = str(item.get("decision") or "drop")
        if decision not in {"keep", "drop", "add_if_natural"}:
            decision = "drop"
        decisions.append(
            {
                "id": link_id,
                "decision": decision,
                "reason": str(item.get("reason") or ""),
            }
        )
    try:
        accuracy = float(parsed.get("task_accuracy_percent") or 0.0)
    except (TypeError, ValueError):
        accuracy = 0.0
    return {
        "audit_parseable": True,
        "audit_parse_error": "",
        "audit_pass": bool(parsed.get("audit_pass")),
        "task_accuracy_percent": max(0.0, min(100.0, accuracy)),
        "summary": str(parsed.get("summary") or ""),
        "issues": issues,
        "mask_decisions": decisions,
        "rewrite_plan": str(parsed.get("rewrite_plan") or ""),
    }


def run_bcc_visual_audit_batch(
    config: dict[str, Any],
    run_dir: str | Path,
    row_groups: list[list[dict[str, Any]]],
    *,
    captioner: QwenCaptioner | None = None,
    mock: bool = False,
) -> Path:
    """Audit every BCC draft with the same complete visual packet."""
    run_dir = Path(run_dir)
    candidate_path = run_dir / "image_caption_candidates.jsonl"
    raw_path = run_dir / "bcc_visual_audit_raw.jsonl"
    audit_path = run_dir / "bcc_visual_audits.jsonl"
    candidates = {
        str(row.get("image_id") or ""): row
        for row in (read_jsonl_indexed(candidate_path) if candidate_path.exists() else [])
        if _is_current_correspondence_record(row)
        and row.get("contract_version") == ONE_REWRITE_CONTRACT_VERSION
    }
    completed = {
        str(row.get("image_id") or "")
        for row in (read_jsonl_indexed(audit_path) if audit_path.exists() else [])
        if row.get("audit_version") == VISUAL_AUDIT_VERSION
        and row.get("contract_version") == ONE_REWRITE_CONTRACT_VERSION
    }
    persisted_raw = {
        str(row.get("image_id") or ""): row
        for row in (read_jsonl_indexed(raw_path) if raw_path.exists() else [])
        if row.get("audit_version") == VISUAL_AUDIT_VERSION
        and row.get("contract_version") == ONE_REWRITE_CONTRACT_VERSION
        and row.get("image_id")
    }
    stage = config.get("image_caption_audit", {})
    seed_base = int(config.get("random_seed", 17)) + int(
        stage.get("seed_offset", 350000)
    )
    prepared: list[dict[str, Any]] = []
    for position, rows in enumerate(row_groups):
        if not rows:
            continue
        image_id = str(rows[0].get("image_id") or "")
        candidate = candidates.get(image_id)
        if candidate is None or image_id in completed:
            continue
        persisted = persisted_raw.get(image_id)
        if persisted is None:
            packet, manifest = build_caption_image_packet(
                rows, str(candidate.get("correspondence_overlay_path") or "")
            )
            visual_tokens = _estimate_bcc_visual_tokens(packet)
        else:
            packet = []
            manifest = list(candidate.get("bcc_input_manifest") or [])
            visual_tokens = 0
        prepared.append(
            {
                "image_id": image_id,
                "rows": rows,
                "candidate": candidate,
                "packet": packet,
                "manifest": manifest,
                "visual_tokens": visual_tokens,
                "prompt": build_visual_audit_prompt(candidate, rows, manifest),
                "seed": seed_base + position,
                "persisted": persisted,
            }
        )
    if not prepared:
        return audit_path
    if not mock and captioner is None:
        captioner = create_captioner(config, "image_caption_audit")
    for batch in _bcc_packet_batches(prepared, stage):
        pending = [item for item in batch if item["persisted"] is None]
        if mock:
            generated = [
                {
                    "raw": json.dumps(
                        {
                            "audit_pass": True,
                            "task_accuracy_percent": 100,
                            "summary": "mock audit",
                            "issues": [],
                            "mask_decisions": [],
                            "rewrite_plan": "Preserve the valid draft.",
                        }
                    ),
                    "backend": "mock",
                }
                for _ in pending
            ]
        elif pending:
            runtime = bcc_generation_config(
                config,
                "image_caption_audit",
                max(len(item["rows"]) for item in pending),
            )
            runtime["json_schema"] = BCC_VISUAL_AUDIT_SCHEMA
            generated = captioner.generate_many_bcc(
                [item["packet"] for item in pending],
                [item["prompt"] for item in pending],
                [item["seed"] for item in pending],
                batch_size=len(pending),
                generation_config=runtime,
            )
        else:
            generated = []
        generated_by_id = {
            item["image_id"]: result for item, result in zip(pending, generated)
        }
        for item in batch:
            persisted = item["persisted"]
            result = persisted or generated_by_id[item["image_id"]]
            raw = str(result.get("raw") or "")
            if persisted is None:
                append_jsonl(
                    {
                        "image_id": item["image_id"],
                        "raw": raw,
                        **_generation_metrics(result),
                        "backend": result.get("backend", "transformers"),
                        "batched": len(pending) > 1,
                        "estimated_visual_tokens": item["visual_tokens"],
                        "prompt_version": BCC_PROMPT_VERSION,
                        "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                        "stage_version": PIPELINE_STAGE_VERSION,
                        "contract_version": ONE_REWRITE_CONTRACT_VERSION,
                        "audit_version": VISUAL_AUDIT_VERSION,
                        "bcc_input_manifest": item["manifest"],
                    },
                    raw_path,
                    durable=True,
                )
            normalized = normalize_visual_audit(raw, item["rows"])
            append_jsonl(
                {
                    "image_id": item["image_id"],
                    "source_image_path": item["candidate"].get(
                        "source_image_path"
                    ),
                    "prompt_version": BCC_PROMPT_VERSION,
                    "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                    "stage_version": PIPELINE_STAGE_VERSION,
                    "contract_version": ONE_REWRITE_CONTRACT_VERSION,
                    "audit_version": VISUAL_AUDIT_VERSION,
                    "model": str(
                        stage.get(
                            "model_name",
                            config.get("image_caption", {}).get(
                                "model_name", "Qwen/Qwen3.8-27B"
                            ),
                        )
                    ),
                    "audit_raw": raw,
                    "simple_deterministic_findings": simple_rewrite_findings(
                        item["candidate"], item["rows"]
                    ),
                    **normalized,
                },
                audit_path,
                durable=True,
            )
    return audit_path
