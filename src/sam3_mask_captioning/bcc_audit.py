from __future__ import annotations

import hashlib
import re
from collections import Counter
from statistics import mean, median
from typing import Any, Iterable


CHECKER_AUDIT_VERSION = "bcc-deterministic-audit-v4-oewn-onehop-2026-08-25"

_INTEGER_ID_RE = re.compile(r"(?:mask|link|group|ID)\s+(?:ID\s+)?(?P<id>\d+)", re.IGNORECASE)
_SPAN_RE = re.compile(r"\[(?P<start>\d+)\s*,\s*(?P<end>\d+)\)")


def _issue_code(message: str) -> tuple[str, str]:
    lower = message.lower()
    rules: list[tuple[tuple[str, ...], str, str]] = [
        (("unmasked concrete noun phrase", "unsupported-object evasion"), "unmasked_entity", "fatal"),
        (("unknown mask", "unknown link", "unknown id", "output no unknown"), "unknown_entity_id", "fatal"),
        (
            (
                "semantically incompatible",
                "identity noun",
                "body-part identity phrase",
                "identity phrase is incompatible",
                "incompatible with mask subject",
                "subject anchor",
                "wrong mask",
            ),
            "identity_mismatch",
            "fatal",
        ),
        (("fused or repeated possessive", "word boundaries"), "style_possessive_boundary", "nonfatal"),
        (("pronoun", "possessive", "reflexive", "corefer"), "coreference", "fatal"),
        (("contact relation", "hold/carry", "grasp", "clutch", "unsupported relation", "spatial relation"), "unsupported_relation", "fatal"),
        (("pass-one reject", "pass-two keep", "model decision"), "model_decision", "fatal"),
        (("accepted masks are missing", "required link ids", "missing group", "unlinked"), "missing_mask_link", "nonfatal"),
        (("stock", "is visible", "is shown", "is present", "appears"), "style_stock_predicate", "nonfatal"),
        (("ordinal", "first ", "second ", "third "), "style_ordinal_catalog", "nonfatal"),
        (("inventory", "one-sentence-per-mask", "jointly describe"), "style_inventory", "nonfatal"),
        (("punctuation", "malformed"), "malformed_punctuation", "nonfatal"),
        (("composite mask", "composite outer", "child id", "collective span"), "composite_link", "nonfatal"),
        (("grammatical number", "plural collection", "singular", "same-type instance"), "instance_number", "nonfatal"),
        (("span", "occurrence", "overlap", "nested", "tag", "text entry"), "span_alignment", "nonfatal"),
        (("caption cleanup", "caption quality", "style", "natural"), "style_naturalness", "nonfatal"),
    ]
    for needles, code, severity in rules:
        if any(needle in lower for needle in needles):
            return code, severity
    return "other_checker_finding", "nonfatal"


def issue_records(
    errors: Iterable[str], rows: list[dict[str, Any]], *, scope: str = "caption"
) -> list[dict[str, Any]]:
    """Convert legacy checker strings into stable, queryable issue records."""
    mask_ids = [str(row.get("mask_id") or "") for row in rows]
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in errors:
        message = str(raw).strip()
        if not message:
            continue
        code, severity = _issue_code(message)
        key = (code, message)
        if key in seen:
            continue
        seen.add(key)
        linked_ids: list[str] = []
        for match in _INTEGER_ID_RE.finditer(message):
            index = int(match.group("id")) - 1
            if 0 <= index < len(mask_ids) and mask_ids[index] not in linked_ids:
                linked_ids.append(mask_ids[index])
        spans = [
            [int(match.group("start")), int(match.group("end"))]
            for match in _SPAN_RE.finditer(message)
        ]
        digest = hashlib.sha1(
            f"{CHECKER_AUDIT_VERSION}\0{code}\0{message}".encode("utf-8")
        ).hexdigest()[:16]
        records.append(
            {
                "issue_id": digest,
                "code": code,
                "severity": severity,
                "scope": scope,
                "message": message,
                "mask_ids": linked_ids,
                "spans": spans,
                "checker_version": CHECKER_AUDIT_VERSION,
            }
        )
    return records


def compare_issue_sets(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    before_parseable: bool,
    after_parseable: bool,
) -> dict[str, Any]:
    def key(item: dict[str, Any]) -> tuple[str, str]:
        return str(item.get("code") or ""), str(item.get("message") or "")

    before_by_key = {key(item): item for item in before}
    after_by_key = {key(item): item for item in after}
    resolved = [before_by_key[item] for item in before_by_key.keys() - after_by_key.keys()]
    persisting = [after_by_key[item] for item in before_by_key.keys() & after_by_key.keys()]
    new = [after_by_key[item] for item in after_by_key.keys() - before_by_key.keys()]
    before_fatal = sum(item.get("severity") == "fatal" for item in before)
    after_fatal = sum(item.get("severity") == "fatal" for item in after)
    if not before_parseable and after_parseable:
        outcome = "recovered_parse"
    elif before_parseable and not after_parseable:
        outcome = "lost_parse"
    elif len(after) < len(before):
        outcome = "improved"
    elif len(after) > len(before):
        outcome = "worsened"
    else:
        outcome = "same"
    return {
        "outcome": outcome,
        "before_parseable": before_parseable,
        "after_parseable": after_parseable,
        "before_issue_count": len(before),
        "after_issue_count": len(after),
        "issue_delta": len(after) - len(before),
        "before_fatal_count": before_fatal,
        "after_fatal_count": after_fatal,
        "fatal_delta": after_fatal - before_fatal,
        "resolved": resolved,
        "persisting": persisting,
        "new": new,
    }


def quality_tier(issues: list[dict[str, Any]]) -> str:
    if any(item.get("severity") == "fatal" for item in issues):
        return "fatal_flagged"
    if issues:
        return "nonfatal_flagged"
    return "clean"


def composite_statistics(
    rows: list[dict[str, Any]], final_groups: list[dict[str, Any]]
) -> dict[str, Any]:
    raw_count = len(rows) + sum(
        len(row.get("bcc_duplicate_mask_aliases") or []) for row in rows
    )
    duplicate_count = sum(
        len(row.get("bcc_duplicate_mask_aliases") or []) for row in rows
    )
    composites = [
        group
        for group in final_groups
        if group.get("bcc_composite_mask_children")
        or int(group.get("bcc_significant_component_count") or 1) > 1
    ]
    union_ious = [
        float(group["bcc_composite_union_iou"])
        for group in composites
        if group.get("bcc_composite_union_iou") is not None
    ]
    coverages = [
        float(group["bcc_composite_coverage"])
        for group in composites
        if group.get("bcc_composite_coverage") is not None
    ]
    return {
        "consistency_passed_mask_count": raw_count,
        "canonical_mask_count": len(rows),
        "final_linked_mask_count": len(final_groups),
        "duplicate_alias_count": duplicate_count,
        "composite_group_count": len(composites),
        "composite_child_count": sum(
            len(group.get("bcc_composite_mask_children") or []) for group in composites
        ),
        "composite_union_iou_mean": mean(union_ious) if union_ious else None,
        "composite_coverage_mean": mean(coverages) if coverages else None,
        "significant_component_count": sum(
            int(group.get("bcc_significant_component_count") or 1)
            for group in final_groups
        ),
    }


def aggregate_audits(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    improvements = [row.get("rewrite_metrics") or {} for row in rows]
    before_counts = [int(item.get("before_issue_count") or 0) for item in improvements]
    after_counts = [int(item.get("after_issue_count") or 0) for item in improvements]
    outcomes = Counter(str(item.get("outcome") or "unknown") for item in improvements)
    quality = Counter(str(row.get("quality_tier") or "excluded") for row in rows)
    codes_before = Counter(
        issue.get("code")
        for row in rows
        for issue in ((row.get("validation") or {}).get("before_rewrite") or {}).get("issues", [])
    )
    codes_after = Counter(
        issue.get("code")
        for row in rows
        for issue in ((row.get("validation") or {}).get("after_rewrite") or {}).get("issues", [])
    )
    composite_rows = [row.get("composite_statistics") or {} for row in rows]
    additive_composite_keys = (
        "consistency_passed_mask_count",
        "canonical_mask_count",
        "final_linked_mask_count",
        "duplicate_alias_count",
        "composite_group_count",
        "composite_child_count",
        "significant_component_count",
    )
    rewrite_total = sum(outcomes.values())
    return {
        "record_count": len(rows),
        "included_count": sum(bool(row.get("included")) for row in rows),
        "excluded_count": sum(not bool(row.get("included")) for row in rows),
        "after_parseable_count": sum(
            bool(item.get("after_parseable")) for item in improvements
        ),
        "rewrite_outcomes": dict(sorted(outcomes.items())),
        "rewrite_outcome_rates": {
            key: value / rewrite_total if rewrite_total else 0.0
            for key, value in sorted(outcomes.items())
        },
        "quality_tiers": dict(sorted(quality.items())),
        "before_issue_mean": mean(before_counts) if before_counts else 0.0,
        "before_issue_median": median(before_counts) if before_counts else 0.0,
        "after_issue_mean": mean(after_counts) if after_counts else 0.0,
        "after_issue_median": median(after_counts) if after_counts else 0.0,
        "mean_issue_delta": (
            mean(after - before for before, after in zip(before_counts, after_counts))
            if before_counts
            else 0.0
        ),
        "issue_codes_before": dict(sorted((str(k), v) for k, v in codes_before.items())),
        "issue_codes_after": dict(sorted((str(k), v) for k, v in codes_after.items())),
        "fatal_issue_count_after": sum(
            int(item.get("after_fatal_count") or 0) for item in improvements
        ),
        "omitted_mask_count": sum(len(row.get("omitted_masks") or []) for row in rows),
        "composite_totals": {
            key: sum(int(item.get(key) or 0) for item in composite_rows)
            for key in additive_composite_keys
        },
        "composite_union_iou_record_mean": mean(
            float(item["composite_union_iou_mean"])
            for item in composite_rows
            if item.get("composite_union_iou_mean") is not None
        )
        if any(item.get("composite_union_iou_mean") is not None for item in composite_rows)
        else None,
        "composite_coverage_record_mean": mean(
            float(item["composite_coverage_mean"])
            for item in composite_rows
            if item.get("composite_coverage_mean") is not None
        )
        if any(item.get("composite_coverage_mean") is not None for item in composite_rows)
        else None,
    }
