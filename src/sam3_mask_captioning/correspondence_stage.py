from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import json
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .caption_cleanup import (
    caption_contact_relations,
    caption_entity_mentions,
    semantic_noun_lemmas,
)
from .caption_stage import QwenCaptioner, _generation_metrics, qwen_model_config
from .io_utils import append_jsonl, read_jsonl_indexed, write_jsonl
from .json_utils import extract_json
from .mask_utils import color_for_index
from .semantic_taxonomy import (
    normalize_semantic_term,
    semantic_is_a,
    semantic_matches,
    semantic_terms_compatible,
    taxonomy_alternatives,
)


CORRESPONDENCE_SCHEMA_VERSION = "bcc-image-text-v7-wordnet-onehop-min10"
PIPELINE_STAGE_VERSION = (
    "bcc-qwen38-allstage-draft-audit-rewrite-v3-natural-coverage-2026-08-25"
)
BCC_PROMPT_VERSION = (
    "bcc-caption-v30-qwen38-natural-coverage-wordnet-2026-08-25"
)
_FORBIDDEN_IMAGE_CONTEXT_RE = re.compile(
    r"\b(?:background|backdrop|foreground|synthetic\s+(?:fill|color)|"
    r"(?:image|photo(?:graph)?|picture|crop|frame)\s+(?:shows?|depicts?|contains?|of)|"
    r"(?:shown|seen|pictured|visible)\s+in\s+(?:the|this)\s+(?:image|photo(?:graph)?|picture|crop|frame)|"
    r"(?:close[- ]?up|cropped|profile|side)\s+view)\b",
    re.IGNORECASE,
)
_DENSE_BACKGROUND_ROLE_RE = re.compile(
    r"\s+(?:is|are|was|were)\s+(?:"
    r"part\s+of|"
    r"(?:visible|seen|shown|located|positioned)\s+in"
    r")\s+the\s+background\b",
    re.IGNORECASE,
)
_DENSE_BACKGROUND_PHRASE_RES = (
    (
        re.compile(
            r"^\s*(?:in|within)\s+(?:the\s+)?(?:immediate\s+)?"
            r"(?:background|foreground)\b\s*,?\s*",
            re.IGNORECASE,
        ),
        "",
    ),
    (
        re.compile(
            r"^\s*(?:(?:an?|the)\s+)?(?:low[- ]angle|high[- ]angle|close[- ]?up)\s+"
            r"view\s+(?:captures?|shows?|depicts?)\s+",
            re.IGNORECASE,
        ),
        "There is ",
    ),
    (
        re.compile(
            r"\b(?:hangs?|stands?|sits?|appears?|remains?)\s+in\s+the\s+"
            r"(?:background|foreground)\b",
            re.IGNORECASE,
        ),
        "is visible",
    ),
    (
        re.compile(
            r"\s+(?:in|against|before)\s+(?:the\s+)?"
            r"(?:immediate\s+)?(?:background|backdrop|foreground)\b",
            re.IGNORECASE,
        ),
        "",
    ),
    (
        re.compile(r"\s+toward\s+(?:the\s+)?scene\b", re.IGNORECASE),
        "",
    ),
)

_DENSE_BACKGROUND_PREDICATE_RE = re.compile(
    r"\b(?P<verb>fills?|lines?)\s+(?:the\s+)?"
    r"(?:background|backdrop|foreground)\b",
    re.IGNORECASE,
)
_INVENTORY_PREDICATE_RE = re.compile(
    r"\b(?:(?:is|are|was|were)\s+(?:clearly\s+)?(?:visible|seen|shown|present|depicted)|"
    r"appears?|is\s+displayed)\b",
    re.IGNORECASE,
)
_ORDINAL_SELECTOR_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|"
    r"tenth|eleventh|twelfth|\d+(?:st|nd|rd|th))\b",
    re.IGNORECASE,
)
_MALFORMED_CAPTION_PUNCTUATION_RE = re.compile(
    r"(?:[.!?]\s*,|,\s*[.!?])"
)
_FUSED_POSSESSIVE_RE = re.compile(
    r"\b(?:her|his|their|its)(?:(?:her|his|their|its))?"
    r"(?:hair|face|nose|hand|arm|leg|head|eye|ear|mouth|neck|shirt|pants|"
    r"shoe|coat|dress|body|torso|foot|feet)\b",
    re.IGNORECASE,
)
_REPEATED_POSSESSIVE_RE = re.compile(
    r"\b(?P<possessive>her|his|their|its)\s+(?P=possessive)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_OBJECT_EVASION_RE = re.compile(
    r"\b(?:(?:holds?|carries?|grasps?|clutches?)\s+nothing|empty[- ]handed)\b",
    re.IGNORECASE,
)
_SELF_BODY_MANIPULATION_RE = re.compile(
    r"\b(?:holds?|holding|carries?|carrying|grasps?|grasping|clutches?|clutching|"
    r"grips?|gripping)\s+(?:one\s+of\s+)?(?:his|her|their|its)\s+(?:other\s+)?"
    r"(?:hands?|arms?|shoulders?|heads?|hair|necks?|faces?|mouths?|torsos?|bodies?|"
    r"legs?|knees?|feet|foot)\b",
    re.IGNORECASE,
)
_SELF_BODY_CONTACT_RE = re.compile(
    r"\b(?:one\s+of\s+)?(?:his|her|their|its)\s+(?:other\s+)?"
    r"(?:hands?|arms?)\s+"
    r"(?:(?:is|are|was|were)\s+(?:positioned|placed|resting)\s+(?:on|against)|"
    r"(?:rests?|rested|lies?|lay|presses?|pressed)\s+(?:on|against)|"
    r"(?:touches?|touched))\s+"
    r"(?:his|her|their|its)\s+(?:other\s+)?"
    r"(?:heads?|hair|faces?|necks?|shoulders?|torsos?|bodies?)\b",
    re.IGNORECASE,
)
_TRANSITIVE_SELF_BODY_CONTACT_RE = re.compile(
    r"\b(?:rests?|resting|places?|placing|positions?|positioning|presses?|pressing)\s+"
    r"(?:one\s+of\s+)?(?:his|her|their|its)\s+(?:other\s+)?(?:hands?|arms?)\s+"
    r"(?:on|against)\s+(?:his|her|their|its)\s+(?:other\s+)?"
    r"(?:heads?|hair|faces?|necks?|shoulders?|torsos?|bodies?)\b",
    re.IGNORECASE,
)
_IMPOSSIBLE_OBJECT_ACTION_RE = re.compile(
    r"\b(?:plays?|playing|played)\s+(?:(?:a|an|the|his|her|their|its)\s+)?"
    r"microphones?\b",
    re.IGNORECASE,
)
_MALFORMED_SELECTOR_WORDS = {
    "one",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
}
_PLURAL_ONLY_IDENTITY_SURFACES = {
    "eyeglasses",
    "glasses",
    "goggles",
    "jeans",
    "leggings",
    "pants",
    "shorts",
    "sunglasses",
    "trousers",
}

_CLOTHING_TERMS = {
    normalize_semantic_term(value)
    for value in (
        "shoe", "sneaker", "boot", "sandal", "pants", "trouser", "jean", "shorts",
        "shirt", "jersey", "blouse", "top", "sweater", "hoodie", "coat", "jacket",
    )
}
_BODY_PART_TERMS = {
    normalize_semantic_term(value)
    for value in (
        "arm", "beard", "ear", "eye", "face", "foot", "hair", "hand",
        "head", "leg", "mouth", "neck", "nose", "shoulder",
    )
}
_HYPHENATED_SIDE_SELECTOR_RE = re.compile(r"\b(?:left|right)-side\b", re.IGNORECASE)
_REFERENCE_ONLY_TERMS = {
    "he", "her", "hers", "him", "his", "it", "its", "one", "other",
    "she", "that", "them", "their", "theirs", "they", "this",
}
_PERSON_ONLY_REFERENCE_TERMS = {"he", "her", "hers", "him", "his", "she"}
_ARTICLE_LED_COLLECTIVE_HEADS = {
    "audience",
    "band",
    "bunch",
    "class",
    "cluster",
    "collection",
    "couple",
    "crowd",
    "family",
    "flock",
    "group",
    "herd",
    "line",
    "pair",
    "row",
    "set",
    "stack",
    "team",
}


@lru_cache(maxsize=65_536)
def _cached_image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _load_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def write_correspondence_overlay(
    source_image_path: str | Path,
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    alpha: float = 0.38,
) -> Path:
    image = Image.open(source_image_path).convert("RGB")
    base = np.asarray(image).astype(np.float32)
    draw_labels: list[tuple[int, int, str, tuple[int, int, int]]] = []
    for index, row in enumerate(rows):
        mask = _load_mask(row["mask_path"])
        if mask.shape != base.shape[:2]:
            mask = np.asarray(
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(image.size, Image.NEAREST)
            ) > 0
        color = color_for_index(index)
        rgb = np.asarray(color, dtype=np.float32)
        base[mask] = (1.0 - alpha) * base[mask] + alpha * rgb
        x, y, _, _ = [int(value) for value in row["bbox"]]
        draw_labels.append((x, y, str(index + 1), color))
    overlay = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for x, y, label, color in draw_labels:
        box = draw.textbbox((x, y), label, font=font, stroke_width=1)
        padded = (box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2)
        draw.rectangle(padded, fill=(16, 19, 24), outline=color, width=2)
        draw.text((x, y), label, fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    return output_path


def _image_region(bbox: list[int], image_width: int, image_height: int) -> str:
    center_x = float(bbox[0]) + float(bbox[2]) / 2.0
    center_y = float(bbox[1]) + float(bbox[3]) / 2.0
    horizontal = ("left", "center", "right")[min(2, int(3 * center_x / max(1, image_width)))]
    vertical = ("upper", "middle", "lower")[min(2, int(3 * center_y / max(1, image_height)))]
    return f"{vertical}-{horizontal}"


def _identity_surface(
    row: dict[str, Any],
    fallback: str,
    *,
    prefer_instance_head: bool = False,
) -> str:
    """Choose a grammatical identity surface while preserving plural-only nouns."""
    expected = _row_expected_semantic_terms(row)
    object_surface = re.sub(
        r"^(?:a|an|the)\s+",
        "",
        re.sub(r"\s+", " ", str(row.get("object") or "").strip().casefold()),
    )
    preserve_reviewed_plural = object_surface in _PLURAL_ONLY_IDENTITY_SURFACES
    keys = (
        ("main_candidate", "object")
        if prefer_instance_head and not preserve_reviewed_plural
        else ("object", "main_candidate")
    )
    for key in keys:
        candidate = re.sub(
            r"^(?:a|an|the)\s+",
            "",
            re.sub(r"\s+", " ", str(row.get(key) or "").strip().casefold()),
        )
        if not candidate or len(candidate.split()) > 5:
            continue

        if not re.fullmatch(r"[a-z0-9]+(?:[- ][a-z0-9]+)*", candidate):
            continue
        candidate_terms = _canonical_semantic_terms(candidate)
        if not expected or semantic_terms_compatible(expected, candidate_terms):
            return candidate
    return re.sub(r"\s+", " ", str(fallback or "object").strip().casefold())


def _bbox_contained_fraction(inner: list[int], outer: list[int]) -> float:
    inner_x0, inner_y0, inner_w, inner_h = [float(value) for value in inner]
    outer_x0, outer_y0, outer_w, outer_h = [float(value) for value in outer]
    inner_x1, inner_y1 = inner_x0 + inner_w, inner_y0 + inner_h
    outer_x1, outer_y1 = outer_x0 + outer_w, outer_y0 + outer_h
    intersection = max(0.0, min(inner_x1, outer_x1) - max(inner_x0, outer_x0)) * max(
        0.0, min(inner_y1, outer_y1) - max(inner_y0, outer_y0)
    )
    return intersection / max(1.0, inner_w * inner_h)


def _body_part_owner_ids(
    rows: list[dict[str, Any]],
    boxes: list[list[int]],
    person_ids: list[int],
) -> list[int | None]:
    """Infer an owner only when body-part/person geometry is unambiguous."""
    owners: list[int | None] = [None] * len(rows)
    for row_index, row in enumerate(rows):
        if not _terms_are_body_parts(_row_expected_semantic_terms(row)):
            continue
        if len(person_ids) == 1:
            owners[row_index] = person_ids[0]
            continue
        scores = sorted(
            (
                (_bbox_contained_fraction(boxes[row_index], boxes[person_id - 1]), person_id)
                for person_id in person_ids
                if person_id - 1 != row_index
            ),
            reverse=True,
        )
        if not scores or scores[0][0] < 0.80:
            continue
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        if runner_up <= 0.55 or scores[0][0] - runner_up >= 0.20:
            owners[row_index] = scores[0][1]
    return owners


def _owner_possessive(row: dict[str, Any]) -> str:
    anchor = str(
        row.get("main_candidate")
        or row.get("object")
        or row.get("source_prompt")
        or ""
    ).casefold()
    words = set(re.findall(r"[a-z]+", anchor))
    if words & {"woman", "girl"}:
        return "her"
    if words & {"man", "boy"}:
        return "his"
    return "their"


def _relative_descriptors(boxes: list[list[int]]) -> list[str]:
    centers_x = [float(box[0]) + float(box[2]) / 2.0 for box in boxes]
    centers_y = [float(box[1]) + float(box[3]) / 2.0 for box in boxes]
    horizontal = max(centers_x) - min(centers_x) >= max(centers_y) - min(centers_y)
    centers = centers_x if horizontal else centers_y
    order = sorted(range(len(boxes)), key=lambda index: centers[index])
    count = len(boxes)
    if horizontal and count == 2:
        labels = ["left-side", "right-side"]
    elif not horizontal and count == 2:
        labels = ["upper", "lower"]
    elif horizontal and count == 3:
        labels = ["left-side", "central", "right-side"]
    elif not horizontal and count == 3:
        labels = ["upper", "middle", "lower"]
    else:
        direction = "left" if horizontal else "top"
        labels = [f"{index + 1}{_ordinal_suffix(index + 1)}-from-{direction}" for index in range(count)]
    output = [""] * count
    for rank, original_index in enumerate(order):
        output[original_index] = labels[rank]
    return output


def _ordinal_suffix(index: int) -> str:
    if index % 100 in {11, 12, 13}:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(index % 10, "th")


def _deduplicate_safe_context(
    context: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    boxes: list[list[int]],
    subject_keys: list[str],
) -> None:
    duplicate_sets: dict[tuple[str, int | None, str], list[int]] = {}
    for index, item in enumerate(context):
        owner_id = item.get("required_owner_id")
        key = (
            subject_keys[index],
            int(owner_id) if owner_id is not None else None,
            str(item["safe_tag_phrase"]).casefold(),
        )
        duplicate_sets.setdefault(key, []).append(index)
    for indexes in duplicate_sets.values():
        if len(indexes) < 2:
            continue
        descriptors = _relative_descriptors([boxes[index] for index in indexes])
        for index, descriptor in zip(indexes, descriptors, strict=True):
            item = context[index]
            noun = str(item["surface_identity_noun"])
            if noun == "hair":
                noun = "hairstyle"
            phrase = f"the {descriptor} {noun}"
            owner_id = item.get("required_owner_id")
            possessive = (
                _owner_possessive(rows[int(owner_id) - 1])
                if owner_id is not None
                else "their"
            )
            item["fallback_unique_selector"] = f"the {descriptor}"
            item["safe_tag_phrase"] = phrase
            item["safe_tagged_phrase"] = _safe_tagged_phrase(
                link_id=int(item["id"]),
                phrase=phrase,
                owner_id=int(owner_id) if owner_id is not None else None,
                owner_possessive=possessive,
            )


def _mask_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    if not rows:
        return context
    overlay_by_mask_id = {
        str(row.get("mask_id") or ""): index + 1 for index, row in enumerate(rows)
    }
    boxes = [
        [int(value) for value in row.get("bbox", [0, 0, 1, 1])]
        for row in rows
    ]
    image_regions: list[str] = []
    source_path = str(rows[0].get("source_image_path") or "")
    if source_path and Path(source_path).exists():
        image_width, image_height = _cached_image_size(source_path)
    else:
        image_width = max(1, max(box[0] + box[2] for box in boxes))
        image_height = max(1, max(box[1] + box[3] for box in boxes))
    image_regions = [
        _image_region(box, image_width, image_height) for box in boxes
    ]
    subjects = [
        str(row.get("main_candidate") or row.get("object") or row.get("source_prompt") or "").strip()
        for row in rows
    ]
    subject_keys = [
        _semantic_subject_key(row, subject)
        for row, subject in zip(rows, subjects, strict=True)
    ]
    subject_totals = Counter(subject_keys)
    subject_seen: Counter[str] = Counter()
    person_ids = [
        index + 1
        for index, row in enumerate(rows)
        if _terms_include_category(_row_expected_semantic_terms(row), "person")
    ]
    owner_ids = _body_part_owner_ids(rows, boxes, person_ids)
    owner_subject_totals = Counter(
        (owner_id, subject_keys[index])
        for index, owner_id in enumerate(owner_ids)
        if owner_id is not None
    )
    owner_subject_seen: Counter[tuple[int, str]] = Counter()
    for index, row in enumerate(rows):
        subject = subjects[index]
        subject_key = subject_keys[index]
        allowed_identity_nouns = _allowed_identity_nouns(row)
        subject_seen[subject_key] += 1
        required_owner_id = owner_ids[index]
        bbox = boxes[index]
        image_region = image_regions[index]
        same_subject_index = subject_seen[subject_key]
        same_subject_total = subject_totals[subject_key]
        surface_identity_noun = _identity_surface(
            row,
            subject,
            prefer_instance_head=(
                same_subject_total > 1
                and int(row.get("bcc_significant_component_count") or 1) <= 1
            ),
        )
        phrase_index = same_subject_index
        phrase_total = same_subject_total
        if required_owner_id is not None:
            owner_key = (required_owner_id, subject_key)
            owner_subject_seen[owner_key] += 1
            phrase_index = owner_subject_seen[owner_key]
            phrase_total = owner_subject_totals[owner_key]
        safe_tag_phrase = _safe_tag_phrase(
            surface_identity_noun,
            allowed_identity_nouns,
            phrase_index,
            phrase_total,
            image_region=image_region,
            prefer_spatial_selector=bool(
                _terms_are_body_parts(_row_expected_semantic_terms(row))
            ) and required_owner_id is None,
        )
        owner_possessive = (
            _owner_possessive(rows[required_owner_id - 1])
            if required_owner_id is not None
            else "their"
        )
        context.append(
            {
                "id": index + 1,
                "crop_image": index + 3,
                "bbox_xywh": bbox,
                "image_region": image_region,
                "same_subject_index": same_subject_index,
                "same_subject_total": same_subject_total,
                "subject_anchor": subject,
                "surface_identity_noun": surface_identity_noun,
                "significant_component_count": int(row.get("bcc_significant_component_count") or 1),
                "composite_of_ids": [
                    overlay_by_mask_id[str(child.get("mask_id") or "")]
                    for child in row.get("bcc_composite_mask_children", [])
                    if str(child.get("mask_id") or "") in overlay_by_mask_id
                ],
                "composite_union_iou": row.get("bcc_composite_union_iou"),
                "composite_coverage": row.get("bcc_composite_coverage"),
                "allowed_identity_nouns": allowed_identity_nouns,
                "fallback_unique_selector": _required_unique_selector(
                    same_subject_index, same_subject_total
                ),
                "safe_tag_phrase": safe_tag_phrase,
                "required_owner_id": required_owner_id,
                "safe_tagged_phrase": _safe_tagged_phrase(
                    link_id=index + 1,
                    phrase=safe_tag_phrase,
                    owner_id=required_owner_id,
                    owner_possessive=owner_possessive,
                ),
                "sam3_proposal": row.get("source_prompt", ""),
                "optional_description": row.get("caption", ""),
                "optional_attributes": row.get("attributes", []),
                "synthetic_fill_rgb": row.get("inverse_background_rgb"),
            }
        )
    _deduplicate_safe_context(context, rows, boxes, subject_keys)
    context_by_id = {int(item["id"]): item for item in context}
    for item in context:
        child_ids = [int(value) for value in item.get("composite_of_ids") or []]
        if len(child_ids) < 2:
            continue
        tagged_children = [
            str(context_by_id[child_id]["safe_tagged_phrase"])
            for child_id in child_ids
        ]
        if len(tagged_children) == 2:
            collective = " and ".join(tagged_children)
        else:
            collective = ", ".join(tagged_children[:-1]) + ", and " + tagged_children[-1]
        example = f"[{item['id']}]{collective}[/{item['id']}]"
        item["safe_tagged_phrase"] = example
        item["composite_link_example"] = example
    return context


def _model_mask_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return visual identity context without legacy per-instance wording.

    ``_mask_context`` retains deterministic fallback phrases for internal
    normalization and diagnostics.  Those phrases often contain ordinal
    selectors, however, so exposing them to Qwen defeats the selective-link
    instruction.  The model instead receives explicit semantic peer IDs and
    may ground them with one shared plural span or omit any difficult member.
    """
    internal = _mask_context(rows)
    peer_ids_by_key: dict[tuple[str, int | None], list[int]] = defaultdict(list)
    keys: list[tuple[str, int | None]] = []
    for row, item in zip(rows, internal, strict=True):
        subject = str(item.get("subject_anchor") or "")
        owner = item.get("required_owner_id")
        key = (
            _semantic_subject_key(row, subject),
            int(owner) if owner is not None else None,
        )
        keys.append(key)
        peer_ids_by_key[key].append(int(item["id"]))

    hidden = {
        "same_subject_index",
        "fallback_unique_selector",
        "safe_tag_phrase",
        "safe_tagged_phrase",
        "composite_link_example",
    }
    output: list[dict[str, Any]] = []
    for item, key in zip(internal, keys, strict=True):
        visible = {name: value for name, value in item.items() if name not in hidden}
        peers = peer_ids_by_key[key]
        if len(peers) > 1:
            visible["collective_candidate_ids"] = peers
        output.append(visible)
    return output


def bcc_authoritative_rulebook() -> str:
    """Compact model-facing contract distilled from the paper and annotation guide."""
    return """# Authoritative BCC rulebook (concise)

1. The output is a complete set of correspondences for the caption you choose to write: each used visual
   instance has one group, and that group contains every textual mention of that same instance.
2. A group may contain several non-contiguous mentions (noun phrase, repeated name, pronoun, possessive, or
   reflexive). Cross-group overlap is valid; within one group, spans may not overlap.
3. Every concrete, tangible, visually groundable noun phrase that remains in the caption must link to a compatible
   supplied mask. If it has no mask, remove the noun phrase and any relation that depends on it. Meta words such as
   image/background/scene and abstract spatial regions are not entities and should not be written.
4. Span boundaries include the article/determiner, count, possessive, and pre-head attributes with the head noun.
   They exclude post-head prepositional phrases, participial clauses, relative clauses, and punctuation.
5. Possessives intentionally overlap: in "her dog", link "her" to the woman and the full "her dog" to the dog.
   Body parts are separate entities when a corresponding body-part mask exists.
6. One natural plural span may link multiple same-type instance masks. Use perfectly nested same-boundary tags;
   do not manufacture first/second/third inventories merely to consume masks.
7. Count plus selectors may split instances and share a plural anchor; a vague plural without instance markers is
   one group. Pronouns corefer with an existing instance, while an anaphoric "one" introduces another instance.
8. Stuff such as sky/water/grass is one group per type. Countable natural things such as trees, rocks, clouds, and
   mountains follow ordinary instance rules.
9. Masks and their descriptions are imperfect proposals, not commands. Inspect the original image and each mask,
   choose the subset that yields the most truthful and natural caption, and omit dubious or awkward masks.
10. Write connected, compositional scene prose—not a mask inventory. Use supported actions, relations, attributes,
    coordination, and coreference. Sentences must flow normally, with ordinary spacing and punctuation; never insert
    stray periods or commas inside a phrase or between a tag and its word.
11. Inline tags are only an encoding layer. Removing them must leave polished natural prose. Each opening tag uses a
    supplied one-based ID, tags are balanced and properly nested, and every later mention of a used entity is tagged.
12. The mask description is optional context. Never copy it mechanically, infer content from the synthetic inverse-
    crop exterior, or mention masks, overlays, crops, fill colors, or the annotation process.

Paper/guide-derived structural few-shots (form only; facts must come from the current images):
- Repeated/coreferential person mention, following the paper's Figure 1 umbrella example:
  [1]A person[/1] carries [2]a red umbrella[/2]. [1]She[/1] walks with [3][1]her[/1] dog[/3].
- One plural anchor shared by two instances, following the annotation guide's selector rules:
  [4][5]Two people[/5][/4] cross [6]a field[/6]; [4]one[/4] leads while [5]the other[/5] follows.
- Natural compact relation, following the paper's Figure 8 tennis example rather than a numbered list:
  On [7]a clay court[/7], [8]a player[/8] grips [9]a tennis racket[/9].
- Counter/content and possessive overlap, following the annotation guide's overlap rules:
  [10][11]A cup of coffee[/11][/10] sits beside [12][8]her[/8] bag[/12].
"""


def _annotation_rules(*, qa: bool = False) -> str:
    response_schema = (
        '{"keep":true,"reason_code":"ok","tagged_caption":"[1]A woman[/1] ..."}'
        if qa
        else '{"reject":false,"tagged_caption":"[1]A woman[/1] ..."}'
    )
    return f"""# Task: Natural caption with selective bidirectional correspondences

Create one training pair for Bidirectional Concept Correspondence (BCC): fluent scene text whose grounded
references link directly to the accepted visual instances they denote. The image already passed a separate
pre-caption gate with at least ten accepted masks. Ten is not a caption-coverage target.

{bcc_authoritative_rulebook()}

Coverage policy (important):
- Try to incorporate as many accepted masks as can be described accurately and naturally, ideally all of them.
- Deliberately inspect every accepted ID before omitting it. Prefer adding another accurately grounded mask when it
  can strengthen the same fluent description through a collective span, coordination, relation, or coreference.
- You do not need to mention every mask individually. Correct, natural prose outranks mask-count coverage.
- The supplied masks are imperfect proposals. If a mask is ambiguous, redundant, fragmentary, visually wrong, or
  would require an awkward phrase, simply leave its ID out.
  Do not force it into an ordinal catalog, malformed possessive, or invented relation. The pipeline will derive
  and report every omitted ID outside the caption; do not discuss omissions in the caption itself.
- One natural collective text span may link several visual masks. For example,
  [5][6][7]three cups[/7][/6][/5] gives IDs 5, 6, and 7 the same exact span. That collective span is sufficient;
  do not follow it with "the first cup, the second cup, the third cup" merely to give each mask separate wording.

Visual evidence:
- IMAGE 1 is the authoritative original image for the scene, actions, attributes, and relations.
- IMAGE 2 overlays a number on every accepted SAM3 instance mask.
- IMAGE 3 onward are matching inverse-mask crops in ID order. Only retained object pixels are evidence; every
  solid RGB exterior is synthetic and must be ignored.
- subject_anchor is the SAM3+spaCy identity prior and is strong identity context.
- optional_description and optional_attributes are context only. Rephrase or omit them; never assume every detail
  is true and never copy them merely to fill the caption.
- bbox_xywh and image_region distinguish instances. Never describe a region as an entity or background.
- No per-mask fallback wording is supplied. collective_candidate_ids identifies semantically compatible proposals
  that may share one natural plural span; it is permission to group them, not a requirement to use the whole group.
- composite_of_ids means a retained mask is a union of child IDs, not another physical object. If you use that
  composite ID, put its outer tag around a collective mention of its children; never invent an extra instance.
- If required_owner_id is present, preserve a natural owner/body-part relationship. Tags must not fuse words:
  write [8][1]her[/1] hair[/8], never "herhair", "herhernose", "theirface", or "her her nose".
- A person's body-part mask cannot denote an object-part compound. Omit an unsupported compound or action rather
  than relabeling a neck, face, or hand or inventing contact with the person's own body.

Caption rules:
- Write coherent, genuinely compositional prose using visually supported actions, relations, and attributes.
- allowed_identity_nouns across ACCEPTED_MASK_CONTEXT are the closed concrete-noun vocabulary. Omit an unmasked
  concrete entity even if it is visible or appears in an optional description.
- Every concrete noun phrase you do write must be tagged to one or more semantically compatible accepted IDs.
  A tangible noun phrase such as "a black jacket" therefore needs an accepted jacket ID; if none exists, delete
  that phrase and its dependent relation rather than leaving it untagged or attaching it to a person ID. Ordinary
  attributes such as "smiling" or "red" need no separate mask when they remain inside a supported entity phrase.
  Abstract descriptors such as color, texture, pattern, pose, and style need no separate mask.
- The mask description is optional context, not required text.
- When omitting an unmasked object, also remove its object-dependent relation. Never replace it with "nothing",
  an empty-handed claim, or invented self-contact.
- Favor informative attributes, fluent relations, coordination, possessives, and clear coreference.
- Do not concatenate descriptions, write one isolated sentence per mask, or repeat "is visible", "is present",
  "is shown", or "appears". At least one sentence should jointly describe multiple linked entities.
- Do not enumerate same-type objects with first/second/third or 1st/2nd/3rd labels. Use a natural collective span,
  visible distinctions where genuinely useful, or omit difficult instances. Never use ordinal labels for body parts.
- Proofread the tag-stripped prose for ordinary sentence flow, token boundaries, and punctuation. Fused possessives,
  stray periods inside phrases, fragments caused by deletion, and sequences such as ".," or ",." are invalid.
- Treat mask geometry as a relation sanity check. Claim holding, carrying, grasping, or contact only when the
  relevant accepted masks touch or lie immediately beside one another.
- Never mention an image, photo, crop, mask, overlay, fill color, background, backdrop, foreground, frame,
  scene area, or standalone spatial region.

Inline correspondence encoding:
- Wrap each linked occurrence as [ID]text[/ID], using a one-based ID from ACCEPTED_MASK_CONTEXT.
- Tags disappear from the stored caption and become exact character spans. Do not output a separate links list,
  character offsets, object_count, instance_ids, or long mask IDs.
- Only IDs used in the caption need tags. Unused accepted IDs are allowed and will be recorded automatically.
- A used ID needs a tagged concrete identity phrase, either its own phrase or a valid collective plural phrase.
  Tag every later noun phrase, synonym, pronoun, possessive, or reflexive referring to that used instance.
- To assign the same collective span to multiple masks, use perfectly nested same-boundary tags such as
  [5][6]the two mugs[/6][/5]. A shared singular phrase such as [5][6]a mug[/6][/5] is invalid.
- Tags may otherwise nest for cross-entity overlap but may never cross. If ID 1 is a woman and ID 4 is her dog,
  [4][1]her[/1] dog[/4] links "her" to ID 1 and "her dog" to ID 4.
- Within one ID, tagged spans may not overlap or self-nest. Include articles, counts, demonstratives, possessives,
  and pre-head attributes inside a noun-phrase tag; exclude post-head clauses and modifiers.
- If same-type instances are mentioned separately, use genuinely distinct visible phrases. Do not manufacture
  ordinal selectors. If they are mentioned collectively, one shared plural span is enough.
- Match complete tokens and preserve spaces between adjacent tagged words. Output no unknown IDs, crossing tags,
  unbalanced tags, or aliases absent from the caption.

Before answering, privately compose the natural caption first, then add balanced tags. Check that every concrete
reference is tagged to its true used ID, every collective span is plural, word boundaries remain natural, and you
used as many accepted masks as the evidence and fluent prose support. Do not target a numeric link quota; omit a
difficult mask instead of hallucinating, distorting the caption, or forcing it into prose.

Return exactly one compact JSON object and nothing else:
{response_schema}

Prompt version: {BCC_PROMPT_VERSION}
"""


def build_caption_image_packet(
    rows: list[dict[str, Any]],
    overlay_path: str | Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return the ordered BCC image packet and its explicit index manifest."""
    if not rows:
        raise ValueError("BCC captioning requires at least one accepted mask")
    source_path = str(rows[0].get("source_image_path") or "")
    ordered_paths = [source_path, str(overlay_path)]
    manifest: list[dict[str, Any]] = [
        {"image_number": 1, "role": "original_image"},
        {"image_number": 2, "role": "numbered_mask_overlay"},
    ]
    for index, row in enumerate(rows):
        inverse_path = str(row.get("inverse_crop_path") or "")
        ordered_paths.append(inverse_path)
        manifest.append(
            {
                "image_number": index + 3,
                "role": "inverse_mask_crop",
                "overlay_number": index + 1,
                "mask_id": str(row["mask_id"]),
                "inverse_background_rgb": row.get("inverse_background_rgb"),
            }
        )
    missing = [path for path in ordered_paths if not path or not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            f"BCC image packet is missing {len(missing)} input image(s): {missing[:3]}"
        )
    return ordered_paths, manifest


def _compact_input_manifest(
    input_manifest: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in input_manifest or []:
        out = {"image": item.get("image_number"), "role": item.get("role")}
        if item.get("overlay_number") is not None:
            out["id"] = item.get("overlay_number")
        compact.append(out)
    return compact


def _candidate_links(
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    id_by_mask = {str(row["mask_id"]): index + 1 for index, row in enumerate(rows)}
    return [
        {"id": id_by_mask.get(str(group.get("mask_id") or "")), "text": group.get("text") or []}
        for group in candidate.get("groups", [])
        if id_by_mask.get(str(group.get("mask_id") or "")) is not None
    ]

_INLINE_LINK_TAG_RE = re.compile(r"\[(?P<closing>/)?(?P<id>\d+)\]")
_COMPOSITE_CHILD_CONNECTOR_RE = re.compile(
    r"(?:\s|,|;|&|\band\b|\bor\b)*", re.IGNORECASE
)


def _repair_unclosed_composite_outer_tags(
    parsed: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Close only a geometry-proven composite around all of its child tags.

    Qwen sometimes copies the complete nested safe phrase but drops only the
    final outer close. Child geometry makes that boundary deterministic: the
    smallest outer span ends immediately after the final required child. No
    ordinary tag, partial child set, crossed nesting, or semantic text is
    repaired here.
    """
    tagged = parsed.get("tagged_caption")
    if not isinstance(tagged, str):
        return dict(parsed), []

    repaired = tagged
    repairs: list[dict[str, Any]] = []
    overlay_by_mask_id = {
        str(row.get("mask_id") or ""): index + 1
        for index, row in enumerate(rows)
    }
    for composite_id, row in enumerate(rows, start=1):
        child_ids = {
            overlay_by_mask_id[str(child.get("mask_id") or "")]
            for child in row.get("bcc_composite_mask_children", [])
            if str(child.get("mask_id") or "") in overlay_by_mask_id
        }
        if len(child_ids) < 2:
            continue

        tokens = list(_INLINE_LINK_TAG_RE.finditer(repaired))
        outer_opens = [
            token
            for token in tokens
            if int(token.group("id")) == composite_id
            and not token.group("closing")
        ]
        outer_closes = [
            token
            for token in tokens
            if int(token.group("id")) == composite_id
            and token.group("closing")
        ]
        if len(outer_opens) != 1 or outer_closes:
            continue
        outer = outer_opens[0]
        outer_index = tokens.index(outer)
        stack = [composite_id]
        direct_children: list[int] = []
        last_direct_close_end: int | None = None
        valid = True
        for token in tokens[outer_index + 1 :]:
            link_id = int(token.group("id"))
            closing = bool(token.group("closing"))
            if not closing:
                if len(stack) == 1:
                    gap_start = outer.end() if last_direct_close_end is None else last_direct_close_end
                    gap = repaired[gap_start : token.start()]
                    if (
                        link_id not in child_ids
                        or link_id in direct_children
                        or _COMPOSITE_CHILD_CONNECTOR_RE.fullmatch(gap) is None
                    ):
                        valid = False
                        break
                stack.append(link_id)
                continue
            if len(stack) <= 1 or stack[-1] != link_id:
                valid = False
                break
            stack.pop()
            if len(stack) == 1:
                direct_children.append(link_id)
                last_direct_close_end = token.end()
                if set(direct_children) == child_ids:
                    break

        if (
            not valid
            or set(direct_children) != child_ids
            or len(direct_children) != len(child_ids)
            or stack != [composite_id]
            or last_direct_close_end is None
        ):
            continue
        closing_tag = f"[/{composite_id}]"
        repaired = repaired[:last_direct_close_end] + closing_tag + repaired[last_direct_close_end:]
        repairs.append(
            {
                "id": composite_id,
                "child_ids": sorted(child_ids),
                "inserted_tag": closing_tag,
                "reason": "closed_geometry_proven_composite_after_all_child_tags",
            }
        )

    return {**parsed, "tagged_caption": repaired}, repairs




def _sentence_case_caption(text: str) -> str:
    """Repair lowercase sentence starts without changing any character offsets."""
    return re.sub(
        r"(^|[.!?]\s+)([a-z])",
        lambda match: match.group(1) + match.group(2).upper(),
        text,
    )


def _expand_tagged_caption(
    parsed: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Decode nested [ID]text[/ID] spans into the canonical link schema.

    Keeping correspondence markers next to the words they ground removes the
    model's error-prone second task of reproducing an independent list of exact
    strings. Nested tags retain valid cross-group overlap.
    """
    if "tagged_caption" not in parsed:
        return dict(parsed), [], {"type": "detached_links"}
    tagged = parsed.get("tagged_caption")
    if not isinstance(tagged, str):
        expanded = {**parsed, "caption": "", "links": []}
        return expanded, ["tagged_caption must be a string"], {"type": "inline_tags"}

    output: list[str] = []
    output_length = 0
    stack: list[tuple[int, int]] = []
    spans_by_id: dict[int, list[list[int]]] = {}
    errors: list[str] = []
    cursor = 0
    tag_count = 0
    for match in _INLINE_LINK_TAG_RE.finditer(tagged):
        chunk = tagged[cursor : match.start()]
        output.append(chunk)
        output_length += len(chunk)
        link_id = int(match.group("id"))
        closing = bool(match.group("closing"))
        tag_count += 1
        if not closing:
            if any(open_id == link_id for open_id, _ in stack):
                errors.append(f"link ID {link_id} has nested self-overlap")
            stack.append((link_id, output_length))
        elif not stack:
            errors.append(f"link ID {link_id} has a closing tag without an opening tag")
        elif stack[-1][0] != link_id:
            errors.append(f"inline tags are crossed: expected [/{stack[-1][0]}], got [/{link_id}]")
        else:
            _, start = stack.pop()
            if start == output_length:
                errors.append(f"link ID {link_id} has an empty inline mention")
            else:
                spans_by_id.setdefault(link_id, []).append([start, output_length])
        cursor = match.end()
    output.append(tagged[cursor:])
    raw_caption = "".join(output)
    caption = _sentence_case_caption(raw_caption)
    if stack:
        errors.extend(f"link ID {link_id} has an unclosed opening tag" for link_id, _ in reversed(stack))

    links = []
    for link_id in sorted(spans_by_id):
        spans = sorted(spans_by_id[link_id])
        links.append({"id": link_id, "text": [caption[start:end] for start, end in spans], "char_spans": spans, "_inline_tagged": True})
    expanded = {**parsed, "caption": caption, "links": links}
    metadata = {
        "type": "inline_tags",
        "tag_count": tag_count,
        "linked_id_count": len(links),
        "sentence_case_repaired": caption != raw_caption,
    }
    return expanded, errors, metadata


def _raw_record_matches_rows(
    record: dict[str, Any],
    rows: list[dict[str, Any]],
) -> bool:
    """Allow raw reuse only when its numbered-mask manifest is still identical."""
    if str(record.get("prompt_version") or "") != BCC_PROMPT_VERSION:
        return False
    manifest = record.get("bcc_input_manifest")
    if not isinstance(manifest, list):
        return False
    prior_ids = [
        str(item.get("mask_id") or "")
        for item in manifest
        if isinstance(item, dict) and item.get("role") == "inverse_mask_crop"
    ]
    current_ids = [str(row.get("mask_id") or "") for row in rows]
    return prior_ids == current_ids


def _is_current_correspondence_record(record: dict[str, Any]) -> bool:
    return (
        str(record.get("prompt_version") or "") == BCC_PROMPT_VERSION
        and str(record.get("schema_version") or "") == CORRESPONDENCE_SCHEMA_VERSION
        and str(record.get("stage_version") or "") == PIPELINE_STAGE_VERSION
    )


def build_caption_prompt(
    rows: list[dict[str, Any]],
    input_manifest: list[dict[str, Any]] | None = None,
) -> str:
    return (
        _annotation_rules()
        + "\nORDERED_INPUT_IMAGES:\n"
        + json.dumps(_compact_input_manifest(input_manifest), ensure_ascii=False, separators=(",", ":"))
        + "\nACCEPTED_MASK_CONTEXT:\n"
        + json.dumps(_model_mask_context(rows), ensure_ascii=False, separators=(",", ":"))
    )


def _prioritized_validation_errors(errors: list[str], limit: int = 24) -> list[str]:
    """Keep global completeness errors ahead of repetitive per-link diagnostics."""
    completeness_markers = (
        "accepted masks are missing groups",
        "required link IDs",
        "only ",
    )
    repair_markers = (
        "same-type instance links share",
        "inventory",
        "caption mention coverage",
        "composite mask",
        "grammatical number",
    )

    def priority(message: str) -> int:
        if any(marker in message for marker in completeness_markers):
            return 0
        if any(marker in message for marker in repair_markers):
            return 1
        return 2

    indexed = list(enumerate(str(value) for value in errors))
    indexed.sort(
        key=lambda item: (
            priority(item[1]),
            item[0],
        )
    )
    return [value for _, value in indexed[:limit]]


def build_qa_prompt(
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    input_manifest: list[dict[str, Any]] | None = None,
) -> str:
    mask_context = _model_mask_context(rows)
    candidate_links = _candidate_links(candidate, rows)
    present_ids = {
        int(link["id"]) for link in candidate_links if link.get("id") is not None
    }
    available_link_context = [
        {
            "id": int(item["id"]),
            "subject_anchor": str(item["subject_anchor"]),
            "surface_identity_noun": str(item["surface_identity_noun"]),
            "allowed_identity_nouns": list(item["allowed_identity_nouns"]),
            "required_owner_id": item.get("required_owner_id"),
            "collective_candidate_ids": list(item.get("collective_candidate_ids") or []),
            "composite_of_ids": list(item.get("composite_of_ids") or []),
            "significant_component_count": int(item.get("significant_component_count") or 1),
        }
        for item in mask_context
    ]
    unused_draft_links = [
        item for item in available_link_context if item["id"] not in present_ids
    ]
    proposed = {
        "caption": candidate.get("caption", ""),
        "links": candidate_links,
        "available_link_count": len(available_link_context),
        "available_link_context": available_link_context,
        "unused_draft_links": unused_draft_links,
    }
    proposed["draft_mention_audit"] = _draft_mention_audit(
        str(candidate.get("caption") or ""), rows
    )
    draft_errors = candidate.get("draft_validation_errors")
    if isinstance(draft_errors, list) and draft_errors:
        proposed["draft_validation_errors"] = _prioritized_validation_errors(draft_errors)
    return (
        """Independently verify and rewrite the proposed BCC pair as needed using all supplied images.
The accepted SAM3 masks are strong visual candidates, but the caption does not have to use every one. Preserve
as many grounded links as fit naturally and deliberately reconsider every unused ID. Prefer another accurate mask
when it can join fluent prose through a collective span, coordination, relation, or coreference. Do not target a
numeric link quota; omit an ambiguous, redundant, or awkward mask instead of forcing it into an ordinal inventory
or malformed phrase. optional_description is context, not required text.
The proposed plain caption and links are diagnostic input only. Return a fresh tagged_caption so correspondence
boundaries are encoded next to their words and the pipeline derives exact spans.
If draft_validation_errors are listed, correct every one; they are why this draft needs pass-two repair.
Every draft_mention_audit item marked remove_unmasked_concrete_noun_phrase is a mandatory deletion, not a
suggestion: remove that physical entity and rewrite its whole relation rather than retagging or paraphrasing it.
Every other concrete, tangible noun phrase must likewise have its own semantically compatible accepted ID. For
example, "a black jacket" needs a jacket mask; a person mask does not ground the jacket. If no such ID exists,
remove the jacket phrase and any relation that depends on it rather than leaving it untagged.
Never replace a removed held object with "nothing", an empty-handed claim, or invented contact between the person's
own body parts. Omit the object-dependent relation; use a neutral body-part pose only when IMAGE 1 directly proves it.
A body-part ID cannot absorb an object-part compound: never link a person's neck to "guitar neck" or a person's
face to "clock face". Delete an unmasked object-part compound and rewrite the surrounding clause instead of
retagging it.
Audit every surviving action and relation against IMAGE 1. In particular, a hand on a guitar neck is not a hand
resting on the person's neck, even after the unmasked guitar phrase is deleted.
AVAILABLE_LINK_CONTEXT lists possibilities, not mandatory caption slots. UNUSED_DRAFT_LINKS may remain unused.
When several accepted masks share one natural plural reference, assign that exact span to all of them with nested
same-boundary tags, for example [5][6]the two cups[/6][/5]. Do not add separate first/second labels afterward.
If you use a composite_of_ids entry, its outer tag should enclose the linked collective child mention; the union
mask is not an extra physical object.
Do not preserve draft wording for its own sake. Rewrite inventories and repeated visibility clauses into coherent,
compositional prose. keep:true certifies exact correspondence validity and caption quality.
Return compact JSON only:
{"keep":true,"reason_code":"ok","tagged_caption":"[1]A woman[/1] ..."}

"""
        + _annotation_rules(qa=True)
        + "\nPROPOSED_RECORD:\n"
        + json.dumps(proposed, ensure_ascii=False, separators=(",", ":"))
        + "\nORDERED_INPUT_IMAGES:\n"
        + json.dumps(_compact_input_manifest(input_manifest), ensure_ascii=False, separators=(",", ":"))
        + "\nACCEPTED_MASK_CONTEXT:\n"
        + json.dumps(mask_context, ensure_ascii=False, separators=(",", ":"))
    )


_UNMASKED_ERROR_PHRASE_RE = re.compile(
    r"unmasked concrete noun phrase (?P<quote>['\"])(?P<text>.*?)(?P=quote) at "
)
_UNSUPPORTED_OBJECT_ERROR_PHRASE_RE = re.compile(
    r"unsupported-object evasion (?P<quote>['\"])(?P<text>.*?)(?P=quote);"
)
_ADJACENT_DUPLICATE_ERROR_PHRASE_RE = re.compile(
    r"adjacent duplicate linked phrase (?P<quote>['\"])(?P<text>.*?)(?P=quote);"
)
_PLURAL_SELECTOR_ERROR_PHRASE_RE = re.compile(
    r"plural collection mask ID \d+ needs a descriptive phrase, not (?P<quote>['\"])(?P<text>.*?)(?P=quote)"
)


def _forbidden_error_phrases(errors: list[str]) -> list[str]:
    phrases: list[str] = []
    for error in errors:
        for pattern in (
            _UNMASKED_ERROR_PHRASE_RE,
            _UNSUPPORTED_OBJECT_ERROR_PHRASE_RE,
            _ADJACENT_DUPLICATE_ERROR_PHRASE_RE,
            _PLURAL_SELECTOR_ERROR_PHRASE_RE,
        ):
            match = pattern.search(str(error))
            if match and match.group("text") not in phrases:
                phrases.append(match.group("text"))
    return phrases


def build_schema_repair_prompt(
    raw: str,
    errors: list[str],
    rows: list[dict[str, Any]],
    *,
    qa: bool,
) -> str:
    """Build a focused retry after a prior visual answer made its judgments."""
    mask_context = _model_mask_context(rows)
    available_link_context = [
        {
            "id": int(item["id"]),
            "subject_anchor": str(item["subject_anchor"]),
            "surface_identity_noun": str(item["surface_identity_noun"]),
            "allowed_identity_nouns": list(item["allowed_identity_nouns"]),
            "required_owner_id": item.get("required_owner_id"),
            "collective_candidate_ids": list(item.get("collective_candidate_ids") or []),
            "composite_of_ids": list(item.get("composite_of_ids") or []),
            "significant_component_count": int(item.get("significant_component_count") or 1),
        }
        for item in mask_context
    ]
    allowed_identity_nouns_by_id = [
        {
            "id": int(item["id"]),
            "allowed_identity_nouns": list(item["allowed_identity_nouns"]),
        }
        for item in mask_context
    ]
    available_ids = [item["id"] for item in available_link_context]
    forbidden_exact_phrases = _forbidden_error_phrases(errors)
    prioritized_errors = _prioritized_validation_errors(errors)
    response_shape = (
        '{"keep":true,"reason_code":"corrected_links","tagged_caption":"[1]...[/1]"}'
        if qa
        else '{"reject":false,"tagged_caption":"[1]...[/1]"}'
    )
    previous_caption = ""
    previous_present_ids: set[int] = set()
    try:
        previous_parsed = extract_json(str(raw or ""))
    except Exception:
        previous_parsed = {}
    if isinstance(previous_parsed, dict):
        expanded, _, _ = _expand_tagged_caption(previous_parsed)
        previous_caption = str(expanded.get("caption") or "")
        previous_present_ids = {
            int(link["id"])
            for link in expanded.get("links", [])
            if isinstance(link, dict) and link.get("id") is not None
        }
    unused_previous_ids = [
        link_id for link_id in available_ids if link_id not in previous_present_ids
    ]
    previous_mention_audit = _draft_mention_audit(previous_caption, rows)
    return (
        f"""Repair every JSON, inline-correspondence, and caption-quality error in the previous answer.
The available visual IDs are {available_ids}, but they are not all mandatory. Deliberately reconsider every unused
ID and incorporate as many accepted masks as can be grounded accurately and naturally. Prefer a fluent collective
span, coordination, relation, or coreference over a separate inventory entry. Do not target a numeric link quota.
An ID used in the caption needs a balanced identity or collective phrase, but an awkward or ambiguous ID may still
be omitted. Never repair one link by inventing wording for another.

Rewrite freely, but keep visual claims supported. Use only ALLOWED_IDENTITY_NOUNS_BY_ID for concrete entities.
Every concrete, tangible, visually groundable noun phrase in the final caption must be tagged to one or more
semantically compatible accepted IDs. For example, "a black jacket" requires an accepted jacket ID; a person mask
does not ground the jacket. If no compatible ID exists, delete that noun phrase and every relation that depends on
it. Do not merely remove its tags, transfer it to a nearby mask, or retain it as unlinked scene context. Ordinary
attributes that do not introduce another concrete object may remain within their supported entity phrase.
Delete every unmasked noun phrase named by the mention audit. Every case-insensitive phrase in
FORBIDDEN_EXACT_PHRASES must be absent; delete its surrounding modifier/clause instead of moving, retagging, or
paraphrasing it. Delete a removed held object's object-dependent relation; never replace it with invented contact
between the person's own body parts, "holds nothing", or "empty-handed". A neutral body-part pose is allowed only
when the original image directly proves it. Give same-type IDs distinct, visually supported phrases when they are
described separately. Prefer dropping a difficult
ID over using ordinal fallback wording. Preserve grammatical number and compose entities naturally rather than
making a one-sentence-per-mask inventory.
A body-part ID cannot absorb an object-part compound: never link a person's neck to "guitar neck" or a person's
face to "clock face". Delete an unmasked object-part compound and rewrite its clause instead of retagging it to
the body-part ID.
For an included composite_of_ids entry, use an outer collective span around its included children. Never describe
a composite union as another numbered instance or repeat child phrases solely to satisfy IDs.

Encode links inline beside their grounded words. Tags may nest for cross-entity overlap, such as
[8][1]his[/1] hand[/8], but may never cross or self-nest. Put each entire noun phrase inside its tags: include its
article, count, possessive, and pre-head modifiers; exclude post-head prepositional, relative, and participial
modifiers. Tag every repeated noun phrase, pronoun, possessive, and reflexive occurrence. Return no detached links
or offsets. Several masks may share one natural plural span via same-boundary nested tags, for example
[5][6][7]three cups[/7][/6][/5]; that collective span needs no separate per-mask ordinal phrases. Preserve spaces
between adjacent tagged words and eliminate fused forms such as "herhair", "theirface", and "herhernose".
Before responding, strip the tags and scan every concrete noun phrase/reference: it is correctly tagged, abstract,
or removed. Return exactly this compact shape:
"""
        + response_shape
        + "\nAVAILABLE_LINK_COUNT:\n"
        + str(len(available_ids))
        + "\nAVAILABLE_LINK_IDS:\n"
        + json.dumps(available_ids, separators=(",", ":"))
        + "\nUNUSED_LINK_IDS_AFTER_PREVIOUS_ANSWER:\n"
        + json.dumps(unused_previous_ids, separators=(",", ":"))
        + "\nAVAILABLE_LINK_CONTEXT:\n"
        + json.dumps(available_link_context, ensure_ascii=False, separators=(",", ":"))
        + "\nFORBIDDEN_EXACT_PHRASES:\n"
        + json.dumps(forbidden_exact_phrases, ensure_ascii=False, separators=(",", ":"))
        + "\nVALIDATION_ERRORS:\n"
        + json.dumps(
            prioritized_errors,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\nALLOWED_IDENTITY_NOUNS_BY_ID:\n"
        + json.dumps(
            allowed_identity_nouns_by_id,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\nPREVIOUS_CAPTION_MENTION_AUDIT:\n"
        + json.dumps(
            previous_mention_audit,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\nPREVIOUS_VISUAL_ANSWER:\n"
        + str(raw or "")[:24000]
        + "\nFINAL_CHECK:\n"
        + "Opening-tag IDs are a valid subset of "
        + json.dumps(available_ids, separators=(",", ":"))
        + "; IDs currently unused (which may stay unused) are "
        + json.dumps(unused_previous_ids, separators=(",", ":"))
        + "; forbidden phrases that must be absent are "
        + json.dumps(
            forbidden_exact_phrases,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + """. Every included composite entry is a collective, not an additional instance. Every concrete noun phrase
is wholly tagged to one or more compatible IDs or deleted. Do not place an
article or adjective outside a safe tag: write [6]one patterned shirt[/6], never a patterned [6]one shirt[/6]. Scan
all pronouns and possessives and tag every coreference. Output only """
        + response_shape
    )


def _mention_pattern(text: str) -> str:
    left = r"(?<!\w)" if text and text[0].isalnum() else ""
    right = r"(?!\w)" if text and text[-1].isalnum() else ""
    return left + re.escape(text) + right


def _relaxed_mention_match(caption: str, mention: str) -> re.Match[str] | None:
    """Repair a unique determiner/possessive mismatch without fuzzy guessing."""
    words = mention.split()
    if len(words) < 2 or words[0].casefold() not in {
        "a", "an", "the", "his", "her", "its", "their", "this", "that",
    }:
        return None
    core = " ".join(words[1:])
    matches = list(
        re.finditer(_mention_pattern(core), caption, flags=re.IGNORECASE)
    )
    return matches[0] if len(matches) == 1 else None


def _locate_mentions(
    caption: str,
    mentions: list[str],
) -> tuple[list[str], list[list[int]], list[str]]:
    """Locate exact mentions, repairing only unique leading-determiner drift."""
    canonical_mentions: list[str] = []
    spans: list[list[int]] = []
    cursor_by_text: dict[str, int] = {}
    errors: list[str] = []
    for mention in mentions:
        mention = str(mention).strip().rstrip(".,;:!?")
        if not mention:
            continue
        key = mention.casefold()
        start_at = cursor_by_text.get(key, 0)
        matches = list(re.finditer(_mention_pattern(mention), caption, flags=re.IGNORECASE))
        match = next((candidate for candidate in matches if candidate.start() >= start_at), None)
        if match is None:
            if matches and key in cursor_by_text:
                continue
            match = _relaxed_mention_match(caption, mention)
        if match is None:
            errors.append(f"mention text is absent from caption: {mention!r}")
            continue
        start, end = match.span()
        canonical_mentions.append(caption[start:end])
        spans.append([start, end])
        cursor_by_text[key] = end
    return canonical_mentions, spans, errors
def _drop_lexically_nested_mentions(
    mentions: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Drop redundant noun aliases before ambiguous occurrence matching."""
    cleaned = [str(value).strip().rstrip(".,;:!?") for value in mentions]
    kept: list[str] = []
    repairs: list[dict[str, Any]] = []
    for index, mention in enumerate(cleaned):
        if not mention:
            continue
        if mention.casefold() in _REFERENCE_ONLY_TERMS:
            kept.append(mention)
            continue
        nested_in = next(
            (
                other
                for other_index, other in enumerate(cleaned)
                if other_index != index
                and len(other) > len(mention)
                and re.search(
                    _mention_pattern(mention), other, flags=re.IGNORECASE
                )
            ),
            "",
        )
        if nested_in:
            repairs.append(
                {
                    "dropped_text": mention,
                    "nested_in": nested_in,
                    "reason": "redundant_lexically_nested_model_mention",
                }
            )
            continue
        kept.append(mention)
    return kept, repairs

def _normalize_within_group_mentions(
    texts: list[str],
    spans: list[list[int]],
) -> tuple[list[str], list[list[int]], list[dict[str, Any]], list[str]]:
    """Drop only redundant contained mentions; retain partial-overlap errors."""
    dropped: set[int] = set()
    errors: list[str] = []
    for left_index, left in enumerate(spans):
        for right_index in range(left_index + 1, len(spans)):
            right = spans[right_index]
            if max(left[0], right[0]) >= min(left[1], right[1]):
                continue
            left_contains = left[0] <= right[0] and left[1] >= right[1]
            right_contains = right[0] <= left[0] and right[1] >= left[1]
            if not left_contains and not right_contains:
                errors.append("contains partially overlapping spans")
                continue
            left_length = left[1] - left[0]
            right_length = right[1] - right[0]
            if left_length == right_length:
                dropped.add(right_index)
            elif left_length < right_length:
                dropped.add(left_index)
            else:
                dropped.add(right_index)
    kept = sorted(
        (index for index in range(len(spans)) if index not in dropped),
        key=lambda index: (spans[index][0], spans[index][1]),
    )
    repairs = [
        {
            "dropped_text": texts[index],
            "dropped_span": spans[index],
            "reason": "redundant_nested_within_group_mention",
        }
        for index in sorted(dropped)
    ]
    return (
        [texts[index] for index in kept],
        [spans[index] for index in kept],
        repairs,
        errors,
    )

def _sanitize_dense_caption(caption: str) -> tuple[str, list[str]]:
    """Remove synthetic/background role wording while retaining real entities."""
    rewritten, count = _DENSE_BACKGROUND_ROLE_RE.subn(" is present", caption)
    corrections = [f"removed {count} image-level background-role clause(s)"] if count else []
    rewritten, capture_count = re.subn(r"\bcaptured in motion\b", "moving", rewritten, flags=re.IGNORECASE)
    if capture_count:
        corrections.append(f"removed {capture_count} image-capture phrase(s)")
    for pattern, replacement in _DENSE_BACKGROUND_PHRASE_RES:
        rewritten, pattern_count = pattern.subn(replacement, rewritten)
        if pattern_count:
            corrections.append(f"removed {pattern_count} background phrase(s)")
    rewritten, predicate_count = _DENSE_BACKGROUND_PREDICATE_RE.subn(
        lambda match: (
            "is visible"
            if match.group("verb").casefold().endswith("s")
            else "are visible"
        ),
        rewritten,
    )
    if predicate_count:
        corrections.append(
            f"removed {predicate_count} image-level background predicate(s)"
        )
    rewritten = re.sub(r"\s+([.,;:!?])", r"\1", rewritten)
    # A removed sentence-leading phrase such as ``In the foreground,`` used
    # to leave ``.,`` between the surrounding clauses.  This is generated by
    # our deterministic cleanup, not by the model.  Collapse only objectively
    # malformed punctuation pairs, then restore sentence case.  Group spans
    # are realigned from their inline-tag text below, so these edits cannot
    # silently shift a correspondence.
    rewritten, punctuation_count = re.subn(
        r"([.!?])\s*[,;:]+\s*", r"\1 ", rewritten
    )
    rewritten, reverse_punctuation_count = re.subn(
        r"[,;:]\s*([.!?])", r"\1", rewritten
    )
    if punctuation_count or reverse_punctuation_count:
        corrections.append(
            "repaired "
            f"{punctuation_count + reverse_punctuation_count} malformed punctuation pair(s)"
        )
    rewritten = re.sub(r"[ \t]{2,}", " ", rewritten).strip()
    sentence_cased = _sentence_case_caption(rewritten)
    if sentence_cased != rewritten:
        corrections.append("restored sentence-initial capitalization after cleanup")
    rewritten = sentence_cased
    return rewritten, corrections


def _resolve_mask_id(mask_id: str, valid_mask_ids: set[str]) -> str:
    """Repair a corrupted shared image prefix only when the mask suffix is unique."""
    if mask_id in valid_mask_ids:
        return mask_id
    suffix_match = re.search(r"(_p\d+_m\d+)$", mask_id)
    if suffix_match is None:
        return mask_id
    suffix = suffix_match.group(1)
    matches = [candidate for candidate in valid_mask_ids if candidate.endswith(suffix)]
    return matches[0] if len(matches) == 1 else mask_id


def _resolve_group_mask_id(
    group: dict[str, Any],
    rows: list[dict[str, Any]],
    valid_mask_ids: set[str],
) -> tuple[str, str]:
    """Resolve compact overlay IDs while retaining legacy mask-ID compatibility."""
    if group.get("id") is not None or group.get("overlay_number") is not None:
        raw = group.get("id", group.get("overlay_number"))
        try:
            overlay_id = int(raw)
        except (TypeError, ValueError):
            return "", str(raw)
        if 1 <= overlay_id <= len(rows):
            return str(rows[overlay_id - 1]["mask_id"]), str(raw)
        return "", str(raw)
    raw_mask_id = str(group.get("mask_id") or "").strip()
    return _resolve_mask_id(raw_mask_id, valid_mask_ids), raw_mask_id


def _canonicalize_semantic_term(term: str) -> str:
    # Keep this private name for compatibility with older tests/callers, but
    # normalization no longer consults a hand-authored alias dictionary.
    return normalize_semantic_term(term)


@lru_cache(maxsize=32_768)
def _cached_canonical_semantic_terms(text: str) -> frozenset[str]:
    """Cache normalized spaCy noun lemmas without collapsing taxonomy senses."""
    terms = {
        _canonicalize_semantic_term(term)
        for term in semantic_noun_lemmas(text)
        if len(term) > 1
        if term.casefold() not in _REFERENCE_ONLY_TERMS
    }
    # spaCy singularizes plural-only visual identities such as ``eyeglasses``
    # into a different WordNet sense (a monocle). Preserve those exact forms.
    terms.update(
        _canonicalize_semantic_term(raw_term)
        for raw_term in re.findall(r"[A-Za-z]+", str(text or "").casefold())
        if raw_term in _PLURAL_ONLY_IDENTITY_SURFACES
    )
    return frozenset(terms)


def _canonical_semantic_terms(text: str) -> set[str]:
    """Return an independently mutable view of cached semantic identity terms."""
    return set(_cached_canonical_semantic_terms(str(text or "")))


def _body_part_selector_terms(text: str) -> set[str]:
    """Return spatial selector nouns that do not change body-part identity."""
    return {"side"} if _HYPHENATED_SIDE_SELECTOR_RE.search(str(text or "")) else set()


def _row_expected_semantic_terms(row: dict[str, Any]) -> set[str]:
    """Return the first authoritative semantic identity carried by a mask row."""
    for key in ("main_candidate", "object", "source_prompt", "caption"):
        terms = _canonical_semantic_terms(str(row.get(key) or "").strip())
        if terms:
            return terms
    return set()


def _semantic_terms_match(left: set[str], right: set[str]) -> bool:
    return semantic_terms_compatible(left, right)


def _semantic_common_terms(left: set[str], right: set[str]) -> set[str]:
    """Return left-side labels supported by exact/synset/one-hop evidence."""
    return {
        term
        for term in left
        if semantic_terms_compatible({term}, right)
    }


def _terms_include_category(terms: set[str], category: str) -> bool:
    # This bounded ancestry query is only a structural classifier (for example
    # choosing person pronouns). Link compatibility itself remains exactly one
    # hop as enforced by semantic_matches().
    return any(semantic_is_a(term, category) for term in terms)


def _terms_are_body_parts(terms: set[str]) -> bool:
    return bool(terms & _BODY_PART_TERMS) or any(
        semantic_is_a(term, "body part", max_depth=5) for term in terms
    )


def _terms_are_clothing(terms: set[str]) -> bool:
    return bool(terms & _CLOTHING_TERMS) or any(
        semantic_is_a(term, "clothing", max_depth=5)
        or semantic_is_a(term, "garment", max_depth=5)
        for term in terms
    )


def _semantic_subject_key(row: dict[str, Any], fallback: str = "") -> str:
    terms = _row_expected_semantic_terms(row)
    base = "|".join(sorted(terms)) if terms else str(fallback).casefold()
    if _terms_include_category(terms, "person"):
        anchor_words = set(re.findall(r"[a-z]+", str(fallback).casefold()))
        specific = anchor_words & {"boy", "girl", "man", "woman"}
        if len(specific) == 1:
            return f"{base}:{next(iter(specific))}"
    return base


def _allowed_identity_nouns(row: dict[str, Any]) -> list[str]:
    """Expose the checker's generated same-synset/one-hop noun vocabulary."""
    expected = _row_expected_semantic_terms(row)
    allowed = set(expected)
    for term in expected:
        allowed.update(taxonomy_alternatives(term))
    anchor = str(
        row.get("main_candidate")
        or row.get("object")
        or row.get("source_prompt")
        or ""
    )
    allowed.update(
        token.casefold()
        for token in re.findall(r"[A-Za-z]+", anchor)
        if len(token) > 1
        if token.casefold() not in _REFERENCE_ONLY_TERMS
    )
    return sorted(allowed)


def _required_unique_selector(index: int, total: int) -> str:
    if total <= 1:
        return ""
    if total == 2 and index == 1:
        return "one"
    if total == 2 and index == 2:
        return "the other"
    ordinal_words = (
        "",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
    )
    if 0 < index < len(ordinal_words):
        return f"the {ordinal_words[index]}"
    suffix = "th"
    if index % 100 not in {11, 12, 13}:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(index % 10, "th")
    return f"the {index}{suffix}"


def _safe_tag_phrase(
    subject: str,
    allowed_identity_nouns: list[str],
    index: int,
    total: int,
    *,
    image_region: str = "",
    prefer_spatial_selector: bool = False,
) -> str:
    """Give the model a grammatical, validator-safe fallback phrase per ID."""
    noun = re.sub(r"\s+", " ", str(subject or "").strip().casefold())
    if not noun:
        noun = next(iter(allowed_identity_nouns), "object")
    if total > 1 and image_region and (
        _surface_is_grammatically_plural(noun) or prefer_spatial_selector
    ):
        if noun == "hair":
            noun = "hairstyle"
        adjective = {
            "upper-left": "upper-left",
            "upper-center": "upper",
            "upper-right": "upper-right",
            "middle-left": "left-side",
            "middle-center": "central",
            "middle-right": "right-side",
            "lower-left": "lower-left",
            "lower-center": "lower",
            "lower-right": "lower-right",
        }.get(image_region, image_region)
        return f"the {adjective} {noun}"
    if total <= 1:
        return f"the {noun}"
    selector = _required_unique_selector(index, total)
    return f"{selector} {noun}".strip() if selector else f"the {noun}"


def _surface_is_grammatically_plural(surface: str) -> bool:
    head = str(surface or "").casefold().rsplit(" ", 1)[-1]
    return (
        str(surface or "").casefold() in _PLURAL_ONLY_IDENTITY_SURFACES
        or head in {"children", "feet", "men", "people", "teeth", "women"}
        or (head.endswith("s") and not head.endswith(("ss", "us", "is")))
    )


def _pluralize_identity_surface(noun: str) -> str:
    parts = noun.rsplit(" ", 1)
    head = parts[-1]
    irregular = {
        "child": "children",
        "foot": "feet",
        "man": "men",
        "mouse": "mice",
        "person": "people",
        "tooth": "teeth",
        "woman": "women",
    }
    plural = irregular.get(head)
    if plural is None:
        if head.endswith(("s", "x", "z", "ch", "sh")):
            plural = head + "es"
        elif len(head) > 1 and head.endswith("y") and head[-2] not in "aeiou":
            plural = head[:-1] + "ies"
        else:
            plural = head + "s"
    parts[-1] = plural
    return " ".join(parts)


def _safe_tagged_phrase(
    *,
    link_id: int,
    phrase: str,
    owner_id: int | None,
    owner_possessive: str,
) -> str:
    if owner_id is None:
        return f"[{link_id}]{phrase}[/{link_id}]"
    if phrase.casefold().startswith("one "):
        owned_noun = phrase[4:]
        return (
            f"[{link_id}]one of [{owner_id}]{owner_possessive}[/{owner_id}] "
            f"{_pluralize_identity_surface(owned_noun)}[/{link_id}]"
        )
    owned_phrase = re.sub(r"^the\s+", "", phrase, flags=re.IGNORECASE)
    return (
        f"[{link_id}][{owner_id}]{owner_possessive}[/{owner_id}] "
        f"{owned_phrase}[/{link_id}]"
    )


def _draft_mention_audit(
    caption: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Give repair passes an occurrence-level checklist without trusting the draft links."""
    person_ids = [
        index + 1
        for index, row in enumerate(rows)
        if _terms_include_category(_row_expected_semantic_terms(row), "person")
    ]
    audited: list[dict[str, Any]] = []
    occurrence_by_surface: Counter[str] = Counter()
    for mention in caption_entity_mentions(caption):
        text = str(mention["text"])
        occurrence_by_surface[text.casefold()] += 1
        if mention["kind"] == "person_reference":
            compatible_ids = (
                person_ids
                if text.casefold() in _PERSON_ONLY_REFERENCE_TERMS
                else list(range(1, len(rows) + 1))
            )
        else:
            compatible_ids = [
                index + 1
                for index, row in enumerate(rows)
                if _mention_semantically_compatible(mention, row)
            ]
        if compatible_ids:
            instruction = "link_exact_occurrence"
        elif str(mention.get("head") or "") in _NON_ENTITY_MENTION_HEADS:
            instruction = "abstract_or_spatial_not_linked"
        else:
            instruction = "remove_unmasked_concrete_noun_phrase"
        audited.append(
            {
                "text": text,
                "occurrence": occurrence_by_surface[text.casefold()],
                "compatible_ids": compatible_ids,
                "instruction": instruction,
            }
        )
    return audited


def _semantic_link_evidence(
    texts: list[str],
    row: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    expected_source = ""
    expected_text = ""
    expected: set[str] = set()
    for key in ("main_candidate", "object", "source_prompt", "caption"):
        candidate_text = str(row.get(key) or "").strip()
        candidate_terms = _canonical_semantic_terms(candidate_text)
        if candidate_terms:
            expected_source = key
            expected_text = candidate_text
            expected = candidate_terms
            break
    joined_mentions = " ".join(texts)
    mentioned = _canonical_semantic_terms(joined_mentions)
    # spaCy can tag the semantic head supplied by SAM3 as a compound modifier
    # (for example, ``fern`` in ``green fern fronds``). Preserve noun-based
    # extraction, but also admit an expected term when it occurs as an exact
    # lexical token in the linked phrase. Whole-token matching avoids the old
    # substring failure mode (for example ``he`` inside ``the``).
    mention_tokens = {
        token.casefold() for token in re.findall(r"[A-Za-z]+", joined_mentions)
    }
    mentioned.update(
        _canonicalize_semantic_term(token)
        for token in mention_tokens
        if semantic_terms_compatible(expected, {_canonicalize_semantic_term(token)})
    )
    taxonomy_matches = semantic_matches(expected, mentioned)
    matched = {match.expected for match in taxonomy_matches}
    conflicting_compound_terms = (
        sorted(
            term
            for term in mentioned - _body_part_selector_terms(joined_mentions)
            if not semantic_terms_compatible(expected, {term})
        )
        if _terms_are_body_parts(expected) else []
    )
    compatible = (bool(matched) if expected else bool(mentioned)) and not conflicting_compound_terms
    evidence = {
        "expected_source": expected_source,
        "expected_text": expected_text,
        "expected_terms": sorted(expected),
        "mention_terms": sorted(mentioned),
        "matched_terms": sorted(matched),
        "taxonomy_policy": "oewn:2025 same synset or direct hypernym/hyponym",
        "taxonomy_matches": [match.as_dict() for match in taxonomy_matches],
        "compatible": compatible,
        "conflicting_compound_terms": conflicting_compound_terms,
    }
    if not mentioned:
        return evidence, "has no concrete identity noun phrase"
    if conflicting_compound_terms:
        return evidence, (
            "body-part identity phrase contains a conflicting concrete compound noun "
            f"{conflicting_compound_terms}"
        )
    if expected and not compatible:
        return evidence, (
            "identity phrase is incompatible with mask subject "
            f"(expected {sorted(expected)}, got {sorted(mentioned)})"
        )
    return evidence, None


_GENERIC_SEMANTIC_HEADS = {"group", "pair", "part", "piece", "row", "set"}


def _mention_semantically_compatible(
    mention: dict[str, Any],
    row: dict[str, Any],
) -> bool:
    """Match an entity's head, not an embedded modifier such as car in car mirror."""
    expected = _row_expected_semantic_terms(row)
    head = str(mention.get("head") or "").casefold()
    head_terms = _canonical_semantic_terms(head)
    if not expected:
        return bool(head_terms)
    if _semantic_terms_match(expected, head_terms):
        noun_terms = {
            _canonicalize_semantic_term(str(term))
            for term in mention.get("noun_terms") or []
            if len(str(term)) > 1
        }
        if (
            _terms_are_body_parts(expected)
            and noun_terms
            and any(
                not semantic_terms_compatible(expected, {term})
                for term in noun_terms
                - _body_part_selector_terms(str(mention.get("text") or ""))
            )
        ):
            return False
        return True
    if _terms_are_clothing(expected):
        if _terms_are_clothing(head_terms):
            return True
    if head in _GENERIC_SEMANTIC_HEADS:
        evidence, error = _semantic_link_evidence([str(mention.get("text") or "")], row)
        return error is None and bool(evidence.get("compatible"))
    return False


def _shared_span_phrase(group: dict[str, Any], span: tuple[int, int]) -> str:
    for text, candidate in zip(
        group.get("text") or [], group.get("char_spans") or [], strict=True
    ):
        if tuple(int(value) for value in candidate) == span:
            return str(text)
    return ""


def _is_collective_phrase(phrase: str, common_type: set[str]) -> bool:
    """Accept an exact shared span only when its wording is genuinely collective."""
    normalized = re.sub(r"\s+", " ", str(phrase or "").strip().casefold())
    if not normalized:
        return False
    singular_lead = re.match(r"^(?:a|an|one)\b", normalized)
    article_led_collective = any(
        re.match(
            rf"^(?:a|an|one)\s+(?:[a-z-]+\s+){{0,3}}{re.escape(head)}\s+of\b",
            normalized,
        )
        for head in _ARTICLE_LED_COLLECTIVE_HEADS
    )
    # A singular article normally proves that one span cannot identify several
    # instances ("a cup").  Collective counter/content constructions are the
    # exception: TEXT_ANNOTATION.md keeps phrases such as "a herd of cows" as
    # one fixed span, and that span may anchor the constituent instance masks.
    if singular_lead and not article_led_collective:
        return False
    # Semantic validation stores canonical identities (for example ``sneaker``
    # is canonicalized to ``shoe``), while the caption should remain free to use
    # any natural member of that alias set.  Test plural surface tokens through
    # the same canonicalizer instead of comparing only literal pluralizations
    # of the canonical label.
    for token in re.findall(r"[a-z]+", normalized):
        if (
            _surface_is_grammatically_plural(token)
            and semantic_terms_compatible(common_type, _canonical_semantic_terms(token))
        ):
            return True
    plural_terms = {
        _pluralize_identity_surface(term)
        for term in common_type
        if term and _pluralize_identity_surface(term) != term
    }
    if any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized)
        for term in plural_terms
    ):
        return True
    plural_surface_terms = {
        term for term in common_type if _surface_is_grammatically_plural(term)
    }
    return bool(
        re.search(
            r"\b(?:both|several|multiple|many|few|pair|group|row|cluster|"
            r"two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
            normalized,
        )
        and any(
            re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized)
            for term in plural_surface_terms
        )
    )


def _groups_share_collective_span(
    left: dict[str, Any], right: dict[str, Any]
) -> bool:
    common_type = _semantic_common_terms(
        set(left.get("semantic_validation", {}).get("expected_terms") or []),
        set(right.get("semantic_validation", {}).get("expected_terms") or []),
    )
    if not common_type:
        return False
    right_spans = {
        tuple(int(value) for value in span) for span in right.get("char_spans") or []
    }
    for raw_span in left.get("char_spans") or []:
        span = tuple(int(value) for value in raw_span)
        if span in right_spans and _is_collective_phrase(
            _shared_span_phrase(left, span), common_type
        ):
            return True
    return False


def _shared_instance_span_errors(groups: list[dict[str, Any]]) -> list[str]:
    """Permit shared plural spans while rejecting a singular span reused for instances."""
    span_users: dict[tuple[int, int], list[int]] = {}
    for group_index, group in enumerate(groups):
        for span in group.get("char_spans", []):
            span_users.setdefault((int(span[0]), int(span[1])), []).append(group_index)
    errors: list[str] = []
    for span, users in span_users.items():
        if len(users) < 2:
            continue
        expected_sets = [
            set(groups[index].get("semantic_validation", {}).get("expected_terms", []))
            for index in users
        ]
        common_type = set(expected_sets[0]) if expected_sets and all(expected_sets) else set()
        for expected in expected_sets[1:]:
            common_type = _semantic_common_terms(common_type, expected)
        if not common_type:
            continue
        phrase = _shared_span_phrase(groups[users[0]], span)
        if not _is_collective_phrase(phrase, common_type):
            errors.append(
                f"same-type instance links share singular/noncollective phrase {phrase!r}; "
                "use one plural collective span or separate natural phrases"
            )
    return errors


def _substring_occurrences(caption: str, text: str) -> list[list[int]]:
    occurrences: list[list[int]] = []
    start = 0
    while text and start <= len(caption) - len(text):
        found = caption.find(text, start)
        if found < 0:
            break
        occurrences.append([found, found + len(text)])
        start = found + 1
    return occurrences


def _repair_group_spans(
    caption: str,
    texts: list[str],
    spans: list[Any],
) -> tuple[list[list[int]], list[str]]:
    """Repair stale offsets without collapsing repeated coreferential mentions."""
    repaired: list[list[int]] = []
    errors: list[str] = []
    for span_index, text in enumerate(texts):
        requested_start: int | None = None
        candidate = spans[span_index] if span_index < len(spans) else None
        if isinstance(candidate, (list, tuple)) and len(candidate) == 2:
            try:
                requested_start = int(candidate[0])
                requested_end = int(candidate[1])
            except (TypeError, ValueError):
                requested_start = None
            else:
                if (
                    0 <= requested_start < requested_end <= len(caption)
                    and caption[requested_start:requested_end] == text
                    and all(
                        max(requested_start, prior[0]) >= min(requested_end, prior[1])
                        for prior in repaired
                    )
                ):
                    repaired.append([requested_start, requested_end])
                    continue
        available = [
            occurrence
            for occurrence in _substring_occurrences(caption, text)
            if all(
                max(occurrence[0], prior[0]) >= min(occurrence[1], prior[1])
                for prior in repaired
            )
        ]
        if not available:
            # Cleanup can promote a linked lowercase phrase to the start of a
            # sentence (``the beach`` -> ``The beach``).  Recover that same
            # occurrence case-insensitively, but retain all overlap and
            # nearest-original-offset checks below; no fuzzy lexical matching
            # is introduced.
            available = [
                [match.start(), match.end()]
                for match in re.finditer(re.escape(text), caption, flags=re.IGNORECASE)
                if all(
                    max(match.start(), prior[0]) >= min(match.end(), prior[1])
                    for prior in repaired
                )
            ]
        if not available:
            errors.append(f"span {span_index} does not extract {text!r}")
            continue
        if requested_start is None:
            chosen = available[0]
        else:
            chosen = min(available, key=lambda item: (abs(item[0] - requested_start), item[0]))
        repaired.append(chosen)
    return repaired, errors


_NON_ENTITY_MENTION_HEADS = {
    "area",
    "atmosphere",
    "background",
    "center",
    "color",
    "corner",
    "detail",
    "design",
    "edge",
    "form",
    "foreground",
    "group",
    "kind",
    "left",
    "manner",
    "middle",
    "moment",
    "mood",
    "pair",
    "part",
    "pattern",
    "pose",
    "right",
    "row",
    "scene",
    "set",
    "setting",
    "side",
    "style",
    "surface",
    "texture",
    "time",
    "type",
    "view",
    "way",
}


def _group_covers_mention(
    group: dict[str, Any], start: int, end: int
) -> bool:
    return any(
        int(span[0]) <= start and int(span[1]) >= end
        for span in group.get("char_spans") or []
    )


def _group_collectively_covers_mention(
    group: dict[str, Any], start: int, end: int
) -> bool:
    """Treat a nested collective head as covered by its full linked phrase."""
    expected = set(
        group.get("semantic_validation", {}).get("expected_terms") or []
    )
    if not expected:
        return False
    return any(
        int(span[0]) <= start
        and int(span[1]) >= end
        and _is_collective_phrase(str(text), expected)
        for text, span in zip(
            group.get("text") or [], group.get("char_spans") or [], strict=True
        )
    )


def _complete_unambiguous_mentions(
    caption: str,
    groups: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Add only occurrence links whose accepted-mask identity is unique."""
    row_by_mask = {str(row["mask_id"]): row for row in rows}
    overlay_by_mask = {
        str(row["mask_id"]): index + 1 for index, row in enumerate(rows)
    }
    group_by_mask = {
        str(group.get("mask_id") or ""): group for group in groups
    }
    person_masks = [
        str(row["mask_id"])
        for row in rows
        if _terms_include_category(_row_expected_semantic_terms(row), "person")
    ]
    repairs: list[dict[str, Any]] = []
    for mention in mentions:
        start = int(mention["start"])
        end = int(mention["end"])
        text = str(mention["text"])
        if any(_group_covers_mention(group, start, end) for group in groups):
            continue
        if mention["kind"] == "person_reference":
            compatible_masks = (
                person_masks
                if text.casefold() in _PERSON_ONLY_REFERENCE_TERMS
                else [str(row["mask_id"]) for row in rows]
            )
        else:
            if str(mention.get("head") or "") in _NON_ENTITY_MENTION_HEADS:
                continue
            if re.search(r"\b(?:and|or)\b", text, flags=re.IGNORECASE):
                continue
            compatible_masks = [
                str(row["mask_id"])
                for row in rows
                if _mention_semantically_compatible(mention, row)
            ]
        if len(compatible_masks) != 1:
            continue
        mask_id = compatible_masks[0]
        group = group_by_mask.get(mask_id)
        if group is None:
            if mention["kind"] == "person_reference":
                continue
            semantic_evidence, semantic_error = _semantic_link_evidence(
                [text], row_by_mask[mask_id]
            )
            if semantic_error:
                continue
            group = {
                "overlay_number": overlay_by_mask[mask_id],
                "mask_id": mask_id,
                "char_spans": [],
                "text": [],
                "object_count": 1,
                "instance_ids": [mask_id],
                "semantic_validation": semantic_evidence,
                "mention_repairs": [],
            }
            groups.append(group)
            group_by_mask[mask_id] = group
        if _group_covers_mention(group, start, end):
            continue
        spans = [
            [int(span[0]), int(span[1])]
            for span in group.get("char_spans") or []
        ]
        partial_overlap = any(
            max(start, span[0]) < min(end, span[1])
            and not (start <= span[0] and end >= span[1])
            for span in spans
        )
        if partial_overlap:
            continue
        pairs = [
            (str(prior_text), span)
            for prior_text, span in zip(
                group.get("text") or [], spans, strict=True
            )
            if not (start <= span[0] and end >= span[1])
        ]
        pairs.append((text, [start, end]))
        pairs.sort(key=lambda item: (item[1][0], item[1][1]))
        group["text"] = [item[0] for item in pairs]
        group["char_spans"] = [item[1] for item in pairs]
        semantic_evidence, _ = _semantic_link_evidence(
            group["text"], row_by_mask[mask_id]
        )
        group["semantic_validation"] = semantic_evidence
        repair = {
            "mask_id": mask_id,
            "overlay_number": overlay_by_mask[mask_id],
            "text": text,
            "char_span": [start, end],
            "reason": "unambiguous_spacy_mention_completion",
        }
        group.setdefault("mention_repairs", []).append(repair)
        repairs.append(repair)
    groups.sort(key=lambda group: int(group["overlay_number"]))
    return groups, repairs

def _expand_tagged_identity_noun_phrases(
    caption: str,
    groups: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    mentions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand a uniquely tagged identity to its complete enclosing noun phrase."""
    repairs: list[dict[str, Any]] = []
    for mention in mentions:
        if mention.get("kind") != "noun_phrase":
            continue
        start = int(mention["start"])
        end = int(mention["end"])
        text = str(mention["text"])
        if re.search(r"\b(?:and|or)\b", text, flags=re.IGNORECASE):
            continue
        if any(_group_covers_mention(group, start, end) for group in groups):
            continue

        candidates: list[
            tuple[dict[str, Any], list[list[int]], list[int]]
        ] = []
        for group in groups:
            row = row_by_id.get(str(group.get("mask_id") or ""), {})
            if not _mention_semantically_compatible(mention, row):
                continue
            spans = [
                [int(span[0]), int(span[1])]
                for span in group.get("char_spans") or []
            ]
            contained = [
                index
                for index, span in enumerate(spans)
                if start <= span[0] and span[1] <= end
            ]
            if not contained:
                continue
            has_partial_overlap = any(
                max(start, span[0]) < min(end, span[1])
                and not (start <= span[0] and span[1] <= end)
                for span in spans
            )
            if has_partial_overlap:
                continue
            candidates.append((group, spans, contained))
        if len(candidates) != 1:
            continue

        group, spans, contained = candidates[0]
        prior_pairs = list(
            zip(group.get("text") or [], spans, strict=True)
        )
        pairs = [
            (str(prior_text), span)
            for index, (prior_text, span) in enumerate(prior_pairs)
            if index not in contained
        ]
        if any(
            max(start, span[0]) < min(end, span[1])
            for _, span in pairs
        ):
            continue
        pairs.append((text, [start, end]))
        pairs.sort(key=lambda item: (item[1][0], item[1][1]))
        group["text"] = [item[0] for item in pairs]
        group["char_spans"] = [item[1] for item in pairs]
        semantic_evidence, _ = _semantic_link_evidence(
            group["text"], row_by_id[str(group["mask_id"])]
        )
        group["semantic_validation"] = semantic_evidence
        repair = {
            "mask_id": str(group["mask_id"]),
            "overlay_number": int(group["overlay_number"]),
            "text": text,
            "char_span": [start, end],
            "replaced_spans": [spans[index] for index in contained],
            "reason": "tagged_identity_expanded_to_full_noun_phrase",
        }
        group.setdefault("mention_repairs", []).append(repair)
        repairs.append(repair)
    return groups, repairs


def _caption_mention_coverage(
    caption: str,
    groups: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    mentions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Require each concrete noun phrase/coreference occurrence to be linked."""
    person_groups = [
        group
        for group in groups
        if _terms_include_category(
            set(group.get("semantic_validation", {}).get("expected_terms", [])),
            "person",
        )
    ]
    errors: list[str] = []
    unlinked: list[dict[str, Any]] = []
    ignored_abstract: list[dict[str, Any]] = []
    for mention in mentions:
        start = int(mention["start"])
        end = int(mention["end"])
        text = str(mention["text"])
        if mention["kind"] == "person_reference":
            covered = [
                group
                for group in groups
                if _group_covers_mention(group, start, end)
            ]
            if covered:
                continue
            compatible_reference_groups = (
                person_groups
                if text.casefold() in _PERSON_ONLY_REFERENCE_TERMS
                else groups
            )
            ids = [
                int(group["overlay_number"])
                for group in compatible_reference_groups
            ]
            detail = {
                **mention,
                "compatible_link_ids": ids,
                "reason": "unlinked_reference",
            }
            unlinked.append(detail)
            errors.append(
                "caption mention coverage: unlinked referring expression "
                f"{text!r} at [{start},{end}); possible link IDs: {ids}"
            )
            continue

        compatible: list[dict[str, Any]] = []
        for group in groups:
            row = row_by_id.get(str(group.get("mask_id") or ""), {})
            if _mention_semantically_compatible(mention, row):
                compatible.append(group)
        covered = [
            group
            for group in compatible
            if _group_covers_mention(group, start, end)
        ]
        if not covered:
            covered = [
                group
                for group in groups
                if _group_collectively_covers_mention(group, start, end)
            ]
        if covered:
            continue
        if compatible:
            ids = [int(group["overlay_number"]) for group in compatible]
            detail = {
                **mention,
                "compatible_link_ids": ids,
                "reason": "unlinked_accepted_entity_mention",
            }
            unlinked.append(detail)
            errors.append(
                "caption mention coverage: unlinked accepted-entity noun phrase "
                f"{text!r} at [{start},{end}); compatible link IDs: {ids}"
            )
        elif str(mention.get("head") or "") in _NON_ENTITY_MENTION_HEADS:
            ignored_abstract.append(mention)
        else:
            detail = {**mention, "reason": "unmasked_concrete_entity"}
            unlinked.append(detail)
            errors.append(
                "caption mention coverage: unmasked concrete noun phrase "
                f"{text!r} at [{start},{end}); omit it because no accepted mask/link exists"
            )
    return {
        "valid": not errors,
        "checked_mention_count": len(mentions),
        "unlinked_mentions": unlinked,
        "ignored_abstract_mentions": ignored_abstract,
    }, errors




def _composite_group_link_errors(
    caption: str,
    groups: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[str]:
    by_id = {str(group.get("mask_id") or ""): group for group in groups}
    overlay_by_id = {
        str(row.get("mask_id") or ""): index + 1 for index, row in enumerate(rows)
    }
    errors: list[str] = []
    for row in rows:
        composite_children = row.get("bcc_composite_mask_children") or []
        if len(composite_children) < 2:
            continue
        composite_id = str(row.get("mask_id") or "")
        child_ids = [
            str(child.get("mask_id") or "") for child in composite_children
        ]
        composite_group = by_id.get(composite_id)
        child_groups = [by_id.get(child_id) for child_id in child_ids]
        if composite_group is None or any(group is None for group in child_groups):
            continue
        composite_spans = [
            (int(span[0]), int(span[1]))
            for span in composite_group.get("char_spans") or []
        ]
        enclosing_spans = [
            (composite_start, composite_end)
            for composite_start, composite_end in composite_spans
            if all(
                any(
                    child_start >= composite_start and child_end <= composite_end
                    for child_start, child_end in (
                        (int(span[0]), int(span[1]))
                        for span in child_group.get("char_spans") or []
                    )
                )
                for child_group in child_groups
            )
        ]
        if not enclosing_spans:
            errors.append(
                f"composite mask ID {overlay_by_id[composite_id]} must link one collective span "
                f"enclosing a linked mention of every child ID {[overlay_by_id[value] for value in child_ids]}; do not describe the union mask as an additional instance"
            )
            continue
        for composite_start, composite_end in enclosing_spans:
            repeated_before: list[bool] = []
            repeated_after: list[bool] = []
            for child_group in child_groups:
                child_spans = [
                    (int(span[0]), int(span[1]))
                    for span in child_group.get("char_spans") or []
                ]
                repeated_before.append(
                    any(
                        child_end <= composite_start
                        and composite_start - child_end <= 96
                        and not re.search(r"[.!?]", caption[child_end:composite_start])
                        for _, child_end in child_spans
                    )
                )
                repeated_after.append(
                    any(
                        child_start >= composite_end
                        and child_start - composite_end <= 96
                        and not re.search(r"[.!?]", caption[composite_end:child_start])
                        for child_start, _ in child_spans
                    )
                )
            if all(repeated_before) or all(repeated_after):
                errors.append(
                    f"composite mask ID {overlay_by_id[composite_id]} repeats every child immediately outside its collective span; mention each child once inside the composite link"
                )
                break
    return errors



def _instance_number_errors(
    groups: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[str]:
    group_by_id = {str(group.get("mask_id") or ""): group for group in groups}
    context = _mask_context(rows)
    errors: list[str] = []
    for row, item in zip(rows, context, strict=True):
        surface = str(item.get("surface_identity_noun") or "").casefold()
        group = group_by_id.get(str(row.get("mask_id") or ""))
        if group is None:
            continue
        group_mentions = [str(value) for value in group.get("text") or []]
        if (
            int(item.get("same_subject_total") or 1) > 1
            and not item.get("composite_of_ids")
            and _surface_is_grammatically_plural(surface)
        ):
            ordinal_mentions = [
                mention
                for mention in group_mentions
                if re.search(
                    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|\d+(?:st|nd|rd|th))\b",
                    mention,
                    re.IGNORECASE,
                )
                and re.search(
                    rf"(?<!\w){re.escape(surface)}(?!\w)",
                    mention,
                    re.IGNORECASE,
                )
            ]
            if ordinal_mentions:
                errors.append(
                    f"grammatical selector: plural collection mask ID {item['id']} needs a descriptive phrase, not {ordinal_mentions[0]!r}"
                )
        if (
            int(item.get("same_subject_total") or 1) <= 1
            or int(item.get("significant_component_count") or 1) > 1
            or item.get("composite_of_ids")
            or surface in _PLURAL_ONLY_IDENTITY_SURFACES
        ):
            continue
        plural_surface = _pluralize_identity_surface(surface)
        if not surface or plural_surface == surface:
            continue
        if any(
            re.search(rf"(?<!\w){re.escape(surface)}(?!\w)", mention, re.IGNORECASE)
            for mention in group_mentions
        ):
            continue
        plural_mentions = [
            (mention, tuple(int(value) for value in span))
            for mention, span in zip(
                group_mentions, group.get("char_spans") or [], strict=True
            )
            if re.search(rf"(?<!\w){re.escape(plural_surface)}(?!\w)", mention, re.IGNORECASE)
            and not re.search(r"\bone\s+of\b", mention, re.IGNORECASE)
            and not any(
                tuple(int(value) for value in other_span)
                == tuple(int(value) for value in span)
                and _groups_share_collective_span(group, other_group)
                for other_group in groups
                if other_group is not group
                for other_span in other_group.get("char_spans") or []
            )
        ]
        if plural_mentions:
            errors.append(
                f"grammatical number: single-component repeated mask ID {item['id']} needs singular identity {surface!r}, not {plural_mentions[0][0]!r}"
            )
    return errors


def _group_identity_phrases(group: dict[str, Any]) -> list[str]:
    expected = set(group.get("semantic_validation", {}).get("expected_terms") or [])
    phrases = []
    for text in group.get("text") or []:
        if expected and not semantic_terms_compatible(
            _canonical_semantic_terms(str(text)), expected
        ):
            continue
        phrases.append(str(text))
    return phrases


def _identity_style_errors(groups: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    by_subject: dict[tuple[str, ...], list[tuple[dict[str, Any], set[str]]]] = {}
    body_part_ordinals: list[dict[str, Any]] = []
    ordinal_by_subject: dict[tuple[str, ...], list[int]] = {}
    for group in groups:
        expected = tuple(
            sorted(group.get("semantic_validation", {}).get("expected_terms") or [])
        )
        if not expected:
            continue
        phrases = _group_identity_phrases(group)
        normalized = {
            re.sub(r"\s+", " ", phrase).strip().casefold()
            for phrase in phrases
            if phrase.strip()
        }
        by_subject.setdefault(expected, []).append((group, normalized))
        ordinal_phrases = [
            phrase for phrase in phrases if _ORDINAL_SELECTOR_RE.search(phrase)
        ]
        if ordinal_phrases:
            ordinal_by_subject.setdefault(expected, []).append(
                int(group["overlay_number"])
            )
            if _terms_are_body_parts(set(expected)):
                body_part_ordinals.append(
                    {
                        "overlay_number": int(group["overlay_number"]),
                        "phrase": ordinal_phrases[0],
                    }
                )

    duplicate_identity_phrases: list[dict[str, Any]] = []
    for entries in by_subject.values():
        for left_index, (left_group, left_phrases) in enumerate(entries):
            for right_group, right_phrases in entries[left_index + 1 :]:
                if not left_phrases or left_phrases != right_phrases:
                    continue
                if _groups_share_collective_span(left_group, right_group):
                    continue
                duplicate_identity_phrases.append(
                    {
                        "overlay_numbers": [
                            int(left_group["overlay_number"]),
                            int(right_group["overlay_number"]),
                        ],
                        "phrases": sorted(left_phrases),
                    }
                )

    ordinal_catalogs = [
        {"expected_terms": list(subject), "overlay_numbers": overlay_numbers}
        for subject, overlay_numbers in ordinal_by_subject.items()
        if len(overlay_numbers) >= 2
    ]
    errors = [
        "caption quality: duplicate identity wording for link IDs "
        f"{item['overlay_numbers']}: {item['phrases']}; repeated instances need distinct natural phrases"
        for item in duplicate_identity_phrases
    ]
    errors.extend(
        "caption quality: ordinal body-part identity for link ID "
        f"{item['overlay_number']}: {item['phrase']!r}; use ownership or a visible distinction"
        for item in body_part_ordinals
    )
    errors.extend(
        "caption quality: mechanical ordinal catalog for "
        f"{item['expected_terms']} across link IDs {item['overlay_numbers']}; use connected prose and visible selectors"
        for item in ordinal_catalogs
    )
    return {
        "duplicate_identity_phrases": duplicate_identity_phrases,
        "body_part_ordinal_phrases": body_part_ordinals,
        "ordinal_catalogs": ordinal_catalogs,
    }, errors


def _span_group(
    groups: list[dict[str, Any]],
    span: list[int] | None,
    *,
    person_only: bool = False,
) -> dict[str, Any] | None:
    if span is None:
        return None
    start, end = [int(value) for value in span]
    candidates = []
    for group in groups:
        expected = set(group.get("semantic_validation", {}).get("expected_terms") or [])
        if person_only and not _terms_include_category(expected, "person"):
            continue
        for group_span in group.get("char_spans") or []:
            group_start, group_end = [int(value) for value in group_span]
            if group_start <= start and group_end >= end:
                candidates.append((group_end - group_start, group))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _linear_relation_groups(
    caption: str,
    groups: list[dict[str, Any]],
    verb_span: list[int],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    verb_start, verb_end = [int(value) for value in verb_span]
    sentence_start = max(caption.rfind(mark, 0, verb_start) for mark in ".!?") + 1
    following = [
        caption.find(mark, verb_end)
        for mark in ".!?"
        if caption.find(mark, verb_end) >= 0
    ]
    sentence_end = min(following) + 1 if following else len(caption)
    before = []
    after = []
    for group in groups:
        expected = set(group.get("semantic_validation", {}).get("expected_terms") or [])
        for span in group.get("char_spans") or []:
            start, end = [int(value) for value in span]
            if (
                sentence_start <= start
                and end <= verb_start
                and _terms_include_category(expected, "person")
            ):
                before.append((end, group))
            if (
                verb_end <= start < sentence_end
                and not _terms_include_category(expected, "person")
            ):
                after.append((start, group))
    subject = max(before, key=lambda item: item[0])[1] if before else None
    obj = min(after, key=lambda item: item[0])[1] if after else None
    return subject, obj


def _bbox_gap(left: list[int], right: list[int]) -> tuple[float, float]:
    left_x0, left_y0, left_w, left_h = [float(value) for value in left]
    right_x0, right_y0, right_w, right_h = [float(value) for value in right]
    left_x1, left_y1 = left_x0 + left_w, left_y0 + left_h
    right_x1, right_y1 = right_x0 + right_w, right_y0 + right_h
    return (
        max(0.0, left_x0 - right_x1, right_x0 - left_x1),
        max(0.0, left_y0 - right_y1, right_y0 - left_y1),
    )


def _contact_relation_geometry(
    caption: str,
    groups: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    row_by_id = {str(row.get("mask_id") or ""): row for row in rows}
    boxes = [
        [int(value) for value in row.get("bbox", [0, 0, 1, 1])]
        for row in rows
    ]
    source_path = str(rows[0].get("source_image_path") or "") if rows else ""
    if source_path and Path(source_path).exists():
        image_width, image_height = _cached_image_size(source_path)
    else:
        image_width = max(1, max((box[0] + box[2] for box in boxes), default=1))
        image_height = max(1, max((box[1] + box[3] for box in boxes), default=1))
    image_diagonal = max(1.0, float(np.hypot(image_width, image_height)))
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    for relation in caption_contact_relations(caption):
        subject = _span_group(
            groups, relation.get("subject_span"), person_only=True
        )
        obj = _span_group(groups, relation.get("object_span"))
        if subject is None or obj is None:
            fallback_subject, fallback_object = _linear_relation_groups(
                caption, groups, relation["verb_span"]
            )
            subject = subject or fallback_subject
            obj = obj or fallback_object
        if subject is None or obj is None or subject is obj:
            continue
        subject_row = row_by_id.get(str(subject.get("mask_id") or ""))
        object_row = row_by_id.get(str(obj.get("mask_id") or ""))
        if not subject_row or not object_row:
            continue
        subject_box = [
            int(value) for value in subject_row.get("bbox", [0, 0, 1, 1])
        ]
        object_box = [
            int(value) for value in object_row.get("bbox", [0, 0, 1, 1])
        ]
        gap_x, gap_y = _bbox_gap(subject_box, object_box)
        normalized_gap = float(np.hypot(gap_x, gap_y) / image_diagonal)
        supported = normalized_gap <= 0.06
        check = {
            "verb": relation["verb"],
            "subject_overlay_number": int(subject["overlay_number"]),
            "object_overlay_number": int(obj["overlay_number"]),
            "bbox_gap_xy": [round(gap_x, 3), round(gap_y, 3)],
            "normalized_gap": round(normalized_gap, 6),
            "supported_by_proximity": supported,
        }
        checks.append(check)
        if not supported:
            errors.append(
                "caption quality: unsupported contact relation "
                f"{relation['verb']!r} between link IDs {subject['overlay_number']} and "
                f"{obj['overlay_number']}; mask-box gap is {normalized_gap:.3f} image diagonals"
            )
    return checks, errors


def _caption_quality(
    caption: str,
    groups: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Measure high-confidence structural, prose, and relation failure modes."""
    sentences = [
        (match.start(), match.end())
        for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", caption)
        if match.group(0).strip()
    ]
    linked_groups_per_sentence: list[int] = []
    for sentence_start, sentence_end in sentences:
        linked_groups_per_sentence.append(
            sum(
                1
                for group in groups
                if any(
                    int(span[0]) < sentence_end and int(span[1]) > sentence_start
                    for span in group.get("char_spans") or []
                )
            )
        )
    max_linked = max(linked_groups_per_sentence, default=0)
    stock_predicate_count = len(_INVENTORY_PREDICATE_RE.findall(caption))
    evasion_matches = [
        *_UNSUPPORTED_OBJECT_EVASION_RE.finditer(caption),
        *_SELF_BODY_MANIPULATION_RE.finditer(caption),
        *_SELF_BODY_CONTACT_RE.finditer(caption),
        *_TRANSITIVE_SELF_BODY_CONTACT_RE.finditer(caption),
        *_IMPOSSIBLE_OBJECT_ACTION_RE.finditer(caption),
    ]
    unsupported_object_evasions = [
        match.group(0)
        for match in sorted(evasion_matches, key=lambda item: item.start())
    ]
    adjacent_duplicate_link_phrases: list[str] = []
    for group in groups:
        occurrences = sorted(
            (
                int(span[0]),
                int(span[1]),
                caption[int(span[0]) : int(span[1])],
            )
            for span in group.get("char_spans") or []
        )
        for previous, current in zip(occurrences, occurrences[1:]):
            if previous[2].casefold() != current[2].casefold():
                continue
            gap = caption[previous[1] : current[0]]
            if not re.fullmatch(r"\s*[,;:]?\s*", gap):
                continue
            phrase = caption[previous[0] : current[1]]
            if phrase not in adjacent_duplicate_link_phrases:
                adjacent_duplicate_link_phrases.append(phrase)
    malformed_selector_phrases: list[str] = []
    for mention in mentions:
        if mention.get("kind") != "noun_phrase":
            continue
        text = str(mention.get("text") or "")
        words = re.findall(r"[A-Za-z]+", text.casefold())
        if len(words) < 4 or words[0] not in {"a", "an"}:
            continue
        selector_misordered = any(
            token in _MALFORMED_SELECTOR_WORDS
            for token in words[2:-1]
        )
        other_misordered = any(
            words[index : index + 2] == ["the", "other"]
            for index in range(2, len(words) - 2)
        )
        is_malformed = selector_misordered or other_misordered
        if is_malformed and text not in malformed_selector_phrases:
            malformed_selector_phrases.append(text)
    malformed_punctuation = [
        match.group(0) for match in _MALFORMED_CAPTION_PUNCTUATION_RE.finditer(caption)
    ]
    possessive_boundary_errors = [
        match.group(0)
        for pattern in (_FUSED_POSSESSIVE_RE, _REPEATED_POSSESSIVE_RE)
        for match in pattern.finditer(caption)
    ]
    identity_style, identity_style_errors = _identity_style_errors(groups)
    relation_geometry, relation_errors = _contact_relation_geometry(
        caption, groups, rows
    )
    group_count = len(groups)
    inventory_like = group_count >= 4 and max_linked < 2
    stock_threshold = 2
    repetitive_stock_predicates = (
        group_count >= 4 and stock_predicate_count >= stock_threshold
    )
    errors: list[str] = [*identity_style_errors, *relation_errors]
    if inventory_like:
        errors.append(
            "caption quality: one-sentence-per-mask inventory; at least one sentence "
            "must jointly describe two or more linked entities"
        )
    if repetitive_stock_predicates:
        errors.append(
            "caption quality: repeated stock visibility/presence predicates "
            f"({stock_predicate_count}; maximum before rewrite is {stock_threshold - 1})"
        )
    for phrase in unsupported_object_evasions:
        errors.append(
            f"caption quality: unsupported-object evasion {phrase!r}; describe a supported pose instead"
        )
    for phrase in adjacent_duplicate_link_phrases:
        errors.append(
            f"caption quality: adjacent duplicate linked phrase {phrase!r}; mention it once, nesting any composite tag around that single occurrence"
        )
    for phrase in malformed_selector_phrases:
        errors.append(
            "caption quality: malformed repeated-instance selector phrase "
            f"{phrase!r}; put article, modifiers, and selector in natural order"
        )
    for phrase in malformed_punctuation:
        errors.append(
            f"caption quality: malformed punctuation {phrase!r}; proofread the sentence boundary"
        )
    for phrase in possessive_boundary_errors:
        errors.append(
            "caption quality: fused or repeated possessive wording "
            f"{phrase!r}; restore natural word boundaries or omit the awkward mask reference"
        )
    metrics = {
        "valid": not errors,
        "sentence_count": len(sentences),
        "linked_groups_per_sentence": linked_groups_per_sentence,
        "max_linked_groups_per_sentence": max_linked,
        "stock_predicate_count": stock_predicate_count,
        "unsupported_object_evasions": unsupported_object_evasions,
        "adjacent_duplicate_link_phrases": adjacent_duplicate_link_phrases,
        "malformed_selector_phrases": malformed_selector_phrases,
        "malformed_punctuation": malformed_punctuation,
        "possessive_boundary_errors": possessive_boundary_errors,
        **identity_style,
        "contact_relation_geometry": relation_geometry,
        "inventory_like": inventory_like,
        "repetitive_stock_predicates": repetitive_stock_predicates,
    }
    return metrics, errors


def normalize_correspondence(
    parsed: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    min_groups: int,
    require_all_masks: bool = True,
    retain_semantically_invalid_groups: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    parsed, composite_close_repairs = _repair_unclosed_composite_outer_tags(
        parsed, rows
    )
    parsed, inline_errors, correspondence_encoding = _expand_tagged_caption(parsed)
    correspondence_encoding["composite_outer_close_repairs"] = composite_close_repairs
    errors: list[str] = list(inline_errors)
    original_caption = str(parsed.get("caption") or "").strip()
    caption, dense_corrections = _sanitize_dense_caption(original_caption)
    caption_cleanup = {
        "original": original_caption,
        "caption": caption,
        "changed": caption != original_caption,
        "corrections": dense_corrections,
        "valid": bool(caption),
    }
    if not caption:
        errors.append("caption is empty")
    elif forbidden := _FORBIDDEN_IMAGE_CONTEXT_RE.search(caption):
        errors.append(f"caption contains forbidden image/background wording: {forbidden.group(0)!r}")

    valid_mask_ids = {str(row["mask_id"]) for row in rows}
    row_by_id = {str(row["mask_id"]): row for row in rows}
    overlay_by_id = {str(row["mask_id"]): index + 1 for index, row in enumerate(rows)}
    groups_out: list[dict[str, Any]] = []
    encountered_mask_ids: set[str] = set()
    normalized_mask_ids: set[str] = set()
    mask_id_repairs: list[dict[str, str]] = []
    link_repairs: list[dict[str, Any]] = []
    raw_groups = parsed.get("links") if "links" in parsed else parsed.get("groups")
    if raw_groups is None:
        raw_groups = []
    if not isinstance(raw_groups, list):
        raw_groups = []
        errors.append("links/groups must be a list")

    for index, group in enumerate(raw_groups):
        if not isinstance(group, dict):
            errors.append(f"link list position {index + 1} is not an object")
            continue
        mask_id, provided_reference = _resolve_group_mask_id(group, rows, valid_mask_ids)
        if mask_id not in valid_mask_ids:
            link_repairs.append(
                {
                    "group_index": index,
                    "provided_reference": provided_reference,
                    "reason": "unknown_extra_link_dropped",
                }
            )
            continue
        link_id = overlay_by_id[mask_id]
        link_label = f"link ID {link_id}"
        raw_mask_id = str(group.get("mask_id") or "").strip()
        if raw_mask_id and mask_id != raw_mask_id:
            mask_id_repairs.append({"provided": raw_mask_id, "resolved": mask_id})
        if mask_id in encountered_mask_ids:
            errors.append(f"{link_label} appears more than once")
            continue
        encountered_mask_ids.add(mask_id)

        texts = group.get("text") or group.get("mentions") or []
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(value) for value in texts if str(value)]
        if not texts:
            errors.append(f"{link_label} has no referential text")
            continue
        inline_tagged = bool(group.get("_inline_tagged"))
        if inline_tagged:
            lexical_repairs: list[dict[str, Any]] = []
            normalized_spans, span_errors = _repair_group_spans(
                caption, texts, group.get("char_spans") or []
            )
            canonical_texts = [caption[start:end] for start, end in normalized_spans]
        else:
            texts, lexical_repairs = _drop_lexically_nested_mentions(texts)
            canonical_texts, normalized_spans, span_errors = _locate_mentions(caption, texts)
        location_repairs: list[dict[str, Any]] = []
        if canonical_texts:
            retained_errors: list[str] = []
            for message in span_errors:
                if message.startswith("mention text is absent from caption:"):
                    location_repairs.append(
                        {
                            "detail": message,
                            "reason": "unmatched_model_mention_dropped",
                        }
                    )
                else:
                    retained_errors.append(message)
            span_errors = retained_errors
        canonical_texts, normalized_spans, nested_repairs, overlap_errors = (
            _normalize_within_group_mentions(canonical_texts, normalized_spans)
        )
        mention_repairs = lexical_repairs + location_repairs + nested_repairs
        span_errors.extend(overlap_errors)
        group_valid = not span_errors
        errors.extend(f"{link_label} {message}" for message in span_errors)

        semantic_evidence, semantic_error = _semantic_link_evidence(
            canonical_texts, row_by_id[mask_id]
        )
        if semantic_error:
            errors.append(f"{link_label} {semantic_error}")
            if retain_semantically_invalid_groups and group_valid:
                semantic_evidence = {
                    **semantic_evidence,
                    "valid": False,
                    "retained_with_error": True,
                    "error": semantic_error,
                }
            else:
                group_valid = False
        if not group_valid:
            continue

        normalized_mask_ids.add(mask_id)
        groups_out.append(
            {
                "overlay_number": overlay_by_id[mask_id],
                "mask_id": mask_id,
                "char_spans": normalized_spans,
                "text": canonical_texts,
                "object_count": 1,
                "instance_ids": [mask_id],
                "semantic_validation": semantic_evidence,
                "mention_repairs": mention_repairs,
            }
        )

    caption_mentions = caption_entity_mentions(caption)
    groups_out, mention_completion_repairs = _complete_unambiguous_mentions(
        caption, groups_out, rows, caption_mentions
    )
    groups_out, noun_phrase_span_repairs = (
        _expand_tagged_identity_noun_phrases(
            caption, groups_out, row_by_id, caption_mentions
        )
    )
    normalized_mask_ids = {
        str(group.get("mask_id") or "") for group in groups_out
    }
    errors.extend(_shared_instance_span_errors(groups_out))
    composite_link_errors = _composite_group_link_errors(caption, groups_out, rows)
    errors.extend(composite_link_errors)
    instance_number_errors = _instance_number_errors(groups_out, rows)
    errors.extend(instance_number_errors)
    caption_quality, quality_errors = _caption_quality(
        caption, groups_out, caption_mentions, rows
    )
    errors.extend(quality_errors)
    if composite_link_errors or instance_number_errors:
        caption_quality["valid"] = False
    mention_coverage, mention_coverage_errors = _caption_mention_coverage(
        caption, groups_out, row_by_id, caption_mentions
    )
    caption_quality["mention_coverage"] = mention_coverage
    if mention_coverage_errors:
        caption_quality["valid"] = False
    errors.extend(mention_coverage_errors)
    if len(groups_out) < min_groups:
        errors.append(f"only {len(groups_out)} valid groups; need at least {min_groups}")
    if require_all_masks:
        missing = sorted(
            valid_mask_ids - normalized_mask_ids,
            key=lambda value: overlay_by_id[value],
        )
        if missing:
            missing_details: list[str] = []
            for missing_id in missing[:12]:
                source = row_by_id[missing_id]
                subject = str(
                    source.get("main_candidate")
                    or source.get("object")
                    or source.get("source_prompt")
                    or "unknown"
                ).strip()
                missing_details.append(f"{overlay_by_id[missing_id]}:{subject}")
            errors.append(
                f"{len(missing)} accepted masks are missing groups; required link IDs: "
                + ", ".join(missing_details)
                + ("…" if len(missing) > 12 else "")
            )
    normalized = {
        "caption": caption,
        "groups": groups_out,
        "caption_cleanup": caption_cleanup,
        "mask_id_repairs": mask_id_repairs,
        "caption_quality": caption_quality,
        "link_repairs": link_repairs,
        "mention_completion_repairs": mention_completion_repairs,
        "noun_phrase_span_repairs": noun_phrase_span_repairs,
        "correspondence_encoding": correspondence_encoding,
        "composite_link_validation": {
            "valid": not composite_link_errors,
            "errors": composite_link_errors,
        },
        "instance_number_validation": {
            "valid": not instance_number_errors,
            "errors": instance_number_errors,
        },
        "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
        "stage_version": PIPELINE_STAGE_VERSION,
        "prompt_version": BCC_PROMPT_VERSION,
    }
    return normalized, errors


def _decision_errors(parsed: dict[str, Any], *, qa: bool) -> list[str]:
    if qa:
        return [] if parsed.get("keep") is True else ["pass-two keep must be true"]
    return [] if parsed.get("reject") is False else ["pass-one reject must be false"]


def _response_content_fingerprint(parsed: dict[str, Any]) -> str:
    keys = ("keep", "reject", "tagged_caption", "caption", "links", "groups")
    content = {key: parsed[key] for key in keys if key in parsed}
    return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _qa_eligible_draft(
    normalized: dict[str, Any], validation_errors: list[str], min_groups: int
) -> bool:
    """Forward useful drafts to strict visual QA instead of discarding them early."""
    return (
        bool(validation_errors)
        and "pass-one reject must be false" not in validation_errors
        and bool(normalized.get("caption_cleanup", {}).get("valid"))
        and len(normalized.get("groups") or []) >= min_groups
    )


def _enrich_groups(record: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["mask_id"]): (index, row) for index, row in enumerate(rows)}
    groups = []
    for group in record["groups"]:
        color_index, source = by_id[group["mask_id"]]
        groups.append(
            {
                **group,
                "color_rgb": list(color_for_index(color_index)),
                "mask_path": source["mask_path"],
                "inverse_crop_path": source.get("inverse_crop_path"),
                "bbox": source["bbox"],
                "source_sam3_prompt": source.get("source_prompt", ""),
                "main_candidate": source.get("main_candidate", ""),
                "mask_object": source.get("object", ""),
                "mask_caption": source.get("caption", ""),
                "mask_attributes": source.get("attributes", []),
                "mask_review_reason": source.get("mask_review_reason", ""),
                "caption_cleanup": source.get("caption_cleanup", {}),
                "qa_caption_cleanup": source.get("qa_caption_cleanup", {}),
                "inverse_background_rgb": source.get("inverse_background_rgb"),
                "sam3_score": source.get("sam3_score", source.get("entityseg_score")),
                "sam3_requery_iou": (source.get("sam3_consistency") or {}).get("best_iou"),
                "bcc_duplicate_mask_aliases": source.get("bcc_duplicate_mask_aliases", []),
                "bcc_composite_mask_children": source.get("bcc_composite_mask_children", []),
                "bcc_composite_union_iou": source.get("bcc_composite_union_iou"),
                "bcc_composite_coverage": source.get("bcc_composite_coverage"),
                "bcc_significant_component_count": int(source.get("bcc_significant_component_count") or 1),
                "bcc_composite_link_valid": not bool(record.get("composite_link_validation", {}).get("errors")),
            }
        )
    return {**record, "groups": groups}


def _mock_record(rows: list[dict[str, Any]], min_groups: int) -> dict[str, Any]:
    selected = rows
    pieces: list[str] = []
    groups: list[dict[str, Any]] = []
    cursor = 0
    for row in selected:
        piece = str(row.get("caption") or row.get("object") or row.get("main_candidate") or "an object").strip()
        piece = piece.rstrip(".")
        if pieces:
            connector = "; "
            cursor += len(connector)
        pieces.append(piece)
        groups.append(
            {
                "mask_id": row["mask_id"],
                "char_spans": [[cursor, cursor + len(piece)]],
                "text": [piece],
                "object_count": 1,
                "instance_ids": [row["mask_id"]],
            }
        )
        cursor += len(piece)
    return {"reject": False, "caption": "; ".join(pieces) + ".", "groups": groups}


def bcc_generation_config(
    config: dict[str, Any],
    section: str,
    mask_count: int,
    *,
    text_only_repair: bool = False,
) -> dict[str, Any]:
    """Scale only BCC output length so complete high-mask records are not truncated."""
    runtime = dict(qwen_model_config(config, section))
    stage = config.get(section, {})
    floor = int(runtime.get("max_new_tokens", 4096))
    cap = int(stage.get("max_new_tokens_cap", 8192))
    per_mask = int(stage.get("tokens_per_mask", 96))
    runtime["max_new_tokens"] = min(cap, max(floor, 1024 + per_mask * int(mask_count)))
    if text_only_repair:
        runtime["temperature"] = float(stage.get("repair_temperature", 0.15))
        runtime["top_p"] = float(stage.get("repair_top_p", 0.90))
    return runtime


def run_image_caption_pass(
    config: dict[str, Any],
    run_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    captioner: QwenCaptioner | None = None,
    mock: bool = False,
    initial_raw: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    candidate_path = run_dir / "image_caption_candidates.jsonl"
    rejected_path = run_dir / "image_caption_rejected.jsonl"
    raw_path = run_dir / "image_caption_raw.jsonl"
    error_path = run_dir / "image_caption_errors.jsonl"
    stage_config = config.get("image_caption", {})
    image_id = str(rows[0]["image_id"]) if rows else ""
    completed = {
        str(row.get("image_id") or "")
        for row in (read_jsonl_indexed(candidate_path) if candidate_path.exists() else [])
        if _is_current_correspondence_record(row)
    } | {
        str(row.get("image_id") or "")
        for row in (read_jsonl_indexed(rejected_path) if rejected_path.exists() else [])
        if str(row.get("reason") or "") != "generation_or_schema_failed"
        and _is_current_correspondence_record(row)
    }
    if bool(config.get("resume", False) or stage_config.get("resume", False)) and image_id in completed:
        return candidate_path
    min_groups = int(stage_config.get("min_groups", 10))
    min_input_masks = int(
        stage_config.get("min_input_masks", stage_config.get("min_groups", 10))
    )
    require_all_masks = bool(stage_config.get("require_all_masks", True))
    if len(rows) < min_input_masks:
        append_jsonl(
            {
                "image_id": image_id,
                "stage": "image_caption",
                "prompt_version": BCC_PROMPT_VERSION,
                "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                "stage_version": PIPELINE_STAGE_VERSION,
                "reason": f"only {len(rows)} consistency-passed masks; need {min_input_masks}",
            },
            rejected_path,
        )
        return candidate_path
    source_path = str(rows[0]["source_image_path"])
    overlay_path = write_correspondence_overlay(
        source_path,
        rows,
        run_dir / "correspondence_overlays" / f"{image_id}.png",
    )
    image_packet, input_manifest = build_caption_image_packet(rows, overlay_path)
    prompt = build_caption_prompt(rows, input_manifest)
    max_attempts = int(stage_config.get("max_attempts", 3))
    seed = int(config.get("random_seed", 17)) + int(stage_config.get("seed_offset", 300000))

    def append_candidate(normalized: dict[str, Any], **extra: Any) -> None:
        record = _enrich_groups(
            {
                "image_id": image_id,
                "source_image_path": source_path,
                "correspondence_overlay_path": str(overlay_path),
                "bcc_input_manifest": input_manifest,
                "prompt_version": BCC_PROMPT_VERSION,
                "model": qwen_model_config(config, "image_caption").get("model_name", "Qwen/Qwen3.5-9B"),
                "pass": 1,
                **extra,
                **normalized,
            },
            rows,
        )
        append_jsonl(record, candidate_path)

    def append_candidate_if_ready(
        normalized: dict[str, Any], validation_errors: list[str], **extra: Any
    ) -> bool:
        strict = not validation_errors
        if not strict and not _qa_eligible_draft(normalized, validation_errors, min_groups):
            return False
        extra["pass1_strict"] = strict
        if validation_errors:
            extra["draft_validation_errors"] = list(validation_errors)
        append_candidate(normalized, **extra)
        return True

    if bool(config.get("resume", False) or stage_config.get("resume", False)) and raw_path.exists():
        for prior in reversed(read_jsonl_indexed(raw_path)):
            if (
                str(prior.get("image_id") or "") != image_id
                or not prior.get("raw")
                or not _raw_record_matches_rows(prior, rows)
            ):
                continue
            try:
                parsed = extract_json(str(prior["raw"]))
            except Exception:
                continue
            if _decision_errors(parsed, qa=False):
                continue
            normalized, prior_errors = normalize_correspondence(
                parsed,
                rows,
                min_groups=min_groups,
                require_all_masks=require_all_masks,
            )
            if append_candidate_if_ready(
                normalized,
                prior_errors,
                recovered_from_prior_raw=True,
                recovered_raw_attempt=prior.get("attempt"),
            ):
                return candidate_path

    if not mock and captioner is None:
        captioner = QwenCaptioner(config, config_section="image_caption")

    last_errors: list[str] = []
    previous_raw = str(initial_raw or "")
    start_attempt = 0
    previous_content_fingerprint = ""
    previous_error_signature: tuple[str, ...] = ()
    retry_stop_reason = ""
    if initial_raw is not None:
        start_attempt = 1
        try:
            parsed = extract_json(previous_raw)
        except Exception as exc:
            last_errors = [f"batched visual response JSON error: {exc!r}"]
        else:
            normalized, normalization_errors = normalize_correspondence(
                parsed,
                rows,
                min_groups=min_groups,
                require_all_masks=require_all_masks,
            )
            last_errors = _decision_errors(parsed, qa=False) + normalization_errors
            previous_content_fingerprint = _response_content_fingerprint(parsed)
            previous_error_signature = tuple(last_errors)
            if append_candidate_if_ready(
                normalized,
                last_errors,
                recovered_from_batched_raw=True,
            ):
                return candidate_path

    for attempt in range(start_attempt, max_attempts):
        raw = ""
        result: dict[str, Any] = {}
        repair_attempt = attempt > 0
        try:
            if mock:
                parsed = _mock_record(rows, min_groups)
                raw = json.dumps(parsed)
            else:
                attempt_prompt = (
                    build_schema_repair_prompt(previous_raw, last_errors, rows, qa=False)
                    if repair_attempt
                    else prompt
                )
                result = captioner.generate(
                    image_packet,
                    attempt_prompt,
                    seed + attempt,
                    generation_config=bcc_generation_config(
                        config,
                        "image_caption",
                        len(rows),
                        text_only_repair=repair_attempt,
                    ),
                )
                raw = result["raw"]
                parsed = extract_json(raw)
        except Exception as exc:
            previous_raw = raw or previous_raw
            if raw:
                append_jsonl(
                    {
                        "image_id": image_id,
                        "attempt": attempt + 1,
                        "raw": raw,
                        **_generation_metrics(result),
                        "text_only_repair": False,
                        "visual_repair": repair_attempt,
                        "prompt_version": BCC_PROMPT_VERSION,
                        "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                        "stage_version": PIPELINE_STAGE_VERSION,
                        "bcc_input_manifest": input_manifest,
                        "parse_error": repr(exc),
                    },
                    raw_path,
                )
            append_jsonl(
                {
                    "image_id": image_id,
                    "attempt": attempt + 1,
                    "stage": "image_caption",
                    "text_only_repair": False,
                    "visual_repair": repair_attempt,
                    "retryable": True,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                error_path,
            )
            last_errors = [f"attempt {attempt + 1} generation/JSON error: {exc!r}"]
            continue
        previous_raw = raw
        append_jsonl(
            {
                "image_id": image_id,
                "attempt": attempt + 1,
                "raw": raw,
                **_generation_metrics(result),
                "text_only_repair": False,
                "visual_repair": repair_attempt,
                "prompt_version": BCC_PROMPT_VERSION,
                "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                "stage_version": PIPELINE_STAGE_VERSION,
                "bcc_input_manifest": input_manifest,
            },
            raw_path,
        )
        normalized, normalization_errors = normalize_correspondence(
            parsed,
            rows,
            min_groups=min_groups,
            require_all_masks=require_all_masks,
        )
        current_errors = _decision_errors(parsed, qa=False) + normalization_errors
        current_content_fingerprint = _response_content_fingerprint(parsed)
        no_progress = (
            repair_attempt
            and current_content_fingerprint == previous_content_fingerprint
            and tuple(current_errors) == previous_error_signature
        )
        last_errors = current_errors
        previous_content_fingerprint = current_content_fingerprint
        previous_error_signature = tuple(current_errors)
        if append_candidate_if_ready(
            normalized,
            last_errors,
            visual_schema_repair=repair_attempt,
        ):
            return candidate_path
        if no_progress:
            retry_stop_reason = "identical_repair_and_validation_errors"
            break
    append_jsonl(
        {
            "image_id": image_id,
            "stage": "image_caption",
            "prompt_version": BCC_PROMPT_VERSION,
            "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
            "stage_version": PIPELINE_STAGE_VERSION,
            "reason": "generation_or_schema_failed",
            "validation_errors": last_errors,
            **({"retry_stop_reason": retry_stop_reason} if retry_stop_reason else {}),
        },
        rejected_path,
    )
    return candidate_path


def run_image_caption_qa(
    config: dict[str, Any],
    run_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    captioner: QwenCaptioner | None = None,
    mock: bool = False,
    initial_raw: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    candidates_path = run_dir / "image_caption_candidates.jsonl"
    final_path = run_dir / "image_text_pairs.jsonl"
    rejected_path = run_dir / "image_caption_qa_rejected.jsonl"
    raw_path = run_dir / "image_caption_qa_raw.jsonl"
    error_path = run_dir / "image_caption_qa_errors.jsonl"
    stage_config = config.get("image_caption_qa", {})
    candidates = read_jsonl_indexed(candidates_path) if candidates_path.exists() else []
    image_id = str(rows[0]["image_id"]) if rows else ""
    candidate = next(
        (
            row
            for row in reversed(candidates)
            if str(row.get("image_id")) == image_id
            and _is_current_correspondence_record(row)
        ),
        None,
    )
    if candidate is None:
        return final_path
    completed = {
        str(row.get("image_id") or "")
        for row in (read_jsonl_indexed(final_path) if final_path.exists() else [])
        if _is_current_correspondence_record(row)
    } | {
        str(row.get("image_id") or "")
        for row in (read_jsonl_indexed(rejected_path) if rejected_path.exists() else [])
        if str(row.get("reason") or "") != "generation_or_schema_failed"
        and _is_current_correspondence_record(row)
    }
    if bool(config.get("resume", False) or stage_config.get("resume", False)) and image_id in completed:
        return final_path
    min_groups = int(config.get("image_caption", {}).get("min_groups", 10))
    require_all_masks = bool(config.get("image_caption", {}).get("require_all_masks", True))
    max_attempts = int(stage_config.get("max_attempts", 3))
    seed = int(config.get("random_seed", 17)) + int(stage_config.get("seed_offset", 400000))
    image_packet, input_manifest = build_caption_image_packet(
        rows, candidate["correspondence_overlay_path"]
    )
    prompt = build_qa_prompt(candidate, rows, input_manifest)

    def append_final(normalized: dict[str, Any], reason: str, **extra: Any) -> None:
        final = _enrich_groups(
            {
                "image_id": image_id,
                "source_image_path": candidate["source_image_path"],
                "correspondence_overlay_path": candidate["correspondence_overlay_path"],
                "bcc_input_manifest": input_manifest,
                "prompt_version": BCC_PROMPT_VERSION,
                "model": qwen_model_config(config, "image_caption_qa").get("model_name", "Qwen/Qwen3.5-9B"),
                "pass": 2,
                "qa_reason": reason,
                "first_pass_caption": candidate["caption"],
                "first_pass_groups": candidate["groups"],
                **extra,
                **normalized,
            },
            rows,
        )
        append_jsonl(final, final_path)

    if bool(config.get("resume", False) or stage_config.get("resume", False)) and raw_path.exists():
        for prior in reversed(read_jsonl_indexed(raw_path)):
            if (
                str(prior.get("image_id") or "") != image_id
                or not prior.get("raw")
                or not _raw_record_matches_rows(prior, rows)
            ):
                continue
            try:
                parsed = extract_json(str(prior["raw"]))
            except Exception:
                continue
            if _decision_errors(parsed, qa=True):
                continue
            normalized, prior_errors = normalize_correspondence(
                parsed,
                rows,
                min_groups=min_groups,
                require_all_masks=require_all_masks,
            )
            if not prior_errors:
                append_final(
                    normalized,
                    str(parsed.get("reason") or ""),
                    recovered_from_prior_qa_raw=True,
                    recovered_qa_raw_attempt=prior.get("attempt"),
                )
                return final_path

    if not mock and captioner is None:
        captioner = QwenCaptioner(config, config_section="image_caption_qa")

    def qa_reason(parsed: dict[str, Any]) -> str:
        return str(parsed.get("reason_code") or parsed.get("reason") or "")

    last_errors: list[str] = []
    previous_raw = str(initial_raw or "")
    start_attempt = 0
    previous_content_fingerprint = ""
    previous_error_signature: tuple[str, ...] = ()
    retry_stop_reason = ""
    if initial_raw is not None:
        start_attempt = 1
        try:
            parsed = extract_json(previous_raw)
        except Exception as exc:
            last_errors = [f"batched visual QA response JSON error: {exc!r}"]
        else:
            normalized, normalization_errors = normalize_correspondence(
                parsed,
                rows,
                min_groups=min_groups,
                require_all_masks=require_all_masks,
            )
            last_errors = _decision_errors(parsed, qa=True) + normalization_errors
            previous_content_fingerprint = _response_content_fingerprint(parsed)
            previous_error_signature = tuple(last_errors)
            if not last_errors:
                append_final(normalized, qa_reason(parsed), recovered_from_batched_raw=True)
                return final_path

    for attempt in range(start_attempt, max_attempts):
        raw = ""
        result: dict[str, Any] = {}
        repair_attempt = attempt > 0
        try:
            if mock:
                parsed = {
                    "keep": True,
                    "reason_code": "mock",
                    "caption": candidate["caption"],
                    "groups": candidate["groups"],
                }
                raw = json.dumps(parsed)
            else:
                attempt_prompt = (
                    build_schema_repair_prompt(previous_raw, last_errors, rows, qa=True)
                    if repair_attempt
                    else prompt
                )
                result = captioner.generate(
                    image_packet,
                    attempt_prompt,
                    seed + attempt,
                    generation_config=bcc_generation_config(
                        config,
                        "image_caption_qa",
                        len(rows),
                        text_only_repair=repair_attempt,
                    ),
                )
                raw = result["raw"]
                parsed = extract_json(raw)
        except Exception as exc:
            previous_raw = raw or previous_raw
            if raw:
                append_jsonl(
                    {
                        "image_id": image_id,
                        "attempt": attempt + 1,
                        "raw": raw,
                        **_generation_metrics(result),
                        "text_only_repair": False,
                        "visual_repair": repair_attempt,
                        "prompt_version": BCC_PROMPT_VERSION,
                        "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                        "stage_version": PIPELINE_STAGE_VERSION,
                        "bcc_input_manifest": input_manifest,
                        "parse_error": repr(exc),
                    },
                    raw_path,
                )
            append_jsonl(
                {
                    "image_id": image_id,
                    "attempt": attempt + 1,
                    "stage": "image_caption_qa",
                    "text_only_repair": False,
                    "visual_repair": repair_attempt,
                    "retryable": True,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                error_path,
            )
            last_errors = [f"attempt {attempt + 1} generation/JSON error: {exc!r}"]
            continue
        previous_raw = raw
        append_jsonl(
            {
                "image_id": image_id,
                "attempt": attempt + 1,
                "raw": raw,
                **_generation_metrics(result),
                "text_only_repair": False,
                "visual_repair": repair_attempt,
                "prompt_version": BCC_PROMPT_VERSION,
                "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                "stage_version": PIPELINE_STAGE_VERSION,
                "bcc_input_manifest": input_manifest,
            },
            raw_path,
        )
        normalized, normalization_errors = normalize_correspondence(
            parsed,
            rows,
            min_groups=min_groups,
            require_all_masks=require_all_masks,
        )
        current_errors = _decision_errors(parsed, qa=True) + normalization_errors
        current_content_fingerprint = _response_content_fingerprint(parsed)
        no_progress = (
            repair_attempt
            and current_content_fingerprint == previous_content_fingerprint
            and tuple(current_errors) == previous_error_signature
        )
        last_errors = current_errors
        previous_content_fingerprint = current_content_fingerprint
        previous_error_signature = tuple(current_errors)
        if not last_errors:
            append_final(
                normalized,
                qa_reason(parsed),
                visual_schema_repair=repair_attempt,
            )
            return final_path
        if no_progress:
            retry_stop_reason = "identical_repair_and_validation_errors"
            break
    append_jsonl(
        {
            "image_id": image_id,
            "stage": "image_caption_qa",
            "prompt_version": BCC_PROMPT_VERSION,
            "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
            "stage_version": PIPELINE_STAGE_VERSION,
            "reason": "generation_or_schema_failed",
            "validation_errors": last_errors,
            **({"retry_stop_reason": retry_stop_reason} if retry_stop_reason else {}),
        },
        rejected_path,
    )
    return final_path


def _estimate_bcc_visual_tokens(image_paths: list[str]) -> int:
    """Conservative cached estimate used to bucket BCC visual packets."""
    total = 0
    for path in image_paths:
        width, height = _cached_image_size(str(path))
        pixels = max(65_536, min(int(width) * int(height), 16_777_216))
        total += (pixels + 783) // 784
    return total


def _bcc_packet_batches(
    items: list[dict[str, Any]],
    stage_config: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    batch_size = max(1, int(stage_config.get("batch_size", 2) or 2))
    visual_ceiling = max(1, int(stage_config.get("max_visual_tokens_per_batch", 16_384)))
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    ordered = sorted(
        items,
        key=lambda item: (
            len(item.get("rows") or []),
            int(item["visual_tokens"]),
            str(item.get("image_id") or ""),
        ),
    )
    for item in ordered:
        estimate = int(item["visual_tokens"])
        if current and (
            len(current) >= batch_size or current_tokens + estimate > visual_ceiling
        ):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += estimate
    if current:
        batches.append(current)
    return batches


def run_image_caption_pass_batch(
    config: dict[str, Any],
    run_dir: str | Path,
    row_groups: list[list[dict[str, Any]]],
    *,
    captioner: QwenCaptioner | None = None,
    mock: bool = False,
) -> Path:
    """Run BCC pass one in visual-budgeted batches, then retry failures individually."""
    run_dir = Path(run_dir)
    candidate_path = run_dir / "image_caption_candidates.jsonl"
    if mock:
        for rows in row_groups:
            run_image_caption_pass(config, run_dir, rows, captioner=captioner, mock=True)
        return candidate_path
    if captioner is None:
        captioner = QwenCaptioner(config, config_section="image_caption")
    stage_config = config.get("image_caption", {})
    min_groups = int(stage_config.get("min_groups", 10))
    min_input_masks = int(
        stage_config.get("min_input_masks", stage_config.get("min_groups", 10))
    )
    require_all_masks = bool(stage_config.get("require_all_masks", True))
    raw_path = run_dir / "image_caption_raw.jsonl"
    rejected_path = run_dir / "image_caption_rejected.jsonl"
    completed = {
        str(row.get("image_id") or "")
        for row in (read_jsonl_indexed(candidate_path) if candidate_path.exists() else [])
        if _is_current_correspondence_record(row)
    } | {
        str(row.get("image_id") or "")
        for row in (read_jsonl_indexed(rejected_path) if rejected_path.exists() else [])
        if str(row.get("reason") or "") != "generation_or_schema_failed"
        and _is_current_correspondence_record(row)
    }
    seed_base = int(config.get("random_seed", 17)) + int(stage_config.get("seed_offset", 300000))
    prepared: list[dict[str, Any]] = []
    for position, rows in enumerate(row_groups):
        if not rows:
            continue
        image_id = str(rows[0]["image_id"])
        if image_id in completed:
            continue
        if len(rows) < min_input_masks:
            run_image_caption_pass(config, run_dir, rows, captioner=captioner, mock=False)
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
            }
        )
    for batch in _bcc_packet_batches(prepared, stage_config):
        generation_config = bcc_generation_config(
            config, "image_caption", max(len(item["rows"]) for item in batch)
        )
        try:
            results = captioner.generate_many_bcc(
                [item["packet"] for item in batch],
                [item["prompt"] for item in batch],
                [item["seed"] for item in batch],
                batch_size=len(batch),
                generation_config=generation_config,
            )
        except Exception as exc:
            append_jsonl(
                {
                    "stage": "image_caption_batch",
                    "image_ids": [item["image_id"] for item in batch],
                    "batch_size": len(batch),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                run_dir / "bcc_batch_errors.jsonl",
            )
            for item in batch:
                run_image_caption_pass(
                    config, run_dir, item["rows"], captioner=captioner, mock=False
                )
            continue
        for item, result in zip(batch, results):
            raw = result["raw"]
            append_jsonl(
                {
                    "image_id": item["image_id"],
                    "attempt": 1,
                    "raw": raw,
                    **_generation_metrics(result),
                    "batched": len(batch) > 1,
                    "batch_size": len(batch),
                    "estimated_visual_tokens": item["visual_tokens"],
                    "prompt_version": BCC_PROMPT_VERSION,
                    "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                    "stage_version": PIPELINE_STAGE_VERSION,
                    "bcc_input_manifest": item["manifest"],
                },
                raw_path,
            )
            try:
                parsed = extract_json(raw)
            except Exception:
                run_image_caption_pass(
                    config, run_dir, item["rows"], captioner=captioner, mock=False,
                    initial_raw=raw,
                )
                continue
            normalized, errors = normalize_correspondence(
                parsed,
                item["rows"],
                min_groups=min_groups,
                require_all_masks=require_all_masks,
            )
            errors = _decision_errors(parsed, qa=False) + errors
            if errors:
                run_image_caption_pass(
                    config, run_dir, item["rows"], captioner=captioner, mock=False,
                    initial_raw=raw,
                )
                continue
            record = _enrich_groups(
                {
                    "image_id": item["image_id"],
                    "source_image_path": item["source_path"],
                    "correspondence_overlay_path": item["overlay_path"],
                    "bcc_input_manifest": item["manifest"],
                    "prompt_version": BCC_PROMPT_VERSION,
                    "model": generation_config.get("model_name", "Qwen/Qwen3.5-9B"),
                    "pass": 1,
                    "batched": len(batch) > 1,
                    **normalized,
                },
                item["rows"],
            )
            append_jsonl(record, candidate_path)
    return candidate_path


def run_image_caption_qa_batch(
    config: dict[str, Any],
    run_dir: str | Path,
    row_groups: list[list[dict[str, Any]]],
    *,
    captioner: QwenCaptioner | None = None,
    mock: bool = False,
) -> Path:
    """Run BCC pass two in visual-budgeted batches, then retry failures individually."""
    run_dir = Path(run_dir)
    final_path = run_dir / "image_text_pairs.jsonl"
    if mock:
        for rows in row_groups:
            run_image_caption_qa(config, run_dir, rows, captioner=captioner, mock=True)
        return final_path
    if captioner is None:
        captioner = QwenCaptioner(config, config_section="image_caption_qa")
    candidates_path = run_dir / "image_caption_candidates.jsonl"
    candidates = read_jsonl_indexed(candidates_path) if candidates_path.exists() else []
    candidate_by_id = {
        str(row.get("image_id") or ""): row
        for row in candidates
        if str(row.get("image_id") or "")
        and _is_current_correspondence_record(row)
    }
    rejected_path = run_dir / "image_caption_qa_rejected.jsonl"
    raw_path = run_dir / "image_caption_qa_raw.jsonl"
    completed = {
        str(row.get("image_id") or "")
        for row in (read_jsonl_indexed(final_path) if final_path.exists() else [])
        if _is_current_correspondence_record(row)
    } | {
        str(row.get("image_id") or "")
        for row in (read_jsonl_indexed(rejected_path) if rejected_path.exists() else [])
        if str(row.get("reason") or "") != "generation_or_schema_failed"
        and _is_current_correspondence_record(row)
    }
    stage_config = config.get("image_caption_qa", {})
    min_groups = int(config.get("image_caption", {}).get("min_groups", 10))
    require_all_masks = bool(config.get("image_caption", {}).get("require_all_masks", True))
    seed_base = int(config.get("random_seed", 17)) + int(stage_config.get("seed_offset", 400000))
    prepared: list[dict[str, Any]] = []
    for position, rows in enumerate(row_groups):
        if not rows:
            continue
        image_id = str(rows[0]["image_id"])
        candidate = candidate_by_id.get(image_id)
        if candidate is None or image_id in completed:
            continue
        packet, manifest = build_caption_image_packet(
            rows, candidate["correspondence_overlay_path"]
        )
        prepared.append(
            {
                "image_id": image_id,
                "rows": rows,
                "candidate": candidate,
                "packet": packet,
                "manifest": manifest,
                "prompt": build_qa_prompt(candidate, rows, manifest),
                "seed": seed_base + position,
                "visual_tokens": _estimate_bcc_visual_tokens(packet),
            }
        )
    for batch in _bcc_packet_batches(prepared, stage_config):
        generation_config = bcc_generation_config(
            config, "image_caption_qa", max(len(item["rows"]) for item in batch)
        )
        try:
            results = captioner.generate_many_bcc(
                [item["packet"] for item in batch],
                [item["prompt"] for item in batch],
                [item["seed"] for item in batch],
                batch_size=len(batch),
                generation_config=generation_config,
            )
        except Exception as exc:
            append_jsonl(
                {
                    "stage": "image_caption_qa_batch",
                    "image_ids": [item["image_id"] for item in batch],
                    "batch_size": len(batch),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                run_dir / "bcc_batch_errors.jsonl",
            )
            for item in batch:
                run_image_caption_qa(
                    config, run_dir, item["rows"], captioner=captioner, mock=False
                )
            continue
        for item, result in zip(batch, results):
            raw = result["raw"]
            append_jsonl(
                {
                    "image_id": item["image_id"],
                    "attempt": 1,
                    "raw": raw,
                    **_generation_metrics(result),
                    "batched": len(batch) > 1,
                    "batch_size": len(batch),
                    "estimated_visual_tokens": item["visual_tokens"],
                    "prompt_version": BCC_PROMPT_VERSION,
                    "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                    "stage_version": PIPELINE_STAGE_VERSION,
                    "bcc_input_manifest": item["manifest"],
                },
                raw_path,
            )
            try:
                parsed = extract_json(raw)
            except Exception:
                run_image_caption_qa(
                    config, run_dir, item["rows"], captioner=captioner, mock=False,
                    initial_raw=raw,
                )
                continue
            normalized, errors = normalize_correspondence(
                parsed,
                item["rows"],
                min_groups=min_groups,
                require_all_masks=require_all_masks,
            )
            errors = _decision_errors(parsed, qa=True) + errors
            if errors:
                run_image_caption_qa(
                    config, run_dir, item["rows"], captioner=captioner, mock=False,
                    initial_raw=raw,
                )
                continue
            candidate = item["candidate"]
            final = _enrich_groups(
                {
                    "image_id": item["image_id"],
                    "source_image_path": candidate["source_image_path"],
                    "correspondence_overlay_path": candidate["correspondence_overlay_path"],
                    "bcc_input_manifest": item["manifest"],
                    "prompt_version": BCC_PROMPT_VERSION,
                    "model": generation_config.get("model_name", "Qwen/Qwen3.5-9B"),
                    "pass": 2,
                    "qa_reason": str(parsed.get("reason_code") or parsed.get("reason") or ""),
                    "first_pass_caption": candidate["caption"],
                    "first_pass_groups": candidate["groups"],
                    "batched": len(batch) > 1,
                    **normalized,
                },
                item["rows"],
            )
            append_jsonl(final, final_path)
    return final_path


def final_success_count(run_dir: str | Path) -> int:
    path = Path(run_dir) / "image_text_pairs.jsonl"
    if not path.exists():
        return 0
    return len({
        str(row.get("image_id") or "")
        for row in read_jsonl_indexed(path)
        if _is_current_correspondence_record(row)
    })


def ensure_correspondence_outputs(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    for name in (
        "image_caption_candidates.jsonl",
        "image_caption_rejected.jsonl",
        "image_caption_qa_rejected.jsonl",
        "image_text_pairs.jsonl",
    ):
        path = run_dir / name
        if not path.exists():
            write_jsonl([], path)
