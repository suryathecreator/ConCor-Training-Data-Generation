from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


BACKGROUND_NOUNS = {
    "background",
    "backdrop",
    "foreground",
    "scene",
    "setting",
    "surrounding",
    "surroundings",
}
VIEW_NOUNS = {
    "close-up",
    "closeup",
    "frame",
    "image",
    "photo",
    "photograph",
    "picture",
    "view",
}
GENERIC_HEADS = {
    "area",
    "content",
    "element",
    "item",
    "object",
    "part",
    "piece",
    "region",
    "section",
    "shape",
    "thing",
}
_BANNED_ATTRIBUTE_TERMS = BACKGROUND_NOUNS | VIEW_NOUNS | {
    "black area",
    "masked area",
    "removed area",
    "silhouette",
}

_LEADING_VIEW_RE = re.compile(
    r"^\s*(?:an?\s+|the\s+)?"
    r"(?:close[- ]?up(?:\s+view)?|cropped\s+view|detailed\s+view|"
    r"image|photo(?:graph)?|picture|view)\s+"
    r"(?:of|showing|depicting)\s+",
    re.IGNORECASE,
)
_BACKGROUND_CLAUSE_RES = (
    re.compile(
        r"\s*,?\s*(?:set|shown|seen|standing|sitting|placed|positioned|displayed)?\s*"
        r"(?:against|before|on|over|in\s+front\s+of)\s+"
        r"(?:(?:an?|the)\s+)?[^,.;]{0,64}?\b(?:background|backdrop)\b[^,.;]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*,?\s*(?:with|featuring|showing)\s+"
        r"(?:(?:an?|the)\s+)?[^,.;]{0,64}?\b(?:background|backdrop)\b[^,.;]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*,?\s*(?:surrounded\s+by|amid|among)\s+[^,.;]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*,?\s*(?:shown|seen|viewed|pictured)\s+"
        r"(?:from|in|at)\s+[^,.;]{0,48}?\bview\b[^,.;]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*,?\s*(?:in|within)\s+(?:(?:an?|the|this)\s+)?"
        r"(?:image|photo(?:graph)?|picture|frame|scene)\b[^,.;]*",
        re.IGNORECASE,
    ),
)


@lru_cache(maxsize=1)
def _english_pipeline():
    import spacy

    try:
        return spacy.load("en_core_web_sm", exclude=["ner"])
    except OSError:
        # The production environment installs en_core_web_sm. A blank English
        # pipeline keeps deterministic tokenization available for lightweight
        # environments and unit tests.
        return spacy.blank("en")


def _normalize_caption_punctuation(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*([.;])", r"\1", text)
    text = re.sub(r"\s*,\s*$", "", text)
    text = re.sub(r"\b(?:and|with|while|against)\s*$", "", text, flags=re.IGNORECASE).strip(" ,")
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _spacy_background_ranges(text: str) -> list[tuple[int, int, str]]:
    """Find background/view noun phrases and their introducing prepositions."""
    doc = _english_pipeline()(text)
    if not doc.has_annotation("DEP"):
        return []
    ranges: list[tuple[int, int, str]] = []
    for chunk in doc.noun_chunks:
        root_lemma = chunk.root.lemma_.lower()
        chunk_lower = chunk.text.lower()
        banned = root_lemma in BACKGROUND_NOUNS or (
            root_lemma in VIEW_NOUNS
            and any(term in chunk_lower for term in ("close", "profile", "side", "cropped", "image", "photo", "picture"))
        )
        if not banned:
            continue
        start = chunk.start_char
        end = chunk.end_char
        head = chunk.root.head
        if head.pos_ == "ADP" or head.lemma_.lower() in {
            "against",
            "amid",
            "among",
            "before",
            "in",
            "on",
            "over",
            "with",
        }:
            start = head.idx
            subtree = list(head.subtree)
            if subtree:
                end = max(token.idx + len(token) for token in subtree)
        ranges.append((start, end, chunk.text))
    return ranges


def _remove_ranges(text: str, ranges: list[tuple[int, int, str]]) -> tuple[str, list[str]]:
    if not ranges:
        return text, []
    merged: list[tuple[int, int, list[str]]] = []
    for start, end, label in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end), merged[-1][2] + [label])
        else:
            merged.append((start, end, [label]))
    out = text
    corrections: list[str] = []
    for start, end, labels in reversed(merged):
        corrections.append(f"removed synthetic-context phrase: {out[start:end].strip()}")
        out = out[:start] + out[end:]
    return out, list(reversed(corrections))


def clean_caption(text: str) -> dict[str, Any]:
    """Remove model-facing crop/background leakage from a mask caption."""
    original = str(text or "").strip()
    cleaned = original
    corrections: list[str] = []

    leading = _LEADING_VIEW_RE.match(cleaned)
    if leading:
        corrections.append(f"removed crop/view preamble: {leading.group(0).strip()}")
        cleaned = cleaned[leading.end() :]

    for pattern in _BACKGROUND_CLAUSE_RES:
        while True:
            match = pattern.search(cleaned)
            if not match:
                break
            corrections.append(f"removed background/view clause: {match.group(0).strip()}")
            cleaned = cleaned[: match.start()] + cleaned[match.end() :]

    cleaned, spacy_corrections = _remove_ranges(cleaned, _spacy_background_ranges(cleaned))
    corrections.extend(spacy_corrections)
    cleaned = _normalize_caption_punctuation(cleaned)
    return {
        "original": original,
        "caption": cleaned,
        "changed": cleaned != original,
        "corrections": corrections,
        "valid": bool(cleaned),
    }


def clean_attributes(attributes: list[Any] | Any) -> dict[str, Any]:
    if isinstance(attributes, str):
        values = [attributes]
    else:
        values = list(attributes or [])
    kept: list[str] = []
    removed: list[str] = []
    for value in values:
        text = str(value).strip()
        lowered = text.lower()
        if not text:
            continue
        if any(term in lowered for term in _BANNED_ATTRIBUTE_TERMS):
            removed.append(text)
        elif text not in kept:
            kept.append(text)
    return {"attributes": kept, "removed": removed, "changed": bool(removed)}


def _noun_candidate(text: str) -> str:
    doc = _english_pipeline()(str(text or "").strip())
    if not doc:
        return ""
    if doc.has_annotation("DEP"):
        chunks = list(doc.noun_chunks)
        if chunks:
            chunk = chunks[0]
            root = chunk.root
            compound = [
                token
                for token in chunk
                if token.i <= root.i and token.dep_ in {"compound", "nmod"} and token.is_alpha
            ]
            parts = [token.lemma_.lower() for token in compound] + [root.lemma_.lower()]
            return " ".join(dict.fromkeys(part for part in parts if part))
        roots = [token for token in doc if token.dep_ == "ROOT" and token.pos_ in {"NOUN", "PROPN"}]
        if roots:
            return roots[0].lemma_.lower()
    tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    return tokens[-1] if tokens else ""


def semantic_noun_lemmas(text: str) -> set[str]:
    """Return concrete noun lemmas for deterministic mask/link checking.

    The production spaCy model supplies POS tags and lemmas. The token-only
    fallback intentionally stays conservative so unit tests do not require the
    downloaded model.
    """
    doc = _english_pipeline()(str(text or "").strip())
    if not doc:
        return set()
    if doc.has_annotation("POS"):
        terms = {
            (token.lemma_ or token.text).casefold()
            for token in doc
            if token.pos_ in {"NOUN", "PROPN"} and token.is_alpha
        }
        if terms:
            return terms
    return {
        token.text.casefold()
        for token in doc
        if token.is_alpha and not token.is_stop
    }


_PERSON_REFERENCE_FORMS = {
    "he",
    "her",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "she",
    "their",
    "theirs",
    "them",
    "themselves",
    "they",
}

_TRAILING_FINITE_VERB_FORMS = {
    "appears",
    "carries",
    "checks",
    "clusters",
    "contains",
    "covers",
    "crosses",
    "cuts",
    "displays",
    "drifts",
    "extends",
    "floats",
    "faces",
    "features",
    "fills",
    "forms",
    "frames",
    "glides",
    "gestures",
    "grips",
    "hangs",
    "holds",
    "hovers",
    "leans",
    "lies",
    "looks",
    "moves",
    "plays",
    "reaches",
    "rests",
    "rises",
    "runs",
    "shows",
    "sits",
    "stands",
    "steadies",
    "walks",
    "watches",
    "wears",
    "swings",
    "tilts",
    "wraps",
}


def caption_entity_mentions(text: str) -> list[dict[str, Any]]:
    """Return spaCy noun phrases and person-reference tokens with exact spans."""
    doc = _english_pipeline()(str(text or ""))
    if not doc:
        return []
    mentions: list[dict[str, Any]] = []
    if doc.has_annotation("DEP"):
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ not in {"NOUN", "PROPN"}:
                continue
            chunk_tokens = list(chunk)
            start = chunk.start_char
            end = chunk.end_char
            head = (chunk.root.lemma_ or chunk.root.text).casefold()
            if (
                len(chunk_tokens) > 1
                and chunk_tokens[-1].text.casefold() in _TRAILING_FINITE_VERB_FORMS
            ):
                end = chunk_tokens[-1].idx
                prior_alpha = next(
                    (token for token in reversed(chunk_tokens[:-1]) if token.is_alpha),
                    chunk.root,
                )
                head = (prior_alpha.lemma_ or prior_alpha.text).casefold()
            mention_text = doc.text[start:end].rstrip()
            end = start + len(mention_text)
            noun_terms = sorted(
                {
                    (token.lemma_ or token.text).casefold()
                    for token in chunk_tokens
                    if token.idx < end
                    and token.pos_ in {"NOUN", "PROPN"}
                    and token.is_alpha
                }
            )
            mentions.append(
                {
                    "kind": "noun_phrase",
                    "text": mention_text,
                    "start": start,
                    "end": end,
                    "noun_terms": noun_terms,
                    "head": head,
                }
            )
    for token in doc:
        lowered = token.text.casefold()
        if lowered in _PERSON_REFERENCE_FORMS:
            mentions.append(
                {
                    "kind": "person_reference",
                    "text": token.text,
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "head": lowered,
                }
            )
    return sorted(mentions, key=lambda item: (item["start"], item["end"], item["kind"]))


_CONTACT_RELATION_LEMMAS = {"carry", "clutch", "grasp", "grip", "hold"}
_CONTACT_RELATION_FORMS = {
    "carried": "carry",
    "carries": "carry",
    "carry": "carry",
    "carrying": "carry",
    "clutch": "clutch",
    "clutched": "clutch",
    "clutches": "clutch",
    "clutching": "clutch",
    "grasp": "grasp",
    "grasped": "grasp",
    "grasping": "grasp",
    "grasps": "grasp",
    "grip": "grip",
    "gripped": "grip",
    "gripping": "grip",
    "grips": "grip",
    "held": "hold",
    "hold": "hold",
    "holding": "hold",
    "holds": "hold",
}


def caption_contact_relations(text: str) -> list[dict[str, Any]]:
    """Return high-confidence spaCy subject/verb/object contact relations.

    Token spans let the correspondence validator map grammatical arguments
    back to linked mask spans. A token-only spaCy fallback still returns verb
    locations for a deliberately conservative linear fallback in the caller.
    """
    doc = _english_pipeline()(str(text or ""))
    relations: list[dict[str, Any]] = []
    for token in doc:
        raw = token.text.casefold()
        lemma = _CONTACT_RELATION_FORMS.get(
            (token.lemma_ or "").casefold(), _CONTACT_RELATION_FORMS.get(raw, raw)
        )
        if lemma not in _CONTACT_RELATION_LEMMAS:
            continue
        subjects = []
        objects = []
        if doc.has_annotation("DEP"):
            subjects = [
                child
                for child in token.children
                if child.dep_ in {"csubj", "nsubj", "nsubjpass"}
            ]
            objects = [
                child
                for child in token.children
                if child.dep_ in {"attr", "dobj", "obj", "oprd"}
            ]
            if not subjects and token.dep_ == "conj":
                subjects = [
                    child
                    for child in token.head.children
                    if child.dep_ in {"csubj", "nsubj", "nsubjpass"}
                ]
        subject = subjects[0] if len(subjects) == 1 else None
        obj = objects[0] if len(objects) == 1 else None
        relations.append(
            {
                "verb": token.text,
                "lemma": lemma,
                "verb_span": [token.idx, token.idx + len(token.text)],
                "subject_span": (
                    [subject.idx, subject.idx + len(subject.text)]
                    if subject is not None
                    else None
                ),
                "object_span": (
                    [obj.idx, obj.idx + len(obj.text)]
                    if obj is not None
                    else None
                ),
            }
        )
    return relations


def extract_main_candidate(
    *,
    object_text: str = "",
    caption: str = "",
    source_prompt: str = "",
) -> dict[str, str]:
    """Extract the concrete SAM3 re-query concept with spaCy noun parsing."""
    object_candidate = _noun_candidate(object_text)
    caption_candidate = _noun_candidate(caption)
    source_candidate = _noun_candidate(source_prompt)
    candidate = object_candidate or caption_candidate or source_candidate
    source = "object"
    if not candidate or candidate in GENERIC_HEADS | BACKGROUND_NOUNS | VIEW_NOUNS:
        candidate = source_candidate or caption_candidate
        source = "source_prompt" if source_candidate else "caption"
    if not candidate:
        candidate = str(source_prompt or object_text or caption).strip().lower()
        source = "fallback"
    return {
        "candidate": candidate,
        "source": source,
        "object_candidate": object_candidate,
        "caption_candidate": caption_candidate,
        "source_prompt_candidate": source_candidate,
    }
