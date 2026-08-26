from __future__ import annotations

"""Reproducible noun compatibility backed by Open English WordNet.

The BCC checker deliberately permits only three lexical relationships:

* the normalized words are identical;
* the words occur in the same noun synset; or
* one noun synset is a direct (one-edge) hypernym/hyponym of the other.

This replaces the former hand-maintained alias families.  It is intentionally
not a transitive thesaurus lookup: accepting arbitrary ancestors would make
``object`` compatible with almost every visible thing and hide bad links.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Any, Iterable

# `wn` resolves its database directory when it is imported. Default to the
# repository-owned scrubbed copy before that import so CLI utilities, tests,
# and checkpoint-only re-finalization never fall back to ~/.wn_data. Explicit
# WN_DATA_DIR values still win.
_default_wordnet_dir = Path(__file__).resolve().parents[2] / ".runtime" / "wordnet"
_default_wordnet_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WN_DATA_DIR", str(_default_wordnet_dir))

try:
    import wn
except ImportError:  # SAM3-only workers never invoke the lexical checker.
    wn = None  # type: ignore[assignment]


WORDNET_LEXICON = os.environ.get("BCC_WORDNET_LEXICON", "oewn:2025")

# Direct relations ending at these very broad nouns are not useful evidence
# that two masks have the same visible identity.  Same-synset matches still
# work (for example ``physical object``/``object`` when explicitly intended).
_GENERIC_TAXONOMY_ENDPOINTS = frozenset(
    {
        "abstraction",
        "artifact",
        "entity",
        "matter",
        "object",
        "physical entity",
        "physical object",
        "thing",
        "unit",
        "whole",
    }
)


@dataclass(frozen=True)
class SemanticMatch:
    expected: str
    mentioned: str
    relation: str
    expected_synset: str | None = None
    mentioned_synset: str | None = None
    bridge_synset: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def normalize_semantic_term(term: str) -> str:
    """Normalize a spaCy lemma or short surface noun for WordNet lookup."""
    value = re.sub(r"[_\s]+", " ", str(term or "").strip().casefold())
    value = value.replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    # These fallbacks matter when this function is called on a raw token rather
    # than a spaCy lemma.  WordNet itself performs additional morphology.
    irregular = {
        "children": "child",
        "feet": "foot",
        "men": "man",
        "people": "person",
        "teeth": "tooth",
        "women": "woman",
    }
    plural_only = {
        "eyeglasses",
        "glasses",
        "jeans",
        "pants",
        "shorts",
        "sunglasses",
        "trousers",
    }
    if value in irregular:
        value = irregular[value]
    elif value in plural_only:
        value = value
    elif value.endswith("ies") and len(value) > 4:
        value = value[:-3] + "y"
    elif value.endswith("s") and len(value) > 3 and not value.endswith("ss"):
        value = value[:-1]
    return value


@lru_cache(maxsize=1)
def _wordnet() -> Any:
    if wn is None:
        raise RuntimeError(
            "The BCC semantic checker requires wn==1.1.0; run "
            "scripts/install_wordnet.sh in the Qwen environment."
        )
    try:
        return wn.Wordnet(WORDNET_LEXICON, expand="")
    except Exception as error:  # pragma: no cover - environment diagnostic
        data_dir = os.environ.get("WN_DATA_DIR", "<wn default>")
        raise RuntimeError(
            f"Pinned WordNet lexicon {WORDNET_LEXICON!r} is unavailable in "
            f"WN_DATA_DIR={data_dir}. Run scripts/install_wordnet.sh first."
        ) from error


@lru_cache(maxsize=65_536)
def _noun_synsets(term: str) -> tuple[Any, ...]:
    normalized = normalize_semantic_term(term)
    if not normalized:
        return ()
    candidates = [normalized]
    underscored = normalized.replace(" ", "_")
    if underscored != normalized:
        candidates.append(underscored)
    found: dict[str, Any] = {}
    for candidate in candidates:
        for synset in _wordnet().synsets(candidate, pos="n"):
            found[synset.id] = synset
    return tuple(found[key] for key in sorted(found))


def _lemma_surfaces(synset: Any) -> set[str]:
    return {
        normalize_semantic_term(lemma)
        for lemma in synset.lemmas()
        if normalize_semantic_term(lemma)
    }


@lru_cache(maxsize=131_072)
def semantic_match(expected: str, mentioned: str) -> SemanticMatch | None:
    """Return exact/synonym/direct-taxonomy evidence, never transitive evidence."""
    left = normalize_semantic_term(expected)
    right = normalize_semantic_term(mentioned)
    if not left or not right:
        return None
    if left == right:
        return SemanticMatch(left, right, "exact_lemma")

    left_synsets = _noun_synsets(left)
    right_synsets = _noun_synsets(right)
    right_by_id = {synset.id: synset for synset in right_synsets}
    for left_synset in left_synsets:
        if left_synset.id in right_by_id:
            return SemanticMatch(
                left,
                right,
                "same_synset",
                expected_synset=left_synset.id,
                mentioned_synset=left_synset.id,
            )

    if left in _GENERIC_TAXONOMY_ENDPOINTS or right in _GENERIC_TAXONOMY_ENDPOINTS:
        return None

    right_ids = set(right_by_id)
    for left_synset in left_synsets:
        for parent in left_synset.hypernyms():
            if parent.id in right_ids:
                return SemanticMatch(
                    left,
                    right,
                    "expected_is_direct_hyponym",
                    expected_synset=left_synset.id,
                    mentioned_synset=parent.id,
                    bridge_synset=parent.id,
                )
        for child in left_synset.hyponyms():
            if child.id in right_ids:
                return SemanticMatch(
                    left,
                    right,
                    "expected_is_direct_hypernym",
                    expected_synset=left_synset.id,
                    mentioned_synset=child.id,
                    bridge_synset=left_synset.id,
                )
    return None


def semantic_matches(
    expected_terms: Iterable[str], mentioned_terms: Iterable[str]
) -> list[SemanticMatch]:
    matches: list[SemanticMatch] = []
    seen: set[tuple[str, str]] = set()
    for expected in expected_terms:
        for mentioned in mentioned_terms:
            match = semantic_match(str(expected), str(mentioned))
            if match is None or (match.expected, match.mentioned) in seen:
                continue
            seen.add((match.expected, match.mentioned))
            matches.append(match)
    return matches


def semantic_terms_compatible(
    expected_terms: Iterable[str], mentioned_terms: Iterable[str]
) -> bool:
    return bool(semantic_matches(expected_terms, mentioned_terms))


@lru_cache(maxsize=32_768)
def taxonomy_alternatives(term: str, limit: int = 32) -> tuple[str, ...]:
    """Keep the prompt anchor compact while validation uses the taxonomy.

    A word can have many unrelated senses (``hand`` and ``short`` are classic
    examples). Dumping every WordNet lemma into the visual prompt would invite
    hallucinations. Qwen sees the reviewed subject anchor and may naturally
    choose a synonym/specific child; the checker then verifies that choice and
    stores its WordNet evidence.
    """
    normalized = normalize_semantic_term(term)
    return (normalized,) if normalized else ()


@lru_cache(maxsize=65_536)
def semantic_is_a(term: str, category: str, max_depth: int = 4) -> bool:
    """Taxonomy-only category test used for pronouns/geometry, not link acceptance."""
    normalized_term = normalize_semantic_term(term)
    normalized_category = normalize_semantic_term(category)
    if normalized_term == normalized_category:
        return True
    category_ids = {synset.id for synset in _noun_synsets(normalized_category)}
    frontier = list(_noun_synsets(normalized_term))
    seen = {synset.id for synset in frontier}
    for _ in range(max(0, int(max_depth))):
        next_frontier: list[Any] = []
        for synset in frontier:
            for parent in synset.hypernyms():
                if parent.id in category_ids:
                    return True
                if parent.id not in seen:
                    seen.add(parent.id)
                    next_frontier.append(parent)
        frontier = next_frontier
        if not frontier:
            break
    return False
