from __future__ import annotations

import json
import re
from typing import Any



def _repair_truncated_outer_object(fragment: str) -> dict[str, Any] | None:
    """Close at most two trailing containers; never invent string content."""
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for char in fragment:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or stack[-1] != pairs[char]:
                return None
            stack.pop()
            if not stack:
                return None
    if in_string or not stack or stack[0] != "{" or len(stack) > 2:
        return None
    closers = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    try:
        parsed = json.loads(fragment.rstrip() + closers)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None



def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped.strip(), flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for match in re.finditer(r"\{", stripped):
        try:
            parsed, consumed = decoder.raw_decode(stripped[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append((match.start() + consumed, match.start(), parsed))
    for match in re.finditer(r"\{", stripped):
        repaired = _repair_truncated_outer_object(stripped[match.start() :])
        if repaired is not None:
            # Sort after any complete nested object ending at the truncation.
            candidates.append(
                (len(stripped) + 1, match.start(), repaired)
            )

    if not candidates:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    # Thinking-enabled responses may contain earlier brace examples. The final
    # complete top-level object has the greatest absolute end position; for a
    # tie, prefer the earliest opening brace so a nested child is not selected.
    _, _, parsed = max(candidates, key=lambda item: (item[0], -item[1]))
    return parsed
