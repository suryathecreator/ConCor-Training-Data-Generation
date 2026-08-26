from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, sha256_file


EXCLUSION_HEADERS = {
    "identifier",
    "id",
    "image_id",
    "file",
    "file_name",
    "filename",
    "pair_key",
    "source_key",
}


def identifier_variants(value: Any) -> set[str]:
    """Return stable forms accepted by the GPIC/local exclusion list."""
    raw = str(value or "").strip()
    if not raw:
        return set()
    normalized = raw.replace("\\", "/")
    path = Path(normalized)
    values = {raw, normalized, path.name, path.stem}
    if path.suffix:
        values.add(path.with_suffix("").as_posix())
    return {value for value in values if value}


def load_exclusion_csv(path: str | Path | None) -> tuple[set[str], dict[str, Any] | None]:
    """Load a one-column identifier CSV, with or without a header.

    Recognized headers include image_id, file_name, pair_key, and identifier.
    Only a checksum and basename are retained in campaign provenance; a user's
    machine-specific absolute path is never serialized.
    """
    if not path:
        return set(), None
    source = Path(path).expanduser().resolve()
    values: list[str] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return set(), {
            "file_name": source.name,
            "sha256": sha256_file(source),
            "identifier_count": 0,
        }
    header = [cell.strip().lower() for cell in rows[0]]
    selected_column = next(
        (index for index, cell in enumerate(header) if cell in EXCLUSION_HEADERS),
        None,
    )
    start = 1 if selected_column is not None else 0
    column = selected_column if selected_column is not None else 0
    for row in rows[start:]:
        if column < len(row) and row[column].strip():
            values.append(row[column].strip())
    exclusions: set[str] = set()
    for value in values:
        exclusions.update(identifier_variants(value))
    return exclusions, {
        "file_name": source.name,
        "sha256": sha256_file(source),
        "identifier_count": len(values),
    }


def is_excluded(exclusions: set[str], *values: Any) -> bool:
    return bool(exclusions) and any(
        identifier_variants(value) & exclusions for value in values if value is not None
    )


def read_source_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read the public JSONL or CSV source-manifest contract."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = read_jsonl(source)
    elif source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    else:
        raise ValueError("Source manifest must be .jsonl, .ndjson, or .csv")

    normalized: list[dict[str, Any]] = []
    for position, original in enumerate(rows):
        row = {key: value for key, value in dict(original).items() if value not in (None, "")}
        metadata_value = row.pop("metadata_json", None)
        metadata: dict[str, Any] = {}
        if isinstance(metadata_value, str) and metadata_value.strip():
            parsed = json.loads(metadata_value)
            if not isinstance(parsed, dict):
                raise ValueError(f"metadata_json on row {position + 1} must be an object")
            metadata = parsed
        context = dict(row.get("source_context") or {})
        for key in ("source_dataset", "split", "pair_key", "paired_text"):
            if key in row:
                context.setdefault(key, row[key])
        if metadata:
            context.setdefault("metadata", metadata)
        image_id = str(row.get("image_id") or row.get("id") or "").strip()
        file_name = str(row.get("file_name") or row.get("filename") or "").strip()
        if not image_id:
            image_id = Path(file_name).stem if file_name else f"image_{position:09d}"
        context.setdefault("pair_key", str(row.get("pair_key") or image_id))
        row["image_id"] = image_id
        if file_name:
            row["file_name"] = file_name
        row["source_context"] = context
        normalized.append(row)
    return normalized
