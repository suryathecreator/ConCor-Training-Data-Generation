from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_jsonl


def load_records(config: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    manifest_path = Path(dataset["manifest_path"]).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = Path(config.get("project_root", ".")).resolve() / manifest_path
    image_root = dataset.get("image_root")
    image_root_path = None
    if image_root:
        image_root_path = Path(image_root).expanduser()
        if not image_root_path.is_absolute():
            image_root_path = Path(config.get("project_root", ".")).resolve() / image_root_path
    rows = read_jsonl(manifest_path)
    start = int(dataset.get("start_index", 0) or 0)
    configured_limit = dataset.get("limit")
    if limit is None and configured_limit is not None:
        limit = int(configured_limit)
    records = rows[start:] if limit is None else rows[start : start + int(limit)]
    out = []
    for row in records:
        if row.get("image_path"):
            image_path = Path(row["image_path"]).expanduser()
        elif image_root_path is not None and row.get("file_name"):
            image_path = image_root_path / str(row["file_name"])
        else:
            raise ValueError(f"Manifest row needs image_path or file_name with image_root: {row}")
        out.append(
            {
                "image_id": str(row.get("image_id") or image_path.stem),
                "image_path": str(image_path),
                "source_context": row.get("source_context") or {},
                "raw_record": row,
            }
        )
    return out
