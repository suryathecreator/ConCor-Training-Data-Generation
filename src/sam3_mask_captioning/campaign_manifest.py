from __future__ import annotations

import io
import json
import os
import re
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from .selection import is_excluded, load_exclusion_csv, read_source_manifest


CAMPAIGN_SCHEMA_VERSION = "gpic-bcc-campaign-v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "item"


def campaign_paths(root: str | Path) -> dict[str, Path]:
    root = Path(root).expanduser().resolve()
    return {
        "root": root,
        "registry": root / "campaign_registry.json",
        "manifest": root / "source_manifest.jsonl",
        "units": root / "units",
        "published": root / "published",
        "site": root / "site",
    }


def load_registry(root: str | Path) -> dict[str, Any]:
    paths = campaign_paths(root)
    if not paths["registry"].exists():
        return {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "unit_size": 100,
            "source_count": 0,
            "unit_count": 0,
            "extensions": [],
            "selection": {},
        }
    return json.loads(paths["registry"].read_text(encoding="utf-8"))


def initialize_campaign(
    root: str | Path,
    *,
    unit_size: int = 100,
    seed: int = 20260808,
    dataset: str = "stanford-vision-lab/gpic",
    split: str = "train",
    terminal_stage: str = "bcc-rewrite",
    preview_pairs: int = 0,
) -> dict[str, Any]:
    paths = campaign_paths(root)
    paths["units"].mkdir(parents=True, exist_ok=True)
    paths["published"].mkdir(parents=True, exist_ok=True)
    paths["site"].mkdir(parents=True, exist_ok=True)
    registry_exists = paths["registry"].exists()
    registry = load_registry(root)
    if registry_exists:
        if registry.get("source_count", 0) and int(registry.get("unit_size", 0)) != int(unit_size):
            raise ValueError("Cannot change unit_size after sources have been committed")
        return registry
    registry.update(
        {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "unit_size": int(unit_size),
            "seed": int(seed),
            "dataset": dataset,
            "split": split,
            "terminal_stage": str(terminal_stage),
            "preview_pairs": max(0, int(preview_pairs)),
            "updated_at": _utc_now(),
        }
    )
    write_json(registry, paths["registry"])
    if not paths["manifest"].exists():
        write_jsonl([], paths["manifest"])
    return registry


def _source_key(row: dict[str, Any]) -> str:
    context = row.get("source_context") or {}
    return str(context.get("pair_key") or row.get("pair_key") or row.get("image_id") or "")


def _rebuild_source_manifest(
    root: Path, registry: dict[str, Any]
) -> list[dict[str, Any]]:
    """Atomically materialize the logical append-only manifest from commit shards."""
    rows: list[dict[str, Any]] = []
    expected_index = 0
    for extension in registry.get("extensions") or []:
        if extension.get("status") != "complete":
            continue
        shard_value = extension.get("manifest_shard")
        if not shard_value:
            raise RuntimeError(
                f"Complete extension {extension.get('extension_id')} has no manifest shard"
            )
        shard = root / str(shard_value)
        committed = read_jsonl(shard)
        expected_count = int(extension.get("added_count") or 0)
        if len(committed) != expected_count:
            raise RuntimeError(
                f"Extension shard {shard} has {len(committed)} rows, expected {expected_count}"
            )
        for row in committed:
            index = int(row.get("source_manifest_index") or 0)
            if index != expected_index:
                raise RuntimeError(
                    f"Non-contiguous source index {index}; expected {expected_index}"
                )
            expected_index += 1
        rows.extend(committed)
    if expected_index != int(registry.get("source_count") or 0):
        raise RuntimeError(
            f"Registry source_count={registry.get('source_count')} but committed shards contain {expected_index}"
        )
    write_jsonl(rows, campaign_paths(root)["manifest"])
    return rows


def _extension_start(
    root: Path, add_count: int, source_kind: str
) -> tuple[dict[str, Any], dict[str, Any], set[str], set[str]]:
    registry = load_registry(root)
    if registry.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported campaign registry: {registry.get('schema_version')}")
    interrupted = False
    for prior in registry.get("extensions") or []:
        if prior.get("status") == "in_progress":
            prior.update(
                {
                    "status": "interrupted_superseded",
                    "finished_at": _utc_now(),
                    "note": "No commit shard was made authoritative; uncommitted unit paths may be overwritten safely.",
                }
            )
            interrupted = True
    if interrupted:
        registry["updated_at"] = _utc_now()
        write_json(registry, campaign_paths(root)["registry"])
    existing = _rebuild_source_manifest(root, registry)
    extension = {
        "extension_id": len(registry.get("extensions") or []),
        "started_at": _utc_now(),
        "source_kind": source_kind,
        "requested_add_count": int(add_count),
        "start_source_index": len(existing),
        "start_unit_index": int(registry.get("unit_count") or 0),
        "status": "in_progress",
    }
    registry.setdefault("extensions", []).append(extension)
    registry["updated_at"] = _utc_now()
    write_json(registry, campaign_paths(root)["registry"])
    return (
        registry,
        extension,
        {_source_key(row) for row in existing},
        {str(row.get("image_id") or "") for row in existing},
    )


def _write_source_unit(
    root: Path,
    unit_index: int,
    source_index: int,
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unit_dir = campaign_paths(root)["units"] / f"{unit_index:06d}"
    artifacts_dir = unit_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    archive_path = artifacts_dir / "source.tar"
    fd, temporary = tempfile.mkstemp(
        prefix=".source.", suffix=".tar.tmp", dir=artifacts_dir
    )
    os.close(fd)
    rows: list[dict[str, Any]] = []
    try:
        with tarfile.open(temporary, "w") as archive:
            for offset, item in enumerate(items):
                row = dict(item["row"])
                suffix = str(item.get("suffix") or Path(str(row.get("image_path") or "")).suffix or ".jpg").lower()
                if suffix == ".jpeg":
                    suffix = ".jpg"
                image_id = str(row.get("image_id") or f"gpic_train_{source_index + offset:09d}")
                member_name = f"source_images/{_safe_id(image_id)}{suffix}"
                payload = item.get("bytes")
                if payload is None:
                    payload = Path(str(row["image_path"])).read_bytes()
                info = tarfile.TarInfo(member_name)
                info.size = len(payload)
                info.mtime = 0
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(payload))
                global_index = source_index + offset
                source_context = dict(row.get("source_context") or {})
                source_context["source_archive"] = str(archive_path)
                source_context["source_member"] = member_name
                committed = {
                    "image_id": image_id,
                    "image_path": str(unit_dir / member_name),
                    "source_context": source_context,
                    "source_manifest_index": global_index,
                    "campaign_unit": unit_index,
                    "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
                }
                rows.append(committed)
        os.replace(temporary, archive_path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    write_jsonl(rows, unit_dir / "selected_images.jsonl")
    archive_meta = {
        "archive_path": str(archive_path),
        "sha256": sha256_file(archive_path),
        "member_count": len(rows),
    }
    write_json(
        {
            "unit_index": unit_index,
            "source_start": source_index,
            "source_end": source_index + len(rows) - 1,
            "source_count": len(rows),
            "source_archive": archive_meta,
            "created_at": _utc_now(),
        },
        unit_dir / "unit.json",
    )
    return rows, archive_meta


def _commit_extension(
    root: Path,
    registry: dict[str, Any],
    extension: dict[str, Any],
    committed_rows: list[dict[str, Any]],
    units_added: int,
) -> dict[str, Any]:
    shard_relative = Path("extensions") / f"extension-{int(extension['extension_id']):06d}.jsonl"
    shard_path = root / shard_relative
    write_jsonl(committed_rows, shard_path)
    registry["source_count"] = int(registry.get("source_count") or 0) + len(committed_rows)
    registry["unit_count"] = int(registry.get("unit_count") or 0) + int(units_added)
    extension.update(
        {
            "finished_at": _utc_now(),
            "status": "complete",
            "added_count": len(committed_rows),
            "end_source_index": int(registry["source_count"]) - 1,
            "end_unit_index": int(registry["unit_count"]) - 1,
            "manifest_shard": shard_relative.as_posix(),
            "manifest_shard_sha256": sha256_file(shard_path),
        }
    )
    registry["updated_at"] = _utc_now()
    write_json(registry, campaign_paths(root)["registry"])
    _rebuild_source_manifest(root, registry)
    return registry


def extend_from_manifest(
    root: str | Path,
    manifest_path: str | Path,
    *,
    add_images: int | None = None,
    target_total: int | None = None,
    image_root: str | Path | None = None,
    exclude_csv: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if not campaign_paths(root)["registry"].exists():
        initialize_campaign(root)
    registry = load_registry(root)
    current = int(registry.get("source_count") or 0)
    requested = int(add_images) if add_images is not None else int(target_total or 0) - current
    if requested <= 0:
        return registry
    registry, extension, existing_keys, existing_ids = _extension_start(
        root, requested, "local_manifest"
    )
    exclusions, exclusion_provenance = load_exclusion_csv(exclude_csv)
    if exclusion_provenance:
        registry.setdefault("selection", {})["exclusion_list"] = exclusion_provenance
        write_json(registry, campaign_paths(root)["registry"])
    candidates = read_source_manifest(manifest_path)
    selected: list[dict[str, Any]] = []
    for row in candidates:
        key = _source_key(row)
        image_id = str(row.get("image_id") or "")
        if is_excluded(
            exclusions,
            key,
            image_id,
            row.get("image_path"),
            row.get("file_name"),
        ):
            continue
        if not key or key in existing_keys or image_id in existing_ids:
            continue
        image_value = row.get("image_path")
        if not image_value and image_root is not None and row.get("file_name"):
            image_value = Path(image_root) / str(row["file_name"])
        image_path = Path(str(image_value or ""))
        if not image_path.is_file():
            continue
        materialized_row = dict(row)
        materialized_row["image_path"] = str(image_path)
        selected.append({"row": materialized_row, "suffix": image_path.suffix})
        existing_keys.add(key)
        existing_ids.add(image_id)
        if len(selected) >= requested:
            break
    if len(selected) < requested:
        raise RuntimeError(f"Only found {len(selected)} new manifest rows; requested {requested}")
    unit_size = int(registry["unit_size"])
    source_index = int(registry.get("source_count") or 0)
    unit_index = int(registry.get("unit_count") or 0)
    committed: list[dict[str, Any]] = []
    units_added = 0
    for start in range(0, len(selected), unit_size):
        rows, _ = _write_source_unit(
            root, unit_index + units_added, source_index + len(committed), selected[start : start + unit_size]
        )
        committed.extend(rows)
        units_added += 1
    return _commit_extension(root, registry, extension, committed, units_added)


def manifest_target_add_count(
    root: str | Path, *, add_images: int | None, target_total: int | None
) -> int:
    current = int(load_registry(root).get("source_count") or 0)
    if add_images is not None:
        return max(0, int(add_images))
    if target_total is None:
        raise ValueError("Provide --add-images or --target-total")
    return max(0, int(target_total) - current)
