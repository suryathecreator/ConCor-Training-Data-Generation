from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .artifact_store import read_tar_member
from .bcc_audit import aggregate_audits
from .bcc_html_report import write_bcc_html_report
from .campaign_manifest import campaign_paths, load_registry
from .campaign_runner import _success_path, _unit_dir
from .io_utils import append_jsonl, read_jsonl, sha256_file, write_json, write_jsonl
from .one_rewrite_stage import ONE_REWRITE_CONTRACT_VERSION


PUBLISHER_SCHEMA_VERSION = "bcc-corpus-publisher-v3-selective-links-2026-08-14"
SITE_RENDERER_VERSION = "bcc-before-after-checkpoint-preview-v6-2026-08-25"


def _row_source_manifest_index(row: dict[str, Any]) -> int | None:
    """Recover the durable source index from original or reviewed rows.

    Image review preserves the original manifest row under ``raw_record``.
    Consequently, reviewed ``selected_images.jsonl`` rows may no longer carry
    ``source_manifest_index`` at the top level even though the value is still
    present in the nested record.
    """
    current: Any = row
    for _ in range(3):
        if not isinstance(current, dict):
            break
        value = current.get("source_manifest_index")
        if value is not None:
            return int(value)
        current = current.get("raw_record")
    return None


def _unit_source_index_by_image(unit_dir: Path) -> dict[str, int]:
    rows = read_jsonl(unit_dir / "selected_images.jsonl")
    source_start = 0
    unit_metadata = unit_dir / "unit.json"
    if unit_metadata.is_file():
        try:
            source_start = int(
                json.loads(unit_metadata.read_text(encoding="utf-8")).get(
                    "source_start"
                )
                or 0
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            source_start = 0
    result: dict[str, int] = {}
    for offset, row in enumerate(rows):
        image_id = str(row.get("image_id") or "")
        if not image_id:
            continue
        recovered = _row_source_manifest_index(row)
        result[image_id] = (
            int(recovered) if recovered is not None else source_start + offset
        )
    return result


def _preview_pair_index(image_id: str, *, rejected: bool) -> int:
    """Return a stable, collision-resistant asset namespace for live previews."""
    digest = int(hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:12], 16)
    namespace = 1_000_000_000_000_000 if rejected else 2_000_000_000_000_000
    return namespace + digest


def _connect(root: Path) -> sqlite3.Connection:
    database = campaign_paths(root)["published"] / "campaign_state.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS published_units (
            unit_id INTEGER PRIMARY KEY,
            published_at REAL NOT NULL,
            success_sha256 TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS pairs (
            pair_index INTEGER PRIMARY KEY,
            image_id TEXT UNIQUE NOT NULL,
            source_manifest_index INTEGER NOT NULL,
            unit_id INTEGER NOT NULL,
            record_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audits (
            image_id TEXT PRIMARY KEY,
            source_manifest_index INTEGER NOT NULL,
            unit_id INTEGER NOT NULL,
            included INTEGER NOT NULL,
            record_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS milestones (
            milestone INTEGER PRIMARY KEY,
            pair_count INTEGER NOT NULL,
            published_at REAL NOT NULL,
            metadata_json TEXT NOT NULL
        );
        """
    )
    return connection


def _current_records(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path) if path.exists() else []
    current: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("contract_version") != ONE_REWRITE_CONTRACT_VERSION:
            continue
        image_id = str(row.get("image_id") or "")
        if image_id:
            current[image_id] = row
    return current


def _artifact_metadata(pair: dict[str, Any], unit_dir: Path) -> dict[str, Any]:
    pair = copy.deepcopy(pair)
    source_rows = {
        str(row.get("image_id") or ""): row
        for row in read_jsonl(unit_dir / "selected_images.jsonl")
    }
    source = source_rows.get(str(pair.get("image_id") or ""), {})
    context = source.get("source_context") or {}
    pair["source_artifact"] = {
        "archive_path": context.get("source_archive"),
        "member": context.get("source_member"),
    }
    sam3_archive = str(unit_dir / "artifacts" / "sam3.tar")
    for group_key in ("groups", "first_pass_groups"):
        for group in pair.get(group_key) or []:
            mask_id = str(group.get("mask_id") or "")
            group["mask_artifact"] = {
                "archive_path": sam3_archive,
                "member": f"masks/{mask_id}.png",
                "rle_index_path": str(unit_dir / "mask_rle.jsonl"),
            }
            group["inverse_crop_artifact"] = {
                "archive_path": sam3_archive,
                "member": f"inverse_crops/{mask_id}.png",
            }
    for omitted_key in ("omitted_masks", "first_pass_omitted_masks"):
        for item in pair.get(omitted_key) or []:
            mask_id = str(item.get("mask_id") or "")
            item["mask_artifact"] = {
                "archive_path": sam3_archive,
                "member": f"masks/{mask_id}.png",
            }
            item["inverse_crop_artifact"] = {
                "archive_path": sam3_archive,
                "member": f"inverse_crops/{mask_id}.png",
            }
    pair["overlay_artifact"] = {
        "archive_path": str(unit_dir / "artifacts" / "bcc.tar"),
        "member": f"correspondence_overlays/{pair.get('image_id')}.png",
    }
    return pair


def _publish_unit(
    connection: sqlite3.Connection, root: Path, unit_id: int, terminal_stage: str
) -> int:
    unit_dir = _unit_dir(root, unit_id)
    source_index = _unit_source_index_by_image(unit_dir)
    pairs = _current_records(unit_dir / "image_text_pairs.jsonl")
    audits = _current_records(unit_dir / "bcc_validation_audit.jsonl")
    next_pair = int(connection.execute("SELECT COALESCE(MAX(pair_index), -1) + 1 FROM pairs").fetchone()[0])
    for image_id, pair in sorted(pairs.items(), key=lambda item: source_index.get(item[0], 0)):
        enriched = _artifact_metadata(pair, unit_dir)
        enriched.update(
            {
                "pair_index": next_pair,
                "pair_id": f"gpic-bcc-{next_pair:09d}",
                "source_manifest_index": source_index.get(image_id, 0),
                "campaign_unit": unit_id,
                "publisher_schema_version": PUBLISHER_SCHEMA_VERSION,
            }
        )
        cursor = connection.execute(
            "INSERT OR IGNORE INTO pairs(pair_index,image_id,source_manifest_index,unit_id,record_json) VALUES(?,?,?,?,?)",
            (
                next_pair,
                image_id,
                source_index.get(image_id, 0),
                unit_id,
                json.dumps(enriched, ensure_ascii=False, sort_keys=True),
            ),
        )
        if cursor.rowcount:
            next_pair += 1
    for image_id, audit in audits.items():
        stored_audit = (
            _artifact_metadata(audit, unit_dir)
            if audit.get("source_image_path")
            and audit.get("correspondence_overlay_path")
            else audit
        )
        connection.execute(
            "INSERT OR REPLACE INTO audits(image_id,source_manifest_index,unit_id,included,record_json) VALUES(?,?,?,?,?)",
            (
                image_id,
                source_index.get(image_id, 0),
                unit_id,
                1 if audit.get("included") else 0,
                json.dumps(stored_audit, ensure_ascii=False, sort_keys=True),
            ),
        )
    success = _success_path(unit_dir, terminal_stage)
    connection.execute(
        "INSERT INTO published_units(unit_id,published_at,success_sha256) VALUES(?,?,?)",
        (unit_id, time.time(), sha256_file(success)),
    )
    connection.commit()
    return len(pairs)


def _repair_legacy_source_indexes(
    connection: sqlite3.Connection, root: Path
) -> int:
    """Repair rows published while reviewed source indexes resolved to zero."""
    mappings: dict[int, dict[str, int]] = {}

    def expected(unit_id: int, image_id: str) -> int | None:
        if unit_id not in mappings:
            mappings[unit_id] = _unit_source_index_by_image(
                _unit_dir(root, unit_id)
            )
        return mappings[unit_id].get(image_id)

    repaired = 0
    pair_rows = connection.execute(
        "SELECT pair_index,image_id,source_manifest_index,unit_id,record_json FROM pairs"
    ).fetchall()
    for pair_index, image_id, stored_index, unit_id, record_json in pair_rows:
        recovered = expected(int(unit_id), str(image_id))
        if recovered is None or int(stored_index) == recovered:
            continue
        record = json.loads(record_json)
        record["source_manifest_index"] = recovered
        connection.execute(
            "UPDATE pairs SET source_manifest_index=?,record_json=? WHERE pair_index=?",
            (
                recovered,
                json.dumps(record, ensure_ascii=False, sort_keys=True),
                pair_index,
            ),
        )
        repaired += 1

    audit_rows = connection.execute(
        "SELECT image_id,source_manifest_index,unit_id,record_json FROM audits"
    ).fetchall()
    for image_id, stored_index, unit_id, record_json in audit_rows:
        recovered = expected(int(unit_id), str(image_id))
        if recovered is None or int(stored_index) == recovered:
            continue
        record = json.loads(record_json)
        record["source_manifest_index"] = recovered
        connection.execute(
            "UPDATE audits SET source_manifest_index=?,record_json=? WHERE image_id=?",
            (
                recovered,
                json.dumps(record, ensure_ascii=False, sort_keys=True),
                image_id,
            ),
        )
        repaired += 1
    if repaired:
        connection.commit()
    return repaired


def _rewrite_pair_log(connection: sqlite3.Connection, root: Path) -> None:
    records = [
        json.loads(row[0])
        for row in connection.execute(
            "SELECT record_json FROM pairs ORDER BY pair_index"
        )
    ]
    write_jsonl(
        records,
        campaign_paths(root)["published"] / "image_text_pairs.jsonl",
    )


def _repair_append_log(connection: sqlite3.Connection, root: Path) -> int:
    output = campaign_paths(root)["published"] / "image_text_pairs.jsonl"
    existing_indexes: set[int] = set()
    if output.exists():
        try:
            existing_rows = read_jsonl(output)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # SQLite committed first and remains authoritative. A preemption
            # during one append can therefore be repaired without guessing.
            records = [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT record_json FROM pairs ORDER BY pair_index"
                )
            ]
            write_jsonl(records, output)
            return len(records)
        for row in existing_rows:
            if row.get("pair_index") is not None:
                existing_indexes.add(int(row["pair_index"]))
    appended = 0
    for pair_index, record_json in connection.execute(
        "SELECT pair_index,record_json FROM pairs ORDER BY pair_index"
    ):
        if int(pair_index) in existing_indexes:
            continue
        append_jsonl(json.loads(record_json), output, durable=True)
        appended += 1
    return appended


def _write_compressed_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    import zstandard

    path.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstandard.ZstdCompressor(level=7)
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in records
    )
    path.write_bytes(compressor.compress(payload))


def _write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(
        [
            {
                "pair_index": int(row["pair_index"]),
                "pair_id": str(row["pair_id"]),
                "image_id": str(row["image_id"]),
                "source_manifest_index": int(row["source_manifest_index"]),
                "quality_tier": str(row.get("quality_tier") or ""),
                "record_json": json.dumps(row, ensure_ascii=False, sort_keys=True),
            }
            for row in records
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _materialize_artifact(
    destination: Path, artifact: dict[str, Any], *, force: bool = False
) -> Path:
    if not force and destination.is_file() and destination.stat().st_size:
        return destination
    payload = read_tar_member(artifact["archive_path"], artifact["member"])
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return destination


def _copy_pair_assets(
    pair: dict[str, Any], site: Path, *, force: bool = False
) -> dict[str, Any]:
    rendered = copy.deepcopy(pair)
    pair_dir = site / "assets" / f"{int(pair['pair_index']):09d}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    source_artifact = rendered["source_artifact"]
    source_suffix = Path(str(source_artifact["member"])).suffix or ".jpg"
    source_path = _materialize_artifact(
        pair_dir / f"source{source_suffix}", source_artifact, force=force
    )
    rendered["source_image_path"] = str(source_path)
    overlay = _materialize_artifact(
        pair_dir / "overlay.png", rendered["overlay_artifact"], force=force
    )
    rendered["correspondence_overlay_path"] = str(overlay)

    collections = [
        *(rendered.get("groups") or []),
        *(rendered.get("first_pass_groups") or []),
        *(rendered.get("omitted_masks") or []),
        *(rendered.get("first_pass_omitted_masks") or []),
    ]
    mask_artifacts: dict[str, dict[str, Any]] = {}
    inverse_artifacts: dict[str, dict[str, Any]] = {}
    for item in collections:
        mask_id = str(item.get("mask_id") or "")
        if item.get("mask_artifact"):
            mask_artifacts[mask_id] = item["mask_artifact"]
        if item.get("inverse_crop_artifact"):
            inverse_artifacts[mask_id] = item["inverse_crop_artifact"]
    sam3_archive = next(
        (
            str(artifact["archive_path"])
            for artifact in mask_artifacts.values()
            if artifact.get("archive_path")
        ),
        "",
    )

    for group_key in ("groups", "first_pass_groups"):
        for group in rendered.get(group_key) or []:
            mask_id = str(group["mask_id"])
            mask_artifact = mask_artifacts.get(mask_id) or {
                "archive_path": sam3_archive,
                "member": f"masks/{mask_id}.png",
            }
            inverse_artifact = inverse_artifacts.get(mask_id) or {
                "archive_path": sam3_archive,
                "member": f"inverse_crops/{mask_id}.png",
            }
            group["mask_artifact"] = mask_artifact
            group["inverse_crop_artifact"] = inverse_artifact
            mask_path = pair_dir / f"mask-{mask_id}.png"
            prior_diagnostic = pair_dir / f"omitted-{mask_id}.png"
            if not mask_path.exists() and prior_diagnostic.is_file():
                mask_path = prior_diagnostic
            else:
                _materialize_artifact(mask_path, mask_artifact, force=force)
            inverse_path = pair_dir / f"inverse-{mask_id}.png"
            prior_inverse = pair_dir / f"omitted-inverse-{mask_id}.png"
            if not inverse_path.exists() and prior_inverse.is_file():
                inverse_path = prior_inverse
            else:
                _materialize_artifact(inverse_path, inverse_artifact, force=force)
            group["mask_path"] = str(mask_path)
            group["inverse_crop_path"] = str(inverse_path)

    for omitted_key in ("omitted_masks", "first_pass_omitted_masks"):
        for item in rendered.get(omitted_key) or []:
            mask_id = str(item.get("mask_id") or "")
            mask_artifact = mask_artifacts.get(mask_id)
            if not mask_artifact:
                continue
            ordinary = pair_dir / f"mask-{mask_id}.png"
            diagnostic = (
                _materialize_artifact(
                    pair_dir / f"omitted-{mask_id}.png", mask_artifact, force=True
                )
                if force
                else ordinary
                if ordinary.is_file()
                else _materialize_artifact(
                    pair_dir / f"omitted-{mask_id}.png", mask_artifact
                )
            )
            item["diagnostic_mask_path"] = str(diagnostic)
            inverse_artifact = inverse_artifacts.get(mask_id)
            if inverse_artifact:
                ordinary_inverse = pair_dir / f"inverse-{mask_id}.png"
                inverse = (
                    _materialize_artifact(
                        pair_dir / f"omitted-inverse-{mask_id}.png",
                        inverse_artifact,
                        force=True,
                    )
                    if force
                    else ordinary_inverse
                    if ordinary_inverse.is_file()
                    else _materialize_artifact(
                        pair_dir / f"omitted-inverse-{mask_id}.png", inverse_artifact
                    )
                )
                item["inverse_crop_path"] = str(inverse)
    return rendered


def _site_index(
    site: Path,
    milestones: list[dict[str, Any]],
    *,
    preview_path: Path | None = None,
) -> None:
    preview = (
        f'<p><a href="{preview_path.relative_to(site).as_posix()}">Live 10-pair preview, including audit exclusions</a></p>'
        if preview_path is not None and preview_path.is_file()
        else ""
    )
    links = "\n".join(
        f'<li><a href="pages/pairs-{item["start"]:09d}-{item["end"]:09d}.html">Pairs {item["start"]}–{item["end"]}</a> · {(item["stats"].get("included_pair_shard") or {}).get("quality_tiers", {})}</li>'
        for item in milestones
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>BCC corpus milestones</title><style>body{{font:16px/1.5 system-ui;max-width:900px;margin:50px auto;padding:0 20px;background:#f5f1e8;color:#17211c}}a{{color:#174f3a}}li{{margin:.8rem 0}}</style></head><body><h1>BCC corpus milestones</h1><p>Real Qwen3.8-27B/SAM3 records, published in immutable 100-pair pages.</p>{preview}<ol>{links}</ol></body></html>"""
    (site / "index.html").write_text(document, encoding="utf-8")


def _ensure_preview_site(
    connection: sqlite3.Connection, root: Path, preview_pairs: int
) -> Path | None:
    """Render accepted examples without weakening ordered corpus publication.

    The durable pair stream remains a consecutive unit prefix.  For the live
    canary only, completed later unit checkpoints may fill the preview while an
    earlier unit is still running.  Records are sorted by source index and are
    never inserted into the publisher database out of order.
    """
    if preview_pairs <= 0:
        return None
    published_accepted = [
        json.loads(row[0])
        for row in connection.execute(
            "SELECT record_json FROM pairs ORDER BY pair_index LIMIT ?",
            (int(preview_pairs),),
        )
    ]
    accepted_by_image = {
        str(row.get("image_id") or ""): row for row in published_accepted
    }
    audit_by_image = {
        str(row.get("image_id") or ""): row
        for row in [
            json.loads(result[0])
            for result in connection.execute(
                "SELECT record_json FROM audits ORDER BY source_manifest_index"
            )
        ]
    }

    # A straggling low-numbered unit must not hide an already complete canary.
    # Read only units with a durable terminal-stage marker, and leave SQLite's
    # ordered publication state untouched.
    if len(accepted_by_image) < preview_pairs:
        registry = load_registry(root)
        terminal_stage = str(registry.get("terminal_stage") or "bcc-rewrite")
        for unit_id in range(int(registry.get("unit_count") or 0)):
            unit_dir = _unit_dir(root, unit_id)
            if not _success_path(unit_dir, terminal_stage).exists():
                continue
            source_index_by_image = _unit_source_index_by_image(unit_dir)
            for image_id, pair in _current_records(
                unit_dir / "image_text_pairs.jsonl"
            ).items():
                if image_id in accepted_by_image:
                    continue
                enriched = _artifact_metadata(pair, unit_dir)
                source_index = source_index_by_image.get(image_id, 0)
                enriched.update(
                    {
                        "pair_index": _preview_pair_index(
                            image_id, rejected=False
                        ),
                        "pair_id": (
                            f"live-preview-{source_index:09d}-"
                            f"{hashlib.sha256(image_id.encode('utf-8')).hexdigest()[:12]}"
                        ),
                        "source_manifest_index": source_index,
                        "campaign_unit": unit_id,
                        "publisher_schema_version": PUBLISHER_SCHEMA_VERSION,
                    }
                )
                accepted_by_image[image_id] = enriched
            for image_id, audit in _current_records(
                unit_dir / "bcc_validation_audit.jsonl"
            ).items():
                if image_id in audit_by_image:
                    continue
                stored = (
                    _artifact_metadata(audit, unit_dir)
                    if audit.get("source_image_path")
                    and audit.get("correspondence_overlay_path")
                    else audit
                )
                source_index = source_index_by_image.get(image_id, 0)
                stored.update(
                    {
                        "source_manifest_index": source_index,
                        "campaign_unit": unit_id,
                    }
                )
                audit_by_image[image_id] = stored
            if len(accepted_by_image) >= preview_pairs:
                break

    accepted = sorted(
        accepted_by_image.values(),
        key=lambda row: (
            int(row.get("source_manifest_index") or 0),
            str(row.get("image_id") or ""),
        ),
    )[:preview_pairs]
    if len(accepted) < preview_pairs:
        return None
    max_source = max(int(row["source_manifest_index"]) for row in accepted)
    rejected = sorted(
        (
            row
            for row in audit_by_image.values()
            if not row.get("included")
            and int(row.get("source_manifest_index") or 0) <= max_source
        ),
        key=lambda row: int(row.get("source_manifest_index") or 0),
    )[:preview_pairs]
    renderable_rejected: list[dict[str, Any]] = []
    for row in rejected:
        if not row.get("source_artifact") or not row.get("overlay_artifact"):
            continue
        copy = dict(row)
        image_id = str(copy.get("image_id") or "")
        copy["pair_index"] = _preview_pair_index(image_id, rejected=True)
        copy["pair_id"] = (
            f"audit-rejected-{int(copy.get('source_manifest_index') or 0):09d}-"
            f"{hashlib.sha256(image_id.encode('utf-8')).hexdigest()[:12]}"
        )
        renderable_rejected.append(copy)
    site = campaign_paths(root)["site"]
    page = site / "pages" / f"preview-first-{preview_pairs}.html"
    data = site / "data" / f"preview-first-{preview_pairs}.jsonl"
    metadata_path = site / "data" / f"preview-first-{preview_pairs}.metadata.json"
    preview_mode = (
        "published_prefix"
        if len(published_accepted) >= preview_pairs
        else "completed_checkpoints"
    )
    signature_payload = {
        "mode": preview_mode,
        "accepted": [str(row.get("image_id") or "") for row in accepted],
        "rejected": [
            str(row.get("image_id") or "") for row in renderable_rejected
        ],
        "renderer": SITE_RENDERER_VERSION,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    prior_signature = ""
    if metadata_path.is_file():
        try:
            prior_signature = str(
                json.loads(metadata_path.read_text(encoding="utf-8")).get(
                    "signature"
                )
                or ""
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            prior_signature = ""
    if not page.is_file() or prior_signature != signature:
        rendered = [
            _copy_pair_assets(pair, site, force=True)
            for pair in [*accepted, *renderable_rejected]
        ]
        write_jsonl(rendered, data)
        write_bcc_html_report(
            root,
            pairs_path=data,
            output_path=page,
            max_images=0,
            embed_images=False,
        )
        write_json(
            {
                **signature_payload,
                "signature": signature,
                "accepted_pair_count": len(accepted),
                "rendered_rejected_count": len(renderable_rejected),
                "updated_at": time.time(),
            },
            metadata_path,
        )
    return page


def _repair_milestone_views(
    connection: sqlite3.Connection, root: Path
) -> list[dict[str, Any]]:
    milestones = [
        json.loads(row[0])
        for row in connection.execute(
            "SELECT metadata_json FROM milestones ORDER BY milestone"
        )
    ]
    published = campaign_paths(root)["published"]
    write_jsonl(milestones, published / "pair_milestones.jsonl")
    site = campaign_paths(root)["site"]
    registry = load_registry(root)
    preview = _ensure_preview_site(
        connection, root, int(registry.get("preview_pairs") or 0)
    )
    _site_index(site, milestones, preview_path=preview)
    if milestones or preview is not None:
        latest = milestones[-1] if milestones else None
        preview_metadata: dict[str, Any] = {}
        if preview is not None:
            metadata_path = (
                site
                / "data"
                / f"{preview.stem}.metadata.json"
            )
            if metadata_path.is_file():
                preview_metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
        write_json(
            {
                "ready": True,
                "pair_count": int(
                    connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
                ),
                "latest_milestone": (
                    int(latest["milestone"]) if latest is not None else None
                ),
                "preview_page": str(preview) if preview is not None else None,
                "preview_pair_count": int(
                    preview_metadata.get("accepted_pair_count") or 0
                ),
                "preview_mode": preview_metadata.get("mode"),
                "site_renderer_version": SITE_RENDERER_VERSION,
                "site_index": str(site / "index.html"),
                "updated_at": time.time(),
            },
            site / "READY.json",
        )
    return milestones


def _milestone_records(
    connection: sqlite3.Connection, milestone: int
) -> list[dict[str, Any]]:
    start = milestone * 100
    end = start + 99
    records = [
        json.loads(row[0])
        for row in connection.execute(
            "SELECT record_json FROM pairs WHERE pair_index BETWEEN ? AND ? ORDER BY pair_index",
            (start, end),
        )
    ]
    if len(records) != 100:
        raise ValueError(
            f"Milestone {milestone} has {len(records)} records, expected 100"
        )
    return records


def _render_milestone_site(
    root: Path,
    milestone: int,
    records: list[dict[str, Any]],
    *,
    force_assets: bool = False,
) -> Path:
    start = milestone * 100
    end = start + 99
    stem = f"pairs-{start:09d}-{end:09d}"
    site = campaign_paths(root)["site"]
    rendered = [_copy_pair_assets(pair, site, force=force_assets) for pair in records]
    page_data = site / "data" / f"{stem}.jsonl"
    write_jsonl(rendered, page_data)
    page = site / "pages" / f"{stem}.html"
    write_bcc_html_report(
        root,
        pairs_path=page_data,
        output_path=page,
        max_images=0,
        embed_images=False,
    )
    return page


def _publish_milestone(
    connection: sqlite3.Connection, root: Path, milestone: int
) -> dict[str, Any]:
    start = milestone * 100
    end = start + 99
    records = _milestone_records(connection, milestone)
    published = campaign_paths(root)["published"]
    shards = published / "pair_shards"
    stem = f"pairs-{start:09d}-{end:09d}"
    jsonl_zst = shards / f"{stem}.jsonl.zst"
    parquet = shards / f"{stem}.parquet"
    _write_compressed_jsonl(records, jsonl_zst)
    _write_parquet(records, parquet)
    max_source_index = max(int(row["source_manifest_index"]) for row in records)
    all_audits = [
        json.loads(row[0])
        for row in connection.execute(
            "SELECT record_json FROM audits WHERE source_manifest_index <= ? ORDER BY source_manifest_index",
            (max_source_index,),
        )
    ]
    stats = {
        "included_pair_shard": aggregate_audits(records),
        "all_bcc_attempts_through_source_index": aggregate_audits(all_audits),
        "all_bcc_attempt_count": len(all_audits),
        "excluded_bcc_attempt_count": sum(not bool(row.get("included")) for row in all_audits),
    }
    metadata = {
        "milestone": milestone,
        "start": start,
        "end": end,
        "pair_count": 100,
        "jsonl_zst": str(jsonl_zst),
        "jsonl_zst_sha256": sha256_file(jsonl_zst),
        "parquet": str(parquet),
        "parquet_sha256": sha256_file(parquet),
        "stats": stats,
        "published_at": time.time(),
    }
    write_json(metadata, shards / f"{stem}.stats.json")
    page = _render_milestone_site(root, milestone, records, force_assets=True)
    metadata["site_page"] = str(page)
    metadata["site_renderer_version"] = SITE_RENDERER_VERSION
    connection.execute(
        "INSERT INTO milestones(milestone,pair_count,published_at,metadata_json) VALUES(?,?,?,?)",
        (milestone, 100, time.time(), json.dumps(metadata, sort_keys=True)),
    )
    connection.commit()
    return metadata


def _refresh_site_milestones(
    connection: sqlite3.Connection,
    root: Path,
    *,
    milestones: list[int] | None = None,
    force: bool = False,
) -> list[int]:
    requested = set(milestones) if milestones is not None else None
    rows = [
        (int(row[0]), json.loads(row[1]))
        for row in connection.execute(
            "SELECT milestone,metadata_json FROM milestones ORDER BY milestone"
        )
    ]
    available = {milestone for milestone, _ in rows}
    if requested is not None:
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"Unpublished site milestones requested: {unknown}")
    refreshed: list[int] = []
    for milestone, metadata in rows:
        if requested is not None and milestone not in requested:
            continue
        page = Path(str(metadata.get("site_page") or ""))
        stale = (
            metadata.get("site_renderer_version") != SITE_RENDERER_VERSION
            or not page.is_file()
        )
        if not force and not stale:
            continue
        rendered_page = _render_milestone_site(
            root, milestone, _milestone_records(connection, milestone)
        )
        metadata.update(
            {
                "site_page": str(rendered_page),
                "site_renderer_version": SITE_RENDERER_VERSION,
                "site_refreshed_at": time.time(),
            }
        )
        connection.execute(
            "UPDATE milestones SET metadata_json=? WHERE milestone=?",
            (json.dumps(metadata, sort_keys=True), milestone),
        )
        connection.commit()
        refreshed.append(milestone)
    return refreshed


def rebuild_site(
    root: str | Path, *, milestones: list[int] | None = None
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    connection = _connect(root)
    try:
        refreshed = _refresh_site_milestones(
            connection, root, milestones=milestones, force=True
        )
        _repair_milestone_views(connection, root)
        return {
            "site_renderer_version": SITE_RENDERER_VERSION,
            "refreshed_milestones": refreshed,
        }
    finally:
        connection.close()


def publish_once(root: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    registry = load_registry(root)
    terminal_stage = str(registry.get("terminal_stage") or "bcc-rewrite")
    connection = _connect(root)
    next_unit = int(
        connection.execute("SELECT COALESCE(MAX(unit_id), -1) + 1 FROM published_units").fetchone()[0]
    )
    published_units = 0
    while next_unit < int(registry.get("unit_count") or 0):
        unit_dir = _unit_dir(root, next_unit)
        if not _success_path(unit_dir, terminal_stage).exists():
            break
        _publish_unit(connection, root, next_unit, terminal_stage)
        published_units += 1
        next_unit += 1
    repaired_source_indexes = _repair_legacy_source_indexes(connection, root)
    if repaired_source_indexes:
        _rewrite_pair_log(connection, root)
        appended = 0
    else:
        appended = _repair_append_log(connection, root)
    pair_count = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
    cumulative_audits = [
        json.loads(row[0])
        for row in connection.execute(
            "SELECT record_json FROM audits ORDER BY source_manifest_index"
        )
    ]
    write_json(
        {
            "pair_count": pair_count,
            "published_unit_count": int(
                connection.execute("SELECT COUNT(*) FROM published_units").fetchone()[0]
            ),
            "all_bcc_attempts": aggregate_audits(cumulative_audits),
            "excluded_bcc_attempt_count": sum(
                not bool(row.get("included")) for row in cumulative_audits
            ),
            "updated_at": time.time(),
        },
        campaign_paths(root)["published"] / "composite_statistics.json",
    )
    refreshed_site_milestones = _refresh_site_milestones(connection, root)
    published_milestones = {
        int(row[0]) for row in connection.execute("SELECT milestone FROM milestones")
    }
    new_milestones: list[int] = []
    for milestone in range(pair_count // 100):
        if milestone in published_milestones:
            continue
        _publish_milestone(connection, root, milestone)
        new_milestones.append(milestone)
    _repair_milestone_views(connection, root)
    result = {
        "published_units": published_units,
        "jsonl_rows_appended": appended,
        "source_indexes_repaired": repaired_source_indexes,
        "pair_count": pair_count,
        "refreshed_site_milestones": refreshed_site_milestones,
        "new_milestones": new_milestones,
        "next_unit": next_unit,
        "unit_count": int(registry.get("unit_count") or 0),
    }
    write_json(result, campaign_paths(root)["published"] / "publisher_state.json")
    connection.close()
    return result


def publish_daemon(
    root: str | Path,
    *,
    poll_seconds: int = 30,
    stop_when_complete: bool = True,
) -> dict[str, Any]:
    while True:
        result = publish_once(root)
        if stop_when_complete and result["next_unit"] >= result["unit_count"]:
            return result
        time.sleep(max(1, int(poll_seconds)))
