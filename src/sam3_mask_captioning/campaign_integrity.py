from __future__ import annotations

import json
import os
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .campaign_claims import claim_path, stage_claim_lock
from .campaign_manifest import campaign_paths, load_registry
from .io_utils import read_jsonl, write_json
from .stage_merge import STAGE_OUTPUTS


PRIMARY_STAGE_ORDER = (
    "image-review",
    "sam3",
    "mask-caption",
    "mask-qa",
    "mask-caption-qa",
    "consistency",
    "bcc-draft",
    "bcc-rewrite",
    "bcc",
)

STAGE_EXTRA_OUTPUTS: dict[str, tuple[str, ...]] = {
    "sam3": ("mask_rle.jsonl", "artifacts/sam3.tar", "masks", "inverse_crops"),
    "bcc-draft": ("artifacts/bcc.tar", "correspondence_overlays"),
    "bcc": ("artifacts/bcc.tar", "correspondence_overlays"),
}


def _issue(code: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "detail": detail, **extra}


def _read_rows(path: Path, issues: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    if not path.exists():
        issues.append(_issue(f"missing_{label}", f"Missing {path.name}", path=str(path)))
        return []
    try:
        return read_jsonl(path)
    except Exception as exc:
        issues.append(
            _issue(
                f"unparseable_{label}",
                f"Could not parse {path.name}: {exc}",
                path=str(path),
            )
        )
        return []


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return sorted(duplicate)


def _tar_members(path: Path, issues: list[dict[str, Any]], label: str) -> list[str]:
    if not path.exists():
        issues.append(_issue(f"missing_{label}_archive", f"Missing {path.name}", path=str(path)))
        return []
    try:
        with tarfile.open(path, "r") as handle:
            members = [member.name for member in handle.getmembers() if member.isfile()]
    except Exception as exc:
        issues.append(
            _issue(
                f"unreadable_{label}_archive",
                f"Could not read {path.name}: {exc}",
                path=str(path),
            )
        )
        return []
    duplicates = _duplicates(members)
    if duplicates:
        issues.append(
            _issue(
                f"duplicate_{label}_archive_members",
                f"Archive has {len(duplicates)} duplicate members",
                examples=duplicates[:10],
            )
        )
    return members


def _set_mismatch(
    issues: list[dict[str, Any]],
    *,
    code: str,
    expected: set[str],
    actual: set[str],
) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        issues.append(
            _issue(
                code,
                f"Expected and actual member sets differ ({len(missing)} missing, {len(extra)} extra)",
                missing_count=len(missing),
                extra_count=len(extra),
                missing_examples=missing[:10],
                extra_examples=extra[:10],
            )
        )


def audit_sam3_unit(unit_dir: str | Path) -> dict[str, Any]:
    unit = Path(unit_dir).expanduser().resolve()
    issues: list[dict[str, Any]] = []
    rows = _read_rows(unit / "sam3_masks.jsonl", issues, "sam3_manifest")
    mask_ids = [str(row.get("mask_id") or "") for row in rows]
    empty_ids = sum(not value for value in mask_ids)
    if empty_ids:
        issues.append(_issue("empty_mask_ids", f"Manifest has {empty_ids} rows without mask IDs"))
    duplicate_ids = _duplicates(value for value in mask_ids if value)
    if duplicate_ids:
        issues.append(
            _issue(
                "duplicate_mask_ids",
                f"Manifest has {len(duplicate_ids)} duplicate mask IDs",
                examples=duplicate_ids[:10],
            )
        )
    unique_ids = {value for value in mask_ids if value}

    rle_rows = _read_rows(unit / "mask_rle.jsonl", issues, "mask_rle")
    rle_ids = [str(row.get("mask_id") or "") for row in rle_rows]
    duplicate_rle_ids = _duplicates(value for value in rle_ids if value)
    if duplicate_rle_ids:
        issues.append(
            _issue(
                "duplicate_rle_ids",
                f"RLE index has {len(duplicate_rle_ids)} duplicate mask IDs",
                examples=duplicate_rle_ids[:10],
            )
        )
    invalid_rle = [
        str(row.get("mask_id") or "")
        for row in rle_rows
        if not isinstance(row.get("rle"), dict)
        or not (row.get("rle") or {}).get("data")
        or not (row.get("rle") or {}).get("size")
    ]
    if invalid_rle:
        issues.append(
            _issue(
                "invalid_rle_rows",
                f"RLE index has {len(invalid_rle)} malformed rows",
                examples=invalid_rle[:10],
            )
        )
    _set_mismatch(
        issues,
        code="manifest_rle_mismatch",
        expected=unique_ids,
        actual={value for value in rle_ids if value},
    )

    members = _tar_members(unit / "artifacts" / "sam3.tar", issues, "sam3")
    actual_masks = {value for value in members if value.startswith("masks/")}
    actual_inverse = {value for value in members if value.startswith("inverse_crops/")}
    unexpected = sorted(
        value
        for value in members
        if not value.startswith("masks/") and not value.startswith("inverse_crops/")
    )
    if unexpected:
        issues.append(
            _issue(
                "unexpected_sam3_archive_members",
                f"SAM3 archive has {len(unexpected)} unexpected members",
                examples=unexpected[:10],
            )
        )
    expected_masks = {
        f"masks/{Path(str(row.get('mask_path') or '')).name}"
        for row in rows
        if row.get("mask_id")
    }
    expected_inverse = {
        f"inverse_crops/{Path(str(row.get('inverse_crop_path') or '')).name}"
        for row in rows
        if row.get("mask_id")
    }
    _set_mismatch(
        issues,
        code="manifest_mask_archive_mismatch",
        expected=expected_masks,
        actual=actual_masks,
    )
    _set_mismatch(
        issues,
        code="manifest_inverse_archive_mismatch",
        expected=expected_inverse,
        actual=actual_inverse,
    )
    mask_basenames = {Path(value).name for value in actual_masks}
    inverse_basenames = {Path(value).name for value in actual_inverse}
    _set_mismatch(
        issues,
        code="mask_inverse_pair_mismatch",
        expected=mask_basenames,
        actual=inverse_basenames,
    )
    return {
        "stage": "sam3",
        "valid": not issues,
        "manifest_rows": len(rows),
        "unique_mask_ids": len(unique_ids),
        "rle_rows": len(rle_rows),
        "archive_mask_members": len(actual_masks),
        "archive_inverse_members": len(actual_inverse),
        "issues": issues,
    }


def assert_sam3_integrity(unit_dir: str | Path) -> dict[str, Any]:
    report = audit_sam3_unit(unit_dir)
    if not report["valid"]:
        codes = ", ".join(str(item["code"]) for item in report["issues"][:8])
        raise RuntimeError(f"SAM3 artifact integrity failed: {codes}")
    return report


def audit_bcc_unit(unit_dir: str | Path) -> dict[str, Any]:
    unit = Path(unit_dir).expanduser().resolve()
    issues: list[dict[str, Any]] = []
    members = set(_tar_members(unit / "artifacts" / "bcc.tar", issues, "bcc"))
    referenced: set[str] = set()
    for filename in (
        "image_caption_candidates.jsonl",
        "bcc_validation_audit.jsonl",
        "image_text_pairs.jsonl",
    ):
        path = unit / filename
        if not path.exists():
            continue
        try:
            rows = read_jsonl(path)
        except Exception as exc:
            issues.append(_issue("unparseable_bcc_rows", f"Could not parse {filename}: {exc}"))
            continue
        for row in rows:
            value = row.get("correspondence_overlay_path")
            if value:
                referenced.add(f"correspondence_overlays/{Path(str(value)).name}")
    missing = sorted(referenced - members)
    if missing:
        issues.append(
            _issue(
                "missing_bcc_overlay_members",
                f"BCC archive is missing {len(missing)} referenced overlays",
                examples=missing[:10],
            )
        )
    return {
        "stage": "bcc",
        "valid": not issues,
        "referenced_overlay_count": len(referenced),
        "archive_member_count": len(members),
        "issues": issues,
    }


def assert_bcc_integrity(unit_dir: str | Path) -> dict[str, Any]:
    report = audit_bcc_unit(unit_dir)
    if not report["valid"]:
        codes = ", ".join(str(item["code"]) for item in report["issues"][:8])
        raise RuntimeError(f"BCC artifact integrity failed: {codes}")
    return report


def audit_campaign_integrity(
    campaign_root: str | Path,
    *,
    stages: Iterable[str] = ("sam3", "bcc"),
    unit_ids: Iterable[int] | None = None,
    report_path: str | Path | None = None,
    workers: int = 16,
) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    count = int(load_registry(root).get("unit_count") or 0)
    selected = sorted(set(int(value) for value in (unit_ids if unit_ids is not None else range(count))))
    requested = tuple(dict.fromkeys(stages))
    tasks: list[tuple[int, str, Path]] = []
    for unit_id in selected:
        if unit_id < 0 or unit_id >= count:
            raise ValueError(f"Unit ID outside campaign: {unit_id}")
        unit = campaign_paths(root)["units"] / f"{unit_id:06d}"
        for stage in requested:
            tasks.append((unit_id, stage, unit))

    def audit_task(task: tuple[int, str, Path]) -> dict[str, Any]:
        unit_id, stage, unit = task
        success = unit / "stages" / stage / "_SUCCESS.json"
        if not success.exists():
            return {"unit_id": unit_id, "stage": stage, "status": "incomplete"}
        if stage == "sam3":
            result = audit_sam3_unit(unit)
        elif stage == "bcc":
            result = audit_bcc_unit(unit)
        else:
            raise ValueError(f"Unsupported integrity stage: {stage}")
        return {
            "unit_id": unit_id,
            "status": "valid" if result["valid"] else "invalid",
            **result,
        }

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        records = list(executor.map(audit_task, tasks))
    violations = [row for row in records if row.get("status") == "invalid"]
    report = {
        "schema_version": 1,
        "campaign_root": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scanned_unit_count": len(selected),
        "stages": list(requested),
        "workers": max(1, int(workers)),
        "valid": not violations,
        "violation_count": len(violations),
        "violations": violations,
        "records": records,
    }
    if report_path is not None:
        write_json(report, report_path)
    return report


def _affected_stages(from_stage: str) -> tuple[str, ...]:
    if from_stage not in PRIMARY_STAGE_ORDER:
        raise ValueError(f"Unsupported repair stage: {from_stage}")
    index = PRIMARY_STAGE_ORDER.index(from_stage)
    if from_stage == "sam3":
        return tuple(stage for stage in PRIMARY_STAGE_ORDER[index:] if stage != "image-review")
    return PRIMARY_STAGE_ORDER[index:]


def _owned_paths(unit: Path, stage: str) -> list[Path]:
    # ``mask_rle.jsonl`` is repeated in the BCC merge schema because BCC
    # publishes it, but the durable file is produced and owned by SAM3. A
    # downstream rewind must never remove that upstream input.
    values = [
        unit / value
        for value in STAGE_OUTPUTS.get(stage, ())
        if value != "mask_rle.jsonl" or stage == "sam3"
    ]
    values.extend(unit / value for value in STAGE_EXTRA_OUTPUTS.get(stage, ()))
    values.append(unit / "stages" / stage)
    return values


def _move_to_backup(path: Path, root: Path, backup: Path, moved: list[str]) -> None:
    if not os.path.lexists(path):
        return
    relative = path.relative_to(root)
    destination = backup / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.move(str(path), str(destination))
    moved.append(relative.as_posix())


def repair_campaign_units(
    campaign_root: str | Path,
    *,
    unit_ids: Iterable[int],
    from_stage: str,
    apply: bool = False,
    backup_root: str | Path | None = None,
    reason: str = "integrity_repair",
) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    count = int(load_registry(root).get("unit_count") or 0)
    selected = sorted(set(int(value) for value in unit_ids))
    if not selected:
        raise ValueError("At least one unit ID is required")
    if any(value < 0 or value >= count for value in selected):
        raise ValueError(f"Repair unit IDs must be between 0 and {count - 1}")
    affected = _affected_stages(from_stage)
    with ExitStack() as locks:
        for stage in affected:
            locks.enter_context(stage_claim_lock(root, stage))
        for stage in affected:
            for unit_id in selected:
                path = claim_path(root, stage, unit_id)
                if path.exists():
                    raise RuntimeError(
                        f"Refusing repair: {stage} unit {unit_id:06d} has an active/unresolved claim"
                    )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = (
            Path(backup_root).expanduser().resolve()
            if backup_root is not None
            else root / "repairs" / timestamp
        )
        planned: list[str] = []
        for unit_id in selected:
            unit = campaign_paths(root)["units"] / f"{unit_id:06d}"
            for stage in affected:
                for path in _owned_paths(unit, stage):
                    if os.path.lexists(path):
                        value = path.relative_to(root).as_posix()
                        if value not in planned:
                            planned.append(value)
        for stage in affected:
            for path in (root / "stages" / stage / "_MERGED.json", root / "merged" / stage):
                if os.path.lexists(path):
                    value = path.relative_to(root).as_posix()
                    if value not in planned:
                        planned.append(value)

        result: dict[str, Any] = {
            "schema_version": 1,
            "campaign_root": str(root),
            "unit_ids": selected,
            "from_stage": from_stage,
            "affected_stages": list(affected),
            "reason": reason,
            "apply": apply,
            "backup_root": str(backup),
            "planned_paths": planned,
            "moved_paths": [],
        }
        if not apply:
            return result

        backup.mkdir(parents=True, exist_ok=False)
        moved: list[str] = []
        for unit_id in selected:
            unit = campaign_paths(root)["units"] / f"{unit_id:06d}"
            seen: set[Path] = set()
            for stage in affected:
                for path in _owned_paths(unit, stage):
                    if path in seen:
                        continue
                    seen.add(path)
                    _move_to_backup(path, root, backup, moved)
        for stage in affected:
            _move_to_backup(root / "stages" / stage / "_MERGED.json", root, backup, moved)
            _move_to_backup(root / "merged" / stage, root, backup, moved)
        result["moved_paths"] = moved
        result["applied_at"] = datetime.now(timezone.utc).isoformat()
        write_json(result, backup / "repair.json")
        return result
