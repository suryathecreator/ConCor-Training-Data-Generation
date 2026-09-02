from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifact_store import pack_artifacts
from .bcc_contract import ONE_REWRITE_CONTRACT_VERSION
from .bcc_runtime_limits import (
    BCC_INPUT_TOO_LARGE_REASON,
    bcc_input_limit_error,
)
from .campaign_claims import claim_path, stage_claim_lock
from .campaign_integrity import assert_bcc_integrity
from .campaign_manifest import load_registry
from .io_utils import append_jsonl, read_jsonl, sha256_file, write_json


def _rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _ids(path: Path) -> set[str]:
    return {
        str(row.get("image_id") or "")
        for row in _rows(path)
        if row.get("image_id")
    }


def finalize_quarantined_bcc_input_limits(
    campaign_root: str | Path,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Finalize BCC units quarantined by known per-prompt size limits.

    This is intentionally narrow. A unit is eligible only when its quarantine
    error matches the same image/context-limit classifier used during normal
    inference. Existing completed audits and exclusions are preserved. Because
    old vLLM batches did not identify which neighbor triggered validation, all
    still-unfinished BCC-eligible images in that quarantined unit receive an
    explicit interrupted-tail exclusion; successful rows are never removed.
    """
    root = Path(campaign_root).expanduser().resolve()
    unit_count = int(load_registry(root).get("unit_count") or 0)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = root / "repairs" / f"bcc-input-limit-skip-{timestamp}"
    plans: list[dict[str, Any]] = []

    with stage_claim_lock(root, "bcc"):
        for unit_id in range(unit_count):
            unit = root / "units" / f"{unit_id:06d}"
            success = unit / "stages" / "bcc" / "_SUCCESS.json"
            quarantine = unit / "stages" / "bcc" / "_QUARANTINED.json"
            if success.exists() or not quarantine.exists():
                continue
            active_claim = claim_path(root, "bcc", unit_id)
            if active_claim.exists():
                raise RuntimeError(
                    f"Refusing to finalize unit {unit_id:06d} with active BCC claim"
                )
            payload = json.loads(quarantine.read_text(encoding="utf-8"))
            error = str(payload.get("error") or "")
            if bcc_input_limit_error(ValueError(error)) is None:
                continue

            counts = Counter(
                str(row.get("image_id") or "")
                for row in _rows(unit / "bcc_canonical_captions.jsonl")
                if row.get("image_id")
            )
            minimum = 10
            eligible = {image_id for image_id, count in counts.items() if count >= minimum}
            terminal = _ids(unit / "bcc_validation_audit.jsonl") | _ids(
                unit / "bcc_exclusions.jsonl"
            )
            unfinished = sorted(eligible - terminal)
            plan = {
                "unit_id": unit_id,
                "eligible_images": len(eligible),
                "existing_terminal_images": len(eligible & terminal),
                "skipped_unfinished_images": len(unfinished),
                "unfinished_image_ids": unfinished,
                "trigger_error": error,
            }
            plans.append(plan)
            if not apply:
                continue

            artifact = pack_artifacts(
                unit,
                ["correspondence_overlays"],
                unit / "artifacts" / "bcc.tar",
            )
            integrity = assert_bcc_integrity(unit)
            for image_id in unfinished:
                append_jsonl(
                    {
                        "image_id": image_id,
                        "stage": "bcc_quarantine_recovery",
                        "reason_code": "bcc_input_too_large_or_interrupted_batch_tail",
                        "trigger_reason_code": BCC_INPUT_TOO_LARGE_REASON,
                        "included": False,
                        "mask_count": int(counts[image_id]),
                        "diagnostic": error,
                        "contract_version": ONE_REWRITE_CONTRACT_VERSION,
                        "recovered_at": datetime.now(timezone.utc).isoformat(),
                    },
                    unit / "bcc_exclusions.jsonl",
                    durable=True,
                )

            write_json(
                {
                    "stage": "bcc",
                    "unit_id": unit_id,
                    "worker_id": "operator:bcc-input-limit-skip",
                    "claim_token": "operator-recovery",
                    "claim_generation": -1,
                    "elapsed_seconds": 0.0,
                    "completed_at": time.time(),
                    "details": {
                        "recovery": "skip_legacy_oversized_bcc_batch_tail",
                        "skipped_unfinished_images": len(unfinished),
                        "artifact": artifact,
                        "integrity": integrity,
                    },
                    "input_manifest_sha256": sha256_file(
                        unit / "selected_images.jsonl"
                    ),
                },
                success,
            )
            backup = backup_root / "units" / f"{unit_id:06d}"
            backup.mkdir(parents=True, exist_ok=True)
            for name in ("_QUARANTINED.json", "attempt-state.json"):
                path = unit / "stages" / "bcc" / name
                if path.exists():
                    shutil.copy2(path, backup / name)
                    path.unlink()

    report = {
        "schema_version": 1,
        "campaign_root": str(root),
        "applied": apply,
        "matched_quarantined_units": len(plans),
        "skipped_unfinished_images": sum(
            int(plan["skipped_unfinished_images"]) for plan in plans
        ),
        "backup_root": str(backup_root) if apply and plans else "",
        "units": plans,
    }
    if apply:
        write_json(report, backup_root / "report.json")
    return report
