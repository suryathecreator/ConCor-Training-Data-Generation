from __future__ import annotations

import copy
import base64
import json
import os
import shutil
import time
import traceback
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .artifact_store import hydrate_archive, pack_artifacts, remove_hydrated_files
from .bcc_canonicalization import canonicalize_bcc_rows
from .campaign_claims import (
    ClaimHandle,
    ClaimHeartbeat,
    ClaimOwnershipLost,
    commit_claim_json,
    finish_claim,
    run_if_claim_owned,
    try_claim,
)
from .campaign_integrity import assert_bcc_integrity, assert_sam3_integrity
from .campaign_manifest import campaign_paths, load_registry
from .caption_stage import create_captioner, run_captioning, run_mask_review
from .consistency_stage import run_sam3_consistency
from .image_review_stage import run_image_review
from .io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from .one_rewrite_stage import run_bcc_draft_batch, run_bcc_one_rewrite_batch
from .sam3_stage import _load_processor, run_sam3
from .visual_audit_stage import run_bcc_visual_audit_batch


STAGES = (
    "image-review",
    "sam3",
    "mask-caption-qa",
    "bcc",
    "mask-caption",
    "mask-qa",
    "consistency",
    "bcc-draft",
    "bcc-rewrite",
)
PREREQUISITE = {
    "image-review": None,
    "sam3": "image-review",
    "mask-caption-qa": "sam3",
    "bcc": "consistency",
    "mask-caption": "sam3",
    "mask-qa": "mask-caption",
    # New campaigns share one Qwen engine across mask captioning and QA;
    # imported/legacy campaigns may still carry the two-stage checkpoint.
    "consistency": ("mask-caption-qa", "mask-qa"),
    "bcc-draft": "consistency",
    "bcc-rewrite": "bcc-draft",
}

QWEN_STAGES = frozenset(
    {
        "image-review",
        "mask-caption",
        "mask-qa",
        "mask-caption-qa",
        "bcc-draft",
        "bcc-rewrite",
        "bcc",
    }
)
SAM3_STAGES = frozenset({"sam3", "consistency"})


class SharedStageRuntimeError(RuntimeError):
    """A worker-wide model/runtime initialization failure, not bad unit data."""


def _stage_dir(unit_dir: Path, stage: str) -> Path:
    return unit_dir / "stages" / stage


def _success_path(unit_dir: Path, stage: str) -> Path:
    return _stage_dir(unit_dir, stage) / "_SUCCESS.json"


def _unit_ids(campaign_root: Path) -> list[int]:
    count = int(load_registry(campaign_root).get("unit_count") or 0)
    return list(range(count))


def _unit_dir(campaign_root: Path, unit_id: int) -> Path:
    return campaign_paths(campaign_root)["units"] / f"{unit_id:06d}"


def _prerequisite_satisfied(unit_dir: Path, stage: str) -> bool:
    prerequisite = PREREQUISITE[stage]
    if prerequisite is None:
        return True
    alternatives = (prerequisite,) if isinstance(prerequisite, str) else prerequisite
    return any(_success_path(unit_dir, candidate).exists() for candidate in alternatives)


def _unit_config(config: dict[str, Any], unit_dir: Path) -> dict[str, Any]:
    out = copy.deepcopy(config)
    out["project_root"] = str(Path(config.get("project_root", ".")).resolve())
    out["resume"] = True
    out.setdefault("dataset", {})
    out["dataset"].update(
        {
            "manifest_path": str(unit_dir / "selected_images.jsonl"),
            "image_root": None,
            "start_index": 0,
            "limit": None,
        }
    )
    return out


def _hydrate_for_stage(
    unit_dir: Path,
    stage: str,
    *,
    hydration_root: Path | None = None,
) -> list[str]:
    """Expose packed stage inputs, preferably from node-local scratch.

    JSONL records retain their durable absolute paths below ``unit_dir``.
    Directory symlinks make those paths resolve to node-local files while
    avoiding hundreds of loose-file writes to GPFS. SAM3's own mask output
    directories remain durable because that stage mutates and repacks them.
    """
    hydrated: list[str] = []
    needs_source = stage in {
        "image-review",
        "sam3",
        "consistency",
        "bcc-draft",
        "bcc-rewrite",
        "bcc",
    }
    # A partially imported/resumed SAM3 unit already has packed masks. Restore
    # them before adding unfinished images so the replacement archive and RLE
    # index retain both old and new work.
    needs_sam3 = stage in {
        "mask-caption",
        "mask-caption-qa",
        "mask-qa",
        "consistency",
        "bcc-draft",
        "bcc-rewrite",
        "bcc",
    } or (stage == "sam3" and (unit_dir / "artifacts" / "sam3.tar").exists())
    plans: list[tuple[Path, tuple[str, ...], bool]] = []
    if needs_source:
        plans.append(
            (unit_dir / "artifacts" / "source.tar", ("source_images",), True)
        )
    if needs_sam3:
        plans.append(
            (
                unit_dir / "artifacts" / "sam3.tar",
                ("masks", "inverse_crops"),
                stage != "sam3",
            )
        )
    if stage == "bcc-rewrite":
        plans.append(
            (
                unit_dir / "artifacts" / "bcc.tar",
                ("correspondence_overlays",),
                True,
            )
        )

    existed_before: dict[str, bool] = {}
    try:
        for archive, names, may_use_node_local in plans:
            for name in names:
                managed_entry = unit_dir / name
                # Atomic unit ownership guarantees that no other live worker
                # can own this hydration link. A SIGKILL/preemption can bypass
                # ``finally`` and leave it pointing at another node's /tmp;
                # unlink that stale indirection before resolving tar members.
                # Preserve real legacy directories exactly as before.
                if managed_entry.is_symlink():
                    managed_entry.unlink()
                existed_before.setdefault(name, os.path.lexists(unit_dir / name))
            use_node_local = bool(hydration_root is not None and may_use_node_local)
            # Preserve compatibility with a legacy/preempted unit that already
            # contains real hydrated directories. New workers clean their own
            # hydrated inputs in ``finally``.
            if use_node_local and any(
                os.path.lexists(unit_dir / name) for name in names
            ):
                use_node_local = False
            destination = Path(hydration_root) if use_node_local else unit_dir
            destination.mkdir(parents=True, exist_ok=True)
            hydrate_archive(archive, destination)
            # A valid SAM3 unit can contain no proposals. ``pack_artifacts``
            # deliberately stores files rather than empty directory entries,
            # so its tar is empty in that case. Materialize the two expected
            # directories without treating a genuine zero-mask unit as archive
            # corruption; nonempty manifests retain the strict validation.
            sam3_manifest = unit_dir / "sam3_masks.jsonl"
            empty_sam3_artifact = (
                archive.name == "sam3.tar"
                and sam3_manifest.is_file()
                and sam3_manifest.stat().st_size == 0
            )
            if empty_sam3_artifact:
                for name in names:
                    (destination / name).mkdir(parents=True, exist_ok=True)
            if use_node_local:
                for name in names:
                    local_target = destination / name
                    if not local_target.is_dir():
                        raise FileNotFoundError(
                            f"Archive {archive} did not produce expected directory {name}"
                        )
                    os.symlink(local_target, unit_dir / name, target_is_directory=True)
            hydrated.extend(names)
        return hydrated
    except Exception:
        # Extraction can fail before the caller receives ``hydrated``. Remove
        # only top-level inputs created by this operation, never durable inputs
        # that existed when hydration began.
        cleanup_names = [
            name
            for name, existed in existed_before.items()
            if not existed and os.path.lexists(unit_dir / name)
        ]
        remove_hydrated_files(unit_dir, cleanup_names)
        if hydration_root is not None:
            shutil.rmtree(Path(hydration_root), ignore_errors=True)
        raise


def _cleanup_stage_hydration(
    unit_dir: Path,
    hydrated: list[str],
    hydration_root: Path | None,
) -> None:
    if hydrated:
        remove_hydrated_files(unit_dir, hydrated)
    if hydration_root is not None:
        shutil.rmtree(hydration_root, ignore_errors=True)


def _require_stage_emission(
    unit_dir: Path,
    *,
    stage: str,
    input_path: str,
    emitted_paths: tuple[str, ...],
    error_path: str,
) -> dict[str, int]:
    """Reject a systemic all-row failure instead of checkpointing it as success.

    Per-row failures remain durable diagnostics and mixed batches may continue.
    An input-bearing stage that emits no semantic decision at all, however, is
    almost always missing artifacts, an unavailable model, or a broken schema.
    Such a unit must be retried rather than silently advancing the barrier.
    """
    input_count = (
        len(read_jsonl(unit_dir / input_path))
        if (unit_dir / input_path).exists()
        else 0
    )
    emitted_count = sum(
        len(read_jsonl(unit_dir / name))
        for name in emitted_paths
        if (unit_dir / name).exists()
    )
    error_count = (
        len(read_jsonl(unit_dir / error_path))
        if (unit_dir / error_path).exists()
        else 0
    )
    if input_count > 0 and emitted_count == 0:
        raise RuntimeError(
            f"{stage} emitted 0 semantic decisions for {input_count} inputs "
            f"({error_count} recorded errors); refusing to checkpoint the unit"
        )
    return {
        "input_count": input_count,
        "emitted_count": emitted_count,
        "error_count": error_count,
    }


def _fsync_unit_outputs(unit_dir: Path) -> None:
    """Commit durable metadata/archives before a claim can be released."""
    candidates = [
        path
        for path in unit_dir.iterdir()
        if path.is_file() and path.suffix in {".json", ".jsonl", ".tar"}
    ]
    artifacts = unit_dir / "artifacts"
    if artifacts.exists():
        candidates.extend(path for path in artifacts.iterdir() if path.is_file())
    for path in candidates:
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except FileNotFoundError:
            continue
    for directory in (unit_dir, artifacts, unit_dir / "stages"):
        if not directory.exists():
            continue
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _mask_rle(path: str | Path) -> dict[str, Any]:
    mask = np.asarray(Image.open(path).convert("L")) > 0
    flat = mask.flatten(order="F").astype(np.uint8)
    packed = np.packbits(flat, bitorder="little").tobytes()
    compressed = zlib.compress(packed, level=7)
    return {
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "pixel_count": int(flat.size),
        "order": "F",
        "bitorder": "little",
        "encoding": "zlib-packbits-base64-v1",
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def _decode_mask_index(encoded: dict[str, Any]) -> np.ndarray:
    if encoded.get("encoding") != "zlib-packbits-base64-v1":
        raise ValueError(f"Unsupported mask index encoding: {encoded.get('encoding')}")
    packed = zlib.decompress(base64.b64decode(str(encoded["data"])))
    flat = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8),
        bitorder=str(encoded.get("bitorder") or "little"),
    )[: int(encoded["pixel_count"])]
    return flat.astype(bool).reshape(
        [int(value) for value in encoded["size"]],
        order=str(encoded.get("order") or "F"),
    )


def _add_rle(run_dir: Path) -> None:
    path = run_dir / "sam3_masks.jsonl"
    if not path.exists():
        return
    rows = read_jsonl(path)
    rle_rows: list[dict[str, Any]] = []
    changed = False
    for row in rows:
        if row.get("mask_path"):
            rle_rows.append(
                {
                    "image_id": row.get("image_id"),
                    "mask_id": row.get("mask_id"),
                    "rle": _mask_rle(row["mask_path"]),
                }
            )
        if not row.get("mask_rle_ref") and row.get("mask_path"):
            row["mask_rle_ref"] = "mask_rle.jsonl"
            row["mask_encoding"] = "zlib-packbits-base64-v1"
            changed = True
    write_jsonl(rle_rows, run_dir / "mask_rle.jsonl")
    if changed:
        write_jsonl(rows, path)


def _group_by_image(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        image_id = str(row.get("image_id") or "")
        if image_id not in grouped:
            order.append(image_id)
        grouped[image_id].append(row)
    return [grouped[image_id] for image_id in order]


def _bypass_mask_qa(unit_dir: Path) -> dict[str, int]:
    source = unit_dir / "caption_candidates.jsonl"
    rows = read_jsonl(source) if source.exists() else []
    forwarded = []
    reviews = []
    for row in rows:
        out = dict(row)
        out.update(
            {
                "mask_review_keep": True,
                "mask_review_reason": "quality_filter_disabled_for_throughput",
                "mask_review_failure_modes": [],
                "qa_status": "skipped",
            }
        )
        forwarded.append(out)
        reviews.append(
            {
                "image_id": row.get("image_id"),
                "mask_id": row.get("mask_id"),
                "keep": True,
                "qa_status": "skipped",
                "reason": "quality_filter_disabled_for_throughput",
            }
        )
    write_jsonl(forwarded, unit_dir / "captions.jsonl")
    write_jsonl(reviews, unit_dir / "mask_quality_reviews.jsonl")
    return {"forwarded": len(forwarded)}


def _canonicalize(unit_dir: Path, config: dict[str, Any]) -> list[list[dict[str, Any]]]:
    source = unit_dir / "consistent_captions.jsonl"
    groups = _group_by_image(read_jsonl(source) if source.exists() else [])
    canonical: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    caption_config = config.get("image_caption", {})
    minimum = int(
        caption_config.get("min_input_masks", caption_config.get("min_groups", 10))
    )
    mask_indexes = {
        str(item.get("mask_id") or ""): item["rle"]
        for item in (
            read_jsonl(unit_dir / "mask_rle.jsonl")
            if (unit_dir / "mask_rle.jsonl").exists()
            else []
        )
        if item.get("mask_id") and item.get("rle")
    }
    for rows in groups:
        # Canonicalization only removes/merges proposals. An image already
        # below the BCC minimum cannot become eligible, so preserve its rows
        # for the explicit draft exclusion without decoding every mask.
        if len(rows) < minimum:
            canonical.extend(rows)
            continue
        per_image_arrays = {
            str(row.get("mask_id") or ""): _decode_mask_index(
                mask_indexes[str(row.get("mask_id") or "")]
            )
            for row in rows
            if str(row.get("mask_id") or "") in mask_indexes
        }
        kept, dropped = canonicalize_bcc_rows(
            rows,
            config.get("image_caption", {}),
            mask_arrays=per_image_arrays,
        )
        canonical.extend(kept)
        duplicates.extend(dropped)
    write_jsonl(canonical, unit_dir / "bcc_canonical_captions.jsonl")
    write_jsonl(duplicates, unit_dir / "bcc_duplicate_masks.jsonl")
    return _group_by_image(canonical)


def _run_stage(
    stage: str,
    config: dict[str, Any],
    unit_dir: Path,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    def resource(key: str, factory: Any) -> Any:
        if key not in runtime:
            try:
                runtime[key] = factory()
            except Exception as exc:
                # Loading a shared model/runtime is allocation-wide setup. If
                # it fails (for example, a missing Python dependency), marking
                # the claimed unit as bad eventually quarantines healthy data.
                raise SharedStageRuntimeError(
                    f"Could not initialize shared stage resource {key!r}"
                ) from exc
        return runtime[key]

    if stage == "image-review":
        captioner = resource("captioner", lambda: create_captioner(config, "image_review"))
        path = run_image_review(config, unit_dir, captioner_override=captioner)
        return {"row_count": len(read_jsonl(path)) if path.exists() else 0}
    if stage == "sam3":
        processor = resource("sam3_processor", lambda: _load_processor(config))
        path = run_sam3(config, unit_dir, processor_override=processor)
        _add_rle(unit_dir)
        artifact = pack_artifacts(unit_dir, ["masks", "inverse_crops"], unit_dir / "artifacts" / "sam3.tar")
        integrity = assert_sam3_integrity(unit_dir)
        return {
            "row_count": len(read_jsonl(path)) if path.exists() else 0,
            "artifact": artifact,
            "integrity": integrity,
        }
    if stage == "mask-caption":
        mask_rows = (
            read_jsonl(unit_dir / "sam3_masks.jsonl")
            if (unit_dir / "sam3_masks.jsonl").exists()
            else []
        )
        captioner = (
            resource("captioner", lambda: create_captioner(config, "caption"))
            if mask_rows
            else None
        )
        path = run_captioning(config, unit_dir, captioner_override=captioner)
        return {"row_count": len(read_jsonl(path)) if path.exists() else 0}
    if stage == "mask-caption-qa":
        # Both calls use the same Qwen3.8 checkpoint. Keeping one engine alive
        # removes an otherwise repeated 55.6-GB load/compile per worker.
        mask_rows = (
            read_jsonl(unit_dir / "sam3_masks.jsonl")
            if (unit_dir / "sam3_masks.jsonl").exists()
            else []
        )
        captioner = (
            resource("captioner", lambda: create_captioner(config, "caption"))
            if mask_rows
            else None
        )
        run_captioning(config, unit_dir, captioner_override=captioner)
        caption_emission = _require_stage_emission(
            unit_dir,
            stage="mask captioning",
            input_path="sam3_masks.jsonl",
            emitted_paths=("caption_candidates.jsonl", "caption_rejected_masks.jsonl"),
            error_path="caption_errors.jsonl",
        )
        if not bool(config.get("quality_filter", {}).get("enabled", True)):
            details = _bypass_mask_qa(unit_dir)
        else:
            path = run_mask_review(config, unit_dir, captioner_override=captioner)
            details = {"forwarded": len(read_jsonl(path)) if path.exists() else 0}
            details["mask_qa_emission"] = _require_stage_emission(
                unit_dir,
                stage="mask description QA",
                input_path="caption_candidates.jsonl",
                emitted_paths=("mask_quality_reviews.jsonl",),
                error_path="mask_review_errors.jsonl",
            )
        details["caption_emission"] = caption_emission
        details["shared_qwen_engine"] = True
        return details
    if stage == "mask-qa":
        if not bool(config.get("quality_filter", {}).get("enabled", True)):
            return _bypass_mask_qa(unit_dir)
        candidates = (
            read_jsonl(unit_dir / "caption_candidates.jsonl")
            if (unit_dir / "caption_candidates.jsonl").exists()
            else []
        )
        captioner = (
            resource(
                "captioner", lambda: create_captioner(config, "quality_filter")
            )
            if candidates
            else None
        )
        path = run_mask_review(config, unit_dir, captioner_override=captioner)
        return {"row_count": len(read_jsonl(path)) if path.exists() else 0}
    if stage == "consistency":
        rows = read_jsonl(unit_dir / "captions.jsonl") if (unit_dir / "captions.jsonl").exists() else []
        processor = (
            resource("sam3_processor", lambda: _load_processor(config))
            if rows
            else None
        )
        if not rows:
            run_sam3_consistency(config, unit_dir, rows=[], processor=None)
        for image_rows in _group_by_image(rows):
            run_sam3_consistency(config, unit_dir, rows=image_rows, processor=processor)
        passed = unit_dir / "consistent_captions.jsonl"
        return {"row_count": len(read_jsonl(passed)) if passed.exists() else 0}
    if stage == "bcc-draft":
        captioner = resource("captioner", lambda: create_captioner(config, "image_caption"))
        groups = _canonicalize(unit_dir, config)
        path = run_bcc_draft_batch(config, unit_dir, groups, captioner=captioner)
        artifact = pack_artifacts(
            unit_dir,
            ["correspondence_overlays"],
            unit_dir / "artifacts" / "bcc.tar",
        )
        integrity = assert_bcc_integrity(unit_dir)
        return {
            "row_count": len(read_jsonl(path)) if path.exists() else 0,
            "artifact": artifact,
            "integrity": integrity,
        }
    if stage == "bcc-rewrite":
        captioner = resource("captioner", lambda: create_captioner(config, "image_caption_qa"))
        canonical = unit_dir / "bcc_canonical_captions.jsonl"
        groups = _group_by_image(read_jsonl(canonical) if canonical.exists() else [])
        path = run_bcc_one_rewrite_batch(config, unit_dir, groups, captioner=captioner)
        return {"row_count": len(read_jsonl(path)) if path.exists() else 0}
    if stage == "bcc":
        # Draft, visual audit, and exactly one rewrite share one long-lived
        # multimodal engine. Raw responses are fsynced by each substage, so a
        # requeue re-finalizes them without paying for duplicate model calls.
        groups = _canonicalize(unit_dir, config)
        caption_config = config.get("image_caption", {})
        minimum = int(
            caption_config.get(
                "min_input_masks", caption_config.get("min_groups", 10)
            )
        )
        captioner = (
            resource(
                "captioner", lambda: create_captioner(config, "image_caption")
            )
            if any(len(group) >= minimum for group in groups)
            else None
        )
        run_bcc_draft_batch(config, unit_dir, groups, captioner=captioner)
        run_bcc_visual_audit_batch(config, unit_dir, groups, captioner=captioner)
        path = run_bcc_one_rewrite_batch(config, unit_dir, groups, captioner=captioner)
        artifact = pack_artifacts(
            unit_dir,
            ["correspondence_overlays"],
            unit_dir / "artifacts" / "bcc.tar",
        )
        integrity = assert_bcc_integrity(unit_dir)
        audits = unit_dir / "bcc_validation_audit.jsonl"
        return {
            "row_count": len(read_jsonl(path)) if path.exists() else 0,
            "audit_count": len(read_jsonl(audits)) if audits.exists() else 0,
            "shared_qwen_engine": True,
            "artifact": artifact,
            "integrity": integrity,
        }
    raise ValueError(f"Unknown campaign stage: {stage}")


def _attempt_state_path(unit_dir: Path, stage: str) -> Path:
    return _stage_dir(unit_dir, stage) / "attempt-state.json"


def _quarantine_path(unit_dir: Path, stage: str) -> Path:
    return _stage_dir(unit_dir, stage) / "_QUARANTINED.json"


def _persistent_attempt_count(unit_dir: Path, stage: str) -> int:
    path = _attempt_state_path(unit_dir, stage)
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("attempt_count") or 0)
    except Exception:
        return 0


def _record_stage_failure(
    unit_dir: Path,
    stage: str,
    *,
    worker_id: str,
    claim_token: str,
    max_unit_attempts: int,
    exc: Exception,
) -> int:
    stage_dir = _stage_dir(unit_dir, stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    attempt = _persistent_attempt_count(unit_dir, stage) + 1
    payload = {
        "stage": stage,
        "unit_id": int(unit_dir.name),
        "worker_id": worker_id,
        "claim_token": claim_token,
        "attempt_count": attempt,
        "max_unit_attempts": max_unit_attempts,
        "failed_at": time.time(),
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }
    write_json(payload, stage_dir / f"attempt-{time.time_ns()}-{claim_token[:12]}.error.json")
    write_json(payload, _attempt_state_path(unit_dir, stage))
    if attempt >= max_unit_attempts:
        write_json(
            {
                **payload,
                "quarantined": True,
                "reason": "persistent_unit_attempt_limit_reached",
            },
            _quarantine_path(unit_dir, stage),
        )
    return attempt


def _clear_attempt_state(unit_dir: Path, stage: str) -> None:
    for path in (_attempt_state_path(unit_dir, stage), _quarantine_path(unit_dir, stage)):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def run_stage_worker(
    config: dict[str, Any],
    campaign_root: str | Path,
    stage: str,
    *,
    worker_index: int = 0,
    max_units: int | None = None,
    lease_seconds: int = 21_600,
    heartbeat_seconds: float = 60.0,
    orphan_grace_seconds: int = 120,
    max_unit_attempts: int = 3,
    stop_claiming_at_epoch: float | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage}; expected one of {STAGES}")
    campaign_root = Path(campaign_root).expanduser().resolve()
    slurm_job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID", str(worker_index))
    worker_id = f"{slurm_job}:{slurm_task}:{os.getpid()}"
    units = _unit_ids(campaign_root)
    if units:
        rotate = int(worker_index) % len(units)
        units = units[rotate:] + units[:rotate]
    runtime: dict[str, Any] = {}
    completed_count = 0
    failed_units: list[int] = []
    drained = False
    max_attempts = max(1, int(max_unit_attempts))
    while max_units is None or completed_count < int(max_units):
        if stop_claiming_at_epoch is not None and time.time() >= stop_claiming_at_epoch:
            drained = True
            break
        claimed: tuple[int, ClaimHandle] | None = None
        for unit_id in units:
            unit_dir = _unit_dir(campaign_root, unit_id)
            if _success_path(unit_dir, stage).exists():
                continue
            if _quarantine_path(unit_dir, stage).exists():
                continue
            if _persistent_attempt_count(unit_dir, stage) >= max_attempts:
                continue
            if not _prerequisite_satisfied(unit_dir, stage):
                continue
            claim = try_claim(
                campaign_root,
                stage,
                unit_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                orphan_grace_seconds=orphan_grace_seconds,
            )
            if claim is not None:
                claimed = unit_id, claim
                break
        if claimed is None:
            break
        unit_id, claim = claimed
        unit_dir = _unit_dir(campaign_root, unit_id)
        stage_dir = _stage_dir(unit_dir, stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        heartbeat = ClaimHeartbeat(claim, interval_seconds=heartbeat_seconds).start()
        hydrated: list[str] = []
        node_hydration_base = os.environ.get("BCC_NODE_HYDRATION_ROOT", "").strip()
        hydration_root = (
            Path(node_hydration_base) / stage / f"{unit_id:06d}" / claim.token
            if node_hydration_base
            else None
        )
        started = time.perf_counter()
        committed = False
        try:
            hydrated = _hydrate_for_stage(
                unit_dir,
                stage,
                hydration_root=hydration_root,
            )
            unit_config = _unit_config(config, unit_dir)
            details = _run_stage(stage, unit_config, unit_dir, runtime)
            heartbeat.assert_owned()
            payload = {
                "stage": stage,
                "unit_id": unit_id,
                "worker_id": worker_id,
                "claim_token": claim.token,
                "claim_generation": claim.generation,
                "elapsed_seconds": time.perf_counter() - started,
                "completed_at": time.time(),
                "details": details,
                "input_manifest_sha256": sha256_file(unit_dir / "selected_images.jsonl"),
            }
            _fsync_unit_outputs(unit_dir)
            commit_claim_json(claim, _success_path(unit_dir, stage), payload)
            committed = True
            try:
                _clear_attempt_state(unit_dir, stage)
            except OSError:
                # The stage commit is already durable. Stale attempt metadata
                # is diagnostic and must not downgrade a successful unit.
                pass
            completed_count += 1
        except SharedStageRuntimeError:
            # Release the claim in ``finally`` and fail the worker. Successor
            # jobs may retry after the allocation/runtime is repaired, but the
            # unit's persistent attempt count must remain unchanged.
            raise
        except Exception as exc:
            failed_units.append(unit_id)
            if not isinstance(exc, ClaimOwnershipLost):
                run_if_claim_owned(
                    claim,
                    lambda: _record_stage_failure(
                        unit_dir,
                        stage,
                        worker_id=worker_id,
                        claim_token=claim.token,
                        max_unit_attempts=max_attempts,
                        exc=exc,
                    ),
                )
        finally:
            heartbeat.stop()

            def cleanup_owned_artifacts() -> None:
                cleanup_names = list(hydrated)
                if committed and stage == "sam3":
                    cleanup_names.extend(("masks", "inverse_crops"))
                if committed and stage in {"bcc-draft", "bcc"}:
                    cleanup_names.append("correspondence_overlays")
                _cleanup_stage_hydration(unit_dir, cleanup_names, hydration_root)

            cleaned_and_released = finish_claim(
                claim,
                cleanup_owned_artifacts,
            )
            if not cleaned_and_released and hydration_root is not None:
                shutil.rmtree(hydration_root, ignore_errors=True)
    return {
        "stage": stage,
        "worker_id": worker_id,
        "completed_units": completed_count,
        "failed_units": failed_units,
        "drained": drained,
        "stop_claiming_at_epoch": stop_claiming_at_epoch,
    }


def merge_stage(campaign_root: str | Path, stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(stage)
    campaign_root = Path(campaign_root).expanduser().resolve()
    missing: list[int] = []
    quarantined: list[int] = []
    successes: list[dict[str, Any]] = []
    for unit_id in _unit_ids(campaign_root):
        path = _success_path(_unit_dir(campaign_root, unit_id), stage)
        if not path.exists():
            missing.append(unit_id)
            if _quarantine_path(_unit_dir(campaign_root, unit_id), stage).exists():
                quarantined.append(unit_id)
        else:
            successes.append(json.loads(path.read_text(encoding="utf-8")))
    if quarantined:
        raise RuntimeError(
            f"Stage {stage} has {len(quarantined)} quarantined unit(s): {quarantined[:20]}"
        )
    if missing:
        raise RuntimeError(
            f"Stage {stage} is incomplete for {len(missing)} unit(s): {missing[:20]}"
        )
    merged = {
        "stage": stage,
        "unit_count": len(successes),
        "elapsed_seconds_sum": sum(float(row.get("elapsed_seconds") or 0) for row in successes),
        "merged_at": time.time(),
    }
    from .run_ledger import write_run_ledger
    from .stage_merge import merge_stage_jsonl

    merged["outputs"] = merge_stage_jsonl(campaign_root, stage)
    merged["ledger"] = write_run_ledger(campaign_root)
    path = campaign_root / "stages" / stage / "_MERGED.json"
    write_json(merged, path)
    return merged


def wait_for_stage_merge(
    campaign_root: str | Path,
    stage: str,
    *,
    poll_seconds: int = 30,
) -> dict[str, Any]:
    """Advance as soon as any combination of workers completes all units."""
    while True:
        try:
            return merge_stage(campaign_root, stage)
        except RuntimeError as exc:
            if not str(exc).startswith(f"Stage {stage} is incomplete"):
                raise
        time.sleep(max(1, int(poll_seconds)))


def campaign_status(campaign_root: str | Path) -> dict[str, Any]:
    """Return registry, stage, publication, and site progress without scans of data rows."""
    import sqlite3

    campaign_root = Path(campaign_root).expanduser().resolve()
    registry = load_registry(campaign_root)
    unit_count = int(registry.get("unit_count") or 0)
    stages: dict[str, Any] = {}
    for stage in STAGES:
        complete = 0
        failed_attempts = 0
        for unit_id in range(unit_count):
            unit_dir = _unit_dir(campaign_root, unit_id)
            complete += int(_success_path(unit_dir, stage).exists())
            stage_dir = _stage_dir(unit_dir, stage)
            if stage_dir.exists():
                failed_attempts += sum(
                    1 for _ in stage_dir.glob("attempt-*.error.json")
                )
        claims_dir = campaign_root / "claims" / stage
        active_claims = sum(1 for _ in claims_dir.glob("*.claim")) if claims_dir.exists() else 0
        quarantined = sum(
            int(_quarantine_path(_unit_dir(campaign_root, unit_id), stage).exists())
            for unit_id in range(unit_count)
        )
        stages[stage] = {
            "complete_units": complete,
            "total_units": unit_count,
            "active_claims": active_claims,
            "quarantined_units": quarantined,
            "failed_attempts": failed_attempts,
            "merged": (campaign_root / "stages" / stage / "_MERGED.json").exists(),
        }
    pair_count = 0
    milestone_count = 0
    database = campaign_paths(campaign_root)["published"] / "campaign_state.sqlite3"
    if database.exists():
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            pair_count = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
            milestone_count = int(
                connection.execute("SELECT COUNT(*) FROM milestones").fetchone()[0]
            )
        finally:
            connection.close()
    return {
        **registry,
        "campaign_root": str(campaign_root),
        "stages": stages,
        "published_pair_count": pair_count,
        "published_milestone_count": milestone_count,
        "site_ready": (campaign_paths(campaign_root)["site"] / "READY.json").exists(),
    }
