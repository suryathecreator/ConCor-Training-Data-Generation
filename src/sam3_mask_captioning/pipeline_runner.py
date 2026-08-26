from __future__ import annotations

from collections.abc import Iterator
import json
import traceback
from pathlib import Path
from typing import Any

from .bcc_html_report import write_bcc_html_report
from .bcc_canonicalization import canonicalize_bcc_rows
from .caption_stage import QwenCaptioner, run_captioning, run_mask_review
from .dataset import load_records
from .consistency_stage import run_sam3_consistency
from .correspondence_stage import (
    BCC_PROMPT_VERSION,
    CORRESPONDENCE_SCHEMA_VERSION,
    PIPELINE_STAGE_VERSION,
    ensure_correspondence_outputs,
    run_image_caption_pass,
    run_image_caption_pass_batch,
    run_image_caption_qa,
    run_image_caption_qa_batch,
)
from .image_review_stage import run_image_review
from .io_utils import append_jsonl, read_jsonl, write_json
from .sam3_stage import _load_processor, run_sam3


class _JsonlCheckpointIndex:
    """Incrementally index append-only checkpoints by image ID."""

    def __init__(self) -> None:
        self._state: dict[Path, dict[str, Any]] = {}

    def _refresh(self, path: Path) -> dict[str, Any]:
        path = path.resolve()
        if not path.exists():
            return {"rows": [], "by_image": {}, "offset": 0, "inode": None}
        stat = path.stat()
        state = self._state.get(path)
        if (
            state is None
            or state["inode"] != stat.st_ino
            or stat.st_size < state["offset"]
        ):
            state = {
                "rows": [],
                "by_image": {},
                "offset": 0,
                "inode": stat.st_ino,
            }
            self._state[path] = state
        if stat.st_size == state["offset"]:
            return state
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(state["offset"])
            while True:
                line = handle.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                state["rows"].append(row)
                image_id = str(row.get("image_id") or "")
                if image_id:
                    state["by_image"].setdefault(image_id, []).append(row)
            state["offset"] = handle.tell()
        return state

    def rows(self, path: Path, image_id: str | None = None) -> list[dict[str, Any]]:
        state = self._refresh(path)
        if image_id is None:
            return state["rows"]
        return state["by_image"].get(image_id, [])

    def image_ids(self, path: Path) -> set[str]:
        return set(self._refresh(path)["by_image"])

    def image_count(self, path: Path) -> int:
        return len(self._refresh(path)["by_image"])

    def row_count(self, path: Path) -> int:
        return len(self._refresh(path)["rows"])

    def ordered_image_ids(self, path: Path) -> list[str]:
        return list(self._refresh(path)["by_image"])


_CHECKPOINTS = _JsonlCheckpointIndex()


def _rows_for_image(path: Path, image_id: str) -> list[dict[str, Any]]:
    return _CHECKPOINTS.rows(path, image_id)


_VERSIONED_TERMINAL_STAGES = {"bcc_canonicalization", "image_caption_qa"}


def _terminal_image_ids(path: Path) -> set[str]:
    return {
        str(row.get("image_id") or "")
        for row in _CHECKPOINTS.rows(path)
        if str(row.get("status") or "") in {"accepted", "rejected", "error"}
        and (
            str(row.get("stage") or "") not in _VERSIONED_TERMINAL_STAGES
            or _is_current_pair(row)
        )
    }


def _image_ids(path: Path) -> set[str]:
    return _CHECKPOINTS.image_ids(path)


def _is_current_pair(row: dict[str, Any]) -> bool:
    return (
        str(row.get("prompt_version") or "") == BCC_PROMPT_VERSION
        and str(row.get("schema_version") or "") == CORRESPONDENCE_SCHEMA_VERSION
        and str(row.get("stage_version") or "") == PIPELINE_STAGE_VERSION
    )


def _current_successful_ids(path: Path) -> list[str]:
    successful: list[str] = []
    seen: set[str] = set()
    for row in _CHECKPOINTS.rows(path):
        image_id = str(row.get("image_id") or "")
        if not image_id or image_id in seen or not _is_current_pair(row):
            continue
        seen.add(image_id)
        successful.append(image_id)
    return successful


def _current_pair_image_ids(path: Path) -> set[str]:
    return set(_current_successful_ids(path))


def _success_count(run_dir: Path) -> int:
    return len(_current_successful_ids(run_dir / "image_text_pairs.jsonl"))


def _record_bcc_duplicates(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = run_dir / "bcc_duplicate_masks.jsonl"
    existing = {
        str(row.get("dropped_mask_id") or "")
        for row in _CHECKPOINTS.rows(path)
    }
    for row in rows:
        if str(row.get("dropped_mask_id") or "") in existing:
            continue
        append_jsonl(row, path)


def _state_payload(
    run_dir: Path,
    *,
    target_successes: int,
    last_image_id: str = "",
    stopped_early: bool = False,
) -> dict[str, Any]:
    final_path = run_dir / "image_text_pairs.jsonl"
    successful_ids = _current_successful_ids(final_path)
    status_path = run_dir / "image_pipeline_status.jsonl"
    return {
        "target_successes": target_successes,
        "successful_images": len(successful_ids),
        "successful_image_ids": successful_ids,
        "terminal_attempt_count": _CHECKPOINTS.row_count(status_path),
        "last_image_id": last_image_id,
        "stopped_early": stopped_early,
        "site_report": str(run_dir / "site" / "report.html"),
        "site_ready": (run_dir / "site" / "report.html").exists(),
    }


def _flush_bcc_ready(
    config: dict[str, Any],
    run_dir: Path,
    row_groups: list[list[dict[str, Any]]],
    *,
    captioner: QwenCaptioner | None,
    mock: bool,
    target: int,
    status_path: Path,
    terminal_ids: set[str],
) -> None:
    if not row_groups:
        return
    run_image_caption_pass_batch(
        config,
        run_dir,
        row_groups,
        captioner=captioner,
        mock=mock,
    )
    run_image_caption_qa_batch(
        config,
        run_dir,
        row_groups,
        captioner=captioner,
        mock=mock,
    )
    accepted_any = False
    for rows in row_groups:
        image_id = str(rows[0]["image_id"])
        accepted = any(
            _is_current_pair(row)
            for row in _rows_for_image(run_dir / "image_text_pairs.jsonl", image_id)
        )
        append_jsonl(
            {
                "image_id": image_id,
                "status": "accepted" if accepted else "rejected",
                "stage": "image_caption_qa",
                "prompt_version": BCC_PROMPT_VERSION,
                "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                "stage_version": PIPELINE_STAGE_VERSION,
                "sam3_mask_count": len(_rows_for_image(run_dir / "sam3_masks.jsonl", image_id)),
                "qwen_mask_qa_count": len(_rows_for_image(run_dir / "captions.jsonl", image_id)),
                "sam3_consistency_count": len(rows) + sum(len(row.get("bcc_duplicate_mask_aliases") or []) for row in rows),
                "bcc_canonical_mask_count": len(rows),
                "bcc_duplicate_mask_count": sum(len(row.get("bcc_duplicate_mask_aliases") or []) for row in rows),
                "successful_images_after": _success_count(run_dir),
            },
            status_path,
        )
        terminal_ids.add(image_id)
        accepted_any = accepted_any or accepted

    if accepted_any:
        write_bcc_html_report(
            run_dir,
            output_path=run_dir / "site" / "report.html",
            max_images=target,
        )


def _iter_accepted_reviews(
    config: dict[str, Any],
    run_dir: Path,
    *,
    limit: int | None,
    captioner: QwenCaptioner | None,
    mock: bool,
) -> Iterator[dict[str, Any]]:
    """Review bounded manifest windows and stream accepted rows downstream."""
    total_records = len(load_records(config, limit=limit))
    if total_records <= 0:
        return

    runner_config = config.get("pipeline", {})
    review_config = config.get("image_review", {})
    resume = bool(config.get("resume", False) or review_config.get("resume", False))
    requested_window = max(1, int(runner_config.get("image_review_window", 64)))
    window = requested_window if resume else total_records
    selected_path = run_dir / "selected_images.jsonl"
    existing_selected = len(read_jsonl(selected_path)) if selected_path.exists() else 0
    reviewed_limit = min(total_records, max(window, existing_selected))
    yielded_ids: set[str] = set()

    while reviewed_limit > 0:
        run_image_review(
            config,
            run_dir,
            mock=mock,
            limit=reviewed_limit,
            captioner_override=captioner,
        )
        selected = read_jsonl(selected_path) if selected_path.exists() else []
        reviews_path = run_dir / "image_reviews.jsonl"
        reviews = read_jsonl(reviews_path) if reviews_path.exists() else []
        review_by_id = {str(row.get("image_id") or ""): row for row in reviews}
        for record in selected:
            image_id = str(record.get("image_id") or "")
            review = review_by_id.get(image_id)
            if not image_id or image_id in yielded_ids or not review:
                continue
            yielded_ids.add(image_id)
            if review.get("accepted"):
                yield review
        if reviewed_limit >= total_records:
            break
        reviewed_limit = min(total_records, reviewed_limit + window)


def run_checkpointed_pipeline(
    config: dict[str, Any],
    run_dir: str | Path,
    *,
    limit: int | None = None,
    target_successes: int | None = None,
    mock: bool = False,
) -> Path:
    """Process accepted images end-to-end, stopping after N pass-two records."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    runner_config = config.get("pipeline", {})
    target = int(target_successes or runner_config.get("target_successful_images", 10))
    status_path = run_dir / "image_pipeline_status.jsonl"
    state_path = run_dir / "pipeline_state.json"

    ensure_correspondence_outputs(run_dir)
    if _success_count(run_dir) >= target:
        write_bcc_html_report(run_dir, output_path=run_dir / "site" / "report.html", max_images=target)
        write_json(_state_payload(run_dir, target_successes=target, stopped_early=True), state_path)
        return run_dir / "image_text_pairs.jsonl"

    shared_qwen = None if mock else QwenCaptioner(config, config_section="caption")
    sam3_processor = None
    terminal_ids = _terminal_image_ids(status_path)
    bcc_ready: list[list[dict[str, Any]]] = []
    bcc_window = max(1, int(config.get("pipeline", {}).get("bcc_ready_window", 4)))
    for review in _iter_accepted_reviews(
        config,
        run_dir,
        limit=limit,
        captioner=shared_qwen,
        mock=mock,
    ):
        image_id = str(review["image_id"])
        if image_id in terminal_ids:
            continue
        if _success_count(run_dir) >= target:
            break
        try:
            if sam3_processor is None and not mock:
                sam3_processor = _load_processor(config)
            run_sam3(
                config,
                run_dir,
                mock=mock,
                reviews_override=[review],
                processor_override=sam3_processor,
            )
            mask_rows = _rows_for_image(run_dir / "sam3_masks.jsonl", image_id)
            if not mask_rows:
                raise RuntimeError("SAM3 produced no heuristic-passed masks")

            run_captioning(
                config,
                run_dir,
                rows_override=mask_rows,
                mock=mock,
                captioner_override=shared_qwen,
            )
            candidate_rows = _rows_for_image(run_dir / "caption_candidates.jsonl", image_id)
            if not candidate_rows:
                append_jsonl(
                    {
                        "image_id": image_id,
                        "status": "rejected",
                        "stage": "caption",
                        "reason": "no masks passed per-mask caption generation",
                    },
                    status_path,
                )
                terminal_ids.add(image_id)
                write_json(_state_payload(run_dir, target_successes=target, last_image_id=image_id), state_path)
                continue
            run_mask_review(
                config,
                run_dir,
                rows_override=candidate_rows,
                mock=mock,
                captioner_override=shared_qwen,
            )
            qa_rows = _rows_for_image(run_dir / "captions.jsonl", image_id)
            if not qa_rows:
                append_jsonl(
                    {
                        "image_id": image_id,
                        "status": "rejected",
                        "stage": "per_mask_qa",
                        "reason": "no masks passed Qwen mask QA",
                    },
                    status_path,
                )
                terminal_ids.add(image_id)
                write_json(_state_payload(run_dir, target_successes=target, last_image_id=image_id), state_path)
                continue

            run_sam3_consistency(
                config,
                run_dir,
                rows=qa_rows,
                processor=sam3_processor,
                mock=mock,
            )
            consistency_rows = _rows_for_image(run_dir / "consistent_captions.jsonl", image_id)
            image_caption_config = config.get("image_caption", {})
            consistent_rows, duplicate_rows = canonicalize_bcc_rows(
                consistency_rows, image_caption_config
            )
            _record_bcc_duplicates(run_dir, duplicate_rows)
            min_input_masks = int(
                image_caption_config.get(
                    "min_input_masks", image_caption_config.get("min_groups", 10)
                )
            )
            if len(consistent_rows) < min_input_masks:
                append_jsonl(
                    {
                        "image_id": image_id,
                        "stage": "image_caption",
                        "prompt_version": BCC_PROMPT_VERSION,
                        "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                        "stage_version": PIPELINE_STAGE_VERSION,
                        "reason": (
                            f"only {len(consistent_rows)} BCC-canonical masks; "
                            f"need {min_input_masks}"
                        ),
                    },
                    run_dir / "image_caption_rejected.jsonl",
                )
                append_jsonl(
                    {
                        "image_id": image_id,
                        "status": "rejected",
                        "stage": "bcc_canonicalization",
                        "prompt_version": BCC_PROMPT_VERSION,
                        "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                        "stage_version": PIPELINE_STAGE_VERSION,
                        "reason": (
                            f"only {len(consistent_rows)} BCC-canonical masks; "
                            f"need {min_input_masks}"
                        ),
                        "sam3_mask_count": len(mask_rows),
                        "qwen_mask_qa_count": len(qa_rows),
                        "sam3_consistency_count": len(consistency_rows),
                        "bcc_canonical_mask_count": len(consistent_rows),
                        "bcc_duplicate_mask_count": len(duplicate_rows),
                    },
                    status_path,
                )
                terminal_ids.add(image_id)
                write_json(_state_payload(run_dir, target_successes=target, last_image_id=image_id), state_path)
                continue
            bcc_ready.append(consistent_rows)
            remaining = max(1, target - _success_count(run_dir))
            if len(bcc_ready) >= min(bcc_window, remaining):
                _flush_bcc_ready(
                    config,
                    run_dir,
                    bcc_ready,
                    captioner=shared_qwen,
                    mock=mock,
                    target=target,
                    status_path=status_path,
                    terminal_ids=terminal_ids,
                )
                bcc_ready = []
            write_json(_state_payload(run_dir, target_successes=target, last_image_id=image_id), state_path)
        except Exception as exc:
            append_jsonl(
                {
                    "image_id": image_id,
                    "status": "error",
                    "stage": "pipeline",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                status_path,
            )
            terminal_ids.add(image_id)
            write_json(_state_payload(run_dir, target_successes=target, last_image_id=image_id), state_path)
            if not config.get("continue_on_error", True):
                raise

    _flush_bcc_ready(
        config,
        run_dir,
        bcc_ready,
        captioner=shared_qwen,
        mock=mock,
        target=target,
        status_path=status_path,
        terminal_ids=terminal_ids,
    )
    write_bcc_html_report(
        run_dir,
        output_path=run_dir / "site" / "report.html",
        max_images=target,
    )
    stopped_early = _success_count(run_dir) >= target
    write_json(
        _state_payload(run_dir, target_successes=target, stopped_early=stopped_early),
        state_path,
    )
    return run_dir / "image_text_pairs.jsonl"


def run_correspondence_recovery(
    config: dict[str, Any],
    run_dir: str | Path,
    *,
    target_successes: int = 3,
    mock: bool = False,
) -> Path:
    """Reuse consistency-passed rows and recover only the two image-caption passes."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_correspondence_outputs(run_dir)
    target = int(target_successes)
    if target <= 0:
        raise ValueError("target_successes must be positive")
    consistent_path = run_dir / "consistent_captions.jsonl"
    if not consistent_path.exists():
        raise FileNotFoundError(f"Missing consistency checkpoint: {consistent_path}")
    consistent_rows = read_jsonl(consistent_path)
    by_image: dict[str, list[dict[str, Any]]] = {}
    for row in consistent_rows:
        by_image.setdefault(str(row.get("image_id") or ""), []).append(row)
    image_caption_config = config.get("image_caption", {})
    for image_id, image_rows in list(by_image.items()):
        canonical_rows, duplicate_rows = canonicalize_bcc_rows(
            image_rows, image_caption_config
        )
        _record_bcc_duplicates(run_dir, duplicate_rows)
        by_image[image_id] = canonical_rows
    min_input_masks = int(
        image_caption_config.get(
            "min_input_masks", image_caption_config.get("min_groups", 10)
        )
    )
    selected_path = run_dir / "selected_images.jsonl"
    selected = read_jsonl(selected_path) if selected_path.exists() else []
    manifest_order = {
        str(row.get("image_id") or ""): index
        for index, row in enumerate(selected)
    }
    current_candidate_ids = {
        str(row.get("image_id") or "")
        for row in _CHECKPOINTS.rows(run_dir / "image_caption_candidates.jsonl")
        if str(row.get("prompt_version") or "") == BCC_PROMPT_VERSION
        and str(row.get("schema_version") or "")
        == CORRESPONDENCE_SCHEMA_VERSION
        and str(row.get("stage_version") or "")
        == PIPELINE_STAGE_VERSION
    }

    eligible = sorted(
        (
            (image_id, rows)
            for image_id, rows in by_image.items()
            if image_id and len(rows) >= min_input_masks
        ),
        key=lambda item: (
            item[0] not in current_candidate_ids,
            len(item[1]),
            manifest_order.get(item[0], 10**9),
            item[0],
        ),
    )
    status_path = run_dir / "correspondence_recovery_status.jsonl"
    state_path = run_dir / "correspondence_recovery_state.json"
    captioner = None if mock else QwenCaptioner(config, config_section="image_caption")

    def write_state(last_image_id: str = "") -> None:
        final_path = run_dir / "image_text_pairs.jsonl"
        successful_ids = _current_successful_ids(final_path)
        write_json(
            {
                "target_successes": target,
                "successful_images": len(successful_ids),
                "successful_image_ids": successful_ids,
                "eligible_checkpoint_images": len(eligible),
                "recovery_attempt_count": _CHECKPOINTS.row_count(status_path),
                "last_image_id": last_image_id,
                "stopped_early": _success_count(run_dir) >= target,
                "site_report": str(run_dir / "site" / "report.html"),
                "site_ready": (run_dir / "site" / "report.html").exists(),
            },
            state_path,
        )

    if _success_count(run_dir) >= target:
        write_bcc_html_report(
            run_dir,
            output_path=run_dir / "site" / "report.html",
            max_images=target,
        )
        write_state()
        return run_dir / "image_text_pairs.jsonl"

    qa_rejected_path = run_dir / "image_caption_qa_rejected.jsonl"
    qa_rejected_ids = {
        str(row.get("image_id") or "")
        for row in (read_jsonl(qa_rejected_path) if qa_rejected_path.exists() else [])
        if str(row.get("reason") or "") != "generation_or_schema_failed"
        and str(row.get("prompt_version") or "") == BCC_PROMPT_VERSION
        and str(row.get("schema_version") or "") == CORRESPONDENCE_SCHEMA_VERSION
        and str(row.get("stage_version") or "") == PIPELINE_STAGE_VERSION
    }
    for image_id, rows in eligible:
        if _success_count(run_dir) >= target:
            break
        if image_id in _current_pair_image_ids(run_dir / "image_text_pairs.jsonl") or image_id in qa_rejected_ids:
            continue
        run_image_caption_pass(
            config,
            run_dir,
            rows,
            captioner=captioner,
            mock=mock,
        )
        candidate_rows = [
            row
            for row in _rows_for_image(
                run_dir / "image_caption_candidates.jsonl", image_id
            )
            if str(row.get("prompt_version") or "") == BCC_PROMPT_VERSION
            and str(row.get("schema_version") or "") == CORRESPONDENCE_SCHEMA_VERSION
            and str(row.get("stage_version") or "") == PIPELINE_STAGE_VERSION
        ]
        if not candidate_rows:
            append_jsonl(
                {
                    "image_id": image_id,
                    "status": "rejected_or_retryable_error",
                    "stage": "image_caption",
                    "consistency_passed_masks": len(rows),
                    "successful_images_after": _success_count(run_dir),
                },
                status_path,
            )
            write_state(image_id)
            continue
        run_image_caption_qa(
            config,
            run_dir,
            rows,
            captioner=captioner,
            mock=mock,
        )
        accepted = image_id in _current_pair_image_ids(run_dir / "image_text_pairs.jsonl")
        qa_rejected = image_id in _image_ids(run_dir / "image_caption_qa_rejected.jsonl")
        append_jsonl(
            {
                "image_id": image_id,
                "status": "accepted" if accepted else ("rejected" if qa_rejected else "retryable_error"),
                "stage": "image_caption_qa",
                "consistency_passed_masks": len(rows),
                "successful_images_after": _success_count(run_dir),
            },
            status_path,
        )
        if accepted:
            write_bcc_html_report(
                run_dir,
                output_path=run_dir / "site" / "report.html",
                max_images=target,
            )
        write_state(image_id)

    write_bcc_html_report(
        run_dir,
        output_path=run_dir / "site" / "report.html",
        max_images=target,
    )
    write_state()
    return run_dir / "image_text_pairs.jsonl"
