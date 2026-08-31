from __future__ import annotations

import json
import multiprocessing
import os
import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from sam3_mask_captioning.artifact_store import pack_artifacts
from sam3_mask_captioning.campaign_claims import (
    ClaimOwnershipLost,
    ClaimHeartbeat,
    claim_is_owned,
    claim_path,
    commit_claim_json,
    release_claim,
    stage_claim_lock,
    try_claim,
)
from sam3_mask_captioning.campaign_integrity import (
    audit_sam3_unit,
    repair_campaign_units,
)
from sam3_mask_captioning.campaign_manifest import extend_from_manifest, initialize_campaign
from sam3_mask_captioning.campaign_runner import (
    SharedStageRuntimeError,
    _quarantine_path,
    _record_stage_failure,
    merge_stage,
    run_stage_worker,
    wait_for_stage_merge,
)
from sam3_mask_captioning.io_utils import read_jsonl, write_json, write_jsonl
from sam3_mask_captioning.sam3_stage import run_sam3


def _claim_contender(
    root: str,
    barrier: multiprocessing.Barrier,
    result_queue: multiprocessing.Queue,
    index: int,
) -> None:
    barrier.wait()
    handle = try_claim(
        root,
        "sam3",
        7,
        worker_id=f"200:{index}:{os.getpid()}",
        lease_seconds=21_600,
        orphan_grace_seconds=120,
        job_state=lambda _job: "inactive",
    )
    if handle is not None:
        result_queue.put((index, handle.token))


class CampaignRecoveryTests(unittest.TestCase):
    def test_stale_claim_recovery_has_exactly_one_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stale = claim_path(root, "sam3", 7)
            write_json(
                {
                    "worker_id": "100:0:1",
                    "hostname": "retired-node",
                    "pid": 1,
                    "claimed_at": time.time() - 600,
                    "heartbeat_at": time.time() - 600,
                },
                stale,
            )
            process_count = 8
            barrier = multiprocessing.Barrier(process_count)
            result_queue: multiprocessing.Queue = multiprocessing.Queue()
            processes = [
                multiprocessing.Process(
                    target=_claim_contender,
                    args=(str(root), barrier, result_queue, index),
                )
                for index in range(process_count)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            winners = []
            while True:
                try:
                    winners.append(result_queue.get_nowait())
                except queue.Empty:
                    break
            self.assertEqual(len(winners), 1)
            payload = json.loads(stale.read_text(encoding="utf-8"))
            self.assertEqual(payload["token"], winners[0][1])
            self.assertEqual(len(list(stale.parent.glob("000007.claim.stale.*"))), 1)

    def test_array_siblings_do_not_look_like_requeues(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = try_claim(
                root,
                "sam3",
                0,
                worker_id="100:0:10",
                lease_seconds=21_600,
            )
            assert first is not None
            sibling = try_claim(
                root,
                "sam3",
                0,
                worker_id="100:1:20",
                lease_seconds=21_600,
                job_state=lambda _job: "inactive",
            )
            self.assertIsNone(sibling)
            requeue = try_claim(
                root,
                "sam3",
                0,
                worker_id="100:0:30",
                lease_seconds=21_600,
            )
            self.assertIsNotNone(requeue)
            assert requeue is not None
            self.assertFalse(claim_is_owned(first))
            self.assertTrue(claim_is_owned(requeue))

    def test_local_workers_do_not_look_like_requeues(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = try_claim(
                root,
                "sam3",
                0,
                worker_id="local:0:10",
                lease_seconds=21_600,
            )
            assert first is not None
            competitor = try_claim(
                root,
                "sam3",
                0,
                worker_id="local:0:20",
                lease_seconds=21_600,
            )
            self.assertIsNone(competitor)

    def test_heartbeat_renews_only_owned_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            handle = try_claim(
                root,
                "sam3",
                0,
                worker_id="100:0:10",
                lease_seconds=21_600,
            )
            assert handle is not None
            before = json.loads(handle.path.read_text(encoding="utf-8"))["heartbeat_at"]
            heartbeat = ClaimHeartbeat(handle, interval_seconds=0.05).start()
            time.sleep(0.12)
            heartbeat.stop()
            after = json.loads(handle.path.read_text(encoding="utf-8"))["heartbeat_at"]
            self.assertGreater(after, before)

    def test_scheduler_unknown_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = claim_path(root, "sam3", 0)
            write_json(
                {
                    "worker_id": "100:0:1",
                    "hostname": "remote",
                    "pid": 1,
                    "claimed_at": time.time() - 50_000,
                    "heartbeat_at": time.time() - 50_000,
                },
                path,
            )
            self.assertIsNone(
                try_claim(
                    root,
                    "sam3",
                    0,
                    worker_id="200:0:2",
                    lease_seconds=10,
                    orphan_grace_seconds=0,
                    job_state=lambda _job: "unknown",
                )
            )

    def test_old_fencing_token_cannot_release_new_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = try_claim(
                root,
                "sam3",
                0,
                worker_id="100:0:10",
                lease_seconds=1,
            )
            assert old is not None
            with stage_claim_lock(root, "sam3"):
                payload = json.loads(old.path.read_text(encoding="utf-8"))
                payload.update(
                    {
                        "worker_id": "200:0:20",
                        "allocation_id": "200:0",
                        "scheduler_job_id": "200_0",
                        "token": "new-token",
                    }
                )
                write_json(payload, old.path)
            self.assertFalse(claim_is_owned(old))
            self.assertFalse(release_claim(old))
            self.assertEqual(
                json.loads(old.path.read_text(encoding="utf-8"))["token"],
                "new-token",
            )

    def test_only_current_fencing_token_can_commit_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = try_claim(
                root,
                "sam3",
                0,
                worker_id="100:0:10",
                lease_seconds=21_600,
            )
            assert old is not None
            current = try_claim(
                root,
                "sam3",
                0,
                worker_id="100:0:20",
                lease_seconds=21_600,
            )
            assert current is not None
            success = root / "units" / "000000" / "stages" / "sam3" / "_SUCCESS.json"
            with self.assertRaises(ClaimOwnershipLost):
                commit_claim_json(old, success, {"worker": "old"})
            commit_claim_json(current, success, {"worker": "current"})
            self.assertEqual(json.loads(success.read_text(encoding="utf-8"))["worker"], "current")
            self.assertIsNone(
                try_claim(
                    root,
                    "sam3",
                    0,
                    worker_id="200:0:30",
                    lease_seconds=21_600,
                )
            )

    def test_sam3_integrity_detects_manifest_archive_divergence(self):
        with tempfile.TemporaryDirectory() as temporary:
            unit = Path(temporary)
            (unit / "masks").mkdir()
            (unit / "inverse_crops").mkdir()
            (unit / "artifacts").mkdir()
            mask_id = "image_p000_m0000"
            Image.new("L", (4, 4), 255).save(unit / "masks" / f"{mask_id}.png")
            Image.new("RGB", (4, 4), (255, 255, 255)).save(
                unit / "inverse_crops" / f"{mask_id}.png"
            )
            row = {
                "image_id": "image",
                "mask_id": mask_id,
                "mask_path": str(unit / "masks" / f"{mask_id}.png"),
                "inverse_crop_path": str(unit / "inverse_crops" / f"{mask_id}.png"),
            }
            write_jsonl([row], unit / "sam3_masks.jsonl")
            write_jsonl(
                [{"image_id": "image", "mask_id": mask_id, "rle": {"data": "eA==", "size": [4, 4]}}],
                unit / "mask_rle.jsonl",
            )
            pack_artifacts(unit, ["masks", "inverse_crops"], unit / "artifacts" / "sam3.tar")
            self.assertTrue(audit_sam3_unit(unit)["valid"])

            missing_id = "image_p000_m0001"
            write_jsonl(
                [
                    row,
                    {
                        **row,
                        "mask_id": missing_id,
                        "mask_path": str(unit / "masks" / f"{missing_id}.png"),
                        "inverse_crop_path": str(unit / "inverse_crops" / f"{missing_id}.png"),
                    },
                ],
                unit / "sam3_masks.jsonl",
            )
            report = audit_sam3_unit(unit)
            self.assertFalse(report["valid"])
            codes = {item["code"] for item in report["issues"]}
            self.assertIn("manifest_mask_archive_mismatch", codes)
            self.assertIn("manifest_inverse_archive_mismatch", codes)

    def test_zero_mask_sam3_artifact_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            unit = Path(temporary)
            (unit / "artifacts").mkdir()
            write_jsonl([], unit / "sam3_masks.jsonl")
            write_jsonl([], unit / "mask_rle.jsonl")
            pack_artifacts(unit, ["masks", "inverse_crops"], unit / "artifacts" / "sam3.tar")
            self.assertTrue(audit_sam3_unit(unit)["valid"])

    def test_sam3_resume_removes_partial_rows_and_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "source.png"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(image)
            write_jsonl(
                [
                    {
                        "image_id": "image",
                        "source_image_path": str(image),
                        "accepted": True,
                        "sam3_prompts": ["object"],
                    }
                ],
                root / "image_reviews.jsonl",
            )
            (root / "masks").mkdir()
            partial = root / "masks" / "image_p000_m0000.png"
            Image.new("L", (32, 32), 255).save(partial)
            write_jsonl(
                [
                    {
                        "image_id": "image",
                        "mask_id": "image_p000_m0000",
                        "mask_path": str(partial),
                        "inverse_crop_path": str(root / "inverse_crops" / partial.name),
                    }
                ],
                root / "sam3_masks.jsonl",
            )
            run_sam3(
                {
                    "resume": True,
                    "continue_on_error": False,
                    "filter": {"min_mask_area": 1, "max_masks_per_image": 0},
                },
                root,
                mock=True,
            )
            rows = read_jsonl(root / "sam3_masks.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(len({row["mask_id"] for row in rows}), 1)
            self.assertEqual(len(read_jsonl(root / "sam3_completed_images.jsonl")), 1)

    def test_repair_is_dry_run_then_preserves_review_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "source.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(image)
            manifest = root / "manifest.jsonl"
            write_jsonl([{"image_id": "image", "image_path": str(image)}], manifest)
            campaign = root / "campaign"
            initialize_campaign(campaign, unit_size=1)
            extend_from_manifest(campaign, manifest, target_total=1)
            unit = campaign / "units" / "000000"
            write_jsonl([{"image_id": "image", "accepted": True}], unit / "image_reviews.jsonl")
            write_json({"stage": "image-review"}, unit / "stages" / "image-review" / "_SUCCESS.json")
            write_jsonl([], unit / "sam3_masks.jsonl")
            write_jsonl([], unit / "mask_rle.jsonl")
            (unit / "artifacts").mkdir(exist_ok=True)
            pack_artifacts(unit, ["masks", "inverse_crops"], unit / "artifacts" / "sam3.tar")
            write_json({"stage": "sam3"}, unit / "stages" / "sam3" / "_SUCCESS.json")
            write_json({"stage": "sam3"}, campaign / "stages" / "sam3" / "_MERGED.json")
            (campaign / "merged" / "sam3").mkdir(parents=True)

            dry_run = repair_campaign_units(
                campaign,
                unit_ids=[0],
                from_stage="sam3",
                apply=False,
            )
            self.assertTrue(dry_run["planned_paths"])
            self.assertTrue((unit / "sam3_masks.jsonl").exists())

            backup = root / "backup"
            repair_campaign_units(
                campaign,
                unit_ids=[0],
                from_stage="sam3",
                apply=True,
                backup_root=backup,
            )
            self.assertTrue((unit / "selected_images.jsonl").exists())
            self.assertTrue((unit / "image_reviews.jsonl").exists())
            self.assertTrue((unit / "stages" / "image-review" / "_SUCCESS.json").exists())
            self.assertFalse((unit / "sam3_masks.jsonl").exists())
            self.assertFalse((unit / "stages" / "sam3" / "_SUCCESS.json").exists())
            self.assertTrue((backup / "repair.json").exists())

    def test_consistency_repair_preserves_upstream_mask_rle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "source.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(image)
            manifest = root / "manifest.jsonl"
            write_jsonl([{"image_id": "image", "image_path": str(image)}], manifest)
            campaign = root / "campaign"
            initialize_campaign(campaign, unit_size=1)
            extend_from_manifest(campaign, manifest, target_total=1)
            unit = campaign / "units" / "000000"
            write_jsonl(
                [{"image_id": "image", "mask_id": "mask", "rle": {"data": "eA=="}}],
                unit / "mask_rle.jsonl",
            )
            write_jsonl([], unit / "consistent_captions.jsonl")
            write_json(
                {"stage": "consistency"},
                unit / "stages" / "consistency" / "_SUCCESS.json",
            )

            repair_campaign_units(
                campaign,
                unit_ids=[0],
                from_stage="consistency",
                apply=True,
                backup_root=root / "backup",
            )
            self.assertTrue((unit / "mask_rle.jsonl").exists())
            self.assertFalse((unit / "consistent_captions.jsonl").exists())

    def test_attempt_limit_is_persistent_and_merge_reports_quarantine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "source.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(image)
            manifest = root / "manifest.jsonl"
            write_jsonl([{"image_id": "image", "image_path": str(image)}], manifest)
            campaign = root / "campaign"
            initialize_campaign(campaign, unit_size=1)
            extend_from_manifest(campaign, manifest, target_total=1)
            unit = campaign / "units" / "000000"
            for attempt in range(3):
                count = _record_stage_failure(
                    unit,
                    "sam3",
                    worker_id=f"100:{attempt}:1",
                    claim_token=f"token-{attempt}",
                    max_unit_attempts=3,
                    exc=RuntimeError("broken"),
                )
                self.assertEqual(count, attempt + 1)
            self.assertTrue(_quarantine_path(unit, "sam3").exists())
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                merge_stage(campaign, "sam3")

    def test_shared_runtime_failure_does_not_quarantine_unit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "source.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(image)
            manifest = root / "manifest.jsonl"
            write_jsonl([{"image_id": "image", "image_path": str(image)}], manifest)
            campaign = root / "campaign"
            initialize_campaign(campaign, unit_size=1)
            extend_from_manifest(campaign, manifest, target_total=1)
            unit = campaign / "units" / "000000"
            write_jsonl(
                [{"image_id": "image", "accepted": True, "sam3_prompts": ["object"]}],
                unit / "image_reviews.jsonl",
            )
            write_json(
                {"stage": "image-review"},
                unit / "stages" / "image-review" / "_SUCCESS.json",
            )

            with mock.patch(
                "sam3_mask_captioning.campaign_runner._load_processor",
                side_effect=ModuleNotFoundError("No module named 'iopath'"),
            ):
                with self.assertRaises(SharedStageRuntimeError):
                    run_stage_worker({}, campaign, "sam3", max_units=1)

            self.assertFalse(_quarantine_path(unit, "sam3").exists())
            self.assertFalse((unit / "stages" / "sam3" / "attempt-state.json").exists())
            self.assertFalse(claim_path(campaign, "sam3", 0).exists())

    def test_merge_watcher_waits_for_quarantine_repair(self):
        completed = {"stage": "sam3", "unit_count": 1}
        with mock.patch(
            "sam3_mask_captioning.campaign_runner.merge_stage",
            side_effect=[
                RuntimeError("Stage sam3 has 1 quarantined unit(s): [0]"),
                completed,
            ],
        ) as merge, mock.patch("sam3_mask_captioning.campaign_runner.time.sleep") as sleep:
            self.assertEqual(wait_for_stage_merge("/unused", "sam3"), completed)
        self.assertEqual(merge.call_count, 2)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
