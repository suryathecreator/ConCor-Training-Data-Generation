from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from sam3_mask_captioning.campaign_manifest import extend_from_manifest, initialize_campaign
from sam3_mask_captioning.io_utils import read_jsonl, write_jsonl
from sam3_mask_captioning.stage_merge import merge_stage_jsonl


class ResumableIoTests(unittest.TestCase):
    def test_merge_resumes_after_durable_unit_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_rows = []
            for index in range(2):
                image = root / f"image-{index}.png"
                Image.new("RGB", (4, 4), (index, 2, 3)).save(image)
                manifest_rows.append({"image_id": f"image-{index}", "image_path": str(image)})
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest_rows, manifest)
            campaign = root / "campaign"
            initialize_campaign(campaign, unit_size=1)
            extend_from_manifest(campaign, manifest, target_total=2)
            for index in range(2):
                write_jsonl(
                    [{"image_id": f"image-{index}", "accepted": True}],
                    campaign / "units" / f"{index:06d}" / "image_reviews.jsonl",
                )

            import sam3_mask_captioning.stage_merge as stage_merge

            real_write_json = stage_merge.write_json
            interrupted = False

            def interrupt_after_first_cursor(data, path):
                nonlocal interrupted
                real_write_json(data, path)
                review_state = (data.get("streams") or {}).get("image_reviews.jsonl") or {}
                if (
                    not interrupted
                    and Path(path).name == ".merge_state.json"
                    and int(review_state.get("next_unit_id") or 0) == 1
                ):
                    interrupted = True
                    raise RuntimeError("simulated preemption")

            with mock.patch.dict(os.environ, {"BCC_MERGE_CHECKPOINT_UNITS": "1"}), mock.patch.object(
                stage_merge, "write_json", side_effect=interrupt_after_first_cursor
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated preemption"):
                    merge_stage_jsonl(campaign, "image-review", max_shard_bytes=1024)

            with mock.patch.dict(os.environ, {"BCC_MERGE_CHECKPOINT_UNITS": "1"}):
                result = merge_stage_jsonl(campaign, "image-review", max_shard_bytes=1024)
            output = campaign / "merged" / "image-review" / "image_reviews.jsonl"
            self.assertEqual([row["image_id"] for row in read_jsonl(output)], ["image-0", "image-1"])
            self.assertEqual(result["streams"][0]["rows"], 2)
            state = json.loads(
                (campaign / "merged" / "image-review" / ".merge_state.json").read_text()
            )
            self.assertTrue(state["complete"])


if __name__ == "__main__":
    unittest.main()
