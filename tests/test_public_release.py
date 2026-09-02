from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sam3_mask_captioning.campaign_manifest import extend_from_manifest, initialize_campaign
from sam3_mask_captioning.campaign_runner import _mask_rle, _success_path, merge_stage
from sam3_mask_captioning.dataset_export import _coco_compressed_counts, export_hf_dataset
from sam3_mask_captioning.io_utils import read_jsonl, write_json, write_jsonl
from sam3_mask_captioning.selection import load_exclusion_csv, read_source_manifest


class PublicReleaseTests(unittest.TestCase):
    def test_coco_rle_matches_reference_encoder(self):
        mask = np.zeros((23, 17), dtype=np.uint8, order="F")
        mask[1:8, 3:9] = 1
        mask[10:22, 0:2] = 1
        # Verified byte-for-byte against pycocotools 2.0.10.
        self.assertEqual(_coco_compressed_counts(mask), ":<;0>KG000000000g5")

    def test_csv_manifest_and_identifier_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            for name in ("keep-a.jpg", "skip-me.jpg", "keep-b.jpg"):
                Image.new("RGB", (8, 8), (10, 20, 30)).save(images / name)
            manifest = root / "images.csv"
            manifest.write_text(
                "image_id,file_name,source_dataset,split,pair_key,metadata_json\n"
                'a,keep-a.jpg,toy,train,pair-a,"{\"\"license\"\":\"\"ok\"\"}"\n'
                'skip,skip-me.jpg,toy,train,pair-skip,"{}"\n'
                'b,keep-b.jpg,toy,train,pair-b,"{}"\n',
                encoding="utf-8",
            )
            exclusion = root / "exclude.csv"
            exclusion.write_text("identifier\nskip-me.jpg\n", encoding="utf-8")
            values, provenance = load_exclusion_csv(exclusion)
            self.assertIn("skip-me", values)
            self.assertEqual(provenance["file_name"], "exclude.csv")
            self.assertNotIn(str(root), json.dumps(provenance))
            parsed = read_source_manifest(manifest)
            self.assertEqual(parsed[0]["source_context"]["source_dataset"], "toy")

            campaign = root / "campaign"
            initialize_campaign(campaign, unit_size=2, dataset="toy")
            extend_from_manifest(
                campaign,
                manifest,
                target_total=2,
                image_root=images,
                exclude_csv=exclusion,
            )
            selected = read_jsonl(campaign / "source_manifest.jsonl")
            self.assertEqual([row["image_id"] for row in selected], ["a", "b"])

    def test_merge_writes_semantic_shards_and_per_image_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "one.png"
            Image.new("RGB", (8, 8), (10, 20, 30)).save(image)
            manifest = root / "manifest.jsonl"
            write_jsonl(
                [{"image_id": "one", "image_path": str(image), "source_context": {"pair_key": "one"}}],
                manifest,
            )
            campaign = root / "campaign"
            initialize_campaign(campaign, unit_size=1)
            extend_from_manifest(campaign, manifest, target_total=1)
            unit = campaign / "units" / "000000"
            write_jsonl([{"image_id": "one", "accepted": True, "rationale": "usable"}], unit / "image_reviews.jsonl")
            write_json({"stage": "image-review", "unit_id": 0}, _success_path(unit, "image-review"))
            result = merge_stage(campaign, "image-review")
            self.assertEqual(result["outputs"]["streams"][0]["rows"], 1)
            self.assertTrue((campaign / "merged" / "image-review" / "image_reviews.jsonl").exists())
            ledger = read_jsonl(campaign / "reports" / "run_ledger.jsonl")
            self.assertEqual(ledger[0]["image_review_status"], "accepted")
            self.assertEqual(ledger[0]["last_completed_stage"], "image-review")
            self.assertTrue((campaign / "reports" / "run_ledger.csv").exists())

    def test_hf_views_are_parseability_and_count_only(self):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is optional")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_rows = []
            for index in range(4):
                image = root / f"image-{index}.png"
                Image.new("RGB", (8, 8), (index, 20, 30)).save(image)
                manifest_rows.append(
                    {"image_id": f"image-{index}", "image_path": str(image), "source_context": {"pair_key": f"key-{index}"}}
                )
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest_rows, manifest)
            campaign = root / "campaign"
            initialize_campaign(campaign, unit_size=4, terminal_stage="bcc")
            extend_from_manifest(campaign, manifest, target_total=4)
            unit = campaign / "units" / "000000"

            def audit(index: int, count: int, *, included: bool, parseable: bool = True):
                caption = "Objects are arranged." if parseable else "broken"
                return {
                    "image_id": f"image-{index}",
                    "included": included,
                    "caption": caption,
                    "groups": [
                        {"mask_id": f"mask-{index}-{group}", "text": ["Objects"], "char_spans": [[0, 7]]}
                        for group in range(count)
                    ],
                    "validation": {"after_rewrite": {"parseable": parseable, "issues": []}},
                }

            # Included flags intentionally disagree: routing must still use
            # parseability and mask count only.
            write_jsonl(
                [audit(0, 10, included=False), audit(1, 3, included=True), audit(2, 2, included=True, parseable=False)],
                unit / "bcc_validation_audit.jsonl",
            )
            mask_rows = []
            for index, count in ((0, 10), (1, 3), (2, 2)):
                mask = root / f"mask-{index}.png"
                Image.new("L", (8, 8), 255).save(mask)
                for group in range(count):
                    mask_rows.append(
                        {
                            "image_id": f"image-{index}",
                            "mask_id": f"mask-{index}-{group}",
                            "rle": _mask_rle(mask),
                        }
                    )
            write_jsonl(mask_rows, unit / "mask_rle.jsonl")
            write_json({"stage": "bcc", "unit_id": 0}, _success_path(unit, "bcc"))
            output = root / "export"
            stats = export_hf_dataset(campaign, output, include_image_bytes=False, shard_size=100)
            self.assertEqual(stats["min_10_masks"], 1)
            self.assertEqual(stats["masks_1_to_9"], 1)
            self.assertEqual(stats["parseable_1_plus"], 2)
            self.assertEqual(stats["processed_raw_images"], 4)
            min10 = pq.read_table(output / "data" / "min_10_masks" / "train-00000.parquet").to_pylist()
            low = pq.read_table(output / "data" / "masks_1_to_9" / "train-00000.parquet").to_pylist()
            audit_rows = pq.read_table(output / "data" / "audit_all_processed" / "train-00000.parquet").to_pylist()
            self.assertEqual(min10[0]["image_id"], "image-0")
            self.assertEqual(low[0]["image_id"], "image-1")
            self.assertEqual(len(audit_rows), 4)
            standard = pq.read_table(
                output / "train" / "gpic_min_10-00000.parquet"
            )
            self.assertEqual(
                standard.column_names,
                [
                    "dataset",
                    "split",
                    "image_key",
                    "image_id",
                    "height",
                    "width",
                    "caption",
                    "groups_json",
                    "masks_json",
                ],
            )
            standard_row = standard.to_pylist()[0]
            self.assertEqual(standard_row["height"], 8)
            self.assertEqual(standard_row["width"], 8)
            self.assertEqual(len(json.loads(standard_row["groups_json"])), 10)
            masks = json.loads(standard_row["masks_json"])
            self.assertEqual(len(masks), 10)
            self.assertTrue(all(isinstance(value["counts"], str) for value in masks.values()))


if __name__ == "__main__":
    unittest.main()
