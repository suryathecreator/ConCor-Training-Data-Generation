from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from sam3_mask_captioning.artifact_store import pack_artifacts
from sam3_mask_captioning.campaign_manifest import extend_from_manifest, initialize_campaign
from sam3_mask_captioning.campaign_publish import publish_once, rebuild_site
from sam3_mask_captioning.campaign_runner import _success_path
from sam3_mask_captioning.io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from sam3_mask_captioning.one_rewrite_stage import ONE_REWRITE_CONTRACT_VERSION


try:
    import pyarrow  # noqa: F401
    import zstandard

    HAS_CAMPAIGN_EXTRAS = hasattr(zstandard, "ZstdCompressor")
except ImportError:
    HAS_CAMPAIGN_EXTRAS = False


@unittest.skipUnless(HAS_CAMPAIGN_EXTRAS, "campaign publication extras are optional")
class CampaignPublishTests(unittest.TestCase):
    def test_preview_can_use_completed_units_without_publishing_past_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            manifest_rows = []
            for index in range(3):
                image = source / f"image-{index}.png"
                Image.new("RGB", (8, 8), (index, 30, 60)).save(image)
                manifest_rows.append(
                    {
                        "image_id": f"image-{index}",
                        "image_path": str(image),
                        "source_context": {"pair_key": f"key-{index}"},
                    }
                )
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest_rows, manifest)
            campaign = root / "campaign"
            initialize_campaign(campaign, unit_size=1, preview_pairs=2)
            extend_from_manifest(campaign, manifest, target_total=3)

            # Match reviewed production rows: image review nests the durable
            # source index under raw_record instead of retaining it top-level.
            for index in range(3):
                selected = (
                    campaign
                    / "units"
                    / f"{index:06d}"
                    / "selected_images.jsonl"
                )
                rows = read_jsonl(selected)
                for row in rows:
                    source_index = row.pop("source_manifest_index")
                    row["raw_record"] = {
                        "source_manifest_index": source_index
                    }
                write_jsonl(rows, selected)

            for index in (1, 2):
                unit = campaign / "units" / f"{index:06d}"
                pair = {
                    "image_id": f"image-{index}",
                    "contract_version": ONE_REWRITE_CONTRACT_VERSION,
                    "included": True,
                    "source_image_path": "source.png",
                    "correspondence_overlay_path": "overlay.png",
                    "groups": [],
                    "first_pass_groups": [],
                }
                write_jsonl([pair], unit / "image_text_pairs.jsonl")
                write_jsonl([pair], unit / "bcc_validation_audit.jsonl")
                write_json(
                    {"stage": "bcc-rewrite", "unit_id": index},
                    _success_path(unit, "bcc-rewrite"),
                )

            def fake_report(_root, *, pairs_path, output_path, **_kwargs):
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_text("preview", encoding="utf-8")

            with patch(
                "sam3_mask_captioning.campaign_publish._copy_pair_assets",
                side_effect=lambda pair, *_args, **_kwargs: pair,
            ), patch(
                "sam3_mask_captioning.campaign_publish.write_bcc_html_report",
                side_effect=fake_report,
            ):
                result = publish_once(campaign)

            self.assertEqual(result["next_unit"], 0)
            self.assertEqual(result["pair_count"], 0)
            ready = __import__("json").loads(
                (campaign / "site" / "READY.json").read_text(encoding="utf-8")
            )
            self.assertEqual(ready["preview_pair_count"], 2)
            self.assertEqual(ready["preview_mode"], "completed_checkpoints")
            rows = read_jsonl(
                campaign / "site" / "data" / "preview-first-2.jsonl"
            )
            self.assertEqual([row["image_id"] for row in rows], ["image-1", "image-2"])
            self.assertEqual(
                [row["source_manifest_index"] for row in rows], [1, 2]
            )
            self.assertEqual(len({row["pair_index"] for row in rows}), 2)

    def test_publisher_builds_immutable_hundred_pair_shard_and_site(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            manifest_rows = []
            for index in range(100):
                image = source / f"image-{index}.png"
                Image.new("RGB", (8, 8), (index % 255, 30, 60)).save(image)
                manifest_rows.append(
                    {
                        "image_id": f"image-{index}",
                        "image_path": str(image),
                        "source_context": {"pair_key": f"key-{index}"},
                    }
                )
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest_rows, manifest)
            campaign = root / "campaign"
            initialize_campaign(campaign, unit_size=100)
            extend_from_manifest(campaign, manifest, target_total=100)
            unit = campaign / "units" / "000000"
            (unit / "masks").mkdir()
            (unit / "inverse_crops").mkdir()
            (unit / "correspondence_overlays").mkdir()
            pairs = []
            for index in range(100):
                mask_id = f"mask-{index}"
                Image.new("L", (8, 8), 255).save(unit / "masks" / f"{mask_id}.png")
                Image.new("RGB", (8, 8), (index % 255, 30, 60)).save(
                    unit / "inverse_crops" / f"{mask_id}.png"
                )
                Image.new("RGB", (8, 8), (40, index % 255, 80)).save(
                    unit / "correspondence_overlays" / f"image-{index}.png"
                )
                omitted = []
                if index == 0:
                    omitted_id = "omitted-0"
                    Image.new("L", (8, 8), 255).save(
                        unit / "masks" / f"{omitted_id}.png"
                    )
                    Image.new("RGB", (8, 8), (90, 120, 150)).save(
                        unit / "inverse_crops" / f"{omitted_id}.png"
                    )
                    omitted = [
                        {
                            "mask_id": omitted_id,
                            "overlay_number": 2,
                            "reason": "not_mentioned",
                            "main_candidate": "unused object",
                            "mask_caption": "an unused object",
                            "source_sam3_prompt": "object",
                            "inverse_background_rgb": [90, 120, 150],
                        }
                    ]
                first_groups = [
                    {
                        "mask_id": mask_id,
                        "text": ["The object"],
                        "char_spans": [[0, 10]],
                        "color_rgb": [220, 80, 60],
                        "main_candidate": "object",
                        "sam3_score": 0.9,
                    }
                ]
                first_caption = "The object rests here."
                first_tagged = (
                    f"[{mask_id}]The object[/{mask_id}] rests here."
                )
                if index == 0:
                    first_caption = "The object and the unused object rest here."
                    first_tagged = (
                        f"[{mask_id}]The object[/{mask_id}] and "
                        "[omitted-0]the unused object[/omitted-0] rest here."
                    )
                    first_groups.append(
                        {
                            "mask_id": "omitted-0",
                            "text": ["the unused object"],
                            "char_spans": [[15, 32]],
                            "color_rgb": [40, 120, 210],
                            "main_candidate": "unused object",
                            "sam3_score": 0.8,
                        }
                    )
                pair = {
                    "image_id": f"image-{index}",
                    "contract_version": ONE_REWRITE_CONTRACT_VERSION,
                    "included": True,
                    "quality_tier": "clean",
                    "first_pass_caption": first_caption,
                    "first_pass_raw": (
                        '{"reject":false,"tagged_caption":"' + first_tagged + '"}'
                    ),
                    "first_pass_groups": first_groups,
                    "caption": "The object rests here.",
                    "rewrite_raw": (
                        f'{{"reject":false,"tagged_caption":"[{mask_id}]The object[/{mask_id}] rests here."}}'
                    ),
                    "groups": [
                        {
                            "mask_id": mask_id,
                            "text": ["The object"],
                            "char_spans": [[0, 10]],
                            "color_rgb": [220, 80, 60],
                            "main_candidate": "object",
                            "sam3_score": 0.9,
                        }
                    ],
                    "omitted_masks": omitted,
                    "rewrite_metrics": {
                        "outcome": "same",
                        "before_issue_count": 0,
                        "after_issue_count": 0,
                        "after_fatal_count": 0,
                        "before_parseable": True,
                        "after_parseable": True,
                        "resolved": [],
                        "persisting": [],
                        "new": [],
                    },
                    "validation": {
                        "before_rewrite": {"parseable": True, "issues": []},
                        "after_rewrite": {"parseable": True, "issues": []},
                    },
                    "composite_statistics": {
                        "consistency_passed_mask_count": 1,
                        "canonical_mask_count": 1,
                        "final_linked_mask_count": 1,
                    },
                    "bcc_input_manifest": (
                        [
                            {"image_number": 1, "role": "original_image"},
                            {"image_number": 2, "role": "numbered_mask_overlay"},
                            {
                                "image_number": 3,
                                "role": "inverse_mask_crop",
                                "overlay_number": 1,
                                "mask_id": mask_id,
                            },
                            {
                                "image_number": 4,
                                "role": "inverse_mask_crop",
                                "overlay_number": 2,
                                "mask_id": "omitted-0",
                                "inverse_background_rgb": [90, 120, 150],
                            },
                        ]
                        if index == 0
                        else []
                    ),
                }
                pairs.append(pair)
            write_jsonl(pairs, unit / "image_text_pairs.jsonl")
            write_jsonl(pairs, unit / "bcc_validation_audit.jsonl")
            pack_artifacts(unit, ["masks", "inverse_crops"], unit / "artifacts" / "sam3.tar")
            pack_artifacts(
                unit,
                ["correspondence_overlays"],
                unit / "artifacts" / "bcc.tar",
            )
            write_json(
                {"stage": "bcc-rewrite", "unit_id": 0},
                _success_path(unit, "bcc-rewrite"),
            )

            first = publish_once(campaign)
            self.assertEqual(first["pair_count"], 100)
            self.assertEqual(first["refreshed_site_milestones"], [])
            shard = campaign / "published" / "pair_shards" / "pairs-000000000-000000099.jsonl.zst"
            parquet = shard.with_suffix("").with_suffix(".parquet")
            self.assertTrue(shard.is_file())
            self.assertTrue(parquet.is_file())
            self.assertTrue((campaign / "site" / "READY.json").is_file())
            page = campaign / "site" / "pages" / "pairs-000000000-000000099.html"
            document = page.read_text(encoding="utf-8")
            self.assertIn("One-rewrite checker audit", document)
            self.assertIn("Before rewrite", document)
            self.assertIn("After one rewrite", document)
            self.assertIn(
                "[omitted-0]the unused object[/omitted-0]", document
            )
            self.assertTrue(
                (
                    campaign
                    / "site"
                    / "assets"
                    / "000000000"
                    / "inverse-omitted-0.png"
                ).is_file()
            )
            page_rows = read_jsonl(
                campaign
                / "site"
                / "data"
                / "pairs-000000000-000000099.jsonl"
            )
            before_only = page_rows[0]["first_pass_groups"][1]
            self.assertEqual(before_only["mask_id"], "omitted-0")
            self.assertTrue(Path(before_only["mask_path"]).is_file())
            shard_hash = sha256_file(shard)

            page.write_text("stale", encoding="utf-8")
            rebuilt = rebuild_site(campaign, milestones=[0])
            self.assertEqual(rebuilt["refreshed_milestones"], [0])
            self.assertIn("Before rewrite", page.read_text(encoding="utf-8"))
            self.assertEqual(shard_hash, sha256_file(shard))

            second = publish_once(campaign)
            self.assertEqual(second["pair_count"], 100)
            self.assertEqual(second["refreshed_site_milestones"], [])
            self.assertEqual(shard_hash, sha256_file(shard))
            self.assertEqual(
                len(read_jsonl(campaign / "published" / "image_text_pairs.jsonl")),
                100,
            )


if __name__ == "__main__":
    unittest.main()
