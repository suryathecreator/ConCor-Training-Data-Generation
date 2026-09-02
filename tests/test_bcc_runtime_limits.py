from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sam3_mask_captioning.bcc_runtime_limits import (
    bcc_input_limit_error,
    generate_many_bcc_with_input_isolation,
)
from sam3_mask_captioning.campaign_manifest import extend_from_manifest, initialize_campaign
from sam3_mask_captioning.campaign_skip import finalize_quarantined_bcc_input_limits
from sam3_mask_captioning.io_utils import read_jsonl, write_json, write_jsonl


class InputLimitTests(unittest.TestCase):
    def test_only_context_limited_item_is_skipped(self):
        class Captioner:
            def generate_many_bcc(self, packets, prompts, seeds, **kwargs):
                if any("too-large" in prompt for prompt in prompts):
                    raise ValueError(
                        "The decoder prompt is longer than the maximum model length of 49152"
                    )
                return [{"raw": prompt} for prompt in prompts]

        items = [
            {"image_id": "good", "packet": ["a"], "prompt": "good", "seed": 1},
            {
                "image_id": "bad-context",
                "packet": ["b"],
                "prompt": "too-large",
                "seed": 2,
            },
            {
                "image_id": "bad-images",
                "packet": ["x"] * 129,
                "prompt": "otherwise-good",
                "seed": 3,
            },
        ]
        results, skipped = generate_many_bcc_with_input_isolation(
            Captioner(),
            items,
            generation_config={},
            max_images_per_prompt=128,
        )
        self.assertEqual(set(results), {"good"})
        self.assertEqual(set(skipped), {"bad-context", "bad-images"})
        self.assertIsNone(bcc_input_limit_error(RuntimeError("CUDA out of memory")))

    def test_legacy_quarantine_can_be_finalized_as_explicit_skips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(image)
            manifest = root / "manifest.jsonl"
            write_jsonl(
                [{"image_id": "one", "image_path": str(image), "source_context": {}}],
                manifest,
            )
            campaign = root / "campaign"
            initialize_campaign(campaign, unit_size=1, terminal_stage="bcc")
            extend_from_manifest(campaign, manifest, target_total=1)
            unit = campaign / "units" / "000000"
            write_jsonl(
                [
                    {"image_id": "one", "mask_id": f"mask-{index}"}
                    for index in range(10)
                ],
                unit / "bcc_canonical_captions.jsonl",
            )
            write_json(
                {
                    "stage": "bcc",
                    "error": "VLLMValidationError('At most 128 image(s) may be provided in one prompt.')",
                },
                unit / "stages" / "bcc" / "_QUARANTINED.json",
            )
            dry = finalize_quarantined_bcc_input_limits(campaign)
            self.assertFalse(dry["applied"])
            self.assertEqual(dry["skipped_unfinished_images"], 1)
            result = finalize_quarantined_bcc_input_limits(campaign, apply=True)
            self.assertTrue(result["applied"])
            self.assertTrue((unit / "stages" / "bcc" / "_SUCCESS.json").exists())
            self.assertFalse((unit / "stages" / "bcc" / "_QUARANTINED.json").exists())
            exclusion = read_jsonl(unit / "bcc_exclusions.jsonl")[0]
            self.assertEqual(
                exclusion["reason_code"],
                "bcc_input_too_large_or_interrupted_batch_tail",
            )


if __name__ == "__main__":
    unittest.main()
