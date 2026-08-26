from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from sam3_mask_captioning.caption_stage import run_captioning, run_mask_review
from sam3_mask_captioning.html_report import write_html_report
from sam3_mask_captioning.image_review_stage import _normalize as normalize_image_review
from sam3_mask_captioning.io_utils import append_jsonl, read_jsonl, read_jsonl_indexed, write_jsonl
from sam3_mask_captioning.manifest import write_excluded_sample
from sam3_mask_captioning.mask_utils import filter_candidates, inverse_crop_image


class FakeCaptioner:
    calls: list[tuple[list[list[str]], list[str], list[int]]] = []

    def __init__(self, config, config_section="caption"):
        self.config_section = config_section

    def generate_many(self, image_sets, prompts, seeds, batch_size=None, generation_config=None):
        self.calls.append((image_sets, prompts, seeds))
        if self.config_section == "quality_filter":
            raw = json.dumps({"keep": False, "reason": "too vague", "failure_modes": ["vague_caption"]})
        else:
            raw = json.dumps(
                {
                    "reject": False,
                    "reject_reason": "",
                    "object": "red cup",
                    "caption": "a red plastic cup",
                    "attributes": ["red", "plastic"],
                    "uncertain": False,
                }
            )
        return [{"raw": raw} for _ in image_sets]

    def caption(self, row, seed):
        return self.generate_many([[row["inverse_crop_path"]]], ["caption"], [seed])[0]

    def mask_review(self, row, seed, prompt_template):
        prompt = prompt_template
        for key, value in {
            "caption": row.get("caption", ""),
            "source_prompt": row.get("source_prompt", ""),
            "sam3_score": row.get("sam3_score", ""),
        }.items():
            prompt = prompt.replace("{" + key + "}", str(value))
        return self.generate_many([[row["inverse_crop_path"]]], [prompt], [seed])[0]


class PipelineContractTests(unittest.TestCase):
    def test_image_review_requires_prompts_and_20_entities(self):
        config = {"image_review": {"min_distinct_objects": 20, "borderline_max_distinct_objects": 24}}
        parsed = {
            "worth_segmenting": True,
            "estimated_maskable_entities": 21,
            "image_type": "natural_photo",
            "rationale": "many people and objects",
            "reject_reason": "",
            "sam3_prompts": [{"prompt": "people"}, "chairs", "people"],
        }
        normalized = normalize_image_review(parsed, config, "")
        self.assertTrue(normalized["accepted"])
        self.assertEqual(normalized["estimated_maskable_entities"], 21)
        self.assertEqual(normalized["sam3_prompts"], ["people", "chairs"])

        parsed["sam3_prompts"] = []
        normalized = normalize_image_review(parsed, config, "")
        self.assertFalse(normalized["accepted"])
        self.assertEqual(normalized["reject_reason"], "missing_sam3_prompts")

    def test_filters_keep_overlap_but_remove_near_exact_duplicates(self):
        mask_a = np.zeros((20, 20), dtype=bool)
        mask_a[2:12, 2:12] = True
        mask_b = mask_a.copy()
        mask_c = np.zeros((20, 20), dtype=bool)
        mask_c[5:18, 5:18] = True
        candidates = [
            {"raw_index": 0, "mask": mask_a, "bbox": [2, 2, 10, 10], "area": int(mask_a.sum()), "score": 0.9},
            {"raw_index": 1, "mask": mask_b, "bbox": [2, 2, 10, 10], "area": int(mask_b.sum()), "score": 0.8},
            {"raw_index": 2, "mask": mask_c, "bbox": [5, 5, 13, 13], "area": int(mask_c.sum()), "score": 0.7},
        ]
        kept, rejected = filter_candidates(
            candidates,
            image_area=400,
            min_area=1,
            min_area_fraction=0.0,
            dedupe_iou=0.98,
            min_bbox_fill=0.01,
            max_mask_area_fraction=1.0,
            max_bbox_area_fraction=1.0,
            containment_threshold=1.01,
            bbox_containment_threshold=1.01,
            contained_area_ratio=0.0,
            containment_score_margin=0.0,
            disable_containment=True,
            disable_dedupe_iou=False,
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual([row["reject_reason"] for row in rejected], ["duplicate_iou"])

    def test_inverse_crop_preserves_mask_pixels_and_uses_adaptive_nonmask_fill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = Image.new("RGB", (6, 6), (10, 20, 30))
            image_path = root / "image.png"
            image.save(image_path)
            mask = np.zeros((6, 6), dtype=bool)
            mask[2:4, 2:4] = True
            out_path = root / "inverse.png"
            selection = inverse_crop_image(image_path, mask, [2, 2, 2, 2], out_path, padding=1)
            arr = np.asarray(Image.open(out_path).convert("RGB"))
            self.assertTrue((arr[1:3, 1:3] == [10, 20, 30]).all())
            self.assertTrue((arr[0, 0] == selection["rgb"]).all())
            self.assertNotEqual(selection["rgb"], [10, 20, 30])
            self.assertGreater(selection["distance_q10"], 100)

    def test_caption_and_qa_use_inverse_crop_only_and_preserve_original_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inverse = root / "inverse.jpg"
            Image.new("RGB", (4, 4), (255, 0, 0)).save(inverse)
            mask = root / "mask.png"
            Image.new("L", (4, 4), 255).save(mask)
            rows = [
                {
                    "image_id": "img",
                    "mask_id": "img_m0001",
                    "bbox": [0, 0, 4, 4],
                    "area": 16,
                    "sam3_score": 0.9,
                    "source_prompt": "cups",
                    "source_image_path": str(inverse),
                    "mask_path": str(mask),
                    "full_overlay_path": str(inverse),
                    "crop_overlay_path": str(inverse),
                    "crop_image_path": str(inverse),
                    "inverse_crop_path": str(inverse),
                }
            ]
            run_dir = root / "run"
            write_jsonl(rows, run_dir / "sam3_masks.jsonl")
            config = {
                "random_seed": 7,
                "continue_on_error": False,
                "caption": {
                    "model_name": "Qwen/Qwen3.5-9B",
                    "input_image_keys": ["inverse_crop_path"],
                    "prompt": "caption",
                    "batch_size": 2,
                },
                "quality_filter": {
                    "enabled": True,
                    "model_name": "Qwen/Qwen3.5-9B",
                    "input_image_keys": ["inverse_crop_path"],
                    "mask_review_prompt": 'Return {"keep": true, "reason": "..."} for caption {caption} score {sam3_score} prompt {source_prompt}',
                    "batch_size": 2,
                },
            }
            FakeCaptioner.calls = []
            with mock.patch("sam3_mask_captioning.caption_stage.QwenCaptioner", FakeCaptioner):
                run_captioning(config, run_dir)
                run_mask_review(config, run_dir)
            self.assertEqual(FakeCaptioner.calls[0][0], [[str(inverse)]])
            self.assertEqual(FakeCaptioner.calls[1][0], [[str(inverse)]])
            self.assertIn('Return {"keep": true, "reason": "..."}', FakeCaptioner.calls[1][1][0])
            self.assertIn("caption A red plastic cup.", FakeCaptioner.calls[1][1][0])
            self.assertTrue((run_dir / "captions.jsonl").exists())
            self.assertEqual(read_jsonl(run_dir / "captions.jsonl"), [])
            rejected = read_jsonl(run_dir / "rejected_captions.jsonl")
            self.assertEqual(rejected[0]["original_caption"], "a red plastic cup")

    def test_mask_review_resume_retries_previous_review_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inverse = root / "inverse.jpg"
            Image.new("RGB", (4, 4), (255, 0, 0)).save(inverse)
            mask = root / "mask.png"
            Image.new("L", (4, 4), 255).save(mask)
            row = {
                "image_id": "img",
                "mask_id": "img_m0001",
                "bbox": [0, 0, 4, 4],
                "area": 16,
                "object": "red cup",
                "caption": "a red plastic cup",
                "attributes": ["red", "plastic"],
                "uncertain": False,
                "sam3_score": 0.9,
                "source_prompt": "cups",
                "source_image_path": str(inverse),
                "mask_path": str(mask),
                "full_overlay_path": str(inverse),
                "crop_overlay_path": str(inverse),
                "crop_image_path": str(inverse),
                "inverse_crop_path": str(inverse),
            }
            run_dir = root / "run"
            write_jsonl([row], run_dir / "caption_candidates.jsonl")
            write_jsonl(
                [
                    dict(
                        row,
                        mask_review_keep=False,
                        mask_review_reason='mask_review_error: KeyError("\\"keep\\"")',
                        mask_review_failure_modes=["review_error"],
                    )
                ],
                run_dir / "rejected_captions.jsonl",
            )
            config = {
                "random_seed": 7,
                "resume": True,
                "quality_filter": {
                    "enabled": True,
                    "model_name": "Qwen/Qwen3.5-9B",
                    "input_image_keys": ["inverse_crop_path"],
                    "mask_review_prompt": 'Return {"keep": true} for {caption}',
                    "batch_size": 2,
                },
            }
            FakeCaptioner.calls = []
            with mock.patch("sam3_mask_captioning.caption_stage.QwenCaptioner", FakeCaptioner):
                run_mask_review(config, run_dir)
            self.assertEqual(len(FakeCaptioner.calls), 1)
            self.assertIn("a red plastic cup", FakeCaptioner.calls[0][1][0])

    def test_html_stats_table_uses_sam3_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.jpg"
            Image.new("RGB", (4, 4), (1, 2, 3)).save(image)
            run_dir = root / "run"
            write_jsonl([{"image_id": "img", "image_path": str(image), "source_image_path": str(image)}], run_dir / "selected_images.jsonl")
            write_jsonl([{"image_id": "img", "accepted": True}], run_dir / "image_reviews.jsonl")
            write_jsonl([{"image_id": "img"}], run_dir / "sam3_masks.jsonl")
            write_jsonl([{"image_id": "img", "reject_reason": "small_area"}], run_dir / "sam3_rejected_masks.jsonl")
            write_jsonl([{"image_id": "img", "category": "second_pass_rejected", "source_image_path": str(image)}], run_dir / "image_categories.jsonl")
            out = write_html_report(run_dir, embed_images=False, max_images=0, masks_per_image=0)
            html = out.read_text(encoding="utf-8")
            self.assertIn("SAM3 kept masks", html)
            self.assertIn("SAM3 rejected masks", html)
            self.assertIn("final image acceptance rate", html)

    def test_manifest_exclusion_skips_image_id_and_pair_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.jsonl"
            exclude = root / "exclude.jsonl"
            out = root / "selected.jsonl"
            rows = [
                {"image_id": "skip-image", "image_path": "/x/a.jpg", "source_context": {"pair_key": "a"}},
                {"image_id": "skip-pair", "image_path": "/x/b.jpg", "source_context": {"pair_key": "skip-pair-key"}},
                {"image_id": "keep", "image_path": "/x/c.jpg", "source_context": {"pair_key": "c", "split": "test"}},
            ]
            write_jsonl(rows, raw)
            write_jsonl(
                [
                    {"image_id": "skip-image", "source_context": {"pair_key": "other"}},
                    {"image_id": "other", "source_context": {"pair_key": "skip-pair-key"}},
                ],
                exclude,
            )
            written = write_excluded_sample(raw, out, seed=123, limit=1, exclude_manifests=[exclude])
            selected = read_jsonl(out)
            self.assertEqual(written, 1)
            self.assertEqual(selected[0]["image_id"], "keep")
            self.assertEqual(selected[0]["source_context"]["split"], "test")

    def test_indexed_jsonl_reads_only_appends_and_resets_after_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            write_jsonl([{"value": 1}], path)
            self.assertEqual(read_jsonl_indexed(path), [{"value": 1}])
            append_jsonl({"value": 2}, path)
            self.assertEqual(read_jsonl_indexed(path), [{"value": 1}, {"value": 2}])
            write_jsonl([{"value": 3}], path)
            self.assertEqual(read_jsonl_indexed(path), [{"value": 3}])

if __name__ == "__main__":
    unittest.main()
