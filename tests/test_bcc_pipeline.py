from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sam3_mask_captioning.bcc_html_report import write_bcc_html_report
from sam3_mask_captioning.bcc_canonicalization import canonicalize_bcc_rows
from sam3_mask_captioning.caption_cleanup import (
    caption_contact_relations,
    clean_attributes,
    clean_caption,
    extract_main_candidate,
)
from sam3_mask_captioning.caption_stage import qwen_model_config
from sam3_mask_captioning.caption_iteration import _audit_only_display_record
from sam3_mask_captioning.config import load_config
from sam3_mask_captioning.consistency_stage import run_sam3_consistency
from sam3_mask_captioning.correspondence_stage import (
    BCC_PROMPT_VERSION,
    CORRESPONDENCE_SCHEMA_VERSION,
    PIPELINE_STAGE_VERSION,
    _canonical_semantic_terms,
    _mask_context,
    build_caption_image_packet,
    build_caption_prompt,
    build_qa_prompt,
    build_schema_repair_prompt,
    normalize_correspondence,
    run_image_caption_pass,
    run_image_caption_pass_batch,
    run_image_caption_qa,
    run_image_caption_qa_batch,
)
from sam3_mask_captioning.io_utils import read_jsonl, write_jsonl
from sam3_mask_captioning.json_utils import extract_json
from sam3_mask_captioning.mask_utils import save_mask, select_inverse_background_rgb
from sam3_mask_captioning.pipeline_runner import (
    _iter_accepted_reviews,
    _terminal_image_ids,
    run_checkpointed_pipeline,
    run_correspondence_recovery,
)
from sam3_mask_captioning.sam3_stage import _xyxy_overlaps_regions
from sam3_mask_captioning.vllm_backend import _structured_schema_for


class BccPipelineTests(unittest.TestCase):
    def test_cached_semantic_terms_do_not_share_mutable_state(self):
        first = _canonical_semantic_terms("two wooden doors")
        self.assertIn("door", first)
        first.add("cache-poison")
        self.assertNotIn(
            "cache-poison", _canonical_semantic_terms("two wooden doors")
        )

    def test_bcc_canonicalization_drops_only_supported_same_instance_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks = []
            for name, stop in (("hand-a", 9), ("hand-b", 8), ("tree-a", 9), ("tree-b", 8)):
                mask = np.zeros((10, 10), dtype=bool)
                mask[1:stop, 1:stop] = True
                path = root / f"{name}.png"
                save_mask(mask, path)
                masks.append(path)
            rows = [
                {"image_id": "img", "mask_id": "hand-a", "mask_path": str(masks[0]), "main_candidate": "hand", "object": "hand", "caption": "A curled human hand.", "area": 64, "sam3_consistency": {"best_iou": 0.99}},
                {"image_id": "img", "mask_id": "hand-b", "mask_path": str(masks[1]), "main_candidate": "hand", "object": "hand", "caption": "A curled human hand.", "area": 49, "sam3_consistency": {"best_iou": 0.90}},
                {"image_id": "img", "mask_id": "tree-a", "mask_path": str(masks[2]), "main_candidate": "tree", "object": "trees", "caption": "Three leafy tree branches.", "area": 64, "sam3_consistency": {"best_iou": 0.99}},
                {"image_id": "img", "mask_id": "tree-b", "mask_path": str(masks[3]), "main_candidate": "tree", "object": "trees", "caption": "Several green trees.", "area": 49, "sam3_consistency": {"best_iou": 0.90}},
            ]
            kept, dropped = canonicalize_bcc_rows(rows, {})
            self.assertEqual([row["mask_id"] for row in kept], ["hand-a", "tree-a", "tree-b"])
            self.assertEqual([row["dropped_mask_id"] for row in dropped], ["hand-b"])
            self.assertEqual(kept[0]["bcc_duplicate_mask_aliases"][0]["dropped_mask_id"], "hand-b")
    def test_composite_mask_links_collective_span_instead_of_extra_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = np.zeros((12, 12), dtype=bool)
            right = np.zeros((12, 12), dtype=bool)
            left[2:4, 1:6] = True
            right[7:10, 7:10] = True
            masks = [left, right, np.logical_or(left, right)]
            paths = []
            for index, mask in enumerate(masks):
                path = root / f"ski-{index}.png"
                save_mask(mask, path)
                paths.append(path)
            rows = [
                {
                    "image_id": "img",
                    "mask_id": "ski-left",
                    "mask_path": str(paths[0]),
                    "main_candidate": "ski",
                    "object": "ski",
                    "caption": "One ski.",
                    "area": int(left.sum()),
                },
                {
                    "image_id": "img",
                    "mask_id": "ski-right",
                    "mask_path": str(paths[1]),
                    "main_candidate": "ski",
                    "object": "skis",
                    "caption": "A second ski.",
                    "area": int(right.sum()),
                },
                {
                    "image_id": "img",
                    "mask_id": "skis-both",
                    "mask_path": str(paths[2]),
                    "main_candidate": "ski",
                    "object": "skis",
                    "caption": "Both skis together.",
                    "area": int(masks[2].sum()),
                },
            ]
            kept, dropped = canonicalize_bcc_rows(rows, {})
            self.assertEqual(dropped, [])
            self.assertEqual(len(kept), 3)
            self.assertEqual(
                [
                    child["mask_id"]
                    for child in kept[2]["bcc_composite_mask_children"]
                ],
                ["ski-left", "ski-right"],
            )
            prompt = build_caption_prompt(kept)
            self.assertIn('"surface_identity_noun":"ski"', prompt)
            self.assertIn('"composite_of_ids":[1,2]', prompt)
            self.assertIn('"collective_candidate_ids":[1,2,3]', prompt)
            self.assertNotIn('"safe_tagged_phrase":', prompt)
            bad, bad_errors = normalize_correspondence(
                {
                    "tagged_caption": (
                        "[1]The first ski[/1] and [2]the second ski[/2] are part of "
                        "[3]the third ski[/3]."
                    )
                },
                kept,
                min_groups=3,
            )
            self.assertFalse(bad["composite_link_validation"]["valid"])
            self.assertTrue(any("composite mask ID 3" in error for error in bad_errors))
            good, good_errors = normalize_correspondence(
                {
                    "tagged_caption": (
                        "[3][1][2]Both skis[/2][/1][/3] lie side by side."
                    )
                },
                kept,
                min_groups=3,
            )
            self.assertEqual(good_errors, [])
            self.assertTrue(good["composite_link_validation"]["valid"])

            repaired, repaired_errors = normalize_correspondence(
                {
                    "tagged_caption": (
                        "[3][1]The first ski[/1] and [2]the second ski[/2] "
                        "lie side by side."
                    )
                },
                kept,
                min_groups=3,
            )
            self.assertFalse(
                any("unclosed opening tag" in error for error in repaired_errors),
                repaired_errors,
            )
            self.assertEqual(
                repaired["correspondence_encoding"][
                    "composite_outer_close_repairs"
                ],
                [
                    {
                        "id": 3,
                        "child_ids": [1, 2],
                        "inserted_tag": "[/3]",
                        "reason": (
                            "closed_geometry_proven_composite_after_all_child_tags"
                        ),
                    }
                ],
            )
            partial, partial_errors = normalize_correspondence(
                {
                    "tagged_caption": (
                        "[3][1]The first ski[/1] remains beside "
                        "[2]the second ski[/2]."
                    )
                },
                kept,
                min_groups=3,
            )
            self.assertTrue(
                any(
                    "link ID 3 has an unclosed opening tag" in error
                    for error in partial_errors
                ),
                partial_errors,
            )
            self.assertEqual(
                partial["correspondence_encoding"][
                    "composite_outer_close_repairs"
                ],
                [],
            )

            stutter, stutter_errors = normalize_correspondence(
                {
                    "tagged_caption": (
                        "[1]The first ski[/1], "
                        "[3][1]the first ski[/1] and [2]the second ski[/2][/3] "
                        "lie side by side."
                    )
                },
                kept,
                min_groups=3,
            )
            self.assertTrue(
                any("adjacent duplicate linked phrase" in error for error in stutter_errors),
                stutter_errors,
            )
            self.assertEqual(
                stutter["caption_quality"]["adjacent_duplicate_link_phrases"],
                ["The first ski, the first ski"],
            )
            stutter_repair = build_schema_repair_prompt(
                '{"keep":true,"tagged_caption":"[1]The first ski[/1], '
                '[3][1]the first ski[/1] and [2]the second ski[/2][/3]."}',
                stutter_errors,
                kept,
                qa=True,
            )
            self.assertIn(
                'FORBIDDEN_EXACT_PHRASES:\n["The first ski, the first ski"]',
                stutter_repair,
            )
            collective_stutter, collective_stutter_errors = normalize_correspondence(
                {
                    "tagged_caption": (
                        "[1]The first ski[/1] alongside [2]the second ski[/2] and "
                        "[3][1]the first ski[/1] and [2]the second ski[/2][/3] "
                        "lie side by side."
                    )
                },
                kept,
                min_groups=3,
            )
            self.assertTrue(
                any(
                    "repeats every child immediately outside its collective span" in error
                    for error in collective_stutter_errors
                ),
                collective_stutter_errors,
            )
            self.assertFalse(
                collective_stutter["composite_link_validation"]["valid"]
            )

            plural_rows = [
                {
                    "mask_id": "trees-left",
                    "main_candidate": "tree",
                    "object": "trees",
                    "bbox": [0, 0, 20, 20],
                    "bcc_significant_component_count": 2,
                },
                {
                    "mask_id": "trees-right",
                    "main_candidate": "tree",
                    "object": "trees",
                    "bbox": [80, 80, 20, 20],
                    "bcc_significant_component_count": 2,
                },
            ]
            plural_prompt = build_caption_prompt(plural_rows)
            self.assertIn('"collective_candidate_ids":[1,2]', plural_prompt)
            self.assertNotIn('"safe_tagged_phrase":', plural_prompt)
            _, plural_errors = normalize_correspondence(
                {
                    "tagged_caption": (
                        "[1]The first trees[/1] stand apart from [2]the second trees[/2]."
                    )
                },
                plural_rows,
                min_groups=2,
            )
            self.assertTrue(any("plural collection mask ID" in error for error in plural_errors))


    def test_production_dense_caption_stages_disable_thinking(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "qwen38_27b.yaml")
        self.assertFalse(config["image_caption"]["enable_thinking"])
        self.assertFalse(config["image_caption_qa"]["enable_thinking"])

    def test_all_stage_qwen38_config_uses_27b_min10_and_dense_output_headroom(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs" / "qwen38_27b.yaml"
        )
        qwen_sections = (
            "image_review",
            "caption",
            "quality_filter",
            "image_caption",
            "image_caption_audit",
            "image_caption_qa",
        )
        for section in qwen_sections:
            self.assertEqual(config[section]["model_name"], "Qwen/Qwen3.8-27B")
            self.assertFalse(config[section]["enable_thinking"])
        for section in ("image_caption", "image_caption_audit", "image_caption_qa"):
            self.assertEqual(config[section]["max_new_tokens_cap"], 8192)
        self.assertEqual(config["image_caption"]["min_input_masks"], 10)
        self.assertEqual(
            config["image_caption_qa"]["min_linked_masks_after_caption"], 10
        )

    def test_shared_vllm_engine_routes_structured_schema_per_call(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs" / "qwen38_27b.yaml"
        )
        mask_qa_runtime = qwen_model_config(config, "quality_filter")
        self.assertEqual(mask_qa_runtime["_schema_section"], "quality_filter")
        # The default simulates an engine constructed for mask captioning.
        schema = _structured_schema_for(mask_qa_runtime, "caption")
        self.assertIsNotNone(schema)
        self.assertIn("keep", schema["properties"])
        self.assertNotIn("reject", schema["properties"])

        draft_runtime = qwen_model_config(config, "image_caption")
        draft_schema = _structured_schema_for(draft_runtime, "quality_filter")
        self.assertIn("tagged_caption", draft_schema["properties"])

    def test_json_parser_selects_final_object_after_thinking_text(self):
        raw = (
            "<think>I first considered {\"reject\": true}, but the final record is below.</think>\n"
            '{"reject":false,"caption":"A cup.","groups":[{"mask_id":"cup"}]}'
        )
        parsed = extract_json(raw)
        self.assertFalse(parsed["reject"])
        self.assertEqual(parsed["caption"], "A cup.")


    def test_json_parser_repairs_only_trailing_container_delimiters(self):
        raw = (
            '{"reject":false,"caption":"A cup beside a shoe.",'
            '"links":[{"id":1,"text":["A cup"]},{"id":2,"text":["a shoe"]}]'
        )
        parsed = extract_json(raw)
        self.assertFalse(parsed["reject"])
        self.assertEqual(parsed["caption"], "A cup beside a shoe.")
        self.assertEqual(len(parsed["links"]), 2)
        with self.assertRaises(ValueError):
            extract_json('{"caption":"unterminated')

    def test_spacy_cleanup_removes_real_inverse_crop_leakage_patterns(self):
        cases = {
            "a sign displaying OPEN in white text against a black background": "A sign displaying OPEN in white text.",
            "close-up view of a scratched red metal box": "A scratched red metal box.",
            "a woman shown in profile view": "A woman.",
            "a yellow cup on a dark backdrop": "A yellow cup.",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                cleaned = clean_caption(source)
                self.assertEqual(cleaned["caption"], expected)
                self.assertTrue(cleaned["changed"])
        attributes = clean_attributes(["red", "black background", "profile view", "metal"])
        self.assertEqual(attributes["attributes"], ["red", "metal"])

    def test_spacy_main_candidate_prefers_concrete_head_and_source_fallback(self):
        extracted = extract_main_candidate(
            object_text="red cardboard box",
            caption="A scuffed red cardboard box.",
            source_prompt="boxes",
        )
        self.assertIn("box", extracted["candidate"])
        fallback = extract_main_candidate(
            object_text="object",
            caption="A small object.",
            source_prompt="wooden chairs",
        )
        self.assertIn("chair", fallback["candidate"])

    def test_contact_relation_parser_normalizes_inflected_verbs(self):
        relations = caption_contact_relations(
            "A man holds a cup while carrying a bag and gripping a rail."
        )
        self.assertEqual(
            [relation["lemma"] for relation in relations],
            ["hold", "carry", "grip"],
        )

    def test_adaptive_fill_is_far_from_foreground_distribution(self):
        dark_red = np.tile(np.asarray([[80, 5, 5]], dtype=np.uint8), (500, 1))
        selection = select_inverse_background_rgb(dark_red)
        self.assertNotEqual(selection["rgb"], [80, 5, 5])
        self.assertGreater(selection["distance_q10"], 150)
        self.assertEqual(selection["foreground_sample_count"], 500)

    def test_sam3_consistency_mock_uses_iou_and_same_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            Image.new("RGB", (20, 20), (20, 30, 40)).save(source)
            rows = []
            for index, bounds in enumerate(((2, 2, 8, 8), (11, 11, 17, 17))):
                mask = np.zeros((20, 20), dtype=bool)
                x0, y0, x1, y1 = bounds
                mask[y0:y1, x0:x1] = True
                mask_path = root / f"mask-{index}.png"
                save_mask(mask, mask_path)
                rows.append(
                    {
                        "image_id": "img",
                        "mask_id": f"m{index}",
                        "source_image_path": str(source),
                        "mask_path": str(mask_path),
                        "inverse_crop_path": str(source),
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "object": "person",
                        "caption": "A standing person.",
                        "source_prompt": "people",
                    }
                )
            config = {
                "resume": True,
                "consistency_filter": {
                    "mask_iou_threshold": 0.5,
                    "same_region_min_bbox_overlap": 0.1,
                },
            }
            run_sam3_consistency(config, root / "run", rows=rows, mock=True)
            passed = read_jsonl(root / "run" / "consistent_captions.jsonl")
            reviews = read_jsonl(root / "run" / "sam3_consistency_reviews.jsonl")
            self.assertEqual(len(passed), 2)
            self.assertTrue(all(review["passed"] for review in reviews))
            self.assertTrue(all(review["metric"] == "mask_iou" for review in reviews))
            self.assertTrue(all(review["best_iou"] == 1.0 for review in reviews))

    def test_sam3_region_prefilter_matches_smaller_bbox_overlap_rule(self):
        self.assertTrue(
            _xyxy_overlaps_regions([10, 10, 20, 20], [[12, 12, 4, 4]], 0.1)
        )
        self.assertFalse(
            _xyxy_overlaps_regions([10, 10, 20, 20], [[30, 30, 4, 4]], 0.1)
        )
        self.assertTrue(
            _xyxy_overlaps_regions([], [[30, 30, 4, 4]], 0.1),
            "missing boxes must be retained until mask-based fallback is possible",
        )

    def test_correspondence_repairs_repeated_mentions_and_requires_every_mask(self):
        caption = "A woman holds a cup while the woman raises the cup."
        rows = [{"mask_id": "woman"}, {"mask_id": "cup"}]
        parsed = {
            "caption": caption,
            "groups": [
                {
                    "mask_id": "woman",
                    "text": ["a woman", "a woman", "THE WOMAN"],
                    "char_spans": [[99, 106], [99, 108]],
                },
                {
                    "mask_id": "cup",
                    "text": ["A CUP", "THE CUP"],
                    "char_spans": [[14, 19], [45, 52]],
                },
            ],
        }
        normalized, errors = normalize_correspondence(parsed, rows, min_groups=2)
        self.assertEqual(errors, [])
        for group in normalized["groups"]:
            for span, text in zip(group["char_spans"], group["text"]):
                self.assertEqual(caption[span[0] : span[1]], text)
        _, missing_errors = normalize_correspondence(
            {"caption": "A woman.", "groups": [{"mask_id": "woman", "text": ["A woman"], "char_spans": [[0, 7]]}]},
            rows,
            min_groups=1,
        )
        self.assertTrue(any("accepted masks are missing groups" in value for value in missing_errors))
        _, invalid_group_errors = normalize_correspondence(
            {
                "caption": "A woman holds a cup.",
                "groups": [
                    {"mask_id": "woman", "text": ["A woman"]},
                    {"mask_id": "cup", "text": ["a missing mug"]},
                ],
            },
            rows,
            min_groups=1,
        )
        self.assertTrue(any("required link IDs: 2:unknown" in value for value in invalid_group_errors))


    def test_semantic_validator_rejects_hand_to_sneakers(self):
        rows = [
            {
                "mask_id": "hand",
                "main_candidate": "hand",
                "object": "hand",
                "source_prompt": "hands",
                "caption": "A black sneaker.",
            },
            {"mask_id": "shoe", "main_candidate": "shoe"},
        ]
        parsed = {
            "caption": "A hand rests beside a sneaker.",
            "links": [
                {"id": 1, "text": ["a sneaker"]},
                {"id": 2, "text": ["a sneaker"]},
            ],
        }
        _, errors = normalize_correspondence(parsed, rows, min_groups=2)
        self.assertTrue(
            any("conflicting concrete compound noun" in value for value in errors), errors
        )


    def test_semantic_validator_accepts_fist_for_hand(self):
        rows = [{"mask_id": "hand", "main_candidate": "hand"}]
        parsed = {
            "caption": "A clenched fist rests nearby.",
            "links": [{"id": 1, "text": ["A clenched fist"]}],
        }
        _, errors = normalize_correspondence(parsed, rows, min_groups=1)
        self.assertEqual(errors, [])

    def test_semantic_validator_does_not_confuse_compound_modifier_with_head(self):
        rows = [{"mask_id": "fern", "main_candidate": "fern", "object": "ferns"}]
        parsed = {
            "caption": "Several green fern fronds cluster together.",
            "links": [
                {
                    "id": 1,
                    "text": ["Several green fern fronds"],
                }
            ],
        }
        _, errors = normalize_correspondence(parsed, rows, min_groups=1)
        self.assertTrue(
            any("unmasked concrete noun phrase" in error for error in errors),
            errors,
        )

    def test_body_part_does_not_absorb_object_part_compound(self):
        rows = [
            {
                "mask_id": "human-neck",
                "main_candidate": "neck",
                "object": "neck",
            }
        ]
        normalized, errors = normalize_correspondence(
            {
                "tagged_caption": "The guitar neck sits near [1]the neck[/1].",
            },
            rows,
            min_groups=1,
        )
        self.assertTrue(any("The guitar neck" in error for error in errors), errors)
        self.assertFalse(
            any(
                repair["text"] == "The guitar neck"
                for repair in normalized["mention_completion_repairs"]
            )
        )

        _, explicit_errors = normalize_correspondence(
            {
                "tagged_caption": "[1]The guitar neck[/1] sits near [1]the neck[/1].",
            },
            rows,
            min_groups=1,
        )
        self.assertTrue(
            any("conflicting concrete compound noun" in error for error in explicit_errors),
            explicit_errors,
        )

    def test_semantic_validator_handles_plural_and_generic_clothing_aliases(self):
        rows = [
            {"mask_id": "eyewear", "main_candidate": "eyeglasses"},
            {"mask_id": "clothing", "main_candidate": "clothing"},
        ]
        parsed = {
            "caption": "Dark sunglasses sit beside a loose garment.",
            "links": [
                {"id": 1, "text": ["Dark sunglasses"]},
                {"id": 2, "text": ["a loose garment"]},
            ],
        }
        _, errors = normalize_correspondence(parsed, rows, min_groups=2)
        self.assertEqual(errors, [])

    def test_semantic_validator_uses_wordnet_same_synset_and_one_hop(self):
        rows = [
            {"mask_id": "bracelet", "main_candidate": "bracelet"},
            {"mask_id": "building", "main_candidate": "building"},
        ]
        parsed = {
            "tagged_caption": (
                "[1]A wristband[/1] lies beneath [2]a skyscraper[/2]."
            )
        }
        normalized, errors = normalize_correspondence(parsed, rows, min_groups=2)
        self.assertEqual(errors, [])
        self.assertEqual(
            [
                group["semantic_validation"]["taxonomy_matches"][0]["relation"]
                for group in normalized["groups"]
            ],
            ["same_synset", "expected_is_direct_hypernym"],
        )

    def test_semantic_validator_rejects_unrelated_wordnet_branches(self):
        _, errors = normalize_correspondence(
            {"tagged_caption": "[1]A banner[/1] hangs nearby."},
            [{"mask_id": "signage", "main_candidate": "signage"}],
            min_groups=1,
        )
        self.assertTrue(
            any("identity phrase is incompatible" in error for error in errors),
            errors,
        )


    def test_caption_quality_rejects_inventory_and_accepts_composition(self):
        rows = [
            {"mask_id": subject, "main_candidate": subject}
            for subject in ("cup", "shoe", "hat", "bag")
        ]
        inventory = {
            "caption": (
                "A cup is visible. A shoe is shown. "
                "A hat is present. A bag appears."
            ),
            "links": [
                {"id": 1, "text": ["A cup"]},
                {"id": 2, "text": ["A shoe"]},
                {"id": 3, "text": ["A hat"]},
                {"id": 4, "text": ["A bag"]},
            ],
        }
        normalized, errors = normalize_correspondence(
            inventory, rows, min_groups=4
        )
        self.assertFalse(normalized["caption_quality"]["valid"])
        self.assertTrue(
            any("one-sentence-per-mask inventory" in value for value in errors),
            errors,
        )
        self.assertTrue(
            any("repeated stock visibility" in value for value in errors),
            errors,
        )

        composed = {
            "caption": "A cup rests beside a shoe while a hat hangs from a bag.",
            "links": [
                {"id": 1, "text": ["A cup"]},
                {"id": 2, "text": ["a shoe"]},
                {"id": 3, "text": ["a hat"]},
                {"id": 4, "text": ["a bag"]},
            ],
        }
        normalized, errors = normalize_correspondence(
            composed, rows, min_groups=4
        )
        self.assertEqual(errors, [])
        self.assertTrue(normalized["caption_quality"]["valid"])
        self.assertEqual(
            normalized["caption_quality"]["max_linked_groups_per_sentence"], 4
        )

    def test_caption_quality_rejects_duplicate_identity_and_ordinal_catalog(self):
        duplicate_rows = [
            {"mask_id": "pants-left", "main_candidate": "pants"},
            {"mask_id": "pants-right", "main_candidate": "pants"},
        ]
        normalized, errors = normalize_correspondence(
            {
                "tagged_caption": (
                    "[1]The lower-left pants[/1] hang beside "
                    "[2]the lower-left pants[/2]."
                )
            },
            duplicate_rows,
            min_groups=2,
        )
        self.assertFalse(normalized["caption_quality"]["valid"])
        self.assertTrue(
            any("duplicate identity wording" in error for error in errors), errors
        )

        door_rows = [
            {"mask_id": f"door-{index}", "main_candidate": "door"}
            for index in range(4)
        ]
        _, errors = normalize_correspondence(
            {
                "tagged_caption": (
                    "[1]The first door[/1], [2]the second door[/2], "
                    "[3]the third door[/3], and [4]the fourth door[/4] form a row."
                )
            },
            door_rows,
            min_groups=4,
        )
        self.assertTrue(
            any("mechanical ordinal catalog" in error for error in errors), errors
        )

    def test_collective_plural_span_can_link_multiple_instance_masks(self):
        rows = [
            {"mask_id": f"cup-{index}", "main_candidate": "cup"}
            for index in range(3)
        ]
        normalized, errors = normalize_correspondence(
            {
                "tagged_caption": (
                    "[1][2][3]Several cups[/3][/2][/1] rest together."
                )
            },
            rows,
            min_groups=3,
        )
        self.assertEqual(errors, [])
        self.assertEqual(
            {tuple(group["char_spans"][0]) for group in normalized["groups"]},
            {(0, 12)},
        )

        _, singular_errors = normalize_correspondence(
            {"tagged_caption": "[1][2]A cup[/2][/1] rests nearby."},
            rows[:2],
            min_groups=2,
        )
        self.assertTrue(
            any("singular/noncollective phrase" in error for error in singular_errors),
            singular_errors,
        )

    def test_article_led_collective_span_can_link_multiple_instance_masks(self):
        cases = (
            ("An audience of people", "person"),
            ("A crowd of people", "person"),
            ("A herd of cows", "cow"),
        )
        for phrase, subject in cases:
            with self.subTest(phrase=phrase):
                rows = [
                    {"mask_id": f"{subject}-{index}", "main_candidate": subject}
                    for index in range(2)
                ]
                normalized, errors = normalize_correspondence(
                    {
                        "tagged_caption": (
                            f"[1][2]{phrase}[/2][/1] rests nearby."
                        )
                    },
                    rows,
                    min_groups=2,
                )
                self.assertEqual(errors, [])
                self.assertTrue(normalized["instance_number_validation"]["valid"])
                self.assertEqual(
                    normalized["caption_quality"]["duplicate_identity_phrases"],
                    [],
                )

    def test_caption_quality_rejects_body_part_ordinals_and_repairs_bad_punctuation(self):
        rows = [
            {"mask_id": "hair-left", "main_candidate": "hair"},
            {"mask_id": "hair-right", "main_candidate": "hair"},
        ]
        normalized, errors = normalize_correspondence(
            {
                "tagged_caption": (
                    "[1]The first hair[/1]., [2]the second hair[/2] curls."
                )
            },
            rows,
            min_groups=2,
        )
        self.assertFalse(normalized["caption_quality"]["valid"])
        self.assertTrue(
            any("ordinal body-part identity" in error for error in errors), errors
        )
        self.assertNotIn(".,", normalized["caption"])
        self.assertEqual(
            normalized["caption_quality"]["malformed_punctuation"], []
        )
        self.assertTrue(
            any(
                "repaired 1 malformed punctuation pair" in correction
                for correction in normalized["caption_cleanup"]["corrections"]
            )
        )

    def test_caption_quality_flags_fused_and_repeated_possessives(self):
        rows = [
            {"mask_id": "girl", "main_candidate": "girl"},
            {"mask_id": "hair", "main_candidate": "hair"},
        ]
        normalized, errors = normalize_correspondence(
            {
                "tagged_caption": (
                    "[1]A girl[/1] adjusts [2][1]her[/1] her hair[/2], "
                    "leaving theirhair loose."
                )
            },
            rows,
            min_groups=2,
        )
        self.assertFalse(normalized["caption_quality"]["valid"])
        self.assertIn("her her", normalized["caption_quality"]["possessive_boundary_errors"])
        self.assertIn("theirhair", normalized["caption_quality"]["possessive_boundary_errors"])
        self.assertTrue(
            any("fused or repeated possessive" in error for error in errors), errors
        )

    def test_hyphenated_side_selector_is_valid_for_body_part_identity(self):
        normalized, errors = normalize_correspondence(
            {"tagged_caption": "[1]The right-side hand[/1] rests nearby."},
            [{"mask_id": "hand-right", "main_candidate": "hand"}],
            min_groups=1,
        )
        self.assertEqual(errors, [])
        self.assertTrue(normalized["caption_quality"]["valid"])

    def test_contact_relation_geometry_rejects_only_gross_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.png"
            Image.new("RGB", (200, 100), (80, 100, 120)).save(source)
            base_rows = [
                {
                    "mask_id": "man",
                    "main_candidate": "man",
                    "bbox": [80, 10, 40, 85],
                    "source_image_path": str(source),
                },
                {
                    "mask_id": "bottle",
                    "main_candidate": "bottle",
                    "bbox": [0, 35, 10, 30],
                    "source_image_path": str(source),
                },
            ]
            parsed = {
                "tagged_caption": "[1]A man[/1] holds [2]a bottle[/2]."
            }
            normalized, errors = normalize_correspondence(
                parsed, base_rows, min_groups=2
            )
            self.assertFalse(normalized["caption_quality"]["valid"])
            self.assertTrue(
                any("unsupported contact relation" in error for error in errors),
                errors,
            )
            check = normalized["caption_quality"]["contact_relation_geometry"][0]
            self.assertFalse(check["supported_by_proximity"])
            self.assertGreater(check["normalized_gap"], 0.06)

            near_rows = [dict(row) for row in base_rows]
            near_rows[1]["bbox"] = [105, 35, 10, 30]
            normalized, errors = normalize_correspondence(
                parsed, near_rows, min_groups=2
            )
            self.assertEqual(errors, [])
            self.assertTrue(
                normalized["caption_quality"]["contact_relation_geometry"][0][
                    "supported_by_proximity"
                ]
            )

    def test_mask_context_uses_geometry_owners_and_unique_safe_phrases(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.png"
            Image.new("RGB", (300, 120), (80, 100, 120)).save(source)
            rows = [
                {"mask_id": "man", "main_candidate": "man", "bbox": [0, 0, 100, 115]},
                {"mask_id": "woman", "main_candidate": "woman", "bbox": [150, 0, 100, 115]},
                {"mask_id": "hair-man", "main_candidate": "hair", "bbox": [20, 5, 35, 20]},
                {"mask_id": "hair-woman", "main_candidate": "hair", "bbox": [175, 5, 35, 20]},
                {"mask_id": "pants-a", "main_candidate": "pants", "object": "pants", "bbox": [10, 75, 20, 30]},
                {"mask_id": "pants-b", "main_candidate": "pants", "object": "pants", "bbox": [50, 75, 20, 30]},
            ]
            for row in rows:
                row["source_image_path"] = str(source)
            context = _mask_context(rows)
            self.assertEqual(context[0]["safe_tag_phrase"], "the man")
            self.assertEqual(context[1]["safe_tag_phrase"], "the woman")
            self.assertEqual(context[2]["required_owner_id"], 1)
            self.assertEqual(context[3]["required_owner_id"], 2)
            self.assertEqual(context[2]["safe_tagged_phrase"], "[3][1]his[/1] hair[/3]")
            self.assertEqual(context[3]["safe_tagged_phrase"], "[4][2]her[/2] hair[/4]")
            self.assertNotEqual(
                context[4]["safe_tag_phrase"], context[5]["safe_tag_phrase"]
            )

    def test_terminal_image_ids_reopens_stale_bcc_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "status.jsonl"
            current = {
                "prompt_version": BCC_PROMPT_VERSION,
                "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                "stage_version": PIPELINE_STAGE_VERSION,
            }
            write_jsonl(
                [
                    {"image_id": "stable-mask-reject", "status": "rejected", "stage": "caption"},
                    {"image_id": "old-qa", "status": "accepted", "stage": "image_caption_qa"},
                    {"image_id": "current-qa", "status": "accepted", "stage": "image_caption_qa", **current},
                    {"image_id": "old-canonical", "status": "rejected", "stage": "bcc_canonicalization", "prompt_version": "old"},
                    {"image_id": "current-canonical", "status": "rejected", "stage": "bcc_canonicalization", **current},
                ],
                status_path,
            )
            self.assertEqual(
                _terminal_image_ids(status_path),
                {"stable-mask-reject", "current-qa", "current-canonical"},
            )

    def test_redundant_nested_mentions_keep_the_full_noun_phrase(self):
        rows = [{"mask_id": "cup", "main_candidate": "cup"}]
        parsed = {
            "caption": "A bright green cup rests.",
            "links": [{"id": 1, "text": ["green cup", "A bright green cup"]}],
        }
        normalized, errors = normalize_correspondence(parsed, rows, min_groups=1)
        self.assertEqual(errors, [])
        group = normalized["groups"][0]
        self.assertEqual(group["text"], ["A bright green cup"])
        self.assertEqual(len(group["mention_repairs"]), 1)
    def test_mentions_use_token_boundaries_and_drop_nested_noun_aliases(self):
        rows = [
            {"mask_id": "person", "main_candidate": "person"},
            {"mask_id": "number", "main_candidate": "number"},
            {"mask_id": "ski", "main_candidate": "ski"},
        ]
        caption = "A person checks the number. He carries a long ski."
        parsed = {
            "caption": caption,
            "links": [
                {"id": 1, "text": ["A person", "he"]},
                {"id": 2, "text": ["the number"]},
                {"id": 3, "text": ["ski", "a long ski"]},
            ],
        }
        normalized, errors = normalize_correspondence(parsed, rows, min_groups=3)
        self.assertEqual(errors, [])
        groups = {group["mask_id"]: group for group in normalized["groups"]}
        self.assertEqual(groups["person"]["text"], ["A person", "He"])
        person_span = groups["person"]["char_spans"][1]
        self.assertEqual(caption[person_span[0] : person_span[1]], "He")
        self.assertEqual(groups["ski"]["text"], ["a long ski"])
        self.assertTrue(
            any(
                repair["reason"] == "redundant_lexically_nested_model_mention"
                for repair in groups["ski"]["mention_repairs"]
            )
        )


    def test_schema_noise_drops_unmatched_alias_and_unknown_extra_id(self):
        rows = [{"mask_id": "cup", "main_candidate": "cup"}]
        parsed = {
            "caption": "A green cup rests.",
            "links": [{"id": 1, "text": ["A green cup", "a missing mug"]}, {"id": 99, "text": ["A green cup"]}],
        }
        normalized, errors = normalize_correspondence(parsed, rows, min_groups=1)
        self.assertEqual(errors, [])
        group = normalized["groups"][0]
        self.assertEqual(group["text"], ["A green cup"])
        self.assertTrue(any(item["reason"] == "unmatched_model_mention_dropped" for item in group["mention_repairs"]))
        self.assertEqual(normalized["link_repairs"][0]["reason"], "unknown_extra_link_dropped")

    def test_same_type_instances_allow_collective_or_distinct_mentions(self):
        rows = [
            {"mask_id": "shoe-left", "main_candidate": "shoe"},
            {"mask_id": "shoe-right", "main_candidate": "shoe"},
        ]
        caption = "Two shoes sit together: one white shoe and the other black shoe."
        valid = {
            "caption": caption,
            "links": [{"id": 1, "text": ["Two shoes", "one white shoe"]}, {"id": 2, "text": ["Two shoes", "the other black shoe"]}],
        }
        _, errors = normalize_correspondence(valid, rows, min_groups=2)
        self.assertEqual(errors, [])
        collective = {
            "caption": "Two shoes sit together.",
            "links": [{"id": 1, "text": ["Two shoes"]}, {"id": 2, "text": ["Two shoes"]}],
        }
        _, errors = normalize_correspondence(collective, rows, min_groups=2)
        self.assertEqual(errors, [])
        singular = {
            "caption": "A shoe sits nearby.",
            "links": [{"id": 1, "text": ["A shoe"]}, {"id": 2, "text": ["A shoe"]}],
        }
        _, errors = normalize_correspondence(singular, rows, min_groups=2)
        self.assertTrue(
            any("singular/noncollective phrase" in value for value in errors), errors
        )
        _, malformed_collective_errors = normalize_correspondence(
            {
                "caption": "Both shoe sit nearby.",
                "links": [
                    {"id": 1, "text": ["Both shoe"]},
                    {"id": 2, "text": ["Both shoe"]},
                ],
            },
            rows,
            min_groups=2,
        )
        self.assertTrue(
            any(
                "singular/noncollective phrase" in value
                for value in malformed_collective_errors
            ),
            malformed_collective_errors,
        )

    def test_mention_coverage_rejects_missing_coreference_and_unmasked_entity(self):
        rows = [
            {"mask_id": "man", "main_candidate": "man"},
            {"mask_id": "pants", "main_candidate": "pants"},
        ]
        parsed = {
            "caption": "A man wears dark pants. His loose pants cover a foot.",
            "links": [
                {"id": 1, "text": ["A man"]},
                {"id": 2, "text": ["dark pants"]},
            ],
        }
        normalized, errors = normalize_correspondence(parsed, rows, min_groups=2)
        self.assertFalse(normalized["caption_quality"]["mention_coverage"]["valid"])
        self.assertTrue(
            any("unmasked concrete noun phrase 'a foot'" in value for value in errors),
            errors,
        )
        completed = normalized["mention_completion_repairs"]
        self.assertTrue(any(item["text"] == "His" for item in completed), completed)
        self.assertTrue(
            any(item["text"] == "His loose pants" for item in completed),
            completed,
        )

    def test_mention_completion_never_guesses_between_repeated_masks(self):
        rows = [
            {"mask_id": "shirt-a", "main_candidate": "shirt"},
            {"mask_id": "shirt-b", "main_candidate": "shirt"},
        ]
        parsed = {
            "caption": "The red shirt lies beside the blue shirt; the shirt is folded.",
            "links": [
                {"id": 1, "text": ["The red shirt"]},
                {"id": 2, "text": ["the blue shirt"]},
            ],
        }
        normalized, errors = normalize_correspondence(parsed, rows, min_groups=2)
        self.assertTrue(any("'the shirt'" in value for value in errors), errors)
        self.assertFalse(
            any(
                item["text"] == "the shirt"
                for item in normalized["mention_completion_repairs"]
            )
        )

    def test_mention_coverage_uses_entity_head_and_ignores_abstract_surface(self):
        rows = [{"mask_id": "car", "main_candidate": "car"}]
        _, errors = normalize_correspondence(
            {
                "caption": "A car passes a dark car mirror.",
                "links": [{"id": 1, "text": ["A car"]}],
            },
            rows,
            min_groups=1,
        )
        self.assertTrue(any("'a dark car mirror'" in value for value in errors), errors)
        normalized, errors = normalize_correspondence(
            {
                "caption": "A table has a smooth surface.",
                "links": [{"id": 1, "text": ["A table"]}],
            },
            [{"mask_id": "table", "main_candidate": "table"}],
            min_groups=1,
        )
        self.assertEqual(errors, [])
        ignored = normalized["caption_quality"]["mention_coverage"][
            "ignored_abstract_mentions"
        ]
        self.assertTrue(
            any(item["text"] == "a smooth surface" for item in ignored),
            ignored,
        )

    def test_mention_coverage_accepts_cross_linked_possessive_and_repetition(self):
        rows = [
            {"mask_id": "man", "main_candidate": "man"},
            {"mask_id": "pants", "main_candidate": "pants"},
        ]
        parsed = {
            "caption": "A man wears dark pants. His loose pants suit him.",
            "links": [
                {"id": 1, "text": ["A man", "His", "him"]},
                {"id": 2, "text": ["dark pants", "His loose pants"]},
            ],
        }
        normalized, errors = normalize_correspondence(parsed, rows, min_groups=2)
        self.assertEqual(errors, [])
        coverage = normalized["caption_quality"]["mention_coverage"]
        self.assertTrue(coverage["valid"])
        self.assertEqual(coverage["unlinked_mentions"], [])

    def test_inline_tagged_caption_decodes_nested_exact_spans(self):
        rows = [
            {"mask_id": "man", "main_candidate": "man"},
            {"mask_id": "hand", "main_candidate": "hand"},
        ]
        prompt = build_caption_prompt(rows)
        self.assertNotIn('"safe_tagged_phrase":', prompt)
        self.assertIn('"required_owner_id":1', prompt)
        parsed = {
            "reject": False,
            "tagged_caption": (
                "[1]A man[/1] raises [2][1]his[/1] hand[/2]."
            ),
        }
        normalized, errors = normalize_correspondence(
            parsed, rows, min_groups=2
        )
        self.assertEqual(errors, [])
        self.assertEqual(normalized["caption"], "A man raises his hand.")
        self.assertEqual(
            normalized["correspondence_encoding"]["type"], "inline_tags"
        )
        groups = {group["mask_id"]: group for group in normalized["groups"]}
        self.assertEqual(groups["man"]["text"], ["A man", "his"])
        self.assertEqual(groups["hand"]["text"], ["his hand"])
        for group in groups.values():
            for span, text in zip(
                group["char_spans"], group["text"], strict=True
            ):
                self.assertEqual(
                    normalized["caption"][span[0] : span[1]], text
                )

    def test_inline_caption_repairs_sentence_case_and_trailing_gesture_parse(self):
        rows = [
            {"mask_id": "man", "main_candidate": "man", "object": "man"},
            {"mask_id": "hand", "main_candidate": "hand", "object": "hand"},
        ]
        normalized, errors = normalize_correspondence(
            {
                "tagged_caption": (
                    "[1]A man[/1] waits while [2][1]his[/1] hand[/2] gestures. [1]he[/1] smiles."
                )
            },
            rows,
            min_groups=2,
        )
        self.assertEqual(errors, [])
        self.assertEqual(normalized["caption"], "A man waits while his hand gestures. He smiles.")
        self.assertTrue(
            normalized["correspondence_encoding"]["sentence_case_repaired"]
        )
        groups = {group["mask_id"]: group for group in normalized["groups"]}
        self.assertEqual(groups["man"]["text"], ["A man", "his", "He"])
        self.assertEqual(groups["hand"]["text"], ["his hand"])

    def test_partial_inline_identity_expands_but_malformed_selector_still_fails(self):
        rows = [
            {"mask_id": "man", "main_candidate": "man", "object": "man"},
            {"mask_id": "shirt-a", "main_candidate": "shirt", "object": "shirt"},
            {"mask_id": "shirt-b", "main_candidate": "shirt", "object": "shirt"},
        ]
        normalized, errors = normalize_correspondence(
            {
                "tagged_caption": (
                    "[1]A man[/1] wears one [2]patterned shirt[/2] "
                    "with [3]the other shirt[/3]."
                )
            },
            rows,
            min_groups=3,
        )
        self.assertEqual(errors, [])
        groups = {group["mask_id"]: group for group in normalized["groups"]}
        self.assertEqual(groups["shirt-a"]["text"], ["one patterned shirt"])
        self.assertEqual(
            normalized["noun_phrase_span_repairs"][0]["replaced_spans"],
            [[16, 31]],
        )

        malformed, malformed_errors = normalize_correspondence(
            {
                "tagged_caption": (
                    "[1]A man[/1] wears a patterned [2]one shirt[/2] "
                    "with [3]the other shirt[/3]."
                )
            },
            rows,
            min_groups=3,
        )
        self.assertTrue(
            any(
                "malformed repeated-instance selector phrase" in error
                for error in malformed_errors
            ),
            malformed_errors,
        )
        self.assertEqual(
            malformed["caption_quality"]["malformed_selector_phrases"],
            ["a patterned one shirt"],
        )

    def test_inline_tags_reject_crossing_and_accept_nonhuman_reference(self):
        _, malformed_errors = normalize_correspondence(
            {
                "tagged_caption": "[1]A tree[/2]",
                "links": [{"id": 1, "text": ["A tree"]}],
            },
            [{"mask_id": "tree", "main_candidate": "tree"}],
            min_groups=1,
        )
        self.assertTrue(
            any("inline tags are crossed" in value for value in malformed_errors),
            malformed_errors,
        )
        normalized, errors = normalize_correspondence(
            {
                "tagged_caption": "[1]A tree[/1] sways as [1]it[/1] bends.",
            },
            [{"mask_id": "tree", "main_candidate": "tree"}],
            min_groups=1,
        )
        self.assertEqual(errors, [])
        self.assertEqual(normalized["groups"][0]["text"], ["A tree", "it"])

        normalized, errors = normalize_correspondence(
            {
                "tagged_caption": (
                    "[1]The cloud[/1] hovers above [2]the tree[/2]."
                ),
            },
            [
                {"mask_id": "cloud", "main_candidate": "cloud"},
                {"mask_id": "tree", "main_candidate": "tree"},
            ],
            min_groups=2,
        )
        self.assertEqual(errors, [])
        self.assertEqual(normalized["groups"][0]["text"], ["The cloud"])

    def test_correspondence_prompts_omit_model_generated_offsets(self):
        rows = [
            {
                "mask_id": "cup",
                "source_prompt": "cups",
                "main_candidate": "cup",
                "caption": "A green cup.",
            }
        ]
        caption_prompt = build_caption_prompt(rows)
        self.assertIn("ten is not a caption-coverage target", caption_prompt.casefold())
        self.assertIn("do not target a numeric link quota", caption_prompt.casefold())
        self.assertNotIn("at least ten linked masks", caption_prompt.casefold())
        self.assertNotIn("fewer cannot be published", caption_prompt.casefold())
        self.assertIn("do not output a separate links list", caption_prompt.casefold())
        self.assertIn("character offsets", caption_prompt.casefold())
        self.assertIn('"tagged_caption":', caption_prompt)
        self.assertNotIn('"char_spans":', caption_prompt)
        candidate = {
            "caption": "A green cup.",
            "groups": [
                {
                    "mask_id": "cup",
                    "text": ["a GREEN cup."],
                    "char_spans": [[0, 11]],
                    "mask_path": "/tmp/large-unneeded-mask.png",
                }
            ],
        }
        qa_prompt = build_qa_prompt(candidate, rows)
        self.assertIn("do not target a numeric link quota", qa_prompt.casefold())
        self.assertNotIn("at least ten linked masks", qa_prompt.casefold())
        self.assertNotIn("fewer cannot be published", qa_prompt.casefold())
        self.assertIn(
            '{"keep":true,"reason_code":"ok","tagged_caption":',
            qa_prompt,
        )
        self.assertNotIn('{"reject":false', qa_prompt)
        proposed_record = qa_prompt.split("PROPOSED_RECORD:\n", 1)[1].split("\nACCEPTED_MASK_CONTEXT:", 1)[0]
        self.assertNotIn("char_spans", proposed_record)
        self.assertNotIn("mask_path", proposed_record)
        normalized, errors = normalize_correspondence(candidate, rows, min_groups=1)
        self.assertEqual(errors, [])
        self.assertEqual(normalized["groups"][0]["char_spans"], [[0, 11]])
        self.assertEqual(normalized["groups"][0]["text"], ["A green cup"])
        background_candidate = {
            "caption": "A green cup is visible in the background.",
            "groups": [{"mask_id": "cup", "text": ["a GREEN cup."]}],
        }
        normalized, errors = normalize_correspondence(background_candidate, rows, min_groups=1)
        self.assertEqual(errors, [])
        self.assertEqual(normalized["caption"], "A green cup is present.")
        self.assertEqual(normalized["groups"][0]["char_spans"], [[0, 11]])

        # Removing a forbidden sentence-leading framing phrase must not turn
        # the model's correct punctuation into ``.,`` or stale linked spans.
        normalized, errors = normalize_correspondence(
            {
                "reject": False,
                "tagged_caption": (
                    "[1]Grey clouds[/1] hang over water in the distance. "
                    "In the foreground, [2]the wet beach[/2] slopes gently."
                ),
            },
            [
                {"mask_id": "cloud", "main_candidate": "cloud"},
                {"mask_id": "beach", "main_candidate": "beach"},
            ],
            min_groups=2,
        )
        self.assertFalse(any("span" in error for error in errors))
        self.assertEqual(
            normalized["caption"],
            "Grey clouds hang over water in the distance. The wet beach slopes gently.",
        )
        self.assertNotIn(".,", normalized["caption"])
        beach_group = next(
            group for group in normalized["groups"] if group["mask_id"] == "beach"
        )
        self.assertEqual(beach_group["text"], ["The wet beach"])
        start, end = beach_group["char_spans"][0]
        self.assertEqual(normalized["caption"][start:end], "The wet beach")

        context_cases = {
            "In the foreground, [1]a person[/1] gestures toward the scene.":
                "A person gestures.",
            "[1]Trees[/1] line the background.": "Trees are visible.",
            "A low-angle view captures [1]a tower[/1].": "There is a tower.",
            "In the immediate foreground, [1]a bottle[/1] stands.":
                "A bottle stands.",
        }
        for tagged_caption, expected_caption in context_cases.items():
            with self.subTest(tagged_caption=tagged_caption):
                identity = next(
                    noun
                    for noun in ("person", "tree", "tower", "bottle")
                    if noun in tagged_caption.casefold()
                )
                cleaned, context_errors = normalize_correspondence(
                    {"reject": False, "tagged_caption": tagged_caption},
                    [{"mask_id": "entity", "main_candidate": identity}],
                    min_groups=1,
                )
                self.assertFalse(
                    any("span" in error for error in context_errors),
                    context_errors,
                )
                self.assertEqual(cleaned["caption"], expected_caption)
                group = cleaned["groups"][0]
                span_start, span_end = group["char_spans"][0]
                self.assertEqual(
                    cleaned["caption"][span_start:span_end],
                    group["text"][0],
                )
        repaired, errors = normalize_correspondence(
            {
                "caption": "A green cup.",
                "groups": [{"mask_id": "corrupt_p000_m0000", "text": ["A green cup"]}],
            },
            [{"mask_id": "full-image-hash_p000_m0000"}],
            min_groups=1,
        )
        self.assertEqual(errors, [])
        self.assertEqual(repaired["groups"][0]["mask_id"], "full-image-hash_p000_m0000")
        self.assertEqual(len(repaired["mask_id_repairs"]), 1)

    def test_text_repair_exposes_available_ids_without_requiring_all(self):
        subjects = ["t shirt"] * 5 + ["shorts"] * 5 + ["sock"] * 3
        rows = [
            {
                "mask_id": f"mask-{index + 1}",
                "main_candidate": subject,
                "bbox": [index, index, 2, 2],
            }
            for index, subject in enumerate(subjects)
        ]
        errors = [
            f"link ID {index + 1} mention text is absent from caption"
            for index in range(30)
        ] + [
            (
                "caption mention coverage: unmasked concrete noun phrase "
                "'his mouth' at [1,10); omit it"
            ),
            "only 3 valid groups; need at least 10",
            (
                "10 accepted masks are missing groups; required link IDs: "
                "1:t shirt, 2:t shirt, 3:t shirt, 7:sock, 8:sock, 9:sock, "
                "10:shorts, 11:t shirt, 12:shorts, 13:t shirt"
            ),
        ]
        prompt = build_schema_repair_prompt("{}", errors, rows, qa=True)
        self.assertIn("AVAILABLE_LINK_COUNT:\n13", prompt)
        self.assertIn("AVAILABLE_LINK_IDS:\n[1,2,3,4,5,6,7,8,9,10,11,12,13]", prompt)
        self.assertIn(
            (
                '{"id":13,"subject_anchor":"sock",'
                '"surface_identity_noun":"sock",'
                '"allowed_identity_nouns":["sock"],'
                '"required_owner_id":null,'
                '"collective_candidate_ids":[11,12,13],'
                '"composite_of_ids":[],'
                '"significant_component_count":1}'
            ),
            prompt,
        )
        self.assertIn(
            "UNUSED_LINK_IDS_AFTER_PREVIOUS_ANSWER:\n"
            "[1,2,3,4,5,6,7,8,9,10,11,12,13]",
            prompt,
        )
        self.assertLess(
            prompt.index("accepted masks are missing groups"),
            prompt.index("link ID 1 mention"),
        )
        self.assertIn("they are not all mandatory", prompt)
        self.assertIn("[5][6][7]three cups[/7][/6][/5]", prompt)
        self.assertNotIn('"safe_tagged_phrase":', prompt)
        self.assertNotIn('"fallback_unique_selector":', prompt)
        self.assertIn('"collective_candidate_ids":[1,2,3,4,5]', prompt)
        self.assertIn("ALLOWED_IDENTITY_NOUNS_BY_ID:", prompt)
        self.assertNotIn("\nACCEPTED_MASK_CONTEXT:", prompt)
        self.assertGreater(
            prompt.index("FINAL_CHECK:"),
            prompt.index("PREVIOUS_VISUAL_ANSWER:"),
        )
        self.assertIn("never a patterned [6]one shirt[/6]", prompt)
        self.assertIn('FORBIDDEN_EXACT_PHRASES:\n["his mouth"]', prompt)
        self.assertNotIn('"required_unique_selector":', prompt)
        self.assertNotIn("Three shoes are visible", prompt)
        self.assertNotIn("Three shoes rest together", prompt)

        plural_prompt = build_caption_prompt(
            [{"mask_id": "pants", "main_candidate": "pant", "object": "pants"}]
        )
        self.assertIn('"surface_identity_noun":"pants"', plural_prompt)
        self.assertNotIn('"safe_tag_phrase":', plural_prompt)
        boot_rows = [
            {"mask_id": "left-boot", "main_candidate": "ski boot", "object": "ski boot", "bcc_significant_component_count": 1},
            {"mask_id": "right-boot", "main_candidate": "ski boot", "object": "ski boots", "bcc_significant_component_count": 1},
        ]
        repeated_boot_prompt = build_caption_prompt(boot_rows)
        self.assertIn('"significant_component_count":1', repeated_boot_prompt)
        self.assertEqual(
            repeated_boot_prompt.count('"surface_identity_noun":"ski boot"'),
            2,
        )
        self.assertIn('"collective_candidate_ids":[1,2]', repeated_boot_prompt)
        self.assertNotIn('"safe_tagged_phrase":', repeated_boot_prompt)
        _, bad_number_errors = normalize_correspondence(
            {
                "tagged_caption": (
                    "[1]One ski boot[/1] stands beside "
                    "[2]the other ski boots[/2]."
                )
            },
            boot_rows,
            min_groups=2,
        )
        self.assertTrue(
            any("grammatical number" in error for error in bad_number_errors),
            bad_number_errors,
        )
        _, good_number_errors = normalize_correspondence(
            {
                "tagged_caption": (
                    "[1]One ski boot[/1] stands beside [2]the other ski boot[/2]."
                )
            },
            boot_rows,
            min_groups=2,

        )
        self.assertEqual(good_number_errors, [])


        owned_prompt = build_caption_prompt(
            [
                {"mask_id": "man", "main_candidate": "man", "object": "man"},
                {"mask_id": "left", "main_candidate": "hand", "object": "hand"},
                {"mask_id": "right", "main_candidate": "hand", "object": "hand"},
            ]
        )
        self.assertNotIn('"fallback_unique_selector":', owned_prompt)
        self.assertNotIn('"safe_tagged_phrase":', owned_prompt)
        self.assertIn('"required_owner_id":1', owned_prompt)
        self.assertIn('"collective_candidate_ids":[2,3]', owned_prompt)

    def test_unsupported_object_evasion_is_rejected_and_forbidden_on_repair(self):
        rows = [
            {"mask_id": "man", "main_candidate": "man", "object": "man"},
            {"mask_id": "hand", "main_candidate": "hand", "object": "hand"},
        ]
        raw = (
            '{"tagged_caption":"[1]A man[/1] holds nothing with '
            '[2][1]his[/1] hand[/2]."}'
        )
        parsed = extract_json(raw)
        normalized, errors = normalize_correspondence(parsed, rows, min_groups=2)
        self.assertTrue(
            any("unsupported-object evasion 'holds nothing'" in error for error in errors),
            errors,
        )
        self.assertEqual(
            normalized["caption_quality"]["unsupported_object_evasions"],
            ["holds nothing"],
        )
        repair_prompt = build_schema_repair_prompt(raw, errors, rows, qa=True)
        self.assertIn('FORBIDDEN_EXACT_PHRASES:\n["holds nothing"]', repair_prompt)

        self_body_raw = (
            '{"tagged_caption":"[1]A man[/1] stands while holding '
            '[2]one of [1]his[/1] hands[/2]."}'
        )
        self_body_normalized, self_body_errors = normalize_correspondence(
            extract_json(self_body_raw), rows, min_groups=2
        )
        self.assertTrue(
            any(
                "unsupported-object evasion 'holding one of his hands'" in error
                for error in self_body_errors
            ),
            self_body_errors,
        )
        self.assertEqual(
            self_body_normalized["caption_quality"]["unsupported_object_evasions"],
            ["holding one of his hands"],
        )
        self_body_repair = build_schema_repair_prompt(
            self_body_raw, self_body_errors, rows, qa=True
        )
        self.assertIn('FORBIDDEN_EXACT_PHRASES:\n["holding one of his hands"]', self_body_repair)

        contact_rows = [
            *rows,
            {"mask_id": "neck", "main_candidate": "neck", "object": "neck"},
        ]
        self_contact_raw = (
            '{"tagged_caption":"[1]A man[/1] stands while '
            '[2][1]his[/1] hand[/2] rests on [3][1]his[/1] neck[/3]."}'
        )
        self_contact_normalized, self_contact_errors = normalize_correspondence(
            extract_json(self_contact_raw), contact_rows, min_groups=3
        )
        self.assertTrue(
            any(
                "unsupported-object evasion 'his hand rests on his neck'" in error
                for error in self_contact_errors
            ),
            self_contact_errors,
        )
        self.assertEqual(
            self_contact_normalized["caption_quality"]["unsupported_object_evasions"],
            ["his hand rests on his neck"],
        )
        self_contact_repair = build_schema_repair_prompt(
            self_contact_raw, self_contact_errors, contact_rows, qa=True
        )
        self.assertIn('FORBIDDEN_EXACT_PHRASES:\n["his hand rests on his neck"]', self_contact_repair)

        transitive_contact_raw = (
            '{"tagged_caption":"[1]A man[/1] stands while resting '
            '[2]one of [1]his[/1] hands[/2] on [3][1]his[/1] shoulder[/3]."}'
        )
        transitive_rows = [
            *rows,
            {"mask_id": "shoulder", "main_candidate": "shoulder", "object": "shoulder"},
        ]
        transitive_normalized, transitive_errors = normalize_correspondence(
            extract_json(transitive_contact_raw), transitive_rows, min_groups=3
        )
        self.assertTrue(
            any(
                "unsupported-object evasion 'resting one of his hands on his shoulder'" in error
                for error in transitive_errors
            ),
            transitive_errors,
        )
        self.assertEqual(
            transitive_normalized["caption_quality"]["unsupported_object_evasions"],
            ["resting one of his hands on his shoulder"],
        )

        microphone_rows = [
            {"mask_id": "man", "main_candidate": "man", "object": "man"},
            {
                "mask_id": "microphone",
                "main_candidate": "microphone",
                "object": "microphone",
            },
        ]
        microphone_raw = (
            '{"tagged_caption":"[1]A man[/1] plays [2]a microphone[/2]."}'
        )
        microphone_normalized, microphone_errors = normalize_correspondence(
            extract_json(microphone_raw), microphone_rows, min_groups=2
        )
        self.assertTrue(
            any(
                "unsupported-object evasion 'plays a microphone'" in error
                for error in microphone_errors
            ),
            microphone_errors,
        )
        self.assertEqual(
            microphone_normalized["caption_quality"]["unsupported_object_evasions"],
            ["plays a microphone"],
        )

    def test_bcc_packet_contains_original_overlay_and_every_inverse_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / name for name in ("source.png", "overlay.png", "crop-a.png", "crop-b.png")]
            for index, path in enumerate(paths):
                Image.new("RGB", (8, 8), (20 + index, 30, 40)).save(path)
            rows = [
                {
                    "mask_id": "a",
                    "source_image_path": str(paths[0]),
                    "inverse_crop_path": str(paths[2]),
                    "source_prompt": "cups",
                    "caption": "A mask hint.",
                },
                {
                    "mask_id": "b",
                    "source_image_path": str(paths[0]),
                    "inverse_crop_path": str(paths[3]),
                    "source_prompt": "plates",
                    "caption": "Another mask hint.",
                },
            ]
            packet, manifest = build_caption_image_packet(rows, paths[1])
            self.assertEqual(packet, [str(path) for path in paths])
            self.assertEqual([item["image_number"] for item in manifest], [1, 2, 3, 4])
            self.assertEqual([item.get("mask_id") for item in manifest[2:]], ["a", "b"])
            prompt = build_caption_prompt(rows, manifest)
            self.assertIn("context only", prompt)
            self.assertIn("never copy them merely", prompt)
            self.assertIn("do not need to mention every mask individually", prompt)
            self.assertIn("original image", prompt)

    def test_interactive_report_embeds_real_pair_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            mask_path = root / "mask.png"
            Image.new("RGB", (8, 8), (40, 60, 80)).save(source)
            mask = np.zeros((8, 8), dtype=bool)
            mask[2:6, 2:6] = True
            save_mask(mask, mask_path)
            pair = {
                "image_id": "real-output",
                "source_image_path": str(source),
                "first_pass_caption": "A cup.",
                "first_pass_raw": json.dumps(
                    {"reject": False, "tagged_caption": "[cup]A cup[/cup]."}
                ),
                "first_pass_groups": [
                    {
                        "mask_id": "cup",
                        "char_spans": [[0, 5]],
                        "text": ["A cup"],
                        "color_rgb": [12, 180, 90],
                        "mask_path": str(mask_path),
                        "inverse_crop_path": str(source),
                        "main_candidate": "cup",
                        "sam3_requery_iou": 0.91,
                        "sam3_score": 0.88,
                    }
                ],
                "caption": "A green cup.",
                "rewrite_raw": json.dumps(
                    {"reject": False, "tagged_caption": "[cup]A green cup[/cup]."}
                ),
                "groups": [
                    {
                        "mask_id": "cup",
                        "char_spans": [[0, 11]],
                        "text": ["A green cup"],
                        "color_rgb": [12, 180, 90],
                        "mask_path": str(mask_path),
                        "inverse_crop_path": str(source),
                        "main_candidate": "cup",
                        "sam3_requery_iou": 0.91,
                        "sam3_score": 0.88,
                    }
                ],
            }
            excluded = {
                **pair,
                "image_id": "excluded-output",
                "included": False,
                "quality_tier": "unparseable_excluded",
                "reason_code": "final_rewrite_unparseable",
                "validation": {
                    "after_rewrite": {
                        "parseable": False,
                        "parse_error": "malformed inline correspondence tags",
                        "issues": [],
                    }
                },
            }
            write_jsonl([pair, excluded], root / "image_text_pairs.jsonl")
            output = write_bcc_html_report(root, max_images=10)
            document = output.read_text(encoding="utf-8")
            self.assertIn('class="mention"', document)
            self.assertIn('data-groups="cup"', document)
            self.assertIn('data-version="before"', document)
            self.assertIn('data-version="after"', document)
            self.assertIn("Before rewrite", document)
            self.assertIn("After one rewrite", document)
            self.assertIn("[cup]A cup[/cup].", document)
            self.assertIn("[cup]A green cup[/cup].", document)
            self.assertIn("data:image/png;base64,", document)
            self.assertIn("Show all masks", document)
            self.assertIn("1 usable pairs + 1 audit-only exclusions", document)
            self.assertIn("excluded from training", document)
            self.assertIn("Audit-only exclusion", document)
            self.assertIn("malformed inline correspondence tags", document)

    def test_unparseable_rewrite_is_enriched_for_audit_only_visualization(self):
        rows = [
            {
                "mask_id": "person",
                "main_candidate": "person",
                "object": "person",
                "source_prompt": "person",
                "caption": "A standing person.",
                "mask_path": "/tmp/person.png",
                "inverse_crop_path": "/tmp/person-inverse.png",
            },
            {
                "mask_id": "hat",
                "main_candidate": "hat",
                "object": "hat",
                "source_prompt": "hat",
                "caption": "A dark hat.",
                "mask_path": "/tmp/hat.png",
                "inverse_crop_path": "/tmp/hat-inverse.png",
            },
        ]
        candidate = {
            "image_id": "excluded-output",
            "source_image_path": "/tmp/source.png",
            "correspondence_overlay_path": "/tmp/overlay.png",
            "bcc_input_manifest": [{"role": "original_image"}],
            "draft_raw": '{"reject":false,"tagged_caption":"[1]A person[/1]."}',
            "caption": "A person.",
            "groups": [{"mask_id": "person", "char_spans": [[0, 8]]}],
        }
        audit = {
            "image_id": "excluded-output",
            "included": False,
            "reason_code": "final_rewrite_unparseable",
            "raw": (
                '{"reject":false,"tagged_caption":'
                '"[1][2]A person[/1] in a hat[/2]."}'
            ),
            "validation": {
                "after_rewrite": {
                    "parseable": False,
                    "parse_error": "crossed inline tags",
                }
            },
        }

        display = _audit_only_display_record(candidate, audit, rows)

        self.assertFalse(display["included"])
        self.assertEqual(display["quality_tier"], "unparseable_excluded")
        self.assertEqual(display["first_pass_caption"], "A person.")
        self.assertEqual(display["first_pass_groups"], candidate["groups"])
        self.assertEqual(display["caption"], "A person in a hat.")
        self.assertEqual(display["groups"], [])
        self.assertEqual(
            [row["mask_id"] for row in display["first_pass_omitted_masks"]],
            ["hat"],
        )
        self.assertEqual(
            [row["mask_id"] for row in display["omitted_masks"]],
            ["person", "hat"],
        )
        self.assertIn("[1][2]", display["rewrite_raw"])

    def test_image_caption_retries_a_non_json_response(self):
        class RetryCaptioner:
            def __init__(self):
                self.calls = 0
                self.image_counts = []

            def generate(self, images, prompt, seed, generation_config=None):
                self.calls += 1
                self.image_counts.append(len(images))
                if self.calls == 1:
                    return {"raw": "I will reason about the image but forgot the final object."}
                return {
                    "raw": (
                        '{"reject":false,"caption":"A cup.","groups":'
                        '[{"mask_id":"cup","text":["A cup"]}]}'
                    )
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            mask_path = root / "mask.png"
            Image.new("RGB", (8, 8), (40, 60, 80)).save(source)
            mask = np.zeros((8, 8), dtype=bool)
            mask[2:6, 2:6] = True
            save_mask(mask, mask_path)
            rows = [
                {
                    "image_id": "img",
                    "mask_id": "cup",
                    "source_image_path": str(source),
                    "mask_path": str(mask_path),
                        "inverse_crop_path": str(source),
                    "bbox": [2, 2, 4, 4],
                    "caption": "A cup.",
                }
            ]
            config = {
                "resume": True,
                "random_seed": 7,
                "caption": {"model_name": "Qwen/Qwen3.5-9B"},
                "image_caption": {
                    "model_name": "Qwen/Qwen3.5-9B",
                    "min_groups": 1,
                    "require_all_masks": True,
                    "max_attempts": 2,
                    "enable_thinking": False,
                },
            }
            captioner = RetryCaptioner()
            run_dir = root / "run"
            write_jsonl(
                [
                    {
                        "image_id": "img",
                        "stage": "image_caption",
                        "reason": "generation_or_schema_failed",
                    }
                ],
                run_dir / "image_caption_rejected.jsonl",
            )
            run_image_caption_pass(config, run_dir, rows, captioner=captioner)
            self.assertEqual(captioner.image_counts, [3, 3])
            self.assertEqual(captioner.calls, 2)
            self.assertEqual(len(read_jsonl(run_dir / "image_caption_candidates.jsonl")), 1)
            self.assertEqual(len(read_jsonl(run_dir / "image_caption_errors.jsonl")), 1)

    def test_image_caption_recovers_valid_prior_raw_without_generation(self):
        class NoGenerate:
            def generate(self, *args, **kwargs):
                raise AssertionError("valid checkpoint raw should avoid regeneration")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            source = root / "source.png"
            mask_path = root / "mask.png"
            Image.new("RGB", (8, 8), (40, 60, 80)).save(source)
            mask = np.zeros((8, 8), dtype=bool)
            mask[2:6, 2:6] = True
            save_mask(mask, mask_path)
            rows = [
                {
                    "image_id": "img",
                    "mask_id": "cup",
                    "source_image_path": str(source),
                    "mask_path": str(mask_path),
                    "inverse_crop_path": str(source),
                    "bbox": [2, 2, 4, 4],
                    "caption": "A green cup.",
                }
            ]
            write_jsonl(
                [
                    {
                        "image_id": "img",
                        "prompt_version": BCC_PROMPT_VERSION,
                        "bcc_input_manifest": [
                            {"role": "inverse_mask_crop", "mask_id": "cup"},
                        ],
                        "attempt": 3,
                        "raw": (
                            '{"reject":false,"caption":"A green cup is visible in the background.",'
                            '"groups":[{"mask_id":"cup","text":["A green cup."]}]}'
                        ),
                    }
                ],
                run_dir / "image_caption_raw.jsonl",
            )
            write_jsonl(
                [{"image_id": "img", "reason": "generation_or_schema_failed"}],
                run_dir / "image_caption_rejected.jsonl",
            )
            config = {
                "resume": True,
                "caption": {"model_name": "Qwen/Qwen3.5-9B"},
                "image_caption": {
                    "model_name": "Qwen/Qwen3.5-9B",
                    "min_groups": 1,
                    "require_all_masks": True,
                },
            }
            run_image_caption_pass(config, run_dir, rows, captioner=NoGenerate())
            candidate = read_jsonl(run_dir / "image_caption_candidates.jsonl")[0]
            self.assertTrue(candidate["recovered_from_prior_raw"])
            self.assertEqual(candidate["caption"], "A green cup is present.")

    def test_repairable_draft_reaches_qa_but_keep_false_never_emits(self):
        class NoGenerate:
            def generate(self, *args, **kwargs):
                raise AssertionError("checkpoint recovery should avoid generation")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            source = root / "source.png"
            Image.new("RGB", (12, 12), (40, 60, 80)).save(source)
            rows = []
            for index, subject in enumerate(("cup", "shoe")):
                mask_path = root / f"{subject}.png"
                mask = np.zeros((12, 12), dtype=bool)
                mask[2 + index : 7 + index, 2:7] = True
                save_mask(mask, mask_path)
                rows.append(
                    {
                        "image_id": "img",
                        "mask_id": subject,
                        "source_image_path": str(source),
                        "mask_path": str(mask_path),
                        "inverse_crop_path": str(source),
                        "bbox": [2, 2 + index, 5, 5],
                        "main_candidate": subject,
                        "caption": f"A {subject}.",
                    }
                )
            write_jsonl(
                [
                    {
                        "image_id": "img",
                        "prompt_version": BCC_PROMPT_VERSION,
                        "bcc_input_manifest": [
                            {"role": "inverse_mask_crop", "mask_id": "cup"},
                            {"role": "inverse_mask_crop", "mask_id": "shoe"},
                        ],
                        "attempt": 1,
                        "raw": (
                            '{"reject":false,"caption":"A green cup.",'
                            '"links":[{"id":1,"text":["A green cup"]}]}'
                        ),
                    }
                ],
                run_dir / "image_caption_raw.jsonl",
            )
            config = {
                "resume": True,
                "caption": {"model_name": "Qwen/Qwen3.5-9B"},
                "image_caption": {
                    "model_name": "Qwen/Qwen3.5-9B",
                    "min_groups": 1,
                    "require_all_masks": True,
                },
                "image_caption_qa": {
                    "model_name": "Qwen/Qwen3.5-9B",
                    "max_attempts": 1,
                },
            }
            run_image_caption_pass(config, run_dir, rows, captioner=NoGenerate())
            candidate = read_jsonl(run_dir / "image_caption_candidates.jsonl")[0]
            self.assertFalse(candidate["pass1_strict"])
            self.assertTrue(candidate["draft_validation_errors"])
            qa_prompt = build_qa_prompt(candidate, rows)
            self.assertIn("draft_validation_errors", qa_prompt)
            self.assertIn(
                (
                    '"unused_draft_links":[{"id":2,"subject_anchor":"shoe",'
                    '"surface_identity_noun":"shoe",'
                    '"allowed_identity_nouns":'
                ),
                qa_prompt,
            )
            self.assertNotIn('"safe_tagged_phrase":', qa_prompt)
            self.assertIn("AVAILABLE_LINK_CONTEXT lists possibilities", qa_prompt)

            keep_false = (
                '{"keep":false,"reason_code":"irreparable",'
                '"caption":"A cup and a shoe.","links":['
                '{"id":1,"text":["A cup"]},{"id":2,"text":["a shoe"]}]}'
            )
            run_image_caption_qa(
                config,
                run_dir,
                rows,
                captioner=NoGenerate(),
                initial_raw=keep_false,
            )
            self.assertFalse((run_dir / "image_text_pairs.jsonl").exists())
            rejection = read_jsonl(run_dir / "image_caption_qa_rejected.jsonl")[-1]
            self.assertIn("pass-two keep must be true", rejection["validation_errors"])
    def test_image_caption_qa_stops_an_identical_no_progress_repair(self):
        class RepeatingCaptioner:
            def __init__(self):
                self.image_counts = []
                self.generation_configs = []

            def generate(self, images, prompt, seed, generation_config=None):
                self.image_counts.append(len(images))
                self.generation_configs.append(dict(generation_config or {}))
                return {
                    "raw": (
                        '{"keep":true,"reason_code":"unchanged",'
                        '"tagged_caption":"[1]A cup[/1] sits beside a plate."}'
                    )
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            source = root / "source.png"
            overlay = root / "overlay.png"
            mask_path = root / "mask.png"
            Image.new("RGB", (8, 8), (40, 60, 80)).save(source)
            Image.new("RGB", (8, 8), (50, 70, 90)).save(overlay)
            mask = np.zeros((8, 8), dtype=bool)
            mask[2:6, 2:6] = True
            save_mask(mask, mask_path)
            rows = [
                {
                    "image_id": "img",
                    "mask_id": "cup",
                    "source_image_path": str(source),
                    "mask_path": str(mask_path),
                    "inverse_crop_path": str(source),
                    "bbox": [2, 2, 4, 4],
                    "main_candidate": "cup",
                    "object": "cup",
                    "caption": "A cup.",
                }
            ]
            write_jsonl(
                [
                    {
                        "image_id": "img",
                        "prompt_version": BCC_PROMPT_VERSION,
                        "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                        "stage_version": PIPELINE_STAGE_VERSION,
                        "source_image_path": str(source),
                        "correspondence_overlay_path": str(overlay),
                        "caption": "A cup.",
                        "groups": [
                            {
                                "mask_id": "cup",
                                "text": ["A cup"],
                                "char_spans": [[0, 5]],
                            }
                        ],
                    }
                ],
                run_dir / "image_caption_candidates.jsonl",
            )
            config = {
                "resume": False,
                "random_seed": 7,
                "caption": {"model_name": "Qwen/Qwen3.5-9B"},
                "image_caption": {"min_groups": 1, "require_all_masks": True},
                "image_caption_qa": {
                    "model_name": "Qwen/Qwen3.5-9B",
                    "temperature": 0.0,
                    "repair_temperature": 0.15,
                    "repair_top_p": 0.85,
                    "max_attempts": 3,
                },
            }
            captioner = RepeatingCaptioner()
            run_image_caption_qa(config, run_dir, rows, captioner=captioner)
            self.assertEqual(captioner.image_counts, [3, 3])
            self.assertEqual(captioner.generation_configs[0]["temperature"], 0.0)
            self.assertEqual(captioner.generation_configs[1]["temperature"], 0.15)
            self.assertEqual(captioner.generation_configs[1]["top_p"], 0.85)
            self.assertFalse((run_dir / "image_text_pairs.jsonl").exists())
            rejection = read_jsonl(run_dir / "image_caption_qa_rejected.jsonl")[-1]
            self.assertEqual(
                rejection["retry_stop_reason"],
                "identical_repair_and_validation_errors",
            )

    def test_image_caption_qa_recovers_positive_raw_with_prefix_repair(self):
        class NoGenerate:
            def generate(self, *args, **kwargs):
                raise AssertionError("valid QA checkpoint raw should avoid regeneration")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            source = root / "source.png"
            mask_path = root / "mask.png"
            overlay = root / "overlay.png"
            Image.new("RGB", (8, 8), (40, 60, 80)).save(source)
            Image.new("RGB", (8, 8), (50, 70, 90)).save(overlay)
            mask = np.zeros((8, 8), dtype=bool)
            mask[2:6, 2:6] = True
            save_mask(mask, mask_path)
            mask_id = "full-image-hash_p000_m0000"
            rows = [
                {
                    "image_id": "img",
                    "mask_id": mask_id,
                    "source_image_path": str(source),
                    "mask_path": str(mask_path),
                    "inverse_crop_path": str(source),
                    "bbox": [2, 2, 4, 4],
                    "caption": "A green cup.",
                }
            ]
            write_jsonl(
                [
                    {
                        "image_id": "img",
                        "prompt_version": BCC_PROMPT_VERSION,
                        "schema_version": CORRESPONDENCE_SCHEMA_VERSION,
                        "stage_version": PIPELINE_STAGE_VERSION,
                        "source_image_path": str(source),
                        "correspondence_overlay_path": str(overlay),
                        "caption": "A green cup.",
                        "groups": [{"mask_id": mask_id, "text": ["A green cup"], "char_spans": [[0, 11]]}],
                    }
                ],
                run_dir / "image_caption_candidates.jsonl",
            )
            write_jsonl(
                [
                    {
                        "image_id": "img",
                        "prompt_version": BCC_PROMPT_VERSION,
                        "bcc_input_manifest": [
                            {"role": "inverse_mask_crop", "mask_id": mask_id},
                        ],
                        "attempt": 1,
                        "raw": (
                            '{"keep":true,"reason":"faithful","caption":"A green cup.",'
                            '"groups":[{"mask_id":"corrupt_p000_m0000","text":["A green cup."]}]}'
                        ),
                    }
                ],
                run_dir / "image_caption_qa_raw.jsonl",
            )
            config = {
                "resume": True,
                "caption": {"model_name": "Qwen/Qwen3.5-9B"},
                "image_caption": {"min_groups": 1, "require_all_masks": True},
                "image_caption_qa": {"model_name": "Qwen/Qwen3.5-9B"},
            }
            run_image_caption_qa(config, run_dir, rows, captioner=NoGenerate())
            final = read_jsonl(run_dir / "image_text_pairs.jsonl")[0]
            self.assertTrue(final["recovered_from_prior_qa_raw"])
            self.assertEqual(final["groups"][0]["mask_id"], mask_id)

    def test_bcc_passes_batch_two_caption_ready_images(self):
        class BatchCaptioner:
            def __init__(self):
                self.calls = []

            def generate_many_bcc(
                self,
                image_sets,
                prompts,
                seeds,
                batch_size,
                generation_config=None,
            ):
                self.calls.append((len(image_sets), batch_size))
                outputs = []
                for prompt in prompts:
                    payload = {
                        "caption": "A cup.",
                        "links": [{"id": 1, "text": ["A cup"]}],
                    }
                    if prompt.startswith("Independently verify"):
                        payload.update({"keep": True, "reason_code": "ok"})
                    else:
                        payload.update({"reject": False})
                    outputs.append({"raw": json.dumps(payload)})
                return outputs

            def generate(self, *args, **kwargs):
                raise AssertionError("valid batched responses should not fall back")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row_groups = []
            for index in range(2):
                source = root / f"source-{index}.png"
                mask_path = root / f"mask-{index}.png"
                Image.new("RGB", (8, 8), (40 + index, 60, 80)).save(source)
                mask = np.zeros((8, 8), dtype=bool)
                mask[2:6, 2:6] = True
                save_mask(mask, mask_path)
                row_groups.append(
                    [
                        {
                            "image_id": f"img-{index}",
                            "mask_id": f"m{index}",
                            "source_image_path": str(source),
                            "inverse_crop_path": str(source),
                            "mask_path": str(mask_path),
                            "bbox": [2, 2, 4, 4],
                            "caption": "A cup.",
                        }
                    ]
                )
            config = {
                "random_seed": 7,
                "caption": {"model_name": "Qwen/Qwen3.5-9B"},
                "image_caption": {
                    "model_name": "Qwen/Qwen3.5-9B",
                    "min_groups": 1,
                    "require_all_masks": True,
                    "batch_size": 2,
                    "max_visual_tokens_per_batch": 16384,
                    "enable_thinking": False,
                },
                "image_caption_qa": {
                    "model_name": "Qwen/Qwen3.5-9B",
                    "batch_size": 2,
                    "max_visual_tokens_per_batch": 16384,
                    "enable_thinking": False,
                },
            }
            captioner = BatchCaptioner()
            run_dir = root / "run"
            run_image_caption_pass_batch(config, run_dir, row_groups, captioner=captioner)
            run_image_caption_qa_batch(config, run_dir, row_groups, captioner=captioner)
            self.assertEqual(captioner.calls, [(2, 2), (2, 2)])
            self.assertEqual(len(read_jsonl(run_dir / "image_caption_candidates.jsonl")), 2)
            self.assertEqual(len(read_jsonl(run_dir / "image_text_pairs.jsonl")), 2)
            final = read_jsonl(run_dir / "image_text_pairs.jsonl")[0]
            self.assertEqual(len(final["bcc_input_manifest"]), 3)
            self.assertEqual(final["prompt_version"], BCC_PROMPT_VERSION)

    def test_correspondence_recovery_ignores_old_terminal_status_and_stops_at_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            consistent_rows = []
            selected = []
            for index in range(3):
                image_id = f"img-{index}"
                source = root / f"{image_id}.png"
                mask_path = root / f"{image_id}-mask.png"
                Image.new("RGB", (8, 8), (30 + index, 60, 90)).save(source)
                mask = np.zeros((8, 8), dtype=bool)
                mask[2:6, 2:6] = True
                save_mask(mask, mask_path)
                selected.append({"image_id": image_id, "image_path": str(source)})
                consistent_rows.append(
                    {
                        "image_id": image_id,
                        "mask_id": f"{image_id}-mask",
                        "source_image_path": str(source),
                        "mask_path": str(mask_path),
                        "inverse_crop_path": str(source),
                        "bbox": [2, 2, 4, 4],
                        "caption": f"A cup {index}.",
                        "object": "cup",
                        "source_prompt": "cups",
                        "sam3_score": 0.9,
                        "sam3_consistency": {"best_iou": 0.8},
                    }
                )
            write_jsonl(selected, run_dir / "selected_images.jsonl")
            write_jsonl(consistent_rows, run_dir / "consistent_captions.jsonl")
            write_jsonl(
                [
                    {
                        "image_id": row["image_id"],
                        "status": "rejected",
                        "stage": "image_caption_qa",
                    }
                    for row in consistent_rows
                ],
                run_dir / "image_pipeline_status.jsonl",
            )
            write_jsonl(
                [
                    {
                        "image_id": row["image_id"],
                        "reason": "legacy_model_rejection",
                        "prompt_version": "bcc-caption-v2-legacy",
                        "schema_version": "bcc-image-text-v2",
                        "stage_version": "bcc-normalizer-v2",
                    }
                    for row in consistent_rows
                ],
                run_dir / "image_caption_qa_rejected.jsonl",
            )
            write_jsonl(
                [
                    {
                        "image_id": "img-0",
                        "source_image_path": consistent_rows[0]["source_image_path"],
                        "caption": "A legacy cup.",
                        "groups": [
                            {
                                "mask_id": consistent_rows[0]["mask_id"],
                                "mask_path": consistent_rows[0]["mask_path"],
                                "char_spans": [[0, 12]],
                                "text": ["A legacy cup"],
                                "color_rgb": [255, 63, 63],
                            }
                        ],
                        "prompt_version": "bcc-caption-v2-legacy",
                        "schema_version": "bcc-image-text-v2",
                        "stage_version": "bcc-normalizer-v2",
                    }
                ],
                run_dir / "image_text_pairs.jsonl",
            )
            config = {
                "resume": True,
                "random_seed": 7,
                "caption": {"model_name": "Qwen/Qwen3.5-9B"},
                "image_caption": {
                    "model_name": "Qwen/Qwen3.5-9B",
                    "min_groups": 1,
                    "require_all_masks": True,
                },
                "image_caption_qa": {"model_name": "Qwen/Qwen3.5-9B"},
            }
            run_correspondence_recovery(config, run_dir, target_successes=2, mock=True)
            pairs = read_jsonl(run_dir / "image_text_pairs.jsonl")
            self.assertEqual(len(pairs), 3)
            self.assertEqual(
                sum(row.get("prompt_version") == BCC_PROMPT_VERSION for row in pairs),
                2,
            )
            self.assertEqual(len(read_jsonl(run_dir / "correspondence_recovery_status.jsonl")), 2)
            self.assertTrue((run_dir / "site" / "report.html").exists())
            state = json.loads((run_dir / "correspondence_recovery_state.json").read_text())
            self.assertEqual(state["successful_images"], 2)
            self.assertTrue(state["stopped_early"])

    def test_image_review_streams_in_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for index in range(5):
                image = root / f"image-{index}.png"
                Image.new("RGB", (64, 64), (90 + index, 120, 150)).save(image)
                records.append({"image_id": f"img-{index}", "image_path": str(image)})
            manifest = root / "manifest.jsonl"
            write_jsonl(records, manifest)
            run_dir = root / "run"
            config = {
                "project_root": str(root),
                "resume": True,
                "random_seed": 7,
                "dataset": {"manifest_path": str(manifest), "limit": 5},
                "pipeline": {"image_review_window": 2},
                "image_review": {
                    "min_distinct_objects": 1,
                    "min_side_px": 1,
                    "min_total_pixels": 1,
                    "batch_size": 2,
                },
            }
            iterator = _iter_accepted_reviews(
                config,
                run_dir,
                limit=5,
                captioner=None,
                mock=True,
            )
            first = next(iterator)
            self.assertEqual(first["image_id"], "img-0")
            self.assertEqual(len(read_jsonl(run_dir / "selected_images.jsonl")), 2)
            second = next(iterator)
            self.assertNotEqual(first["image_id"], second["image_id"])
            self.assertEqual(len(read_jsonl(run_dir / "selected_images.jsonl")), 2)
            third = next(iterator)
            self.assertNotIn(third["image_id"], {first["image_id"], second["image_id"]})
            self.assertEqual(len(read_jsonl(run_dir / "selected_images.jsonl")), 4)
            iterator.close()

    def test_full_mock_pipeline_is_checkpointed_and_stops_after_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            Image.new("RGB", (64, 64), (90, 120, 150)).save(image)
            manifest = root / "manifest.jsonl"
            write_jsonl([{"image_id": "img", "image_path": str(image), "selected_index": 0}], manifest)
            run_dir = root / "run"
            config = {
                "project_root": str(root),
                "random_seed": 7,
                "continue_on_error": False,
                "resume": True,
                "dataset": {"manifest_path": str(manifest), "limit": 1},
                "pipeline": {"target_successful_images": 1},
                "image_review": {
                    "min_distinct_objects": 1,
                    "min_side_px": 1,
                    "min_total_pixels": 1,
                    "batch_size": 2,
                },
                "filter": {
                    "min_mask_area": 1,
                    "min_mask_area_fraction": 0.0,
                    "max_masks_per_image": 0,
                    "min_bbox_fill": 0.01,
                    "max_mask_area_fraction": 1.0,
                    "max_bbox_area_fraction": 1.0,
                    "max_components": 12,
                },
                "caption": {
                    "model_name": "Qwen/Qwen3.5-9B",
                    "prompt": "caption {source_prompt} {inverse_background_rgb}",
                    "batch_size": 4,
                    "max_masks_per_image": 0,
                    "max_total_masks": 0,
                },
                "quality_filter": {
                    "enabled": True,
                    "model_name": "Qwen/Qwen3.5-9B",
                    "mask_review_prompt": "review {caption} {source_prompt}",
                    "batch_size": 4,
                },
                "consistency_filter": {"mask_iou_threshold": 0.5},
                "image_caption": {
                    "model_name": "Qwen/Qwen3.5-9B",
                    "min_groups": 1,
                    "require_all_masks": True,
                },
                "image_caption_qa": {"model_name": "Qwen/Qwen3.5-9B"},
            }
            run_checkpointed_pipeline(config, run_dir, limit=1, target_successes=1, mock=True)
            pairs = read_jsonl(run_dir / "image_text_pairs.jsonl")
            statuses = read_jsonl(run_dir / "image_pipeline_status.jsonl")
            self.assertEqual(len(pairs), 1)
            self.assertEqual(statuses[0]["status"], "accepted")
            self.assertEqual(statuses[0]["prompt_version"], BCC_PROMPT_VERSION)
            self.assertEqual(statuses[0]["schema_version"], CORRESPONDENCE_SCHEMA_VERSION)
            self.assertEqual(statuses[0]["stage_version"], PIPELINE_STAGE_VERSION)
            self.assertTrue((run_dir / "pipeline_state.json").exists())
            self.assertTrue((run_dir / "site" / "report.html").exists())
            run_checkpointed_pipeline(config, run_dir, limit=1, target_successes=1, mock=True)
            self.assertEqual(len(read_jsonl(run_dir / "image_text_pairs.jsonl")), 1)


if __name__ == "__main__":
    unittest.main()
