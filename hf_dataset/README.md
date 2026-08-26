---
license: mit
task_categories:
- image-text-to-text
- image-segmentation
language:
- en
tags:
- grounded-captioning
- bidirectional-concept-correspondence
- sam3
- gpic
configs:
- config_name: min_10_masks
  data_files:
  - split: train
    path: data/min_10_masks/*.parquet
- config_name: masks_1_to_9
  data_files:
  - split: train
    path: data/masks_1_to_9/*.parquet
- config_name: parseable_1_plus
  data_files:
  - split: train
    path: data/parseable_1_plus/*.parquet
- config_name: audit_all_processed
  data_files:
  - split: train
    path: data/audit_all_processed/*.parquet
---

# GPIC BCC: SAM 3 + Qwen3.8-27B

This is the dataset release produced by [ConCor Training Data Generation](https://github.com/suryathecreator/ConCor-Training-Data-Generation). It pairs GPIC images with natural captions and exact mask-linked text spans for Bidirectional Concept Correspondence-style training.

The initial release target is a fresh 20k-image GPIC train sample. The public card/schema are in place while the checkpointed campaign fills the Parquet shards.

## Views

- `min_10_masks`: parseable examples with at least 10 final linked masks.
- `masks_1_to_9`: parseable examples with 1–9 final linked masks.
- `parseable_1_plus`: all parseable examples with at least one final linked mask.
- `audit_all_processed`: every selected source image and its stage disposition, including image-review rejection, zero masks, unparseable BCC output, and runtime failure. Audit-only rows are not training examples and do not carry training image bytes.

These views are separated only by final linked-mask count. Style warnings or other nonfatal audit flags do not move an otherwise parseable example between training views.

## What is stored

Each training row includes compressed image bytes, caption, correspondence groups/spans, accepted and post-consistency mask RLEs, proposal/consistency/final counts, disposition, and full audit JSON. `stats/summary.json` and SVG histograms describe the release. The campaign ledger records every source file and its result through each stage.

## Quality note

This is generated research data, not a perfect manual annotation. Our audits suggest roughly 80–90% of pairs are good. Deterministic checks can be wrong, and the Qwen auditor sometimes cites the wrong mask number even when its rewrite is useful. A light manual cleanup pass is recommended for high-stakes evaluation. We found Qwen3.8-27B much stronger than the older Qwen3.5 variants we tested, and froze the best stable prompt for this release rather than continuing to trade one error pattern for another. Suggestions are welcome.
