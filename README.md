# ConCor Training Data Generation

ConCor turns ordinary images into dense image–text training pairs: SAM 3 proposes instance masks, Qwen3.8-27B describes and checks them, and a final multimodal pass writes a natural caption whose text spans point back to the masks they describe. The format supports one entity mentioned several times, overlapping possessives, and one collective phrase linked to several masks.

This is the cleaned research release of the pipeline we used for our first GPIC campaign. It is checkpoint-friendly, supports local images or arbitrary GPIC subsets, can run on one capable node, and includes a 24-worker Slurm launcher for mixed A40/H200 clusters.

The initial 20k GPIC release and dataset card live at [suryadv/gpic-bcc-sam3-qwen38-27b](https://huggingface.co/datasets/suryadv/gpic-bcc-sam3-qwen38-27b). The schema/card are public now; Parquet shards are filled by the 20k campaign.

## Quick start

You need Python 3.10–3.12, CUDA GPUs, access to the gated SAM 3 checkpoint, and enough space for Qwen3.8-27B. Two A40s (tensor parallel) or one H200 works for the Qwen stages.

```bash
git clone https://github.com/suryathecreator/ConCor-Training-Data-Generation.git
cd ConCor-Training-Data-Generation
bash scripts/setup.sh
source .venv/bin/activate
hf auth login
# Optional for cluster/offline runs; downloads about 56 GB plus SAM 3.
bash scripts/download_models.sh
```

For your own images, make a CSV or JSONL manifest with `image_id` plus either `image_path`, or `file_name` together with `IMAGE_ROOT`. `source_dataset`, `split`, `pair_key`, `paired_text`, and `metadata_json` are optional. Then run:

```bash
SOURCE=manifest MANIFEST=/data/images.csv IMAGE_ROOT=/data/images \
TARGET_TOTAL=100 CAMPAIGN_ROOT=$PWD/outputs/my_run \
bash scripts/run_campaign.sh
```

For GPIC, the loader samples deterministically from the requested split. An optional one-column CSV can contain any image ID, filename, path, or GPIC pair key that must not be selected:

```bash
SOURCE=gpic TARGET_TOTAL=100 EXCLUDE_CSV=/data/do_not_use.csv \
CAMPAIGN_ROOT=$PWD/outputs/gpic_100 bash scripts/run_campaign.sh
```

See [docs/DATA.md](docs/DATA.md) for the complete input/output contract and [docs/CLUSTER.md](docs/CLUSTER.md) for Slurm and arbitrary GPU partitions.

## Pipeline

1. **Image review** — Qwen keeps scenes likely to yield many distinct, segmentable entities and returns concrete SAM prompts.
2. **SAM 3 proposals** — prompts are batched per image; geometric filters remove tiny, sparse, duplicate, and implausibly large masks.
3. **Mask caption + QA** — Qwen sees dynamically colored inverse-mask crops, writes a short identity description, and checks it against the proposal and pixels.
4. **Consistency** — spaCy extracts the identity head, SAM 3 is prompted again in the same region, and the mask must match a returned proposal at IoU ≥ 0.5.
5. **BCC captioning** — Qwen sees the original, numbered overlay, every inverse crop, and the accepted-mask context. It writes inline links, independently audits the pair, then performs exactly one rewrite. Deterministic checks validate syntax, span coverage, identity compatibility, and a few observed failure modes.
6. **Publish** — an HTML audit site and per-image CSV/JSONL ledger record every outcome, and ConCor-1-compatible Parquet views are exported directly from completed units. Consolidated JSONL is optional.

The caption does not have to copy mask descriptions or mention every mask. A natural plural phrase can point to multiple masks, and a single mask can link to several text spans such as a noun phrase and later pronoun.

## Dataset views

Training examples are separated by **mask count only**, not by audit style flags:

- `gpic_min_10`: parseable BCC examples with at least 10 final linked masks.
- `gpic_1_to_9`: parseable examples with 1–9 final linked masks.
- `gpic_parseable_1_plus`: the union of all parseable examples with at least one linked mask.
- `audit_all_processed`: every selected source image, including upstream rejection, zero-mask, unparseable, and failed outcomes. These audit-only rows are not training pairs.

The three training configs use the same nine-column caption contract as [UWGZQ/ConCor-1-Data](https://huggingface.co/datasets/UWGZQ/ConCor-1-Data): source keys and dimensions plus `caption`, `groups_json`, and compressed-COCO-RLE `masks_json`. Every run also produces `reports/run_ledger.csv`, `run_ledger.jsonl`, and a summary. Optional consolidated JSONL rolls into bounded numbered shards.

## Checkpoints and recovery

Unit claims are fenced for the full stage transaction: stale-claim replacement is serialized, live claims are heartbeated, and every commit, cleanup, and release verifies the worker's unique ownership token. SAM 3 also checks that its manifest, mask PNGs, inverse crops, RLE rows, and packed archive agree exactly before writing `_SUCCESS.json`.

Merge and export are safe to requeue. A JSONL merge records a durable source-unit cursor and reuses completed shards; pass `--metadata-only` to use the stage barrier without building consolidated JSONL. Cluster launchers do this by default; set `MERGE_OUTPUTS=1` only when those convenience files are wanted.

Export reads completed units directly, independent of merge. It checkpoints fixed unit ranges before writing the public views, atomically publishes each Parquet shard, and skips validated shards after restart. The checkpoint directory is a sibling of the export directory, so it is not uploaded. Hugging Face `upload-large-folder` keeps its own resumable upload state.

```bash
concor campaign-merge outputs/campaigns/my_run --stage bcc --metadata-only
concor campaign-export-hf outputs/campaigns/my_run outputs/campaigns/my_run/hf_publish \
  --no-image-bytes --shard-size 100 --checkpoint-units 100
```

At most one merge checkpoint interval or one export unit batch is recomputed after abrupt preemption; completed Parquet shards are not rewritten.


Audit a campaign without changing it:

```bash
concor campaign-integrity outputs/campaigns/my_run \
  --report outputs/campaigns/my_run/reports/integrity.json
```

If the report identifies a damaged unit, preview a recoverable rewind and then apply it. Source images and completed upstream review data are preserved; affected outputs are moved into a timestamped `repairs/` backup.

```bash
concor campaign-repair outputs/campaigns/my_run \
  --report outputs/campaigns/my_run/reports/integrity.json --from-stage sam3
concor campaign-repair outputs/campaigns/my_run \
  --report outputs/campaigns/my_run/reports/integrity.json --from-stage sam3 --apply
```

Then rerun the normal launcher with `START_STAGE=sam3`. Only rewound units repeat the upstream stages. Persistent failures are quarantined with an explicit diagnostic instead of being retried indefinitely.

BCC packets that exceed the model's image or context limit are handled per image: that image receives `bcc_input_too_large`, while valid neighbors in its batch continue. Legacy quarantines caused by the old batch-wide behavior can be finalized explicitly with `concor campaign-skip-oversized-bcc CAMPAIGN --apply`; the recovery report and prior quarantine metadata remain under `repairs/`.

## Quality, honestly

The output is useful but not perfect. Based on our audits, it is roughly 80–90% good; a small manual cleanup pass would still help. The deterministic checks are a strong approximation built from repeated scorer iterations, but they can miss a real issue or flag something harmless. The Qwen visual auditor is also better at improving the caption than at citing exact mask numbers in its explanation, so those explanations should not be treated as ground truth.

We tried older models, including Qwen3.5, and struggled much more. Qwen3.8-27B does very well here, but it still has a pattern of solving one hard case and failing somewhere else. We considered adding more gates and continuing to tune the prompts; at this point the remaining errors look more like a model/perfection problem that manual annotation can fix. For the initial 20k subset, we froze the prompt that behaves consistently instead of destabilizing it. Suggestions and targeted fixes are very welcome.

## Repository layout

```text
configs/                 final Qwen3.8-27B pipeline config
src/sam3_mask_captioning pipeline, BCC rules, validators, exporter, UI
scripts/                 setup, local run, cache staging, 24-way launcher
slurm/                   portable worker/barrier job templates
schema/                  exported BCC record schema
hf_dataset/              Hugging Face dataset-card template
tests/                   deterministic and checkpoint/publication tests
```

The implementation follows the text-linking conventions used by *Grounding as Concept Correspondence* (especially its examples of repeated mentions, plural groups, and overlapping references), with SAM 3 proposals and GPIC as the image source. Please also cite the upstream BCC, SAM 3, GPIC, and Qwen work when applicable.

## License

Code is released under the MIT license. Upstream models and datasets keep their own licenses and access terms.
