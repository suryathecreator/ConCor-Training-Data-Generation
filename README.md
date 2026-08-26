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
6. **Publish** — units merge into bounded JSONL shards, an HTML audit site is built, a per-image CSV/JSONL ledger records every stage outcome, and Parquet views are exported.

The caption does not have to copy mask descriptions or mention every mask. A natural plural phrase can point to multiple masks, and a single mask can link to several text spans such as a noun phrase and later pronoun.

## Dataset views

Training examples are separated by **mask count only**, not by audit style flags:

- `min_10_masks`: parseable BCC examples with at least 10 final linked masks.
- `masks_1_to_9`: parseable examples with 1–9 final linked masks.
- `parseable_1_plus`: the union of all parseable examples with at least one linked mask.
- `audit_all_processed`: every selected source image, including upstream rejection, zero-mask, unparseable, and failed outcomes. These audit-only rows are not training pairs.

Every run also produces `reports/run_ledger.csv`, `run_ledger.jsonl`, and a summary. Stage outputs are concatenated after each barrier; files roll into numbered shards only after the configured size limit.

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
