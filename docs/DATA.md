# Data contract

## Input manifests

The local loader accepts `.csv`, `.jsonl`, or `.ndjson`.

| Column | Required | Meaning |
|---|---:|---|
| `image_id` | recommended | Stable unique ID. If omitted, the filename stem is used. |
| `image_path` | one of two | Absolute path, or a path relative to the current working directory. |
| `file_name` | one of two | Path relative to `--image-root` / `IMAGE_ROOT`. |
| `source_dataset` | no | Dataset name stored in provenance. |
| `split` | no | Source split. |
| `pair_key` | no | Stable source key used for deduplication; defaults to `image_id`. |
| `paired_text` | no | Existing caption or source text retained as context/provenance. |
| `metadata_json` | no | JSON object with dataset-specific metadata. |

The loader copies selected source bytes into immutable unit tar files. Later jobs do not depend on the original filesystem layout.

## GPIC exclusions

Pass `--exclude-csv path.csv` (or `EXCLUDE_CSV` in a wrapper). The file may be a one-column CSV with or without a header. Accepted headers are `identifier`, `image_id`, `file_name`, `filename`, `pair_key`, and `id`. Matching accepts the full value, normalized path, basename, or filename stem. The campaign registry records only the exclusion file's basename, SHA-256, and row count—not its absolute path.

## Run outputs

Each campaign is split into immutable source units. A stage-level filesystem lock serializes claim replacement, while each unit claim carries a generation, unique fencing token, and heartbeat. Workers must still own that token before committing, cleaning hydrated files, or releasing the claim. A successful stage writes a tiny `_SUCCESS.json` only after semantic outputs and artifacts have been validated and fsynced.

SAM 3 success additionally requires exact one-to-one agreement among unique manifest IDs, mask PNGs, inverse crops, RLE rows, and archive members. Interrupted images are cleared and regenerated at image granularity, preventing a resumed unit from appending duplicate semantic rows. Failed unit attempts are counted persistently across jobs; reaching the configured limit creates `_QUARANTINED.json`, which makes the merge fail with the exact unit IDs instead of polling forever.

After every stage barrier:

- `merged/<stage>/manifest.json` lists concatenated JSONL outputs, row counts, checksums, and any size-based rollover shards.
- `reports/run_ledger.csv` and `.jsonl` contain one source image per row with image review, SAM proposal, mask QA, consistency, BCC, count, and disposition fields.
- `reports/run_ledger_summary.json` provides compact totals.

The Hugging Face export uses Parquet shards (100 rows by default) with image bytes for parseable training examples, COCO-style mask RLE JSON, caption spans/groups, counts, disposition, and full BCC audit JSON. Zero-mask and unparseable rows remain metadata-only in `audit_all_processed` and are never training examples.

## Integrity and repair records

`concor campaign-integrity CAMPAIGN` is read-only. It can scan the complete campaign or selected `--unit-id` values and writes a versioned JSON report when `--report` is supplied. Incomplete stages are reported separately from integrity violations.

`concor campaign-repair CAMPAIGN --from-stage STAGE` is a dry run unless `--apply` is present. It refuses units with unresolved claims, preserves immutable source and earlier checkpoints, and moves affected unit outputs plus stale merged views into `repairs/<timestamp>/`. The resulting `repair.json` records the reason, unit IDs, affected stages, and every moved path.
