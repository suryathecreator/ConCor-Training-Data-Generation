from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


def _read_exclusions(paths: Iterable[str | Path]) -> tuple[set[str], set[str]]:
    image_ids: set[str] = set()
    pair_keys: set[str] = set()
    for path_value in paths:
        path = Path(path_value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Exclude manifest does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                image_id = str(row.get("image_id") or "").strip()
                if image_id:
                    image_ids.add(image_id)
                pair_key = str((row.get("source_context") or {}).get("pair_key") or "").strip()
                if pair_key:
                    pair_keys.add(pair_key)
    return image_ids, pair_keys


def write_excluded_sample(
    raw_manifest: str | Path,
    exp_manifest: str | Path,
    *,
    seed: int,
    limit: int,
    exclude_manifests: Iterable[str | Path] = (),
    shuffle: bool = False,
) -> int:
    raw_manifest = Path(raw_manifest)
    exp_manifest = Path(exp_manifest)
    excluded_image_ids, excluded_pair_keys = _read_exclusions(exclude_manifests)
    exp_manifest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with raw_manifest.open("r", encoding="utf-8") as source:
        candidates = [json.loads(line) for line in source if line.strip()]
    if shuffle:
        random.Random(int(seed)).shuffle(candidates)
    with exp_manifest.open("w", encoding="utf-8") as dest:
        for row in candidates:
            context = row.get("source_context") or {}
            image_id = str(row.get("image_id") or "").strip()
            pair_key = str(context.get("pair_key") or "").strip()
            if image_id in excluded_image_ids or (pair_key and pair_key in excluded_pair_keys):
                continue
            safe_context = {
                "source_dataset": context.get("source_dataset", "stanford-vision-lab/gpic"),
                "source": context.get("source", "gpic"),
                "split": context.get("split", "test"),
                "hf_tar": context.get("hf_tar", ""),
                "pair_key": context.get("pair_key", ""),
            }
            out = {
                "selected_index": written,
                "seed": int(seed),
                "image_id": row["image_id"],
                "file_name": Path(row["image_path"]).name,
                "source_context": safe_context,
            }
            dest.write(json.dumps(out, sort_keys=True) + "\n")
            written += 1
            if written >= int(limit):
                break
    if written < int(limit):
        raise RuntimeError(
            f"Only wrote {written} rows after exclusions, below requested LIMIT={limit}. "
            "Increase POOL_LIMIT/MAX_TARS or reduce exclusions."
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--exp-manifest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--exclude-manifests", default="")
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    exclude = [path for path in args.exclude_manifests.split(":") if path]
    written = write_excluded_sample(
        args.raw_manifest,
        args.exp_manifest,
        seed=args.seed,
        limit=args.limit,
        exclude_manifests=exclude,
        shuffle=args.shuffle,
    )
    print(f"Wrote {written} rows to {args.exp_manifest}")


if __name__ == "__main__":
    main()
