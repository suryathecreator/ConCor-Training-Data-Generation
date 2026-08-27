from __future__ import annotations

import hashlib
import json
import os
import random
import tarfile
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from .campaign_manifest import (
    IMAGE_SUFFIXES,
    _commit_extension,
    _extension_start,
    _safe_id,
    _write_source_unit,
    campaign_paths,
    initialize_campaign,
    load_registry,
    manifest_target_add_count,
)
from .io_utils import write_json
from .selection import is_excluded, load_exclusion_csv


def _materialize_write_workers() -> int:
    configured = os.environ.get("BCC_MATERIALIZE_WRITE_WORKERS")
    if configured:
        return max(1, min(64, int(configured)))
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK") or min(8, os.cpu_count() or 1))
    # Unit creation is dominated by independent shared-filesystem metadata and
    # archive writes, so two I/O threads per allocated CPU is intentional.
    return max(1, min(16, 2 * allocated))


def _candidate_tars(repo_id: str, split: str, token: str | None) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="dataset")
    tars = [path for path in files if path.lower().endswith((".tar", ".tar.gz", ".tgz"))]
    split_lower = split.lower()
    matched = [
        path
        for path in tars
        if split_lower in Path(path).name.lower() or f"/{split_lower}" in path.lower()
    ]
    return matched or tars


def _download_tar(
    repo_id: str, repo_path: str, token: str | None, cache_dir: str | None
) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=repo_path,
            token=token,
            cache_dir=cache_dir,
        )
    )


def _member_key(name: str) -> tuple[str, str]:
    path = Path(name)
    return path.with_suffix("").as_posix(), path.suffix.lower()


def _json_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> dict[str, Any]:
    extracted = archive.extractfile(member)
    if extracted is None:
        return {}
    try:
        return json.loads(extracted.read().decode("utf-8"))
    except Exception:
        return {}


def _paired_text(metadata: dict[str, Any]) -> tuple[str, list[str]]:
    values: list[str] = []
    for key in (
        "caption",
        "text",
        "alt_text",
        "description",
        "short_caption",
        "long_caption",
        "image_caption",
        "title",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, list):
            values.extend(str(item).strip() for item in value if str(item).strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return (deduped[0] if deduped else ""), deduped[:8]


def _stable_key_order(keys: list[str], seed: int, tar_path: str) -> list[str]:
    digest = hashlib.sha256(f"{seed}\0{tar_path}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    ordered = list(keys)
    rng.shuffle(ordered)
    return ordered


def _physical_key_order(
    keys: list[str],
    image_members: dict[str, tarfile.TarInfo],
    json_members: dict[str, tarfile.TarInfo],
) -> list[str]:
    """Read a preselected key set in tar order to avoid GPFS seek amplification."""
    return sorted(
        keys,
        key=lambda key: (
            min(image_members[key].offset, json_members[key].offset),
            key,
        ),
    )


def extend_from_gpic(
    root: str | Path,
    *,
    add_images: int | None = None,
    target_total: int | None = None,
    repo_id: str = "stanford-vision-lab/gpic",
    split: str = "train",
    seed: int = 20260808,
    cache_dir: str | None = None,
    token: str | None = None,
    max_tars: int | None = None,
    exclude_csv: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if not campaign_paths(root)["registry"].exists():
        initialize_campaign(root, seed=seed, dataset=repo_id, split=split)
    requested = manifest_target_add_count(
        root, add_images=add_images, target_total=target_total
    )
    if requested <= 0:
        return load_registry(root)
    registry, extension, existing_keys, existing_ids = _extension_start(
        root, requested, "gpic_remote"
    )
    exclusions, exclusion_provenance = load_exclusion_csv(exclude_csv)
    selection = registry.setdefault("selection", {})
    if exclusion_provenance:
        selection["exclusion_list"] = exclusion_provenance
    tar_paths = list(selection.get("tar_paths") or [])
    if not tar_paths:
        tar_paths = _candidate_tars(repo_id, split, token)
        random.Random(seed).shuffle(tar_paths)
        selection.update(
            {
                "repo_id": repo_id,
                "split": split,
                "seed": int(seed),
                "tar_paths": tar_paths,
                "tar_index": 0,
                "key_index": 0,
            }
        )
        write_json(registry, campaign_paths(root)["registry"])
    tar_index = int(selection.get("tar_index") or 0)
    key_index = int(selection.get("key_index") or 0)
    unit_size = int(registry["unit_size"])
    source_index = int(registry.get("source_count") or 0)
    unit_index = int(registry.get("unit_count") or 0)
    pending: list[dict[str, Any]] = []
    committed: list[dict[str, Any]] = []
    accepted_count = 0
    units_submitted = 0
    next_write_offset = 0
    inspected_tars = 0
    next_tar_index = tar_index
    next_key_index = key_index
    write_workers = _materialize_write_workers()
    inflight: deque[Future[tuple[list[dict[str, Any]], dict[str, Any]]]] = deque()
    executor = ThreadPoolExecutor(
        max_workers=write_workers, thread_name_prefix="gpic-source-unit"
    )

    def drain_one() -> None:
        rows, _ = inflight.popleft().result()
        committed.extend(rows)
        if len(committed) % max(unit_size, 128 * unit_size) == 0:
            print(
                f"[materialize] committed_images={len(committed)}/{requested} "
                f"units={len(committed) // unit_size} writers={write_workers}",
                flush=True,
            )

    def submit_unit(items: list[dict[str, Any]]) -> None:
        nonlocal units_submitted, next_write_offset
        inflight.append(
            executor.submit(
                _write_source_unit,
                root,
                unit_index + units_submitted,
                source_index + next_write_offset,
                list(items),
            )
        )
        units_submitted += 1
        next_write_offset += len(items)
        if len(inflight) >= 2 * write_workers:
            drain_one()

    try:
        while tar_index < len(tar_paths) and accepted_count < requested:
            if max_tars is not None and inspected_tars >= int(max_tars):
                break
            repo_tar = tar_paths[tar_index]
            local_tar = _download_tar(repo_id, repo_tar, token, cache_dir)
            inspected_tars += 1
            with tarfile.open(local_tar, "r") as archive:
                json_members: dict[str, tarfile.TarInfo] = {}
                image_members: dict[str, tarfile.TarInfo] = {}
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    key, suffix = _member_key(member.name)
                    if suffix == ".json":
                        json_members[key] = member
                    elif suffix in IMAGE_SUFFIXES:
                        image_members[key] = member
                keys = _stable_key_order(
                    sorted(set(json_members) & set(image_members)), seed, repo_tar
                )
                cursor = key_index if tar_index == int(selection.get("tar_index") or 0) else 0
                while cursor < len(keys) and accepted_count < requested:
                    remaining = requested - accepted_count
                    candidate_keys: list[str] = []
                    while cursor < len(keys) and len(candidate_keys) < remaining:
                        key = keys[cursor]
                        cursor += 1
                        if key not in existing_keys and not is_excluded(
                            exclusions,
                            key,
                            image_members[key].name,
                            json_members[key].name,
                        ):
                            candidate_keys.append(key)

                    # Selection still uses the exact seeded random order above.
                    # Only physical reads are reordered; emit rows in the original
                    # selection order so IDs and cursor semantics stay unchanged.
                    loaded: dict[str, tuple[bytes, dict[str, Any]]] = {}
                    for key in _physical_key_order(candidate_keys, image_members, json_members):
                        extracted = archive.extractfile(image_members[key])
                        if extracted is None:
                            continue
                        payload = extracted.read()
                        try:
                            with Image.open(__import__("io").BytesIO(payload)) as image:
                                image.verify()
                        except Exception:
                            continue
                        loaded[key] = (payload, _json_member(archive, json_members[key]))

                    for key in candidate_keys:
                        if accepted_count >= requested:
                            break
                        item = loaded.get(key)
                        if item is None:
                            continue
                        payload, metadata = item
                        if is_excluded(
                            exclusions,
                            metadata.get("id"),
                            metadata.get("image_id"),
                            metadata.get("file_name"),
                            metadata.get("filename"),
                        ):
                            continue
                        paired_text, all_texts = _paired_text(metadata)
                        global_index = source_index + accepted_count
                        suffix = Path(image_members[key].name).suffix.lower()
                        image_id = f"gpic_{split}_{global_index:09d}_{_safe_id(Path(key).name)[:40]}"
                        if image_id in existing_ids:
                            continue
                        row = {
                            "image_id": image_id,
                            "source_context": {
                                "source_dataset": repo_id,
                                "source": "gpic",
                                "split": split,
                                "hf_tar": repo_tar,
                                "pair_key": key,
                                "paired_text": paired_text,
                                "all_texts": all_texts,
                            },
                        }
                        pending.append({"row": row, "suffix": suffix, "bytes": payload})
                        accepted_count += 1
                        existing_keys.add(key)
                        existing_ids.add(image_id)
                        if len(pending) == unit_size:
                            submit_unit(pending)
                            pending = []
                    next_tar_index = tar_index
                    next_key_index = cursor
                if cursor >= len(keys):
                    next_tar_index = tar_index + 1
                    next_key_index = 0
            tar_index = next_tar_index
            key_index = next_key_index

        if pending:
            submit_unit(pending)
            pending = []
        while inflight:
            drain_one()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if len(committed) < requested:
        raise RuntimeError(
            f"GPIC materialization committed {len(committed)} of {requested} requested images; "
            f"inspected {inspected_tars} tar(s). Increase --max-tars or remove the limit."
        )
    selection["tar_index"] = next_tar_index
    selection["key_index"] = next_key_index
    selection["last_commit_at"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat()
    return _commit_extension(root, registry, extension, committed, units_submitted)
