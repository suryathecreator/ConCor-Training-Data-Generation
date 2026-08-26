from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_JSONL_INDEX: dict[Path, dict[str, Any]] = {}


def read_jsonl_indexed(path: str | Path) -> list[dict[str, Any]]:
    """Incrementally read an append-only JSONL file, resetting after replacement."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        return []
    stat = resolved.stat()
    state = _JSONL_INDEX.get(resolved)
    if state is None or state["inode"] != stat.st_ino or stat.st_size < state["offset"]:
        state = {"inode": stat.st_ino, "offset": 0, "rows": []}
        _JSONL_INDEX[resolved] = state
    if stat.st_size != state["offset"]:
        with resolved.open("r", encoding="utf-8") as handle:
            handle.seek(state["offset"])
            payload = handle.read()
            state["offset"] = handle.tell()
        for line in payload.splitlines():
            if line.strip():
                state["rows"].append(json.loads(line))
    return list(state["rows"])


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _JSONL_INDEX.pop(path.resolve(), None)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def append_jsonl(
    row: dict[str, Any], path: str | Path, *, durable: bool = False
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        if durable:
            handle.flush()
            os.fsync(handle.fileno())


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def slurm_metadata() -> dict[str, str | None]:
    keys = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_GPUS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_SUBMIT_DIR",
    ]
    return {key.lower(): os.environ.get(key) for key in keys}
