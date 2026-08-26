from __future__ import annotations

from pathlib import Path

from .io_utils import sha256_file


def write_checksums(paths: list[str | Path], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            path = Path(path)
            if path.exists() and path.is_file():
                handle.write(f"{sha256_file(path)}  {path}\n")
