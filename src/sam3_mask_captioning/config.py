from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    seen = set(_seen or set())
    if path in seen:
        raise ValueError(f"Recursive config extends chain at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    parent = data.pop("extends", None)
    if parent:
        parent_path = Path(str(parent)).expanduser()
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        data = _deep_merge(load_config(parent_path, seen), data)
    return data


def project_path(config: dict[str, Any], *parts: str) -> Path:
    root = Path(config.get("project_root", ".")).expanduser().resolve()
    return root.joinpath(*parts)


def output_run_dir(config: dict[str, Any], run_id: str | None = None) -> Path:
    root = Path(config.get("output_root", "outputs")).expanduser()
    if not root.is_absolute():
        root = project_path(config, str(root))
    return root / (run_id or config.get("run_id") or "manual_run")
