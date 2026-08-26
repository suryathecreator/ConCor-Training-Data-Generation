from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from .caption_stage import qwen_model_config
from .io_utils import git_commit, sha256_file, slurm_metadata, write_json


def write_run_metadata(config: dict[str, Any], run_dir: Path) -> None:
    project_root = Path(config.get("project_root", ".")).resolve()
    manifest = Path(config["dataset"]["manifest_path"]).expanduser()
    if not manifest.is_absolute():
        manifest = project_root / manifest
    sam3_config = config.get("sam3", {})
    sam3_value = str(os.environ.get("SAM3_REPO_ROOT") or sam3_config.get("repo_root") or "")
    sam3_repo = Path(sam3_value).expanduser() if sam3_value else None
    if sam3_repo is not None and not sam3_repo.is_absolute():
        sam3_repo = project_root / sam3_repo
    checkpoint_value = sam3_config.get("checkpoint_path") or ""
    checkpoint = Path(checkpoint_value).expanduser() if checkpoint_value else None
    if checkpoint is not None and not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pipeline_commit": git_commit(project_root),
        "sam3_commit": git_commit(sam3_repo) if sam3_repo is not None else None,
        "source_manifest_name": manifest.name,
        "source_manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
        "sam3_checkpoint_sha256": sha256_file(checkpoint) if checkpoint is not None and checkpoint.exists() else None,
        "slurm": slurm_metadata(),
        "env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
        },
    }
    write_json(metadata, run_dir / "run_metadata.json")
    write_json(
        {
            "mask_model": "SAM3",
            "mask_config": sam3_config,
            "image_review_model": qwen_model_config(config, "image_review").get("model_name", "Qwen/Qwen3.5-9B"),
            "caption_model": qwen_model_config(config, "caption").get("model_name", "Qwen/Qwen3.5-9B"),
            "mask_review_model": qwen_model_config(config, "quality_filter").get("model_name", "Qwen/Qwen3.5-9B"),
            "image_caption_model": qwen_model_config(config, "image_caption").get("model_name", "Qwen/Qwen3.5-9B"),
            "image_caption_qa_model": qwen_model_config(config, "image_caption_qa").get("model_name", "Qwen/Qwen3.5-9B"),
            "sam3_consistency_metric": config.get("consistency_filter", {}).get("metric", "mask_iou"),
            "sam3_consistency_threshold": config.get("consistency_filter", {}).get("mask_iou_threshold", 0.5),
            "correspondence_schema_version": "bcc-image-text-v1",
            "caption_hf_home": config["caption"].get("hf_home"),
        },
        run_dir / "model_metadata.json",
    )
