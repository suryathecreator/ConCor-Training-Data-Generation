from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .caption_cleanup import clean_attributes, clean_caption
from .io_utils import append_jsonl, read_jsonl, read_jsonl_indexed, write_jsonl
from .json_utils import extract_json


MODEL_CONFIG_KEYS = {
    "backend",
    "model_name",
    "hf_home",
    "local_files_only",
    "torch_dtype",
    "device_map",
    "max_new_tokens",
    "enable_thinking",
    "temperature",
    "top_p",
    "attn_implementation",
    "low_cpu_mem_usage",
    "max_memory",
    "offload_folder",
    "offload_state_dict",
    "trust_remote_code",
    "json_schema",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_images_per_prompt",
    "enable_prefix_caching",
    "enforce_eager",
    "tensor_parallel_size",
    "mm_processor_cache_gb",
    "skip_mm_profiling",
    "safetensors_load_strategy",
    "gdn_prefill_backend",
    "flash_attn_version",
}


def qwen_model_config(config: dict[str, Any], config_section: str = "caption") -> dict[str, Any]:
    caption_config = dict(config.get("caption", {}))
    if config_section != "caption":
        caption_config = {
            key: caption_config[key] for key in MODEL_CONFIG_KEYS if key in caption_config
        }
        stage_config = config.get(config_section, {})
        for key in MODEL_CONFIG_KEYS:
            if key in stage_config:
                caption_config[key] = stage_config[key]
    # One loaded vLLM engine can serve multiple logical calls. Structured
    # decoding must follow the call, not the section that constructed the
    # engine (for example caption -> quality_filter in the combined mask stage).
    caption_config["_schema_section"] = config_section
    return caption_config


def _configured_image_keys(stage_config: dict[str, Any], default: tuple[str, ...]) -> list[str]:
    value = stage_config.get("input_image_keys") or list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _row_images(row: dict[str, Any], keys: list[str], stage: str) -> list[str]:
    missing = [key for key in keys if not row.get(key)]
    if missing:
        raise ValueError(f"{stage} requires {', '.join(missing)} for mask {row.get('mask_id')}")
    return [str(row[key]) for key in keys]


def _suffix_path(path: Path, output_suffix: str) -> Path:
    suffix = output_suffix.strip(".")
    if not suffix:
        return path
    return path.with_name(f"{path.stem}.{suffix}{path.suffix}")


def _generation_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "batch_size",
            "input_tokens",
            "output_tokens",
            "preprocess_seconds",
            "generation_seconds",
            "do_sample",
            "temperature",
            "top_p",
            "adaptive_batch_limit",
            "oom_backoff_count",
        )
        if key in result
    }


def _mask_prompt_values(row: dict[str, Any]) -> dict[str, str]:
    return {
        "object": str(row.get("object", "")),
        "caption": str(row.get("caption", "")),
        "attributes": ", ".join(row.get("attributes") or []),
        "area": str(row.get("area", "")),
        "bbox": str(row.get("bbox", "")),
        "entityseg_score": str(row.get("entityseg_score", row.get("sam3_score", ""))),
        "sam3_score": str(row.get("sam3_score", row.get("entityseg_score", ""))),
        "source_prompt": str(row.get("source_prompt", "")),
        "inverse_background_rgb": str(row.get("inverse_background_rgb", "")),
    }


def _fill_mask_prompt(template: str, row: dict[str, Any]) -> str:
    prompt = template
    for key, value in _mask_prompt_values(row).items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt


def _sharded_rows(
    rows: list[dict[str, Any]],
    shard_index: int | None,
    shard_count: int | None,
) -> list[dict[str, Any]]:
    if shard_index is None or shard_count is None or int(shard_count) <= 1:
        return [dict(row, _row_index=index) for index, row in enumerate(rows)]
    shard_index = int(shard_index)
    shard_count = int(shard_count)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}); got {shard_index}")
    return [dict(row, _row_index=index) for index, row in enumerate(rows) if index % shard_count == shard_index]


def _existing_mask_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl_indexed(path):
            mask_id = str(row.get("mask_id") or "").strip()
            if mask_id:
                ids.add(mask_id)
    return ids


def _existing_mask_review_ids(captions_path: Path, rejected_path: Path) -> set[str]:
    ids = _existing_mask_ids([captions_path])
    if not rejected_path.exists():
        return ids
    for row in read_jsonl_indexed(rejected_path):
        failure_modes = row.get("mask_review_failure_modes") or []
        if isinstance(failure_modes, str):
            failure_modes = [failure_modes]
        reason = str(row.get("mask_review_reason") or "")
        if "review_error" in failure_modes or reason.startswith("mask_review_error:"):
            continue
        mask_id = str(row.get("mask_id") or "").strip()
        if mask_id:
            ids.add(mask_id)
    return ids


def _mask_score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("sam3_score", row.get("entityseg_score", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _limit_caption_rows(
    rows: list[dict[str, Any]],
    caption_config: dict[str, Any],
) -> list[dict[str, Any]]:
    max_per_image = int(caption_config.get("max_masks_per_image", 0) or 0)
    max_total = int(caption_config.get("max_total_masks", 0) or 0)
    if max_per_image <= 0 and max_total <= 0:
        return rows

    indexed = list(enumerate(rows))
    if max_per_image > 0:
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        image_order: list[str] = []
        for index, row in indexed:
            image_id = str(row.get("image_id") or "")
            if image_id not in grouped:
                grouped[image_id] = []
                image_order.append(image_id)
            grouped[image_id].append((index, row))
        indexed = []
        for image_id in image_order:
            ranked = sorted(
                grouped[image_id],
                key=lambda item: (
                    -_mask_score(item[1]),
                    str(item[1].get("mask_id") or ""),
                ),
            )
            indexed.extend(ranked[:max_per_image])

    if max_total > 0 and len(indexed) > max_total:
        indexed = sorted(
            indexed,
            key=lambda item: (
                -_mask_score(item[1]),
                str(item[1].get("image_id") or ""),
                str(item[1].get("mask_id") or ""),
            ),
        )[:max_total]

    return [row for _, row in sorted(indexed, key=lambda item: item[0])]


def _bucketed_indexed_rows(
    rows: list[dict[str, Any]],
    input_keys: list[str],
) -> list[tuple[int, dict[str, Any]]]:
    """Stably group similarly sized visual inputs to reduce processor padding."""
    from PIL import Image

    def visual_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int]:
        index, row = item
        total_pixels = 0
        aspect_bucket = 0
        try:
            for path in _row_images(row, input_keys, "visual bucketing"):
                with Image.open(path) as image:
                    width, height = image.size
                total_pixels += max(1, width * height)
                aspect_bucket += int(round(4 * width / max(1, height)))
        except Exception:
            return (10**9, 10**9, index)
        return (max(1, total_pixels).bit_length(), aspect_bucket, index)

    return sorted(enumerate(rows), key=visual_key)


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda out of memory" in message or "cublas_status_alloc_failed" in message


def _patch_qwen35_grid_split() -> None:
    try:
        import torch
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except Exception:
        return

    cls = getattr(modeling_qwen3_5, "Qwen3_5Model", None)
    if cls is None or getattr(cls, "_mask_captioner_grid_split_patch", False):
        return

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        pixel_values = pixel_values.type(self.visual.dtype)
        kwargs.pop("return_dict", None)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        image_embeds = vision_output.pooler_output
        split_source = image_grid_thw.detach().cpu()
        split_sizes = (split_source.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        vision_output.pooler_output = torch.split(image_embeds, split_sizes)
        return vision_output

    cls.get_image_features = get_image_features
    cls._mask_captioner_grid_split_patch = True


def _mock_caption(row: dict[str, Any]) -> dict[str, Any]:
    proposal = str(row.get("source_prompt") or "object").strip().casefold()
    subject = {"people": "person", "men": "man", "women": "woman"}.get(
        proposal, proposal[:-1] if proposal.endswith("s") else proposal
    )
    caption = f"A {subject}."
    return {
        "raw": json.dumps(
            {
                "object": subject,
                "caption": caption,
                "attributes": [],
                "uncertain": False,
                "reject": False,
                "reject_reason": "",
            }
        ),
        "parsed": {
            "object": subject,
            "caption": caption,
            "attributes": [],
            "uncertain": False,
            "reject": False,
            "reject_reason": "",
        },
    }


def _mock_mask_review(row: dict[str, Any]) -> dict[str, Any]:
    keep = float(row.get("mask_area_fraction", 0.0)) >= 0.001
    parsed = {
        "keep": keep,
        "reason": "mock review kept the mask" if keep else "mock review rejected a tiny mask",
        "failure_modes": [] if keep else ["too_small"],
    }
    return {"raw": json.dumps(parsed), "parsed": parsed}


class QwenCaptioner:
    def __init__(self, config: dict[str, Any], config_section: str = "caption"):
        self.config = config
        caption = qwen_model_config(config, config_section)
        self.config_section = config_section
        self.stage_config = dict(config.get(config_section, {}))
        hf_home = caption.get("hf_home")
        if hf_home:
            hf_home_path = Path(hf_home).expanduser().resolve()
            os.environ.setdefault("HF_HOME", str(hf_home_path))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home_path / "hub"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home_path / "hub"))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        _patch_qwen35_grid_split()
        self.torch = torch
        self.caption_config = caption
        self.model_name = caption.get("model_name", "Qwen/Qwen3.5-9B")
        local_only = bool(caption.get("local_files_only", True))
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                local_files_only=local_only,
                use_fast=True,
            )
        except (ImportError, TypeError) as exc:
            message = str(exc)
            image_only_fallback = "Qwen3VLVideoProcessor requires the Torchvision library" in message
            tokenizer_fallback = "vocab_file" in message or "NoneType" in message
            if not image_only_fallback and not tokenizer_fallback:
                raise
            self.processor = self._load_qwen3vl_processor(local_only)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        model_kwargs = {
            "torch_dtype": caption.get("torch_dtype", "auto"),
            "device_map": caption.get("device_map", "auto"),
            "local_files_only": local_only,
        }
        for key in (
            "attn_implementation",
            "low_cpu_mem_usage",
            "max_memory",
            "offload_folder",
            "offload_state_dict",
            "trust_remote_code",
        ):
            if key in caption:
                model_kwargs[key] = caption[key]
        if model_kwargs.get("offload_folder"):
            Path(str(model_kwargs["offload_folder"])).expanduser().mkdir(parents=True, exist_ok=True)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
        except ValueError:
            from transformers import AutoModelForMultimodalLM

            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )

    def _load_qwen3vl_processor(self, local_only: bool):
        from huggingface_hub import snapshot_download
        from transformers import BaseVideoProcessor, Qwen2TokenizerFast, Qwen2VLImageProcessorPil, Qwen3VLProcessor

        class ImageOnlyVideoProcessor(BaseVideoProcessor):
            merge_size = 2
            temporal_patch_size = 2

            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                raise RuntimeError("SAM3 Mask Captioning uses image-only Qwen inputs; videos are unsupported.")

        snapshot_dir = Path(snapshot_download(self.model_name, local_files_only=local_only))
        with (snapshot_dir / "tokenizer_config.json").open("r", encoding="utf-8") as handle:
            tokenizer_config = json.load(handle)
        tokenizer = Qwen2TokenizerFast(
            vocab_file=str(snapshot_dir / "vocab.json"),
            merges_file=str(snapshot_dir / "merges.txt"),
            tokenizer_file=str(snapshot_dir / "tokenizer.json"),
            chat_template=tokenizer_config.get("chat_template"),
        )
        image_processor = Qwen2VLImageProcessorPil.from_pretrained(str(snapshot_dir), local_files_only=True)
        return Qwen3VLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=ImageOnlyVideoProcessor(),
            chat_template=tokenizer_config.get("chat_template"),
        )

    def _build_messages(self, images: list[str], prompt: str) -> list[dict[str, Any]]:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _generate_batch(
        self,
        image_sets: list[list[str]],
        prompts: list[str],
        seeds: list[int],
        generation_config: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        self.torch.manual_seed(int(seeds[0]))
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(int(seeds[0]))
        conversations = [self._build_messages(images, prompt) for images, prompt in zip(image_sets, prompts)]
        template_input: Any = conversations[0] if len(conversations) == 1 else conversations
        runtime_config = generation_config or self.caption_config
        preprocess_start = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            template_input,
            add_generation_prompt=True,
            enable_thinking=bool(runtime_config.get("enable_thinking", False)),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        try:
            inputs = inputs.to(self.model.device)
        except Exception:
            pass
        preprocess_seconds = time.perf_counter() - preprocess_start
        do_sample = float(runtime_config.get("temperature", 0.0)) > 0
        generate_kwargs = {
            "max_new_tokens": int(runtime_config.get("max_new_tokens", 384)),
            "do_sample": do_sample,
            "use_cache": True,
        }
        if do_sample:
            generate_kwargs["temperature"] = float(runtime_config.get("temperature", 0.0))
            generate_kwargs["top_p"] = float(runtime_config.get("top_p", 0.9))
        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = int(pad_token_id)
        generate_start = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generate_kwargs)
        generation_seconds = time.perf_counter() - generate_start
        prompt_len = inputs["input_ids"].shape[-1]
        trimmed = generated[:, prompt_len:]
        raws = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        input_counts = inputs.get("attention_mask")
        if input_counts is not None:
            input_counts = input_counts.sum(dim=1).detach().cpu().tolist()
        else:
            input_counts = [prompt_len] * len(raws)
        return [
            {
                "raw": raw,
                "batch_size": len(raws),
                "input_tokens": int(input_counts[index]),
                "output_tokens": int(
                    trimmed[index].shape[-1]
                    if pad_token_id is None
                    else (trimmed[index] != int(pad_token_id)).sum().item()
                ),
                "preprocess_seconds": preprocess_seconds,
                "generation_seconds": generation_seconds,
                "do_sample": do_sample,
                "temperature": float(runtime_config.get("temperature", 0.0)),
                "top_p": float(runtime_config.get("top_p", 0.9)),
            }
            for index, raw in enumerate(raws)
        ]

    def generate_many(
        self,
        image_sets: list[list[str]],
        prompts: list[str],
        seeds: list[int],
        batch_size: int | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        if not image_sets:
            return []
        if len(image_sets) != len(prompts) or len(image_sets) != len(seeds):
            raise ValueError("image_sets, prompts, and seeds must have the same length")
        runtime_config = generation_config or self.caption_config
        configured_batch = int(batch_size or runtime_config.get("batch_size", 1) or 1)
        configured_batch = max(1, configured_batch)
        workload_key = (
            int(runtime_config.get("max_new_tokens", 384)),
            max(len(images) for images in image_sets),
            configured_batch,
        )
        safe_limits = getattr(self, "_safe_batch_size_by_workload", {})
        safe_batch = min(configured_batch, int(safe_limits.get(workload_key, configured_batch)))
        results: list[dict[str, str]] = []
        index = 0
        current_batch = safe_batch
        oom_backoff_count = 0
        while index < len(image_sets):
            size = min(current_batch, len(image_sets) - index)
            try:
                chunk_results = self._generate_batch(
                        image_sets[index : index + size],
                        prompts[index : index + size],
                        seeds[index : index + size],
                        generation_config=runtime_config,
                    )
                results.extend(chunk_results)
                index += size
                current_batch = safe_batch
            except RuntimeError as exc:
                if not _is_cuda_oom(exc) or size == 1:
                    raise
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
                safe_batch = max(1, size // 2)
                safe_limits[workload_key] = safe_batch
                self._safe_batch_size_by_workload = safe_limits
                current_batch = safe_batch
                oom_backoff_count += 1
        for result in results:
            result["adaptive_batch_limit"] = safe_batch
            result["oom_backoff_count"] = oom_backoff_count
        return results

    def generate_many_bcc(
        self,
        image_sets: list[list[str]],
        prompts: list[str],
        seeds: list[int],
        batch_size: int,
        generation_config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Batch BCC packets with an OOM limit remembered per workload."""
        if not image_sets:
            return []
        if len(image_sets) != len(prompts) or len(image_sets) != len(seeds):
            raise ValueError("image_sets, prompts, and seeds must have the same length")
        configured = max(1, int(batch_size or 1))
        runtime_config = generation_config or self.caption_config
        workload_key = (
            int(runtime_config.get("max_new_tokens", 2048)),
            max(len(images) for images in image_sets),
            configured,
        )
        safe_limits = getattr(self, "_bcc_safe_batch_size_by_workload", {})
        safe_batch = min(
            configured, int(safe_limits.get(workload_key, configured))
        )
        results: list[dict[str, Any]] = []
        index = 0
        oom_backoff_count = 0
        while index < len(image_sets):
            size = min(safe_batch, len(image_sets) - index)
            try:
                results.extend(
                    self._generate_batch(
                        image_sets[index : index + size],
                        prompts[index : index + size],
                        seeds[index : index + size],
                        generation_config=runtime_config,
                    )
                )
                index += size
            except RuntimeError as exc:
                if not _is_cuda_oom(exc) or size == 1:
                    raise
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
                safe_batch = max(1, size // 2)
                safe_limits[workload_key] = safe_batch
                self._bcc_safe_batch_size_by_workload = safe_limits
                oom_backoff_count += 1
        for result in results:
            result["adaptive_batch_limit"] = safe_batch
            result["oom_backoff_count"] = oom_backoff_count
        return results

    def generate(
        self,
        images: list[str],
        prompt: str,
        seed: int,
        generation_config: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        return self.generate_many(
            [images],
            [prompt],
            [seed],
            batch_size=1,
            generation_config=generation_config,
        )[0]

    def caption(self, row: dict[str, Any], seed: int) -> dict[str, Any]:
        stage_config = getattr(self, "stage_config", getattr(self, "caption_config", {}))
        prompt = _fill_mask_prompt(stage_config.get("prompt") or self.caption_config["prompt"], row)
        keys = _configured_image_keys(stage_config, ("inverse_crop_path",))
        return self.generate(
            _row_images(row, keys, "captioning"),
            prompt,
            seed,
            generation_config=qwen_model_config(self.config, "caption"),
        )

    def mask_review(self, row: dict[str, Any], seed: int, prompt_template: str) -> dict[str, Any]:
        stage_config = getattr(self, "stage_config", getattr(self, "caption_config", {}))
        prompt = _fill_mask_prompt(prompt_template, row)
        keys = _configured_image_keys(
            stage_config,
            ("inverse_crop_path",),
        )
        return self.generate(
            _row_images(row, keys, "mask review"),
            prompt,
            seed,
            generation_config=qwen_model_config(self.config, "quality_filter"),
        )


def create_captioner(
    config: dict[str, Any], config_section: str = "caption"
) -> QwenCaptioner:
    """Construct the configured Qwen backend behind the stable captioner API."""
    runtime = qwen_model_config(config, config_section)
    backend = str(
        runtime.get("backend")
        or (config.get("inference") or {}).get("backend")
        or "transformers"
    ).strip().lower()
    if backend in {"vllm", "vllm-offline", "offline-vllm"}:
        from .vllm_backend import VLLMCaptioner

        return VLLMCaptioner(config, config_section=config_section)  # type: ignore[return-value]
    if backend not in {"qwen", "hf", "huggingface", "transformers"}:
        raise ValueError(f"Unsupported Qwen inference backend: {backend}")
    return QwenCaptioner(config, config_section=config_section)


def _normalize(parsed: dict[str, Any]) -> dict[str, Any]:
    attributes = parsed.get("attributes") or []
    if isinstance(attributes, str):
        attributes = [attributes]
    reject = bool(parsed.get("reject", False))
    reject_reason = str(parsed.get("reject_reason") or "").strip()
    if reject and not reject_reason:
        reject_reason = "captioner_rejected_mask"
    return {
        "object": str(parsed.get("object") or "").strip(),
        "caption": str(parsed.get("caption") or "").strip(),
        "attributes": [str(item).strip() for item in attributes if str(item).strip()],
        "uncertain": bool(parsed.get("uncertain", False)),
        "reject": reject,
        "reject_reason": reject_reason,
    }


def _normalize_mask_review(parsed: dict[str, Any]) -> dict[str, Any]:
    failure_modes = parsed.get("failure_modes") or []
    if isinstance(failure_modes, str):
        failure_modes = [failure_modes]
    return {
        "keep": bool(parsed.get("keep", False)),
        "reason": str(parsed.get("reason") or "").strip(),
        "failure_modes": [str(item).strip() for item in failure_modes if str(item).strip()],
        "corrected_object": str(parsed.get("corrected_object") or "").strip(),
        "corrected_caption": str(parsed.get("corrected_caption") or "").strip(),
        "corrected_attributes": parsed.get("corrected_attributes") or [],
    }


def _mask_review_output(row: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("original_caption", row.get("caption", ""))
    out["mask_review_keep"] = review["keep"]
    out["mask_review_reason"] = review["reason"]
    out["mask_review_failure_modes"] = review["failure_modes"]
    out["qa_corrected_object"] = review.get("corrected_object") or ""
    out["qa_corrected_caption"] = review.get("corrected_caption") or ""
    corrected_attributes = review.get("corrected_attributes") or []
    if isinstance(corrected_attributes, list):
        out["qa_corrected_attributes"] = [str(item).strip() for item in corrected_attributes if str(item).strip()]
    elif corrected_attributes:
        out["qa_corrected_attributes"] = [str(corrected_attributes).strip()]
    else:
        out["qa_corrected_attributes"] = []
    if review["keep"]:
        if out["qa_corrected_object"]:
            out["object"] = out["qa_corrected_object"]
        if out["qa_corrected_caption"]:
            out["caption_before_qa_correction"] = out.get("caption", "")
            out["caption"] = out["qa_corrected_caption"]
        if out["qa_corrected_attributes"]:
            out["attributes_before_qa_correction"] = out.get("attributes", [])
            out["attributes"] = out["qa_corrected_attributes"]
        caption_cleanup = clean_caption(str(out.get("caption") or ""))
        attribute_cleanup = clean_attributes(out.get("attributes") or [])
        out["caption"] = caption_cleanup["caption"]
        out["attributes"] = attribute_cleanup["attributes"]
        out["qa_caption_cleanup"] = caption_cleanup
        out["qa_attribute_cleanup"] = attribute_cleanup
        if not caption_cleanup["valid"]:
            out["mask_review_keep"] = False
            out["mask_review_reason"] = "spaCy cleanup removed the entire caption"
            out["mask_review_failure_modes"] = list(out["mask_review_failure_modes"]) + ["empty_after_cleanup"]
    return out


def run_captioning(
    config: dict[str, Any],
    run_dir: Path,
    masks_path: str | Path | None = None,
    rows_override: list[dict[str, Any]] | None = None,
    mock: bool = False,
    limit: int | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    output_suffix: str = "",
    captioner_override: QwenCaptioner | None = None,
) -> Path:
    input_path = Path(masks_path or run_dir / "sam3_masks.jsonl")
    rows = list(rows_override) if rows_override is not None else (read_jsonl(input_path) if input_path.exists() else [])
    quality_enabled = bool(config.get("quality_filter", {}).get("enabled", False))
    if not output_suffix and shard_index is not None and shard_count is not None and int(shard_count) > 1:
        output_suffix = f"shard{int(shard_index)}"
    captions_path = _suffix_path(run_dir / ("caption_candidates.jsonl" if quality_enabled else "captions.jsonl"), output_suffix)
    caption_rejected_path = _suffix_path(run_dir / "caption_rejected_masks.jsonl", output_suffix)
    raw_path = _suffix_path(run_dir / "caption_raw.jsonl", output_suffix)
    errors_path = _suffix_path(run_dir / "caption_errors.jsonl", output_suffix)
    if limit is not None:
        rows = rows[: int(limit)]
    caption_config = config.get("caption", {})
    rows = _limit_caption_rows(rows, caption_config)
    rows = _sharded_rows(rows, shard_index, shard_count)
    resume = bool(config.get("resume", False) or caption_config.get("resume", False))
    if resume:
        completed_mask_ids = _existing_mask_ids([captions_path, caption_rejected_path])
        rows = [row for row in rows if str(row.get("mask_id") or "") not in completed_mask_ids]
    else:
        for stale in (captions_path, caption_rejected_path, raw_path, errors_path):
            if stale.exists():
                stale.unlink()
    if not rows:
        if not captions_path.exists():
            write_jsonl([], captions_path)
        if not caption_rejected_path.exists():
            write_jsonl([], caption_rejected_path)
        return captions_path
    captioner = None if mock else (captioner_override or QwenCaptioner(config, config_section="caption"))
    seed_base = int(config.get("random_seed", 17))
    prompt = caption_config.get("prompt", "")
    input_keys = _configured_image_keys(caption_config, ("inverse_crop_path",))
    batch_size = int(caption_config.get("batch_size", 1) or 1)

    def write_result(row: dict[str, Any], idx: int, result: dict[str, Any]) -> None:
        append_jsonl(
            {
                "mask_id": row["mask_id"],
                "raw": result["raw"],
                "_row_index": row.get("_row_index", idx),
                **_generation_metrics(result),
            },
            raw_path,
        )
        parsed_obj = result.get("parsed") or extract_json(result["raw"])
        parsed = _normalize(parsed_obj)
        caption_cleanup = clean_caption(parsed["caption"])
        attribute_cleanup = clean_attributes(parsed["attributes"])
        if not caption_cleanup["valid"] and not parsed["reject"]:
            parsed["reject"] = True
            parsed["reject_reason"] = "empty_after_spacy_cleanup"
        out = {
            "_row_index": row.get("_row_index", idx),
            "image_id": row["image_id"],
            "mask_id": row["mask_id"],
            "bbox": row["bbox"],
            "area": row["area"],
            "bbox_area": row.get("bbox_area"),
            "mask_area_fraction": row.get("mask_area_fraction"),
            "object": parsed["object"],
            "caption": caption_cleanup["caption"],
            "original_caption": parsed["caption"],
            "qwen_caption_before_cleanup": parsed["caption"],
            "caption_cleanup": caption_cleanup,
            "attributes": attribute_cleanup["attributes"],
            "qwen_attributes_before_cleanup": parsed["attributes"],
            "attribute_cleanup": attribute_cleanup,
            "uncertain": parsed["uncertain"],
            "caption_reject": parsed["reject"],
            "caption_reject_reason": parsed["reject_reason"],
            "sam3_score": row.get("sam3_score", row.get("entityseg_score")),
            "entityseg_score": row.get("entityseg_score", row.get("sam3_score")),
            "source_prompt": row.get("source_prompt", ""),
            "model": config["caption"].get("model_name", "Qwen/Qwen3.5-9B"),
            "source_image_path": row["source_image_path"],
            "mask_path": row["mask_path"],
            "full_overlay_path": row.get("full_overlay_path"),
            "crop_overlay_path": row.get("crop_overlay_path"),
            "crop_image_path": row.get("crop_image_path"),
            "inverse_crop_path": row.get("inverse_crop_path"),
            "inverse_background_rgb": row.get("inverse_background_rgb"),
            "inverse_background_selection": row.get("inverse_background_selection"),
        }
        if parsed["reject"]:
            append_jsonl(out, caption_rejected_path)
        else:
            append_jsonl(out, captions_path)

    def write_error(row: dict[str, Any], idx: int, exc: BaseException) -> None:
        append_jsonl(
            {
                "_row_index": row.get("_row_index", idx),
                "image_id": row.get("image_id"),
                "mask_id": row.get("mask_id"),
                "stage": "caption",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
            errors_path,
        )

    if not mock and batch_size > 1:
        bucketed_rows = _bucketed_indexed_rows(rows, input_keys)
        for batch_start in tqdm(range(0, len(rows), batch_size), desc="caption"):
            batch_rows = bucketed_rows[batch_start : batch_start + batch_size]
            image_sets: list[list[str]] = []
            prompts: list[str] = []
            seeds: list[int] = []
            prepared: list[tuple[int, dict[str, Any]]] = []
            for idx, row in batch_rows:
                try:
                    image_sets.append(_row_images(row, input_keys, "captioning"))
                    prompts.append(_fill_mask_prompt(prompt, row))
                    seeds.append(seed_base + int(row.get("_row_index", idx)))
                    prepared.append((idx, row))
                except Exception as exc:
                    write_error(row, idx, exc)
                    if not config.get("continue_on_error", True):
                        raise
            if not prepared:
                continue
            try:
                results = captioner.generate_many(
                    image_sets,
                    prompts,
                    seeds,
                    batch_size=batch_size,
                    generation_config=qwen_model_config(config, "caption"),
                )
                for (idx, row), result in zip(prepared, results):
                    write_result(row, idx, result)
            except Exception as batch_exc:
                append_jsonl(
                    {
                        "stage": "caption_batch",
                        "mask_ids": [row.get("mask_id") for _, row in prepared],
                        "batch_size": len(prepared),
                        "batch_fallback": True,
                        "error": repr(batch_exc),
                        "traceback": traceback.format_exc(),
                    },
                    errors_path,
                )
                for idx, row in prepared:
                    try:
                        result = captioner.caption(row, seed_base + int(row.get("_row_index", idx)))
                        write_result(row, idx, result)
                    except Exception as exc:
                        write_error(row, idx, exc)
                        if not config.get("continue_on_error", True):
                            raise
        return captions_path

    for idx, row in enumerate(tqdm(rows, desc="caption")):
        try:
            result = _mock_caption(row) if mock else captioner.caption(row, seed_base + int(row.get("_row_index", idx)))
            write_result(row, idx, result)
        except Exception as exc:
            write_error(row, idx, exc)
            if not config.get("continue_on_error", True):
                raise
    return captions_path


def run_mask_review(
    config: dict[str, Any],
    run_dir: Path,
    candidates_path: str | Path | None = None,
    rows_override: list[dict[str, Any]] | None = None,
    mock: bool = False,
    limit: int | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    output_suffix: str = "",
    captioner_override: QwenCaptioner | None = None,
) -> Path:
    input_path = Path(candidates_path or run_dir / "caption_candidates.jsonl")
    rows = list(rows_override) if rows_override is not None else (read_jsonl(input_path) if input_path.exists() else [])
    if not output_suffix and shard_index is not None and shard_count is not None and int(shard_count) > 1:
        output_suffix = f"shard{int(shard_index)}"
    captions_path = _suffix_path(run_dir / "captions.jsonl", output_suffix)
    rejected_path = _suffix_path(run_dir / "rejected_captions.jsonl", output_suffix)
    reviews_path = _suffix_path(run_dir / "mask_quality_reviews.jsonl", output_suffix)
    raw_path = _suffix_path(run_dir / "mask_quality_raw.jsonl", output_suffix)
    errors_path = _suffix_path(run_dir / "mask_review_errors.jsonl", output_suffix)
    if limit is not None:
        rows = rows[: int(limit)]
    rows = _sharded_rows(rows, shard_index, shard_count)
    quality_config = config.get("quality_filter", {})
    resume = bool(config.get("resume", False) or quality_config.get("resume", False))
    if resume:
        completed_mask_ids = _existing_mask_review_ids(captions_path, rejected_path)
        rows = [row for row in rows if str(row.get("mask_id") or "") not in completed_mask_ids]
    else:
        for stale in (captions_path, rejected_path, reviews_path, raw_path, errors_path):
            if stale.exists():
                stale.unlink()

    def ensure_output_files() -> None:
        for empty_path in (captions_path, rejected_path, reviews_path):
            if not empty_path.exists():
                write_jsonl([], empty_path)

    if not rows:
        ensure_output_files()
        return captions_path
    captioner = None if mock else (captioner_override or QwenCaptioner(config, config_section="quality_filter"))
    seed_base = int(config.get("random_seed", 17)) + int(config.get("quality_filter", {}).get("mask_review_seed_offset", 200000))
    prompt = quality_config.get("mask_review_prompt", "")
    input_keys = _configured_image_keys(
        quality_config,
        ("inverse_crop_path",),
    )
    batch_size = int(quality_config.get("batch_size", 1) or 1)

    def prompt_for(row: dict[str, Any]) -> str:
        return _fill_mask_prompt(prompt, row)

    def write_result(row: dict[str, Any], idx: int, result: dict[str, Any]) -> None:
        append_jsonl(
            {
                "mask_id": row["mask_id"],
                "raw": result["raw"],
                "_row_index": row.get("_row_index", idx),
                **_generation_metrics(result),
            },
            raw_path,
        )
        parsed_obj = result.get("parsed") or extract_json(result["raw"])
        review = _normalize_mask_review(parsed_obj)
        review_row = {
            "_row_index": row.get("_row_index", idx),
            "image_id": row["image_id"],
            "mask_id": row["mask_id"],
            "keep": review["keep"],
            "reason": review["reason"],
            "failure_modes": review["failure_modes"],
            "corrected_object": review["corrected_object"],
            "corrected_caption": review["corrected_caption"],
            "corrected_attributes": review["corrected_attributes"],
            "model": qwen_model_config(config, "quality_filter").get("model_name", "Qwen/Qwen3.5-9B"),
        }
        append_jsonl(review_row, reviews_path)
        out = _mask_review_output(row, review)
        if out["mask_review_keep"]:
            append_jsonl(out, captions_path)
        else:
            append_jsonl(out, rejected_path)

    def write_error(row: dict[str, Any], idx: int, exc: BaseException) -> None:
        error_review = {
            "_row_index": row.get("_row_index", idx),
            "image_id": row.get("image_id"),
            "mask_id": row.get("mask_id"),
            "stage": "mask_review",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        append_jsonl(error_review, errors_path)
        rejected = dict(row)
        rejected.setdefault("original_caption", row.get("caption", ""))
        rejected["mask_review_keep"] = False
        rejected["mask_review_reason"] = f"mask_review_error: {exc!r}"
        rejected["mask_review_failure_modes"] = ["review_error"]
        rejected["qa_corrected_object"] = ""
        rejected["qa_corrected_caption"] = ""
        rejected["qa_corrected_attributes"] = []
        append_jsonl(rejected, rejected_path)

    if not mock and batch_size > 1:
        bucketed_rows = _bucketed_indexed_rows(rows, input_keys)
        for batch_start in tqdm(range(0, len(rows), batch_size), desc="mask-review"):
            batch_rows = bucketed_rows[batch_start : batch_start + batch_size]
            image_sets: list[list[str]] = []
            prompts: list[str] = []
            seeds: list[int] = []
            prepared: list[tuple[int, dict[str, Any]]] = []
            for idx, row in batch_rows:
                try:
                    image_sets.append(_row_images(row, input_keys, "mask review"))
                    prompts.append(prompt_for(row))
                    seeds.append(seed_base + int(row.get("_row_index", idx)))
                    prepared.append((idx, row))
                except Exception as exc:
                    write_error(row, idx, exc)
                    if not config.get("continue_on_error", True):
                        raise
            if not prepared:
                continue
            try:
                results = captioner.generate_many(
                    image_sets,
                    prompts,
                    seeds,
                    batch_size=batch_size,
                    generation_config=qwen_model_config(config, "quality_filter"),
                )
                for (idx, row), result in zip(prepared, results):
                    write_result(row, idx, result)
            except Exception as batch_exc:
                append_jsonl(
                    {
                        "stage": "mask_review_batch",
                        "mask_ids": [row.get("mask_id") for _, row in prepared],
                        "batch_size": len(prepared),
                        "batch_fallback": True,
                        "error": repr(batch_exc),
                        "traceback": traceback.format_exc(),
                    },
                    errors_path,
                )
                for idx, row in prepared:
                    try:
                        result = captioner.mask_review(row, seed_base + int(row.get("_row_index", idx)), prompt)
                        write_result(row, idx, result)
                    except Exception as exc:
                        write_error(row, idx, exc)
                        if not config.get("continue_on_error", True):
                            raise
        ensure_output_files()
        return captions_path

    for idx, row in enumerate(tqdm(rows, desc="mask-review")):
        try:
            result = _mock_mask_review(row) if mock else captioner.mask_review(
                row,
                seed_base + int(row.get("_row_index", idx)),
                prompt,
            )
            write_result(row, idx, result)
        except Exception as exc:
            write_error(row, idx, exc)
            if not config.get("continue_on_error", True):
                raise
    ensure_output_files()
    return captions_path


def merge_sharded_outputs(run_dir: str | Path, stage: str, shard_count: int) -> dict[str, str]:
    run_dir = Path(run_dir)
    if stage == "caption":
        bases = [
            "caption_candidates.jsonl",
            "captions.jsonl",
            "caption_rejected_masks.jsonl",
            "caption_raw.jsonl",
            "caption_errors.jsonl",
        ]
    elif stage == "mask-review":
        bases = [
            "captions.jsonl",
            "rejected_captions.jsonl",
            "mask_quality_reviews.jsonl",
            "mask_quality_raw.jsonl",
            "mask_review_errors.jsonl",
        ]
    else:
        raise ValueError(f"Unsupported sharded stage: {stage}")

    def sort_key(row: dict[str, Any]) -> tuple[Any, str, str]:
        return (
            row.get("_row_index", 10**12),
            str(row.get("image_id") or ""),
            str(row.get("mask_id") or row.get("reject_id") or ""),
        )

    merged: dict[str, str] = {}
    for base in bases:
        out_path = run_dir / base
        rows: list[dict[str, Any]] = []
        found = False
        for shard_index in range(int(shard_count)):
            shard_path = _suffix_path(out_path, f"shard{shard_index}")
            if shard_path.exists():
                found = True
                rows.extend(read_jsonl(shard_path))
        if not found:
            continue
        write_jsonl(sorted(rows, key=sort_key), out_path)
        merged[base] = str(out_path)
    return merged
