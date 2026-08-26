from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .caption_stage import (
    _configured_image_keys,
    _fill_mask_prompt,
    _row_images,
    qwen_model_config,
)


_STAGE_SCHEMAS: dict[str, dict[str, Any]] = {
    "image_review": {
        "type": "object",
        "properties": {
            "worth_segmenting": {"type": "boolean"},
            "estimated_maskable_entities": {"type": "integer"},
            "image_type": {"type": "string"},
            "rationale": {"type": "string"},
            "reject_reason": {"type": "string"},
            "sam3_prompts": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "worth_segmenting",
            "estimated_maskable_entities",
            "image_type",
            "rationale",
            "reject_reason",
            "sam3_prompts",
        ],
        "additionalProperties": False,
    },
    "caption": {
        "type": "object",
        "properties": {
            "reject": {"type": "boolean"},
            "reject_reason": {"type": "string"},
            "object": {"type": "string"},
            "caption": {"type": "string"},
            "attributes": {"type": "array", "items": {"type": "string"}},
            "uncertain": {"type": "boolean"},
        },
        "required": [
            "reject",
            "reject_reason",
            "object",
            "caption",
            "attributes",
            "uncertain",
        ],
        "additionalProperties": False,
    },
    "quality_filter": {
        "type": "object",
        "properties": {
            "keep": {"type": "boolean"},
            "reason": {"type": "string"},
            "failure_modes": {"type": "array", "items": {"type": "string"}},
            "corrected_object": {"type": "string"},
            "corrected_caption": {"type": "string"},
            "corrected_attributes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "keep",
            "reason",
            "failure_modes",
            "corrected_object",
            "corrected_caption",
            "corrected_attributes",
        ],
        "additionalProperties": False,
    },
    "image_caption": {
        "type": "object",
        "properties": {
            "reject": {"type": "boolean"},
            "tagged_caption": {"type": "string"},
        },
        "required": ["reject", "tagged_caption"],
        "additionalProperties": False,
    },
    "image_caption_qa": {
        "type": "object",
        "properties": {
            "reject": {"type": "boolean"},
            "tagged_caption": {"type": "string"},
        },
        "required": ["reject", "tagged_caption"],
        "additionalProperties": False,
    },
}


def _structured_schema_for(
    runtime: dict[str, Any], default_section: str
) -> dict[str, Any] | None:
    explicit = runtime.get("json_schema")
    if explicit:
        return explicit
    section = str(runtime.get("_schema_section") or default_section)
    return _STAGE_SCHEMAS.get(section)


class VLLMCaptioner:
    """Offline vLLM implementation of the existing QwenCaptioner interface."""

    def __init__(self, config: dict[str, Any], config_section: str = "caption"):
        self.config = config
        self.config_section = config_section
        self.stage_config = dict(config.get(config_section, {}))
        self.caption_config = qwen_model_config(config, config_section)
        self.model_name = str(
            os.environ.get("BCC_QWEN_MODEL_PATH")
            or self.caption_config.get("model_name", "Qwen/Qwen3.5-9B")
        )
        hf_home = self.caption_config.get("hf_home")
        if hf_home:
            hf_home_path = Path(str(hf_home)).expanduser().resolve()
            os.environ.setdefault("HF_HOME", str(hf_home_path))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home_path / "hub"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home_path / "hub"))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from transformers import AutoProcessor
        from vllm import LLM

        local_only = bool(self.caption_config.get("local_files_only", True))
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            local_files_only=local_only,
            trust_remote_code=bool(self.caption_config.get("trust_remote_code", False)),
        )
        max_images = int(self.caption_config.get("max_images_per_prompt", 128))
        env_model_len = os.environ.get("BCC_VLLM_MAX_MODEL_LEN")
        env_max_seqs = os.environ.get("BCC_VLLM_MAX_NUM_SEQS")
        env_memory = os.environ.get("BCC_VLLM_GPU_MEMORY_UTILIZATION")
        env_tensor_parallel = os.environ.get("BCC_VLLM_TENSOR_PARALLEL_SIZE")
        engine_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "dtype": self.caption_config.get("torch_dtype", "bfloat16"),
            "trust_remote_code": bool(self.caption_config.get("trust_remote_code", False)),
            "gpu_memory_utilization": float(env_memory or self.caption_config.get("gpu_memory_utilization", 0.88)),
            "max_model_len": int(env_model_len or self.caption_config.get("max_model_len", 65_536)),
            "max_num_seqs": int(env_max_seqs or self.caption_config.get("max_num_seqs", 16)),
            "tensor_parallel_size": int(
                env_tensor_parallel
                or self.caption_config.get("tensor_parallel_size", 1)
            ),
            "enable_prefix_caching": bool(self.caption_config.get("enable_prefix_caching", True)),
            "enforce_eager": bool(self.caption_config.get("enforce_eager", False)),
            "limit_mm_per_prompt": {"image": max_images},
            "skip_mm_profiling": bool(
                self.caption_config.get("skip_mm_profiling", False)
            ),
            "seed": int(config.get("random_seed", 17)),
        }
        for key in ("max_num_batched_tokens", "mm_processor_cache_gb"):
            value = self.caption_config.get(key)
            if value is not None:
                engine_kwargs[key] = value
        load_strategy = self.caption_config.get("safetensors_load_strategy")
        if load_strategy:
            engine_kwargs["safetensors_load_strategy"] = str(load_strategy)
        gdn_prefill_backend = self.caption_config.get("gdn_prefill_backend")
        if gdn_prefill_backend:
            # FlashInfer's GDN kernel is JIT-compiled with nvcc. Some clusters'
            # compute images do not expose nvcc, so use vLLM's supported
            # in-tree Triton implementation instead.
            engine_kwargs["gdn_prefill_backend"] = str(gdn_prefill_backend)
        flash_attn_version = self.caption_config.get("flash_attn_version")
        if flash_attn_version is not None:
            # Keep mixed GPU cohorts on one attention implementation. The
            # Hopper FA3 path produced materially different greedy text on a
            # 116-mask stress packet, running to the output cap where FA2
            # closed valid JSON. A common FA2 path is both more reproducible
            # and faster end-to-end for that observed workload.
            engine_kwargs["attention_config"] = {
                "flash_attn_version": int(flash_attn_version)
            }
        self.engine = LLM(**engine_kwargs)

    @staticmethod
    def _messages(images: list[Image.Image], prompt: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image} for image in images
        ]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _load_images(paths: list[str]) -> list[Image.Image]:
        loaded: list[Image.Image] = []
        for path in paths:
            with Image.open(path) as handle:
                loaded.append(handle.convert("RGB").copy())
        return loaded

    def _sampling_params(
        self, runtime: dict[str, Any], seed: int
    ) -> Any:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        temperature = float(runtime.get("temperature", 0.0))
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": int(runtime.get("max_new_tokens", 384)),
            "seed": int(seed),
        }
        if temperature > 0:
            kwargs["top_p"] = float(runtime.get("top_p", 0.9))
        schema = _structured_schema_for(runtime, self.config_section)
        if schema:
            kwargs["structured_outputs"] = StructuredOutputsParams(json=schema)
        return SamplingParams(**kwargs)

    def generate_many(
        self,
        image_sets: list[list[str]],
        prompts: list[str],
        seeds: list[int],
        batch_size: int | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not image_sets:
            return []
        if len(image_sets) != len(prompts) or len(image_sets) != len(seeds):
            raise ValueError("image_sets, prompts, and seeds must have the same length")
        runtime = dict(generation_config or self.caption_config)
        preprocess_started = time.perf_counter()
        requests: list[dict[str, Any]] = []
        loaded_by_request: list[list[Image.Image]] = []
        for image_paths, prompt in zip(image_sets, prompts):
            images = self._load_images(image_paths)
            loaded_by_request.append(images)
            messages = self._messages(images, prompt)
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(runtime.get("enable_thinking", False)),
            )
            request: dict[str, Any] = {"prompt": text_prompt}
            if images:
                request["multi_modal_data"] = {
                    "image": images[0] if len(images) == 1 else images
                }
            requests.append(request)
        preprocess_seconds = time.perf_counter() - preprocess_started
        sampling = [
            self._sampling_params(runtime, seed) for seed in seeds
        ]
        generation_started = time.perf_counter()
        outputs = self.engine.generate(requests, sampling_params=sampling, use_tqdm=False)
        generation_seconds = time.perf_counter() - generation_started
        results: list[dict[str, Any]] = []
        for output in outputs:
            choice = output.outputs[0]
            results.append(
                {
                    "raw": choice.text,
                    "batch_size": len(requests),
                    "input_tokens": len(output.prompt_token_ids or []),
                    "output_tokens": len(choice.token_ids or []),
                    "preprocess_seconds": preprocess_seconds,
                    "generation_seconds": generation_seconds,
                    "do_sample": float(runtime.get("temperature", 0.0)) > 0,
                    "temperature": float(runtime.get("temperature", 0.0)),
                    "top_p": float(runtime.get("top_p", 0.9)),
                    "adaptive_batch_limit": int(batch_size or len(requests)),
                    "oom_backoff_count": 0,
                    "backend": "vllm-offline",
                }
            )
        return results

    def generate_many_bcc(
        self,
        image_sets: list[list[str]],
        prompts: list[str],
        seeds: list[int],
        batch_size: int,
        generation_config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.generate_many(
            image_sets,
            prompts,
            seeds,
            batch_size=batch_size,
            generation_config=generation_config,
        )

    def generate(
        self,
        images: list[str],
        prompt: str,
        seed: int,
        generation_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.generate_many(
            [images], [prompt], [seed], batch_size=1, generation_config=generation_config
        )[0]

    def caption(self, row: dict[str, Any], seed: int) -> dict[str, Any]:
        prompt = _fill_mask_prompt(
            self.stage_config.get("prompt") or self.caption_config["prompt"], row
        )
        keys = _configured_image_keys(self.stage_config, ("inverse_crop_path",))
        return self.generate(
            _row_images(row, keys, "captioning"),
            prompt,
            seed,
            generation_config=qwen_model_config(self.config, "caption"),
        )

    def mask_review(
        self, row: dict[str, Any], seed: int, prompt_template: str
    ) -> dict[str, Any]:
        prompt = _fill_mask_prompt(prompt_template, row)
        keys = _configured_image_keys(self.stage_config, ("inverse_crop_path",))
        return self.generate(
            _row_images(row, keys, "mask review"),
            prompt,
            seed,
            generation_config=qwen_model_config(self.config, "quality_filter"),
        )
