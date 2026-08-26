from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from sam3_mask_captioning.caption_stage import create_captioner
from sam3_mask_captioning.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--section", default="image_caption")
    parser.add_argument("--warmup-image")
    parser.add_argument("--warmup-image-count", type=int, default=1)
    args = parser.parse_args()
    started = time.time()
    captioner = create_captioner(load_config(args.config), args.section)
    visual_warmup: dict[str, object] | None = None
    if args.warmup_image:
        warmup_path = Path(args.warmup_image).resolve()
        if not warmup_path.is_file():
            raise FileNotFoundError(f"Visual warmup image does not exist: {warmup_path}")
        image_count = max(1, int(args.warmup_image_count))
        warmup_started = time.time()
        result = captioner.generate(  # type: ignore[attr-defined]
            [str(warmup_path)] * image_count,
            (
                "Runtime warmup only. Inspect the supplied real image and return "
                "the JSON object required by this stage."
            ),
            20260825,
            generation_config={
                "enable_thinking": False,
                "temperature": 0.0,
                "max_new_tokens": 32,
            },
        )
        visual_warmup = {
            "image_count": image_count,
            "elapsed_seconds": time.time() - warmup_started,
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
        }
    print(
        json.dumps(
            {
                "event": "qwen_engine_prewarmed",
                "section": args.section,
                "elapsed_seconds": time.time() - started,
                "model_path": os.environ.get("BCC_QWEN_MODEL_PATH"),
                "visual_warmup": visual_warmup,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    # vLLM 0.26 can spend minutes in distributed teardown after all durable
    # compilation products already exist. This dedicated prewarm process owns
    # no task outputs, so bypassing finalizers is safe.
    os._exit(0)


if __name__ == "__main__":
    main()
