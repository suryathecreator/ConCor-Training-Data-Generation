from __future__ import annotations

import argparse
import json

STAGES = (
    "image-review",
    "sam3",
    "mask-caption-qa",
    "bcc",
    "mask-caption",
    "mask-qa",
    "consistency",
    "bcc-draft",
    "bcc-rewrite",
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="concor")
    parser.add_argument("--config", default="configs/qwen38_27b.yaml")
    parser.add_argument("--run-id", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument(
        "--stage",
        choices=["image-review", "sam3", "caption", "caption-qa", "finalize", "all"],
        default="all",
    )
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--mock", action="store_true")

    image_review_parser = sub.add_parser("image-review")
    image_review_parser.add_argument("--limit", type=int, default=None)
    image_review_parser.add_argument("--mock", action="store_true")

    sam3_parser = sub.add_parser("sam3")
    sam3_parser.add_argument("--limit", type=int, default=None)
    sam3_parser.add_argument("--mock", action="store_true")

    caption_parser = sub.add_parser("caption")
    caption_parser.add_argument("--masks-path", default=None)
    caption_parser.add_argument("--limit", type=int, default=None)
    caption_parser.add_argument("--mock", action="store_true")
    caption_parser.add_argument("--shard-index", type=int, default=None)
    caption_parser.add_argument("--shard-count", type=int, default=None)
    caption_parser.add_argument("--output-suffix", default="")

    qa_parser = sub.add_parser("caption-qa")
    qa_parser.add_argument("--candidates-path", default=None)
    qa_parser.add_argument("--limit", type=int, default=None)
    qa_parser.add_argument("--mock", action="store_true")
    qa_parser.add_argument("--shard-index", type=int, default=None)
    qa_parser.add_argument("--shard-count", type=int, default=None)
    qa_parser.add_argument("--output-suffix", default="")

    pipeline_parser = sub.add_parser("pipeline")
    pipeline_parser.add_argument("--limit", type=int, default=None)
    pipeline_parser.add_argument("--target-successes", type=int, default=None)
    pipeline_parser.add_argument("--mock", action="store_true")

    recovery_parser = sub.add_parser("recover-correspondence")
    recovery_parser.add_argument("--target-successes", type=int, default=3)
    recovery_parser.add_argument("--mock", action="store_true")

    merge_parser = sub.add_parser("merge-shards")
    merge_parser.add_argument("run_dir")
    merge_parser.add_argument("--stage", choices=["caption", "caption-qa"], required=True)
    merge_parser.add_argument("--shard-count", type=int, required=True)

    sub.add_parser("finalize")
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("run_dir")
    html_parser = sub.add_parser("html")
    html_parser.add_argument("run_dir")
    html_parser.add_argument("--captions-path", default=None)
    html_parser.add_argument("--max-images", type=int, default=10)
    html_parser.add_argument("--masks-per-image", type=int, default=10)
    html_parser.add_argument("--max-caption-cards", type=int, default=None)
    html_parser.add_argument("--linked-images", action="store_true")
    html_parser.add_argument("--output-name", default="sam3_mask_captioning_visual_review.html")
    bcc_html_parser = sub.add_parser("bcc-html")
    bcc_html_parser.add_argument("run_dir")
    bcc_html_parser.add_argument("--pairs-path", default=None)
    bcc_html_parser.add_argument("--max-images", type=int, default=10)
    bcc_html_parser.add_argument("--output", default=None)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("captions_path")
    checksum_parser = sub.add_parser("checksums")
    checksum_parser.add_argument("paths", nargs="+")
    checksum_parser.add_argument("--out", default="checksums.txt")

    campaign_init = sub.add_parser("campaign-init")
    campaign_init.add_argument("campaign_root")
    campaign_init.add_argument("--unit-size", type=int, default=100)
    campaign_init.add_argument("--seed", type=int, default=20260808)
    campaign_init.add_argument("--dataset", default="stanford-vision-lab/gpic")
    campaign_init.add_argument("--split", default="train")
    campaign_init.add_argument("--terminal-stage", choices=STAGES, default="bcc-rewrite")
    campaign_init.add_argument("--preview-pairs", type=int, default=0)

    campaign_extend = sub.add_parser("campaign-extend")
    campaign_extend.add_argument("campaign_root")
    campaign_extend.add_argument("--source", choices=["gpic", "manifest"], default="gpic")
    campaign_extend.add_argument("--add-images", type=int, default=None)
    campaign_extend.add_argument("--target-total", type=int, default=None)
    campaign_extend.add_argument("--manifest", default=None)
    campaign_extend.add_argument("--image-root", default=None)
    campaign_extend.add_argument("--repo-id", default="stanford-vision-lab/gpic")
    campaign_extend.add_argument("--split", default="train")
    campaign_extend.add_argument("--seed", type=int, default=20260808)
    campaign_extend.add_argument("--cache-dir", default=None)
    campaign_extend.add_argument("--token-file", default=None)
    campaign_extend.add_argument("--max-tars", type=int, default=None)
    campaign_extend.add_argument(
        "--exclude-csv",
        default=None,
        help="one-column CSV of image IDs, filenames, paths, or GPIC pair keys to skip",
    )

    campaign_worker = sub.add_parser("campaign-worker")
    campaign_worker.add_argument("campaign_root")
    campaign_worker.add_argument("--stage", choices=STAGES, required=True)
    campaign_worker.add_argument("--worker-index", type=int, default=0)
    campaign_worker.add_argument("--max-units", type=int, default=None)
    campaign_worker.add_argument("--lease-seconds", type=int, default=21600)
    campaign_worker.add_argument("--max-unit-attempts", type=int, default=3)
    campaign_worker.add_argument("--stop-claiming-at-epoch", type=float, default=None)

    campaign_merge = sub.add_parser("campaign-merge")
    campaign_merge.add_argument("campaign_root")
    campaign_merge.add_argument("--stage", choices=STAGES, required=True)
    campaign_merge.add_argument("--wait", action="store_true")
    campaign_merge.add_argument("--poll-seconds", type=int, default=30)

    campaign_publish = sub.add_parser("campaign-publish")
    campaign_publish.add_argument("campaign_root")
    campaign_publish.add_argument("--daemon", action="store_true")
    campaign_publish.add_argument("--poll-seconds", type=int, default=30)

    campaign_rebuild_site = sub.add_parser("campaign-rebuild-site")
    campaign_rebuild_site.add_argument("campaign_root")
    campaign_rebuild_site.add_argument("--milestone", action="append", type=int)

    campaign_export = sub.add_parser("campaign-export-hf")
    campaign_export.add_argument("campaign_root")
    campaign_export.add_argument("output_dir")
    campaign_export.add_argument("--shard-size", type=int, default=100)
    campaign_export.add_argument("--no-image-bytes", action="store_true")

    campaign_status_parser = sub.add_parser("campaign-status")
    campaign_status_parser.add_argument("campaign_root")

    vllm_canary = sub.add_parser("vllm-canary")
    vllm_canary.add_argument("fixture_run")
    vllm_canary.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "campaign-init":
        from .campaign_manifest import initialize_campaign

        print(json.dumps(initialize_campaign(
            args.campaign_root,
            unit_size=args.unit_size,
            seed=args.seed,
            dataset=args.dataset,
            split=args.split,
            terminal_stage=args.terminal_stage,
            preview_pairs=args.preview_pairs,
        ), indent=2, sort_keys=True))
        return
    if args.command == "campaign-extend":
        from .campaign_manifest import extend_from_manifest
        from .gpic_materialize import extend_from_gpic

        if args.source == "manifest":
            if not args.manifest:
                parser.error("campaign-extend --source manifest requires --manifest")
            result = extend_from_manifest(
                args.campaign_root,
                args.manifest,
                add_images=args.add_images,
                target_total=args.target_total,
                image_root=args.image_root,
                exclude_csv=args.exclude_csv,
            )
        else:
            token = __import__("os").environ.get("HF_TOKEN") or None
            if args.token_file:
                token = __import__("pathlib").Path(args.token_file).expanduser().read_text(encoding="utf-8").strip() or None
            result = extend_from_gpic(
                args.campaign_root,
                add_images=args.add_images,
                target_total=args.target_total,
                repo_id=args.repo_id,
                split=args.split,
                seed=args.seed,
                cache_dir=args.cache_dir,
                token=token,
                max_tars=args.max_tars,
                exclude_csv=args.exclude_csv,
            )
        from .run_ledger import write_run_ledger

        result = {**result, "ledger": write_run_ledger(args.campaign_root)}
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "campaign-merge":
        from .campaign_runner import merge_stage, wait_for_stage_merge

        result = (
            wait_for_stage_merge(
                args.campaign_root,
                args.stage,
                poll_seconds=args.poll_seconds,
            )
            if args.wait
            else merge_stage(args.campaign_root, args.stage)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "campaign-publish":
        from .campaign_publish import publish_daemon, publish_once

        result = (
            publish_daemon(args.campaign_root, poll_seconds=args.poll_seconds)
            if args.daemon
            else publish_once(args.campaign_root)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "campaign-rebuild-site":
        from .campaign_publish import rebuild_site

        print(
            json.dumps(
                rebuild_site(args.campaign_root, milestones=args.milestone),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "campaign-export-hf":
        from .dataset_export import export_hf_dataset

        print(json.dumps(export_hf_dataset(
            args.campaign_root,
            args.output_dir,
            shard_size=args.shard_size,
            include_image_bytes=not args.no_image_bytes,
        ), indent=2, sort_keys=True))
        return
    if args.command == "campaign-status":
        from .campaign_runner import campaign_status

        print(json.dumps(campaign_status(args.campaign_root), indent=2, sort_keys=True))
        return
    if args.command == "summarize":
        from .summarize import summarize_run

        print(json.dumps(summarize_run(args.run_dir), indent=2, sort_keys=True))
        return
    if args.command == "html":
        from .html_report import write_html_report

        print(
            write_html_report(
                args.run_dir,
                args.captions_path,
                max_images=args.max_images,
                masks_per_image=args.masks_per_image,
                max_caption_cards=args.max_caption_cards,
                embed_images=not args.linked_images,
                output_name=args.output_name,
            )
        )
        return
    if args.command == "bcc-html":
        from .bcc_html_report import write_bcc_html_report

        print(
            write_bcc_html_report(
                args.run_dir,
                pairs_path=args.pairs_path,
                output_path=args.output,
                max_images=args.max_images,
            )
        )
        return
    if args.command == "validate":
        from .validate import validate_caption_rows

        print(json.dumps(validate_caption_rows(args.captions_path), indent=2, sort_keys=True))
        return
    if args.command == "checksums":
        from .checksums import write_checksums

        write_checksums(args.paths, args.out)
        return
    if args.command == "merge-shards":
        from .caption_stage import merge_sharded_outputs

        stage = "mask-review" if args.stage == "caption-qa" else args.stage
        print(json.dumps(merge_sharded_outputs(args.run_dir, stage, args.shard_count), indent=2, sort_keys=True))
        return

    from .config import load_config, output_run_dir

    config = load_config(args.config)
    if args.command == "vllm-canary":
        from .vllm_canary import run_vllm_canary

        print(json.dumps(run_vllm_canary(config, args.fixture_run, args.output), indent=2, sort_keys=True))
        return
    if args.command == "campaign-worker":
        from .campaign_runner import run_stage_worker

        print(
            json.dumps(
                run_stage_worker(
                    config,
                    args.campaign_root,
                    args.stage,
                    worker_index=args.worker_index,
                    max_units=args.max_units,
                    lease_seconds=args.lease_seconds,
                    max_unit_attempts=args.max_unit_attempts,
                    stop_claiming_at_epoch=args.stop_claiming_at_epoch,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    run_dir = output_run_dir(config, args.run_id)
    from .io_utils import ensure_dir
    from .metadata import write_run_metadata

    ensure_dir(run_dir)
    write_run_metadata(config, run_dir)

    if args.command == "image-review":
        from .image_review_stage import run_image_review

        run_image_review(config, run_dir, mock=args.mock, limit=args.limit)
    elif args.command == "sam3":
        from .sam3_stage import run_sam3

        run_sam3(config, run_dir, limit=args.limit, mock=args.mock)
    elif args.command == "caption":
        from .caption_stage import run_captioning

        run_captioning(
            config,
            run_dir,
            masks_path=args.masks_path,
            mock=args.mock,
            limit=args.limit,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            output_suffix=args.output_suffix,
        )
    elif args.command == "caption-qa":
        from .caption_stage import run_mask_review

        run_mask_review(
            config,
            run_dir,
            candidates_path=args.candidates_path,
            mock=args.mock,
            limit=args.limit,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            output_suffix=args.output_suffix,
        )
    elif args.command == "pipeline":
        from .pipeline_runner import run_checkpointed_pipeline

        run_checkpointed_pipeline(
            config,
            run_dir,
            limit=args.limit,
            target_successes=args.target_successes,
            mock=args.mock,
        )
    elif args.command == "recover-correspondence":
        from .pipeline_runner import run_correspondence_recovery

        run_correspondence_recovery(
            config,
            run_dir,
            target_successes=args.target_successes,
            mock=args.mock,
        )
    elif args.command == "finalize":
        from .finalize_stage import finalize_run

        finalize_run(config, run_dir)
    elif args.command == "run":
        from .caption_stage import run_captioning, run_mask_review
        from .finalize_stage import finalize_run
        from .image_review_stage import run_image_review
        from .sam3_stage import run_sam3

        if args.stage in ("image-review", "all"):
            run_image_review(config, run_dir, mock=args.mock, limit=args.limit)
        if args.stage in ("sam3", "all"):
            run_sam3(config, run_dir, mock=args.mock, limit=args.limit)
        if args.stage in ("caption", "all"):
            run_captioning(config, run_dir, mock=args.mock, limit=args.limit)
        if args.stage in ("caption-qa", "all"):
            run_mask_review(config, run_dir, mock=args.mock)
        if args.stage in ("finalize", "all"):
            finalize_run(config, run_dir)
        from .summarize import summarize_run

        summarize_run(run_dir)
        if args.stage in ("caption", "caption-qa", "all"):
            from .html_report import write_html_report

            write_html_report(
                run_dir,
                max_images=0,
                masks_per_image=0,
                embed_images=False,
                output_name="sam3_mask_captioning_visual_review.html",
            )


if __name__ == "__main__":
    main()
