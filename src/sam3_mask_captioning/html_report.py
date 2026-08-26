from __future__ import annotations

import html
import base64
import mimetypes
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

from PIL import Image

from .io_utils import read_jsonl


def _src(path_value: str, report_dir: Path) -> str:
    path = Path(path_value)
    try:
        return html.escape(path.resolve().relative_to(report_dir.resolve()).as_posix())
    except Exception:
        try:
            return html.escape(path.resolve().as_uri())
        except Exception:
            return html.escape(path_value)


def _data_uri(path_value: str) -> str | None:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return None
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _img_src(path_value: str, report_dir: Path, embed_images: bool) -> str:
    if embed_images:
        uri = _data_uri(path_value)
        if uri is not None:
            return uri
    return _src(path_value, report_dir)


def _attrs(row: dict[str, Any]) -> str:
    attrs = row.get("attributes") or []
    if not attrs:
        return ""
    return ", ".join(html.escape(str(item)) for item in attrs)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return html.escape(str(value))


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return ""


def _image_tag(
    path_value: str | None,
    report_dir: Path,
    embed_images: bool,
    label: str,
    class_name: str = "",
) -> str:
    if not path_value:
        return f'<div class="missing-img {class_name}">missing {html.escape(label)}</div>'
    src = _img_src(str(path_value), report_dir, embed_images)
    return f'<img class="{html.escape(class_name)}" src="{src}" loading="lazy" alt="{html.escape(label)}">'


def _even_sample(items: list[Any], limit: int) -> list[Any]:
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    last = len(items) - 1
    indexes = [round(index * last / (limit - 1)) for index in range(limit)]
    return [items[index] for index in indexes]


def _group_by_image(rows: list[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row.get("image_id", "unknown")), []).append(row)
    return grouped


def _sam3_demo_path(run_dir: Path, row: dict[str, Any]) -> Path | None:
    source = row.get("source_image_path")
    if not source:
        return None
    candidate = run_dir / "sam3_all_masks" / f"{str(row.get('image_id') or Path(str(source)).stem)}.jpg"
    return candidate if candidate.exists() else None


def _all_masks_image(run_dir: Path, image_id: str, row: dict[str, Any]) -> str:
    demo_path = _sam3_demo_path(run_dir, row)
    if demo_path is None:
        return ""
    output_path = run_dir / "sam3_all_masks" / f"{image_id}.jpg"
    if output_path.exists() and output_path.stat().st_mtime >= demo_path.stat().st_mtime:
        return str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    demo = Image.open(demo_path).convert("RGB")
    source_path = Path(str(row.get("source_image_path", "")))
    if source_path.exists():
        source = Image.open(source_path)
        source_width, source_height = source.size
        if demo.width == source_width and demo.height >= source_height * 2 - 4:
            crop_box = (0, source_height, demo.width, min(demo.height, source_height * 2))
        else:
            crop_box = (0, demo.height // 2, demo.width, demo.height)
    else:
        crop_box = (0, demo.height // 2, demo.width, demo.height)
    demo.crop(crop_box).save(output_path, quality=92)
    return str(output_path)


def _quality_mask_card(
    row: dict[str, Any],
    run_dir: Path,
    embed_images: bool,
    rejected: bool = False,
) -> str:
    title = html.escape(str(row.get("object") or row.get("mask_id") or "masked region"))
    caption = html.escape(str(row.get("caption") or ""))
    reason = html.escape(str(row.get("caption_reject_reason") or row.get("mask_review_reason") or row.get("reason") or ""))
    failure_modes = ", ".join(html.escape(str(item)) for item in row.get("mask_review_failure_modes") or [])
    reason_label = "caption" if row.get("caption_reject") else "qa"
    correction = str(row.get("qa_corrected_caption") or row.get("corrected_caption") or "").strip()
    reject_class = " rejected" if rejected else ""
    review_block = ""
    if rejected or reason:
        review_block = f"""
              <dt>{reason_label}</dt><dd>{reason}</dd>
              <dt>modes</dt><dd>{failure_modes}</dd>
              <dt>correction</dt><dd>{html.escape(correction)}</dd>
"""
    return f"""
          <article class="quality-mask-card{reject_class}">
            <div class="mask-media-pair">
              <figure>
                {_image_tag(row.get("source_image_path"), run_dir, embed_images, "source image", "mask-thumb")}
                <figcaption>source</figcaption>
              </figure>
              <figure>
                {_image_tag(row.get("crop_overlay_path") or row.get("full_overlay_path"), run_dir, embed_images, "crop mask overlay", "mask-thumb")}
                <figcaption>crop overlay</figcaption>
              </figure>
              <figure>
                {_image_tag(row.get("crop_image_path"), run_dir, embed_images, "plain crop", "mask-thumb")}
                <figcaption>plain crop</figcaption>
              </figure>
              <figure>
                {_image_tag(row.get("inverse_crop_path"), run_dir, embed_images, "inverse crop", "mask-thumb")}
                <figcaption>inverse crop</figcaption>
              </figure>
            </div>
            <div class="mask-copy">
              <h4>{title}</h4>
              <p>{caption}</p>
              <dl>
                <dt>mask</dt><dd>{html.escape(str(row.get("mask_id") or row.get("reject_id") or ""))}</dd>
                <dt>prompt</dt><dd>{html.escape(str(row.get("source_prompt") or ""))}</dd>
                <dt>attrs</dt><dd>{_attrs(row)}</dd>
                <dt>area</dt><dd>{_fmt(row.get("area"))} px</dd>
                <dt>bbox</dt><dd>{html.escape(str(row.get("bbox") or ""))}</dd>
                <dt>bbox area</dt><dd>{_fmt(row.get("bbox_area"))} px</dd>
                <dt>image %</dt><dd>{_pct(row.get("mask_area_fraction"))}</dd>
                <dt>score</dt><dd>{_fmt(row.get("sam3_score", row.get("entityseg_score")))}</dd>
                {review_block}
              </dl>
            </div>
          </article>
"""


def _stats_table_html(
    selected_rows: list[dict[str, Any]],
    image_reviews: list[dict[str, Any]],
    sam3_rows: list[dict[str, Any]],
    sam3_rejected_rows: list[dict[str, Any]],
    caption_candidates: list[dict[str, Any]],
    caption_rejected_rows: list[dict[str, Any]],
    accepted_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    categories: list[dict[str, Any]],
) -> str:
    selected_count = len(selected_rows)
    initial_accepted = sum(1 for row in image_reviews if row.get("accepted"))
    initial_rejected = sum(1 for row in image_reviews if not row.get("accepted"))
    sam3_reasons = Counter(str(row.get("reject_reason") or "unknown") for row in sam3_rejected_rows)
    sam3_reason_text = ", ".join(f"{html.escape(reason)}={count}" for reason, count in sam3_reasons.most_common()) or "none"
    final_images = sum(1 for row in categories if row.get("category") == "accepted_both")
    final_rate = f"{(final_images / selected_count * 100):.2f}%" if selected_count else "0.00%"
    rows = [
        ("selected images", selected_count, ""),
        ("initial accepted", initial_accepted, ""),
        ("initial rejected", initial_rejected, ""),
        ("SAM3 kept masks", len(sam3_rows), ""),
        ("SAM3 rejected masks", len(sam3_rejected_rows), sam3_reason_text),
        ("caption accepted", len(caption_candidates), "caption candidates sent to QA"),
        ("caption rejected", len(caption_rejected_rows), ""),
        ("QA accepted", len(accepted_rows), ""),
        ("QA rejected", len(rejected_rows), ""),
        ("final images with >=10 masks", final_images, ""),
        ("final image acceptance rate", final_rate, "final images / selected images"),
    ]
    body = "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value))}</td><td>{detail}</td></tr>"
        for label, value, detail in rows
    )
    return f"""
    <section class="stats-section">
      <h2>Run Stats</h2>
      <table class="stats-table">
        <tbody>
          {body}
        </tbody>
      </table>
    </section>
"""


def _review_text(review: dict[str, Any] | None) -> str:
    if not review:
        return "<p class=\"review-note\">No initial image review row was found.</p>"
    accepted = "accepted" if review.get("accepted") else "rejected"
    return f"""
          <dl class="image-review-list">
            <dt>initial pass</dt><dd>{accepted}</dd>
            <dt>worth</dt><dd>{html.escape(str(review.get("worth_segmenting")))}</dd>
            <dt>objects</dt><dd>{html.escape(str(review.get("estimated_distinct_objects") or ""))}</dd>
            <dt>type</dt><dd>{html.escape(str(review.get("image_type") or ""))}</dd>
            <dt>rationale</dt><dd>{html.escape(str(review.get("rationale") or ""))}</dd>
            <dt>reject</dt><dd>{html.escape(str(review.get("reject_reason") or ""))}</dd>
          </dl>
"""


def _write_quality_html_report(
    run_dir: Path,
    captions_path: Path,
    max_images: int,
    masks_per_image: int,
    embed_images: bool,
    output_name: str,
) -> Path:
    categories = read_jsonl(run_dir / "image_categories.jsonl")
    all_categories = list(categories)
    selected_rows = read_jsonl(run_dir / "selected_images.jsonl") if (run_dir / "selected_images.jsonl").exists() else []
    sam3_rows = read_jsonl(run_dir / "sam3_masks.jsonl") if (run_dir / "sam3_masks.jsonl").exists() else []
    sam3_rejected_rows = read_jsonl(run_dir / "sam3_rejected_masks.jsonl") if (run_dir / "sam3_rejected_masks.jsonl").exists() else []
    accepted_rows = read_jsonl(captions_path) if captions_path.exists() else []
    caption_candidates = read_jsonl(run_dir / "caption_candidates.jsonl") if (run_dir / "caption_candidates.jsonl").exists() else []
    caption_rejected_rows = read_jsonl(run_dir / "caption_rejected_masks.jsonl") if (run_dir / "caption_rejected_masks.jsonl").exists() else []
    rejected_rows = read_jsonl(run_dir / "rejected_captions.jsonl") if (run_dir / "rejected_captions.jsonl").exists() else []
    image_reviews = read_jsonl(run_dir / "image_reviews.jsonl") if (run_dir / "image_reviews.jsonl").exists() else []
    review_by_image = {str(row.get("image_id")): row for row in image_reviews}
    accepted_by_image = _group_by_image(accepted_rows)
    caption_rejected_by_image = _group_by_image(caption_rejected_rows)
    rejected_by_image = _group_by_image(rejected_rows)

    if max_images > 0:
        categories = categories[:max_images]

    category_labels = OrderedDict(
        [
            ("initial_rejected", "Rejected By Initial Qwen Pass"),
            ("second_pass_rejected", "Rejected By Caption QA / Below 10 Final Masks"),
            ("accepted_both", "Accepted By Image Review And Caption QA"),
        ]
    )
    grouped_categories: OrderedDict[str, list[dict[str, Any]]] = OrderedDict(
        (key, []) for key in category_labels
    )
    for row in categories:
        grouped_categories.setdefault(str(row.get("category") or "other"), []).append(row)

    total_rejected = len(rejected_rows)
    total_caption_rejected = len(caption_rejected_rows)
    stats_table = _stats_table_html(
        selected_rows,
        image_reviews,
        sam3_rows,
        sam3_rejected_rows,
        caption_candidates,
        caption_rejected_rows,
        accepted_rows,
        rejected_rows,
        all_categories,
    )
    category_sections: list[str] = []
    for category_key, label in category_labels.items():
        image_sections: list[str] = []
        for image_index, category_row in enumerate(grouped_categories.get(category_key, []), start=1):
            image_id = str(category_row.get("image_id") or "unknown")
            review = category_row.get("initial_review") or review_by_image.get(image_id)
            accepted_masks = accepted_by_image.get(image_id, [])
            caption_rejected_masks = caption_rejected_by_image.get(image_id, [])
            rejected_masks = rejected_by_image.get(image_id, [])
            if masks_per_image > 0:
                accepted_masks = accepted_masks[:masks_per_image]
                caption_rejected_masks = caption_rejected_masks[:masks_per_image]
                rejected_masks = rejected_masks[:masks_per_image]

            accepted_cards = "".join(
                _quality_mask_card(row, run_dir, embed_images, rejected=False)
                for row in accepted_masks
            )
            rejected_cards = "".join(
                _quality_mask_card(row, run_dir, embed_images, rejected=True)
                for row in rejected_masks
            )
            caption_rejected_cards = "".join(
                _quality_mask_card(row, run_dir, embed_images, rejected=True)
                for row in caption_rejected_masks
            )
            accepted_block = ""
            if accepted_cards:
                accepted_block = f"""
          <section class="mask-subsection">
            <h3>Accepted final masks</h3>
            <div class="quality-mask-grid">{accepted_cards}</div>
          </section>
"""
            caption_rejected_block = ""
            if caption_rejected_cards:
                caption_rejected_block = f"""
          <section class="mask-subsection caption-rejected-subsection">
            <h3>Rejected during caption generation</h3>
            <div class="quality-mask-grid">{caption_rejected_cards}</div>
          </section>
"""
            rejected_block = ""
            if rejected_cards:
                rejected_block = f"""
          <section class="mask-subsection rejected-subsection">
            <h3>Rejected masks in caption QA</h3>
            <div class="quality-mask-grid">{rejected_cards}</div>
          </section>
"""

            overlay_path = str(category_row.get("final_overlay_path") or "")
            overlay_label = "final accepted overlay" if overlay_path else "final accepted overlay"
            image_sections.append(
                f"""
      <section class="quality-image-section">
        <header class="image-header">
          <div>
            <h2>{image_index}. {html.escape(image_id)}</h2>
            <p class="ids">
              {int(category_row.get("accepted_mask_count") or 0)} accepted masks,
              {int(category_row.get("caption_rejected_mask_count") or 0)} caption-time rejected masks,
              {int(category_row.get("rejected_second_pass_mask_count") or 0)} QA rejected masks
            </p>
          </div>
          <span class="badge">{html.escape(str(category_row.get("category") or ""))}</span>
        </header>
        <div class="quality-top-grid">
          <figure>
            {_image_tag(category_row.get("source_image_path"), run_dir, embed_images, "source image")}
            <figcaption>source image</figcaption>
          </figure>
          <figure>
            {_image_tag(overlay_path, run_dir, embed_images, overlay_label)}
            <figcaption>final accepted overlay</figcaption>
          </figure>
          <div class="image-review-card">
            <h3>Initial Qwen image review</h3>
            {_review_text(review)}
          </div>
        </div>
        {accepted_block}
        {caption_rejected_block}
        {rejected_block}
      </section>
"""
            )
        category_sections.append(
            f"""
    <section class="category-section" id="{html.escape(category_key)}">
      <header class="category-header">
        <h2>{html.escape(label)}</h2>
        <span class="pill">{len(grouped_categories.get(category_key, []))} images</span>
      </header>
      {''.join(image_sections) if image_sections else '<p class="empty-note">No images in this category.</p>'}
    </section>
"""
        )

    report_path = run_dir / output_name
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>SAM3 Mask Captioning Visual Review</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f4f6f8; color: #17191c; }}
    main {{ max-width: 1720px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 6px; font-size: 30px; font-weight: 750; }}
    h2 {{ margin: 0; overflow-wrap: anywhere; }}
    h3 {{ margin: 0 0 10px; font-size: 16px; }}
    h4 {{ margin: 0 0 4px; font-size: 15px; }}
    .subhead {{ margin: 0 0 24px; color: #5f6672; overflow-wrap: anywhere; }}
    .run-meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 22px; }}
    .pill {{ border: 1px solid #cbd3dc; border-radius: 999px; padding: 5px 10px; background: #fff; font-size: 13px; color: #343942; }}
    .stats-section {{ background: #fff; border: 1px solid #dde3ea; border-radius: 8px; padding: 14px; margin: 0 0 22px; }}
    .stats-section h2 {{ font-size: 18px; margin: 0 0 10px; }}
    .stats-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .stats-table th, .stats-table td {{ text-align: left; border-top: 1px solid #e4e8ee; padding: 7px 8px; vertical-align: top; }}
    .stats-table tr:first-child th, .stats-table tr:first-child td {{ border-top: 0; }}
    .stats-table th {{ width: 230px; color: #343942; font-weight: 650; }}
    .category-section {{ margin: 0 0 28px; }}
    .category-header {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; margin: 0 0 12px; border-bottom: 1px solid #d9dee5; padding-bottom: 8px; }}
    .category-header h2 {{ font-size: 22px; }}
    .quality-image-section {{ background: #fff; border: 1px solid #dde3ea; border-radius: 8px; padding: 16px; margin-bottom: 18px; }}
    .image-header {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 14px; }}
    .image-header h2 {{ font-size: 19px; }}
    .ids {{ margin: 4px 0 0; font-size: 13px; color: #69707a; overflow-wrap: anywhere; }}
    .badge {{ flex: 0 0 auto; font-size: 12px; border: 1px solid #c9d0d8; border-radius: 999px; padding: 4px 8px; color: #343942; }}
    .quality-top-grid {{ display: grid; grid-template-columns: minmax(280px, 1fr) minmax(280px, 1fr) minmax(260px, 0.65fr); gap: 14px; align-items: start; }}
    figure {{ margin: 0; }}
    figure > img, .missing-img {{ width: 100%; max-height: 520px; object-fit: contain; background: #eef1f4; border: 1px solid #e1e5ea; border-radius: 6px; }}
    .missing-img {{ min-height: 160px; display: grid; place-items: center; color: #69707a; font-size: 13px; }}
    figcaption {{ margin-top: 4px; font-size: 12px; color: #69707a; }}
    .image-review-card {{ border: 1px solid #e1e5ea; border-radius: 8px; padding: 12px; background: #fbfcfd; }}
    .image-review-list, .quality-mask-card dl {{ display: grid; grid-template-columns: 82px minmax(0, 1fr); gap: 4px 10px; font-size: 12px; }}
    dt {{ color: #69707a; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    .review-note, .empty-note {{ margin: 0; color: #69707a; }}
    .mask-subsection {{ margin-top: 16px; }}
    .caption-rejected-subsection h3 {{ color: #7f4d00; }}
    .rejected-subsection h3 {{ color: #8a3c10; }}
    .quality-mask-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .quality-mask-card {{ display: grid; grid-template-columns: minmax(220px, 0.78fr) minmax(0, 1fr); gap: 12px; border: 1px solid #e1e5ea; border-radius: 8px; padding: 10px; background: #fbfcfd; }}
    .quality-mask-card.rejected {{ border-color: #e0b08e; background: #fffaf6; }}
    .mask-media-pair {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }}
    .mask-media-pair figure > img, .mask-media-pair .missing-img {{ aspect-ratio: 1 / 1; max-height: none; }}
    .mask-copy {{ min-width: 0; }}
    .mask-copy p {{ margin: 0 0 8px; font-size: 14px; line-height: 1.35; }}
    @media (max-width: 1240px) {{
      .quality-top-grid {{ grid-template-columns: 1fr 1fr; }}
      .image-review-card {{ grid-column: 1 / -1; }}
      .quality-mask-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 760px) {{
      main {{ padding: 16px; }}
      .quality-top-grid, .quality-mask-card, .mask-media-pair {{ grid-template-columns: 1fr; }}
      .category-header, .image-header {{ flex-direction: column; align-items: flex-start; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>SAM3 Mask Captioning Visual Review</h1>
    <p class="subhead">Quality-filtered image-level review for {html.escape(str(captions_path))}</p>
    <div class="run-meta">
      <span class="pill">{len(categories)} displayed images</span>
      <span class="pill">{len(accepted_rows)} accepted captions</span>
      <span class="pill">{total_caption_rejected} caption-time rejected masks</span>
      <span class="pill">{total_rejected} QA rejected captions</span>
      <span class="pill">{len(image_reviews)} initial image reviews</span>
      <span class="pill">{'embedded images' if embed_images else 'linked images'}</span>
    </div>
    {stats_table}
    {''.join(category_sections)}
  </main>
</body>
</html>
"""
    report_path.write_text(html_doc, encoding="utf-8")
    return report_path


def write_html_report(
    run_dir: str | Path,
    captions_path: str | Path | None = None,
    max_images: int = 10,
    masks_per_image: int = 10,
    max_caption_cards: int | None = None,
    embed_images: bool = True,
    output_name: str = "sam3_mask_captioning_visual_review.html",
) -> Path:
    run_dir = Path(run_dir)
    captions_path = Path(captions_path) if captions_path else run_dir / "captions.jsonl"
    if (run_dir / "image_categories.jsonl").exists():
        return _write_quality_html_report(
            run_dir,
            captions_path,
            max_images=max_images,
            masks_per_image=masks_per_image,
            embed_images=embed_images,
            output_name=output_name,
        )
    rows = read_jsonl(captions_path) if captions_path.exists() else []
    report_path = run_dir / output_name
    grouped = _group_by_image(rows)
    sampled_images = _even_sample(list(grouped.items()), max_images)
    candidate_pairs: list[tuple[str, dict[str, Any]]] = []
    for image_id, image_rows in sampled_images:
        for row in _even_sample(image_rows, masks_per_image):
            candidate_pairs.append((image_id, row))
    if max_caption_cards is not None:
        candidate_pairs = _even_sample(candidate_pairs, max_caption_cards)
    sampled_by_image: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for image_id, row in candidate_pairs:
        sampled_by_image.setdefault(image_id, []).append(row)

    sections = []
    sampled_caption_count = len(candidate_pairs)

    for image_idx, (image_id, image_rows) in enumerate(sampled_images, start=1):
        first_row = image_rows[0]
        sampled_rows = sampled_by_image.get(image_id, [])
        if not sampled_rows:
            continue
        mask_cards = []
        for row in sampled_rows:
            uncertain = " uncertain" if row.get("uncertain") else ""
            mask_cards.append(
                f"""
          <article class="mask-card{uncertain}">
            <img src="{_img_src(row.get("crop_overlay_path", ""), run_dir, embed_images)}" loading="lazy" alt="">
            <div class="mask-copy">
              <h3>{html.escape(str(row.get("object") or "unnamed region"))}</h3>
              <p>{html.escape(str(row.get("caption") or ""))}</p>
              <dl>
                <dt>mask</dt><dd>{html.escape(str(row.get("mask_id")))}</dd>
                <dt>attrs</dt><dd>{_attrs(row)}</dd>
                <dt>bbox</dt><dd>{html.escape(str(row.get("bbox")))}</dd>
                <dt>score</dt><dd>{html.escape(str(row.get("sam3_score", row.get("entityseg_score"))))}</dd>
              </dl>
            </div>
          </article>
"""
            )

        all_masks_path = _all_masks_image(run_dir, image_id, first_row)
        all_masks_src = _src(all_masks_path, run_dir) if all_masks_path else _src(
            str(_sam3_demo_path(run_dir, first_row) or ""), run_dir
        )
        sections.append(
            f"""
      <section class="image-section">
        <header class="image-header">
          <div>
            <h2>{image_idx}. {html.escape(image_id)}</h2>
            <p class="ids">{len(image_rows)} retained masks, {len(sampled_rows)} sampled captions</p>
          </div>
          <span class="badge">{len(image_rows)} masks</span>
        </header>
        <div class="review-grid">
          <figure>
            <img src="{_img_src(first_row.get("source_image_path", ""), run_dir, embed_images)}" loading="lazy" alt="">
            <figcaption>source image</figcaption>
          </figure>
          <figure>
            <img src="{_img_src(all_masks_path, run_dir, embed_images) if all_masks_path else all_masks_src}" loading="lazy" alt="">
            <figcaption>SAM3 all masks</figcaption>
          </figure>
          <div class="sample-panel">
            <h3>Sampled mask captions</h3>
            <div class="mask-grid">
              {''.join(mask_cards)}
            </div>
          </div>
        </div>
      </section>
"""
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SAM3 Mask Captioning Visual Review</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f5f6f8; color: #17191c; }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; font-weight: 700; }}
    .subhead {{ margin: 0 0 24px; color: #5f6672; }}
    .run-meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 22px; }}
    .pill {{ border: 1px solid #cfd5dd; border-radius: 999px; padding: 5px 10px; background: #fff; font-size: 13px; color: #343942; }}
    .image-section {{ background: #fff; border: 1px solid #dfe3e8; border-radius: 8px; padding: 16px; margin-bottom: 18px; }}
    .image-header {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 14px; }}
    h2 {{ margin: 0; font-size: 18px; overflow-wrap: anywhere; }}
    .ids {{ margin: 4px 0 0; font-size: 12px; color: #69707a; overflow-wrap: anywhere; }}
    .badge {{ flex: 0 0 auto; font-size: 12px; border: 1px solid #c9d0d8; border-radius: 999px; padding: 4px 8px; color: #343942; }}
    .review-grid {{ display: grid; grid-template-columns: minmax(260px, 0.85fr) minmax(260px, 0.85fr) minmax(360px, 1.3fr); gap: 14px; align-items: start; }}
    figure {{ margin: 0; }}
    figure > img {{ width: 100%; max-height: 520px; object-fit: contain; background: #eef1f4; border: 1px solid #e1e5ea; border-radius: 6px; }}
    figcaption {{ margin-top: 4px; font-size: 12px; color: #69707a; }}
    .sample-panel {{ min-width: 0; }}
    .sample-panel > h3 {{ margin: 0 0 10px; font-size: 15px; }}
    .mask-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .mask-card {{ display: grid; grid-template-columns: 118px minmax(0, 1fr); gap: 10px; border: 1px solid #e1e5ea; border-radius: 8px; padding: 8px; background: #fbfcfd; }}
    .mask-card.uncertain {{ border-color: #c99000; }}
    .mask-card img {{ width: 118px; height: 118px; object-fit: contain; background: #eef1f4; border: 1px solid #e1e5ea; border-radius: 6px; }}
    .mask-copy {{ min-width: 0; }}
    .mask-copy h3 {{ margin: 0 0 4px; font-size: 14px; }}
    .mask-copy p {{ margin: 0 0 8px; font-size: 13px; line-height: 1.35; }}
    dl {{ display: grid; grid-template-columns: 48px minmax(0, 1fr); gap: 3px 8px; font-size: 11px; }}
    dt {{ color: #69707a; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    @media (max-width: 1180px) {{ .review-grid {{ grid-template-columns: 1fr 1fr; }} .sample-panel {{ grid-column: 1 / -1; }} }}
    @media (max-width: 760px) {{ main {{ padding: 16px; }} .review-grid, .mask-grid {{ grid-template-columns: 1fr; }} .mask-card {{ grid-template-columns: 96px minmax(0, 1fr); }} .mask-card img {{ width: 96px; height: 96px; }} }}
  </style>
</head>
<body>
  <main>
    <h1>SAM3 Mask Captioning Visual Review</h1>
    <p class="subhead">Image-level review for {html.escape(str(captions_path))}</p>
    <div class="run-meta">
      <span class="pill">{len(rows)} total captions</span>
      <span class="pill">{len(grouped)} total images</span>
      <span class="pill">{len(sampled_images)} sampled images</span>
      <span class="pill">{sampled_caption_count} sampled captions</span>
      <span class="pill">max {masks_per_image} masks per sampled image</span>
      <span class="pill">{'embedded images' if embed_images else 'linked images'}</span>
    </div>
    {''.join(sections)}
  </main>
</body>
</html>
"""
    report_path.write_text(html_doc, encoding="utf-8")
    return report_path
