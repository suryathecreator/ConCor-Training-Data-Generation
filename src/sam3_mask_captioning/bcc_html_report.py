from __future__ import annotations

import base64
import html
import json
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl
from .json_utils import extract_json


def _data_uri(
    path_value: str | Path,
    *,
    embed_images: bool = True,
    output_dir: Path | None = None,
) -> str:
    path = Path(path_value)
    if not embed_images:
        base = output_dir or Path.cwd()
        return Path(os.path.relpath(path, base)).as_posix()
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _version_groups(pair: dict[str, Any], version: str) -> list[dict[str, Any]]:
    if version == "before" and "first_pass_groups" in pair:
        return list(pair.get("first_pass_groups") or [])
    return list(pair.get("groups") or [])


def _version_caption(pair: dict[str, Any], version: str) -> str:
    if version == "before" and pair.get("first_pass_caption") is not None:
        return str(pair.get("first_pass_caption") or "")
    return str(pair.get("caption") or "")


def _version_omitted(pair: dict[str, Any], version: str) -> list[dict[str, Any]]:
    if version == "before" and "first_pass_omitted_masks" in pair:
        return list(pair.get("first_pass_omitted_masks") or [])
    return list(pair.get("omitted_masks") or [])


def _tagged_caption(pair: dict[str, Any], version: str) -> str | None:
    raw = pair.get("first_pass_raw" if version == "before" else "rewrite_raw")
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = extract_json(raw)
        except ValueError:
            return None
    else:
        return None
    tagged = parsed.get("tagged_caption")
    return tagged if isinstance(tagged, str) else None


def _viewer_data(
    pair: dict[str, Any], *, embed_images: bool, output_dir: Path
) -> dict[str, Any]:
    groups_by_id: dict[str, dict[str, Any]] = {}
    for version in ("after", "before"):
        for group in _version_groups(pair, version):
            groups_by_id.setdefault(str(group["mask_id"]), group)
    omitted_by_id: dict[str, dict[str, Any]] = {}
    for version in ("after", "before"):
        for item in _version_omitted(pair, version):
            omitted_by_id.setdefault(str(item.get("mask_id") or ""), item)
    return {
        "imageId": pair["image_id"],
        "source": _data_uri(pair["source_image_path"], embed_images=embed_images, output_dir=output_dir),
        "groups": [
            {
                "id": group["mask_id"],
                "color": group["color_rgb"],
                "mask": _data_uri(group["mask_path"], embed_images=embed_images, output_dir=output_dir),
            }
            for group in groups_by_id.values()
        ],
        "versionGroupIds": {
            version: [str(group["mask_id"]) for group in _version_groups(pair, version)]
            for version in ("before", "after")
        },
        "versionOmittedIds": {
            version: [
                str(item.get("mask_id") or "")
                for item in _version_omitted(pair, version)
            ]
            for version in ("before", "after")
        },
        "omitted": [
            {
                "id": str(item.get("mask_id") or ""),
                "mask": _data_uri(
                    item.get("diagnostic_mask_path") or item.get("mask_path"),
                    embed_images=embed_images,
                    output_dir=output_dir,
                ),
            }
            for item in omitted_by_id.values()
            if item.get("diagnostic_mask_path") or item.get("mask_path")
        ],
    }


def _caption_html(pair: dict[str, Any], version: str) -> str:
    caption = _version_caption(pair, version)
    groups = _version_groups(pair, version)
    boundaries = {0, len(caption)}
    spans: list[tuple[int, int, str]] = []
    color_by_id: dict[str, list[int]] = {}
    for group in groups:
        group_id = str(group["mask_id"])
        color_by_id[group_id] = list(group.get("color_rgb") or [255, 80, 80])
        for span in group.get("char_spans") or []:
            start, end = int(span[0]), int(span[1])
            boundaries.add(start)
            boundaries.add(end)
            spans.append((start, end, group_id))
    ordered = sorted(boundaries)
    parts: list[str] = []
    for start, end in zip(ordered, ordered[1:]):
        text = html.escape(caption[start:end])
        ids = [group_id for span_start, span_end, group_id in spans if span_start <= start and end <= span_end]
        if not ids:
            parts.append(text)
            continue
        colors = [color_by_id[group_id] for group_id in ids]
        if len(colors) == 1:
            background = f"rgba({colors[0][0]},{colors[0][1]},{colors[0][2]},.22)"
        else:
            stops = []
            width = 100 / len(colors)
            for index, color in enumerate(colors):
                stops.append(
                    f"rgba({color[0]},{color[1]},{color[2]},.28) {index * width:.2f}% {(index + 1) * width:.2f}%"
                )
            background = "linear-gradient(90deg," + ",".join(stops) + ")"
        parts.append(
            f'<button class="mention" data-version="{version}" data-groups="{html.escape(",".join(ids))}" '
            f'style="background:{background}">{text}</button>'
        )
    return "".join(parts)


def _group_legend(pair: dict[str, Any], version: str) -> str:
    cards: list[str] = []
    for index, group in enumerate(_version_groups(pair, version), start=1):
        color = list(group.get("color_rgb") or [255, 80, 80])
        mentions = ", ".join(f"“{value}”" for value in group.get("text") or [])
        requery_iou = group.get("sam3_requery_iou")
        iou_label = f"IoU {float(requery_iou):.3f}" if requery_iou is not None else "re-query pending"
        cards.append(
            f"""
            <button class="group-card" data-version="{version}" data-groups="{html.escape(str(group['mask_id']))}">
              <span class="swatch" style="--swatch:rgb({color[0]},{color[1]},{color[2]})">{index}</span>
              <span class="group-copy">
                <strong>{html.escape(str(group.get('main_candidate') or group.get('source_sam3_prompt') or 'entity'))}</strong>
                <span>{html.escape(mentions)}</span>
                <small>{iou_label} · SAM3 {float(group.get('sam3_score') or 0):.3f}</small>
              </span>
            </button>
            """
        )
    return "".join(cards)


def _caption_version_card(
    pair: dict[str, Any], viewer_id: str, version: str
) -> str:
    before = version == "before"
    title = "Before rewrite (before audit)" if before else "After one rewrite (audit-guided)"
    note = "Qwen initial draft" if before else "Qwen final rewrite"
    tagged = _tagged_caption(pair, version)
    linked_count = len(_version_groups(pair, version))
    omitted_count = len(_version_omitted(pair, version))
    tagged_html = (
        html.escape(tagged)
        if tagged is not None
        else '<span class="tagged-missing">Not recorded as a parseable model tagged_caption.</span>'
    )
    checked = "" if before else " checked"
    return f"""
      <section class="caption-version{' selected' if not before else ''}" data-version-card="{version}">
        <label class="version-selector">
          <input type="radio" name="{viewer_id}-caption-version" value="{version}"{checked}>
          <span><strong>{title}</strong><small>{note} · {linked_count} linked / {omitted_count} omitted · select for mask highlighting</small></span>
        </label>
        <p class="caption">{_caption_html(pair, version)}</p>
        <div class="tagged-caption">
          <span>Exact model <code>tagged_caption</code></span>
          <code>{tagged_html}</code>
        </div>
      </section>
    """


def _caption_comparison_html(pair: dict[str, Any], viewer_id: str) -> str:
    return f"""
      <div class="caption-versions">
        {_caption_version_card(pair, viewer_id, 'before')}
        {_caption_version_card(pair, viewer_id, 'after')}
      </div>
      <div class="version-legend" data-version-legend="before" hidden>
        <span class="legend-label">Before-rewrite linked entities</span>
        <div class="group-grid">{_group_legend(pair, 'before')}</div>
      </div>
      <div class="version-legend" data-version-legend="after">
        <span class="legend-label">After-rewrite linked entities</span>
        <div class="group-grid">{_group_legend(pair, 'after')}</div>
      </div>
    """


def _issue_list(items: list[dict[str, Any]], empty: str) -> str:
    if not items:
        return f'<p class="issue-empty">{html.escape(empty)}</p>'
    rendered: list[str] = []
    for item in items:
        severity = str(item.get("severity") or "nonfatal")
        code = str(item.get("code") or "checker_finding")
        masks = ", ".join(str(value) for value in item.get("mask_ids") or [])
        rendered.append(
            f'<li class="issue issue-{html.escape(severity)}"><span>{html.escape(severity)}</span>'
            f'<strong>{html.escape(code)}</strong><p>{html.escape(str(item.get("message") or ""))}</p>'
            + (f'<small>Masks: {html.escape(masks)}</small>' if masks else "")
            + "</li>"
        )
    return '<ul class="issue-list">' + "".join(rendered) + "</ul>"


def _model_visual_audit_html(pair: dict[str, Any]) -> str:
    audit = pair.get("model_visual_audit") or {}
    if not audit:
        return '<p class="issue-empty">No middle-stage model audit is stored for this archived record.</p>'
    passed = bool(audit.get("audit_pass"))
    accuracy = max(0.0, min(100.0, float(audit.get("task_accuracy_percent") or 0.0)))
    issues: list[str] = []
    for item in audit.get("issues") or []:
        severity = html.escape(str(item.get("severity") or "rewrite"))
        code = html.escape(str(item.get("code") or "model_audit_finding"))
        evidence = html.escape(str(item.get("caption_evidence") or ""))
        explanation = html.escape(str(item.get("explanation") or ""))
        instruction = html.escape(str(item.get("rewrite_instruction") or ""))
        ids = ", ".join(str(value) for value in item.get("mask_ids") or [])
        issues.append(
            f'<li class="issue issue-{severity}"><span>{severity}</span><strong>{code}</strong>'
            + (f'<p><b>Evidence:</b> {evidence}</p>' if evidence else "")
            + f'<p>{explanation}</p><p><b>Rewrite:</b> {instruction}</p>'
            + (f'<small>Overlay IDs: {html.escape(ids)}</small>' if ids else "")
            + "</li>"
        )
    decisions = "".join(
        f'<li><code>{int(item.get("id") or 0)}</code> · '
        f'<strong>{html.escape(str(item.get("decision") or "unknown"))}</strong> · '
        f'{html.escape(str(item.get("reason") or ""))}</li>'
        for item in audit.get("mask_decisions") or []
    ) or "<li>No per-mask decision recorded</li>"
    disposition = "draft accepted as-is" if passed else "draft sent to rewrite"
    return f"""
      <section class="model-audit-card">
        <header><div><span class="eyebrow">Independent Qwen visual/rules audit</span><h3>{html.escape(disposition)}</h3></div><strong>{accuracy:.1f}% task accuracy</strong></header>
        <div class="accuracy-track"><span style="width:{accuracy:.2f}%"></span></div>
        <p>{html.escape(str(audit.get('summary') or 'No summary recorded.'))}</p>
        <ul class="issue-list">{''.join(issues) or '<li class="issue-empty">No model-audit issues</li>'}</ul>
        <details><summary>Mask decisions and rewrite plan</summary><ul class="omitted-list">{decisions}</ul><p>{html.escape(str(audit.get('rewrite_plan') or ''))}</p></details>
      </section>
    """


def _audit_html(pair: dict[str, Any]) -> str:
    validation = pair.get("validation") or {}
    before = (validation.get("before_rewrite") or {}).get("issues") or []
    after = (validation.get("after_rewrite") or {}).get("issues") or []
    metrics = pair.get("rewrite_metrics") or {}
    resolved = metrics.get("resolved") or []
    new = metrics.get("new") or []
    persisting = metrics.get("persisting") or []
    before_omitted = _version_omitted(pair, "before")
    omitted = _version_omitted(pair, "after")
    included = pair.get("included") is not False
    exclusion_reason = html.escape(
        str(pair.get("reason_code") or pair.get("exclusion_reason") or "")
    )
    parse_error = html.escape(
        str((validation.get("after_rewrite") or {}).get("parse_error") or "")
    )
    disposition = ""
    if not included:
        disposition = (
            '<p class="exclusion-note"><strong>Audit-only exclusion.</strong> '
            + (f"{exclusion_reason}. " if exclusion_reason else "")
            + (f"{parse_error}" if parse_error else "This record is not a training pair.")
            + "</p>"
        )
    def omitted_list(items: list[dict[str, Any]]) -> str:
        return "".join(
        f'<li><code>{html.escape(str(item.get("mask_id") or ""))}</code> · '
        f'{html.escape(str(item.get("main_candidate") or item.get("object") or "entity"))} · '
        f'{html.escape(str(item.get("reason") or "not_mentioned"))}</li>'
        for item in items
        ) or "<li>None</li>"
    composite = pair.get("composite_statistics") or {}
    minimum_linked = int(
        (pair.get("coverage_policy") or {}).get(
            "minimum_linked_masks_after_caption", 5
        )
    )
    return f"""
      <details class="audit-panel" open>
        <summary>Audit trail · model audit + One-rewrite checker audit + final structural gates · {html.escape(str(metrics.get('outcome') or 'unknown'))}</summary>
        {disposition}
        {_model_visual_audit_html(pair)}
        <div class="audit-metrics">
          <span>Before <b>{int(metrics.get('before_issue_count') or len(before))}</b></span>
          <span>After <b>{int(metrics.get('after_issue_count') or len(after))}</b></span>
          <span>Fatal after <b>{int(metrics.get('after_fatal_count') or 0)}</b></span>
          <span>Omitted masks <b>{len(omitted)}</b></span>
        </div>
        <div class="audit-columns">
          <section><h3>Resolved</h3>{_issue_list(resolved, 'No findings resolved')}</section>
          <section><h3>Persisting</h3>{_issue_list(persisting, 'No findings persisted')}</section>
          <section><h3>New</h3>{_issue_list(new, 'No new findings')}</section>
        </div>
        <details><summary>All findings before and after</summary><div class="audit-columns"><section><h3>Before</h3>{_issue_list(before, 'Checker-clean draft')}</section><section><h3>After</h3>{_issue_list(after, 'Checker-clean final caption')}</section></div></details>
        <details><summary>Omitted-mask audit</summary><div class="audit-columns"><section><h3>Before rewrite ({len(before_omitted)})</h3><ul class="omitted-list">{omitted_list(before_omitted)}</ul></section><section><h3>After rewrite ({len(omitted)})</h3><ul class="omitted-list">{omitted_list(omitted)}</ul></section></div></details>
        <details><summary>Composite statistics</summary><pre>{html.escape(json.dumps(composite, indent=2, sort_keys=True))}</pre></details>
        <p class="checker-caveat">The middle visual audit is a Qwen judgment used to guide the single rewrite. The post-rewrite historical checker is retained only as a diagnostic approximation developed through scorer/heuristic iterations; it does not drive this rewrite and may be incomplete or occasionally inapplicable. Final inclusion uses objective parse/tag structure and the configured {minimum_linked}-linked-mask floor.</p>
      </details>
    """


def _packet_image(
    path_value: str | Path,
    alt: str,
    *,
    embed_images: bool,
    output_dir: Path,
) -> str:
    path = Path(path_value)
    if not str(path_value) or not path.is_file():
        return f'<div class="packet-missing">Missing: {html.escape(str(path))}</div>'
    return f'<img src="{_data_uri(path, embed_images=embed_images, output_dir=output_dir)}" loading="lazy" alt="{html.escape(alt)}">'


def _input_packet_html(
    pair: dict[str, Any],
    run_dir: Path,
    pair_index: int,
    *,
    embed_images: bool,
    output_dir: Path,
) -> str:
    """Render the exact ordered visual packet and weak hints used by Qwen."""
    groups_by_id = {
        str(group.get("mask_id")): group
        for group in [
            *(pair.get("groups") or []),
            *(pair.get("first_pass_groups") or []),
            *(pair.get("omitted_masks") or []),
            *(pair.get("first_pass_omitted_masks") or []),
        ]
    }
    manifest = pair.get("bcc_input_manifest") or []
    overlay_path = pair.get("correspondence_overlay_path") or ""
    primary_cards = f"""
      <figure class="packet-primary-card">
        <span class="image-number">IMAGE 1</span>
        {_packet_image(pair.get("source_image_path") or "", "Original input image", embed_images=embed_images, output_dir=output_dir)}
        <figcaption><strong>Original image</strong><span>Authoritative scene context</span></figcaption>
      </figure>
      <figure class="packet-primary-card">
        <span class="image-number">IMAGE 2</span>
        {_packet_image(overlay_path, "Numbered mask overlay", embed_images=embed_images, output_dir=output_dir)}
        <figcaption><strong>Numbered overlay</strong><span>Visual index from each number to one mask and crop</span></figcaption>
      </figure>
    """
    crop_cards: list[str] = []
    for item in manifest:
        if item.get("role") != "inverse_mask_crop":
            continue
        mask_id = str(item.get("mask_id") or "")
        group = groups_by_id.get(mask_id, {})
        image_number = int(item.get("image_number") or 0)
        overlay_number = int(item.get("overlay_number") or 0)
        inverse_path = Path(str(group.get("inverse_crop_path") or run_dir / "inverse_crops" / f"{mask_id}.png"))
        rgb = item.get("inverse_background_rgb") or group.get("inverse_background_rgb")
        rgb_label = ", ".join(str(value) for value in rgb) if rgb else "unknown"
        description = str(group.get("mask_caption") or "No mask description recorded")
        candidate = str(group.get("main_candidate") or group.get("mask_object") or "unknown")
        prompt = str(group.get("source_sam3_prompt") or "")
        crop_cards.append(
            f"""
            <article class="crop-card">
              <div class="crop-visual">
                <span class="image-number">IMAGE {image_number}</span>
                <span class="overlay-badge">overlay #{overlay_number}</span>
                {_packet_image(inverse_path, f"Inverse crop for overlay number {overlay_number}", embed_images=embed_images, output_dir=output_dir)}
              </div>
              <div class="crop-copy">
                <strong>{html.escape(candidate)}</strong>
                <p>{html.escape(description)}</p>
                <small>SAM3 hint: {html.escape(prompt or 'none')} · synthetic RGB: {html.escape(rgb_label)}</small>
              </div>
            </article>
            """
        )
    first_pass = html.escape(str(pair.get("first_pass_caption") or "Not recorded"))
    pass_two = html.escape(str(pair.get("caption") or "Not recorded"))
    return f"""
      <section class="packet-card" aria-labelledby="packet-title-{pair_index}">
        <header class="packet-header">
          <div>
            <span class="eyebrow">Exact Qwen visual packet</span>
            <h2 id="packet-title-{pair_index}">What went into BCC-pair generation</h2>
          </div>
          <span class="packet-count">{len(manifest)} ordered images</span>
        </header>
        <div class="definition-note">
          <strong>The numbered overlay is an index, not a label map.</strong>
          Number <em>n</em> marks the region represented by overlay entry <em>n</em>, inverse crop <em>n</em>,
          and that mask's weak context row. It is neither a class ID nor a confidence ranking.
        </div>
        <div class="pass-flow" aria-label="Caption generation flow">
          <span>Original + overlay + crops</span><b>→</b><span>Qwen draft</span><b>→</b><span>Qwen visual/rules audit + %</span><b>→</b><span>One full-visual rewrite</span>
        </div>
        <div class="packet-primary">{primary_cards}</div>
        <details class="crop-details" open>
          <summary>Images 3 onward: every dynamically colored inverse crop</summary>
          <p class="weak-note">The object pixels are real. The solid color outside the mask is synthetic. The descriptions and SAM3 labels shown here were supplied as weak, optional hints and were not required wording.</p>
          <div class="crop-grid">{''.join(crop_cards)}</div>
        </details>
        <details class="draft-details">
          <summary>Text passed between caption stages</summary>
          <dl><dt>Initial draft</dt><dd>{first_pass}</dd><dt>Audit-guided rewrite</dt><dd>{pass_two}</dd></dl>
        </details>
      </section>
    """


def write_bcc_html_report(
    run_dir: str | Path,
    *,
    pairs_path: str | Path | None = None,
    output_path: str | Path | None = None,
    max_images: int = 10,
    embed_images: bool = True,
) -> Path:
    run_dir = Path(run_dir)
    pairs_path = Path(pairs_path) if pairs_path else run_dir / "image_text_pairs.jsonl"
    pairs = read_jsonl(pairs_path) if pairs_path.exists() else []
    if max_images > 0:
        pairs = pairs[:max_images]
    output_path = Path(output_path) if output_path else run_dir / "bcc_review.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_path.parent.resolve()
    metadata_path = run_dir / "bcc_report_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    first_pair = pairs[0] if pairs else {}
    model_name = str(
        first_pair.get("model")
        or (first_pair.get("backend_provenance") or {}).get("model")
        or "Qwen"
    ).split("/")[-1]
    kicker = html.escape(str(metadata.get("kicker") or f"SAM3 × {model_name}"))
    title = html.escape(str(metadata.get("title") or "BCC inputs and pipeline-completed pairs"))
    subtitle = html.escape(
        str(
            metadata.get("subtitle")
            or "First inspect the exact ordered visual packet Qwen received, then inspect the "
            "mandatory one-rewrite image–text record and its interactive correspondences."
        )
    )
    status_note = html.escape(
        str(
            metadata.get("status_note")
            or "Pipeline-completed means the automated gates passed; it does not mean human-verified "
            "correctness. Review the stored checker findings and quality tier for each pair."
        )
    )
    pass_label = html.escape(
        str(metadata.get("pass_label") or f"{model_name} · audit + one rewrite")
    )
    configured_minimums = {
        int((pair.get("coverage_policy") or {}).get("minimum_linked_masks_after_caption", 5))
        for pair in pairs
    }
    minimum_label = (
        str(next(iter(configured_minimums)))
        if len(configured_minimums) == 1
        else "the configured minimum number of"
    )
    viewer_payloads: list[dict[str, Any]] = []
    accepted_sections: list[str] = []
    rejected_sections: list[str] = []
    for index, pair in enumerate(pairs, start=1):
        viewer_id = f"viewer-{index}"
        included = pair.get("included") is not False
        default_label = (
            f"Usable training pair {index:02d}"
            if included
            else f"Audit-only excluded example {index:02d}"
        )
        pair_label = html.escape(str(pair.get("pair_label") or default_label))
        pair_pass_label = html.escape(str(pair.get("pass_label") or pass_label))
        quality_tier = html.escape(str(pair.get("quality_tier") or "unclassified"))
        disposition_label = "usable pair" if included else "excluded from training"
        disposition_class = "included" if included else "excluded"
        viewer_payloads.append(
            {
                "viewerId": viewer_id,
                **_viewer_data(pair, embed_images=embed_images, output_dir=output_dir),
            }
        )
        rendered_section = (
            _input_packet_html(
                pair,
                run_dir,
                index,
                embed_images=embed_images,
                output_dir=output_dir,
            )
            + f"""
            <article class="pair-card" id="{viewer_id}">
              <header class="pair-header">
                <div>
                  <span class="eyebrow">{pair_label}</span>
                  <h2>{html.escape(str(pair.get('image_id') or 'unknown'))}</h2>
                </div>
                <div class="pair-stats">
                  <span>{len(_version_groups(pair, 'before'))} → {len(_version_groups(pair, 'after'))} grounded groups</span>
                  <span>{pair_pass_label}</span>
                  <span class="disposition-badge disposition-{disposition_class}">{disposition_label}</span>
                  <span class="quality-badge quality-{quality_tier}">{quality_tier}</span>
                </div>
              </header>
              <div class="pair-layout">
                <section class="visual-panel">
                  <div class="canvas-shell">
                    <canvas aria-label="Image with interactive accepted masks"></canvas>
                    <div class="loading-note">Loading masks…</div>
                  </div>
                  <div class="viewer-actions">
                    <button class="show-all" type="button">Show all masks</button>
                    {'<button class="show-omitted" type="button">Show omitted masks for selected caption</button>' if _version_omitted(pair, 'before') or _version_omitted(pair, 'after') else ''}
                    <span>Hover to inspect · click to pin</span>
                  </div>
                </section>
                <section class="text-panel">
                  {_caption_comparison_html(pair, viewer_id)}
                </section>
              </div>
              {_audit_html(pair)}
            </article>
            """
        )
        (accepted_sections if included else rejected_sections).append(rendered_section)
    sections = (
        '<section class="disposition-section disposition-section-accepted"><header class="section-header"><span class="eyebrow">Audit-accepted</span><h2>Final training pairs</h2><p>These rewrites passed the objective schema/tag gates and retained at least '
        + minimum_label
        + ' linked masks.</p></header>'
        + "".join(accepted_sections)
        + ("" if accepted_sections else '<div class="empty"><p>No accepted records yet.</p></div>')
        + "</section>"
        + '<section class="disposition-section disposition-section-rejected"><header class="section-header"><span class="eyebrow">Audit-rejected</span><h2>Excluded attempts</h2><p>These remain visible for diagnosis but are not emitted as BCC training pairs.</p></header>'
        + "".join(rejected_sections)
        + ("" if rejected_sections else '<div class="empty"><p>No rejected records in this run.</p></div>')
        + "</section>"
    )
    empty = "" if pairs else '<section class="empty"><h2>No pipeline-completed one-rewrite pairs yet.</h2><p>The report will populate as checkpointed results arrive.</p></section>'
    payload = json.dumps(viewer_payloads, separators=(",", ":")).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Bidirectional concept correspondence review</title>
  <style>
    :root {{ color-scheme: light; --ink:#17211c; --muted:#667069; --paper:#f5f1e8; --card:#fffdf8; --line:#d9d2c3; --accent:#174f3a; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 12% -10%,#dcebd9 0,transparent 36rem),var(--paper); font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif; }}
    main {{ max-width:1500px; margin:auto; padding:38px 24px 72px; }}
    .masthead {{ display:grid; grid-template-columns:1.4fr .6fr; gap:28px; align-items:end; border-bottom:1px solid var(--line); padding-bottom:24px; margin-bottom:28px; }}
    .kicker,.eyebrow {{ display:block; color:var(--accent); text-transform:uppercase; letter-spacing:.14em; font-weight:750; font-size:11px; }}
    h1 {{ max-width:900px; margin:8px 0 10px; font-family:Georgia,serif; font-size:clamp(36px,5vw,68px); line-height:.98; font-weight:500; }}
    .masthead p {{ max-width:720px; margin:0; color:var(--muted); font-size:16px; line-height:1.55; }}
    .run-note {{ justify-self:end; max-width:300px; padding:16px; border:1px solid var(--line); background:rgba(255,253,248,.7); font-size:13px; line-height:1.5; }}
    .disposition-section {{ margin:0 0 48px; }}
    .section-header {{ margin:0 0 20px; padding:20px 22px; border:1px solid var(--line); border-radius:16px; background:rgba(255,253,248,.78); }}
    .section-header h2 {{ margin:5px 0 5px; font:500 32px/1.1 Georgia,serif; }}
    .section-header p {{ margin:0; color:var(--muted); }}
    .disposition-section-rejected>.section-header {{ border-color:#d8bdd0; background:#fbf2f8; }}
    .quality-badge {{ font-weight:750; text-transform:uppercase; letter-spacing:.06em; }}
    .quality-clean {{ background:#d8f3df!important; color:#185c31; }}
    .quality-nonfatal_flagged {{ background:#fff0bd!important; color:#755300; }}
    .quality-fatal_flagged {{ background:#ffd8d5!important; color:#8b251f; }}
    .quality-unparseable_excluded {{ background:#eadcf8!important; color:#5c277d; }}
    .disposition-badge {{ font-weight:750; text-transform:uppercase; letter-spacing:.05em; }}
    .disposition-included {{ background:#d8f3df!important; color:#185c31!important; }}
    .disposition-excluded {{ background:#eadcf8!important; color:#5c277d!important; }}
    .exclusion-note {{ margin:12px 0; border-left:4px solid #7d3da0; border-radius:6px; background:#f4ecfb; padding:10px 12px; color:#4f2965; font-size:12px; line-height:1.45; }}
    .audit-panel {{ margin:0 22px 22px; border:1px solid var(--line); border-radius:14px; background:#fbf8f0; padding:14px 16px; }}
    .audit-panel>summary {{ cursor:pointer; font-weight:750; }}
    .audit-metrics {{ display:flex; flex-wrap:wrap; gap:8px; margin:14px 0; }}
    .audit-metrics span {{ border:1px solid var(--line); border-radius:999px; padding:6px 10px; background:white; }}
    .audit-columns {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
    .model-audit-card {{ margin:14px 0; border:1px solid #c7d7cf; border-radius:12px; background:#f4faf6; padding:14px; }}
    .model-audit-card>header {{ display:flex; justify-content:space-between; gap:14px; align-items:end; }}
    .model-audit-card h3 {{ margin:4px 0 0; font:500 20px/1.15 Georgia,serif; }}
    .model-audit-card>header>strong {{ color:#14573d; white-space:nowrap; }}
    .accuracy-track {{ height:8px; margin:12px 0; overflow:hidden; border-radius:999px; background:#dce6df; }}
    .accuracy-track span {{ display:block; height:100%; border-radius:inherit; background:linear-gradient(90deg,#bb5b45,#d5a83a,#2d8159); }}
    .audit-columns h3 {{ margin:0 0 8px; font-size:13px; text-transform:uppercase; letter-spacing:.08em; }}
    .issue-list,.omitted-list {{ list-style:none; margin:0; padding:0; }}
    .issue {{ border-left:4px solid #d0a428; background:white; padding:9px; margin:0 0 7px; border-radius:6px; }}
    .issue-fatal {{ border-left-color:#bd4138; }}
    .issue>span {{ float:right; font-size:10px; text-transform:uppercase; }}
    .issue p {{ margin:4px 0; font-size:12px; line-height:1.4; }}
    .issue-empty,.checker-caveat {{ color:var(--muted); font-size:12px; }}
    .audit-panel pre {{ white-space:pre-wrap; font-size:11px; background:#17211c; color:#f8f5ec; padding:12px; border-radius:8px; }}
    .packet-card {{ background:var(--card); border:1px solid var(--line); border-radius:18px; box-shadow:0 16px 40px rgba(48,42,30,.08); margin:0 0 20px; overflow:hidden; }}
    .packet-header {{ display:flex; justify-content:space-between; gap:18px; align-items:end; padding:22px; border-bottom:1px solid var(--line); }}
    .packet-header h2 {{ margin:6px 0 0; font:500 28px/1.1 Georgia,serif; }}
    .packet-count {{ padding:7px 10px; border-radius:999px; background:#e8f0e9; color:#315144; font-size:11px; white-space:nowrap; }}
    .definition-note {{ margin:20px 22px 10px; padding:14px 16px; border-left:4px solid #c2793b; background:#fff4e7; color:#543820; font-size:14px; line-height:1.55; }}
    .pass-flow {{ display:flex; align-items:center; flex-wrap:wrap; gap:8px; margin:14px 22px 22px; color:#315144; font-size:12px; }}
    .pass-flow span {{ padding:7px 9px; border:1px solid #c9d7cd; border-radius:999px; background:#f4f8f4; }}
    .packet-primary {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; padding:0 22px 22px; }}
    .packet-primary-card {{ position:relative; margin:0; padding:10px; border:1px solid #ddd6ca; border-radius:12px; background:#f6f3ec; }}
    .packet-primary-card img {{ display:block; width:100%; max-height:600px; object-fit:contain; border-radius:8px; background:#20231f; }}
    .packet-primary-card figcaption {{ display:flex; justify-content:space-between; gap:12px; padding-top:9px; color:#59625d; font-size:11px; }}
    .packet-primary-card figcaption strong {{ color:#26372f; font-size:12px; }}
    .image-number {{ position:absolute; top:18px; left:18px; z-index:1; padding:5px 7px; border-radius:6px; background:rgba(15,20,17,.88); color:#fff; font-size:10px; font-weight:800; letter-spacing:.08em; }}
    .crop-details,.draft-details {{ border-top:1px solid var(--line); padding:17px 22px 22px; }}
    summary {{ cursor:pointer; color:#244839; font-weight:700; }}
    .weak-note {{ color:var(--muted); font-size:12px; line-height:1.5; }}
    .crop-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
    .crop-card {{ min-width:0; border:1px solid #e1ddd4; border-radius:10px; overflow:hidden; background:#fff; }}
    .crop-visual {{ position:relative; min-height:190px; display:grid; place-items:center; background:#242824; }}
    .crop-visual img {{ display:block; width:100%; height:230px; object-fit:contain; }}
    .crop-visual .image-number {{ top:9px; left:9px; }}
    .overlay-badge {{ position:absolute; top:9px; right:9px; padding:5px 7px; border-radius:6px; background:rgba(255,255,255,.9); color:#24352d; font-size:10px; font-weight:800; }}
    .crop-copy {{ padding:10px; }}
    .crop-copy strong {{ text-transform:capitalize; font-size:13px; }}
    .crop-copy p {{ margin:5px 0 7px; color:#435047; font:400 12px/1.4 Georgia,serif; }}
    .crop-copy small {{ color:#7a827d; font-size:10px; overflow-wrap:anywhere; }}
    .draft-details dl {{ display:grid; grid-template-columns:max-content 1fr; gap:8px 14px; margin:14px 0 0; font-size:12px; }}
    .draft-details dt {{ color:#667069; font-weight:700; }}
    .draft-details dd {{ margin:0; font-family:Georgia,serif; }}
    .packet-missing {{ padding:20px; color:#a14132; font-size:11px; overflow-wrap:anywhere; }}
    .pair-card {{ background:var(--card); border:1px solid var(--line); border-radius:18px; box-shadow:0 16px 40px rgba(48,42,30,.08); margin:0 0 28px; overflow:hidden; }}
    .pair-header {{ display:flex; justify-content:space-between; gap:18px; padding:20px 22px; border-bottom:1px solid var(--line); align-items:end; }}
    .pair-header h2 {{ margin:5px 0 0; max-width:900px; font:500 20px/1.2 Georgia,serif; overflow-wrap:anywhere; }}
    .pair-stats {{ display:flex; gap:7px; flex-wrap:wrap; justify-content:flex-end; }}
    .pair-stats span {{ padding:6px 9px; border-radius:999px; background:#edf2ea; color:#315144; font-size:11px; }}
    .pair-layout {{ display:grid; grid-template-columns:minmax(420px,1.08fr) minmax(380px,.92fr); }}
    .visual-panel {{ padding:22px; border-right:1px solid var(--line); }}
    .canvas-shell {{ position:relative; display:grid; place-items:center; min-height:420px; border-radius:12px; overflow:hidden; background:#1b201d; }}
    canvas {{ display:block; width:100%; max-height:720px; object-fit:contain; }}
    .loading-note {{ position:absolute; color:#dce4df; font-size:12px; }}
    .viewer-actions {{ display:flex; justify-content:space-between; gap:12px; align-items:center; color:var(--muted); font-size:12px; padding-top:10px; }}
    button {{ font:inherit; }}
    .show-all {{ border:1px solid #aeb9b0; background:#f7faf6; color:#214838; padding:7px 11px; border-radius:999px; cursor:pointer; }}
    .show-omitted {{ border:1px dashed #a83e76; background:#fff5fa; color:#7b2452; padding:7px 11px; border-radius:999px; cursor:pointer; }}
    [hidden] {{ display:none!important; }}
    .text-panel {{ padding:20px; }}
    .caption-versions {{ display:grid; gap:12px; }}
    .caption-version {{ min-width:0; padding:14px; border:1px solid #ddd6ca; border-radius:13px; background:#fbf9f3; transition:border-color .15s,box-shadow .15s; }}
    .caption-version.selected {{ border-color:#547f6d; box-shadow:inset 0 0 0 1px #547f6d,0 6px 18px rgba(23,79,58,.08); background:#fff; }}
    .version-selector {{ display:flex; align-items:flex-start; gap:9px; cursor:pointer; color:#244839; }}
    .version-selector input {{ margin-top:3px; accent-color:var(--accent); }}
    .version-selector span {{ display:flex; flex-direction:column; gap:2px; }}
    .version-selector strong {{ font-size:13px; }}
    .version-selector small {{ color:var(--muted); font-size:10px; font-weight:500; }}
    .caption {{ margin:13px 0 14px; font:400 clamp(17px,1.65vw,23px)/1.55 Georgia,serif; }}
    .tagged-caption {{ display:grid; gap:5px; padding-top:11px; border-top:1px dashed #d6d0c5; }}
    .tagged-caption>span {{ color:var(--muted); font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }}
    .tagged-caption>code {{ display:block; max-height:150px; overflow:auto; padding:9px; border-radius:8px; background:#202822; color:#e8f2ea; font:10px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap; overflow-wrap:anywhere; }}
    .tagged-missing {{ color:#f3c5b9; text-transform:none; letter-spacing:0; }}
    .mention {{ display:inline; border:0; color:inherit; padding:2px 3px; margin:0 -1px; border-radius:5px; cursor:pointer; text-decoration:underline; text-decoration-thickness:1px; text-underline-offset:4px; }}
    .mention:hover,.mention.active {{ outline:2px solid rgba(23,79,58,.42); }}
    .version-legend {{ margin-top:16px; padding-top:14px; border-top:1px solid var(--line); }}
    .legend-label {{ display:block; margin-bottom:8px; color:var(--muted); font-size:10px; font-weight:750; text-transform:uppercase; letter-spacing:.08em; }}
    .group-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
    .group-card {{ display:flex; gap:10px; align-items:flex-start; text-align:left; min-width:0; padding:10px; border:1px solid #e1ddd4; border-radius:10px; background:#fff; cursor:pointer; }}
    .group-card:hover,.group-card.active {{ border-color:#789589; box-shadow:inset 0 0 0 1px #789589; }}
    .swatch {{ flex:0 0 25px; height:25px; display:grid; place-items:center; border-radius:7px; color:#fff; background:var(--swatch); text-shadow:0 1px 2px #000; font-size:11px; font-weight:800; }}
    .group-copy {{ display:flex; flex-direction:column; gap:2px; min-width:0; }}
    .group-copy strong {{ font-size:12px; text-transform:capitalize; }}
    .group-copy span {{ color:#48534d; font:400 12px/1.35 Georgia,serif; overflow-wrap:anywhere; }}
    .group-copy small {{ color:#7b837e; font-size:10px; }}
    .empty {{ padding:38px; border:1px dashed var(--line); background:var(--card); }}
    @media (max-width:960px) {{ .masthead,.pair-layout {{ grid-template-columns:1fr; }} .run-note {{ justify-self:start; }} .visual-panel {{ border-right:0; border-bottom:1px solid var(--line); }} .crop-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media (max-width:600px) {{ main {{ padding:24px 12px 48px; }} .pair-header,.packet-header {{ align-items:flex-start; flex-direction:column; }} .group-grid,.packet-primary,.crop-grid {{ grid-template-columns:1fr; }} .canvas-shell {{ min-height:280px; }} .draft-details dl {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header class="masthead">
      <div><span class="kicker">{kicker}</span><h1>{title}</h1><p>{subtitle}</p></div>
      <aside class="run-note"><strong>{sum(pair.get('included') is not False for pair in pairs)} usable pairs + {sum(pair.get('included') is False for pair in pairs)} audit-only exclusions shown.</strong><br>{status_note}</aside>
    </header>
    {sections}
    {empty}
  </main>
  <script id="viewer-data" type="application/json">{payload}</script>
  <script>
  const configs = JSON.parse(document.getElementById("viewer-data").textContent);
  const loadImage = src => new Promise((resolve,reject) => {{ const img=new Image(); img.onload=()=>resolve(img); img.onerror=reject; img.src=src; }});
  for (const config of configs) {{
    const root=document.getElementById(config.viewerId), canvas=root.querySelector("canvas"), ctx=canvas.getContext("2d"), note=root.querySelector(".loading-note");
    const state={{version:"after",pinned:new Set(),hovered:new Set(),source:null,layers:new Map(),omittedLayers:new Map(),showOmitted:false}};
    Promise.all([loadImage(config.source),...config.groups.map(g=>loadImage(g.mask)),...(config.omitted||[]).map(g=>loadImage(g.mask))]).then(([source,...allMasks])=>{{
      state.source=source; canvas.width=source.naturalWidth; canvas.height=source.naturalHeight;
      const masks=allMasks.slice(0,config.groups.length), omittedMasks=allMasks.slice(config.groups.length);
      config.groups.forEach((g,i)=>{{
        const off=document.createElement("canvas"); off.width=canvas.width; off.height=canvas.height; const oc=off.getContext("2d");
        oc.drawImage(masks[i],0,0,canvas.width,canvas.height); const data=oc.getImageData(0,0,canvas.width,canvas.height), c=g.color;
        for(let p=0;p<data.data.length;p+=4){{ const a=data.data[p]; data.data[p]=c[0]; data.data[p+1]=c[1]; data.data[p+2]=c[2]; data.data[p+3]=a; }}
        oc.putImageData(data,0,0); state.layers.set(g.id,off);
      }});
      (config.omitted||[]).forEach((g,i)=>{{
        const off=document.createElement("canvas"); off.width=canvas.width; off.height=canvas.height; const oc=off.getContext("2d");
        oc.drawImage(omittedMasks[i],0,0,canvas.width,canvas.height); const data=oc.getImageData(0,0,canvas.width,canvas.height), src=new Uint8ClampedArray(data.data);
        for(let p=0;p<data.data.length;p+=4){{ const pixel=p/4,x=pixel%canvas.width,y=Math.floor(pixel/canvas.width),inside=src[p]>0; let boundary=false;
          if(inside){{for(const [dx,dy] of [[-1,0],[1,0],[0,-1],[0,1]]){{const nx=x+dx,ny=y+dy;if(nx<0||ny<0||nx>=canvas.width||ny>=canvas.height||src[(ny*canvas.width+nx)*4]===0){{boundary=true;break;}}}}}}
          const dash=((x+y)%12)<7; data.data[p]=194; data.data[p+1]=52; data.data[p+2]=126; data.data[p+3]=boundary&&dash?235:0;
        }} oc.putImageData(data,0,0); state.omittedLayers.set(g.id,off);
      }}); note.remove(); draw();
    }}).catch(()=>{{note.textContent="Could not load one or more image assets."; }});
    function activeIds(){{return state.pinned.size?state.pinned:state.hovered;}}
    function setVersion(version){{
      state.version=version; state.pinned.clear(); state.hovered.clear();
      root.querySelectorAll('input[type="radio"][name$="-caption-version"]').forEach(input=>{{input.checked=input.value===version;}});
      root.querySelectorAll("[data-version-card]").forEach(card=>card.classList.toggle("selected",card.dataset.versionCard===version));
      root.querySelectorAll("[data-version-legend]").forEach(legend=>{{legend.hidden=legend.dataset.versionLegend!==version;}});
      if(omittedButton){{const count=(config.versionOmittedIds[version]||[]).length;omittedButton.textContent=state.showOmitted?`Hide ${{count}} omitted masks`:`Show ${{count}} omitted masks`;}}
      draw();
    }}
    function draw(){{
      if(!state.source)return; ctx.clearRect(0,0,canvas.width,canvas.height); ctx.globalAlpha=1; ctx.drawImage(state.source,0,0,canvas.width,canvas.height);
      const active=activeIds(), visible=new Set(config.versionGroupIds[state.version]||[]);
      for(const g of config.groups){{if(!visible.has(g.id))continue;ctx.globalAlpha=active.size?(active.has(g.id)?.68:.06):.30;ctx.drawImage(state.layers.get(g.id),0,0);}} ctx.globalAlpha=1;
      if(state.showOmitted){{const omittedVisible=new Set(config.versionOmittedIds[state.version]||[]);for(const g of config.omitted||[]){{if(omittedVisible.has(g.id))ctx.drawImage(state.omittedLayers.get(g.id),0,0);}}}}
      root.querySelectorAll("[data-groups]").forEach(el=>{{const ids=el.dataset.groups.split(","),sameVersion=el.dataset.version===state.version;el.classList.toggle("active",sameVersion&&ids.some(id=>active.has(id)));}});
    }}
    root.querySelectorAll('input[type="radio"][name$="-caption-version"]').forEach(input=>input.addEventListener("change",()=>{{if(input.checked)setVersion(input.value);}}));
    root.querySelectorAll("[data-groups]").forEach(el=>{{
      const ids=()=>el.dataset.groups.split(","),version=el.dataset.version||"after";
      el.addEventListener("mouseenter",()=>{{if(version===state.version){{state.hovered=new Set(ids());draw();}}}});
      el.addEventListener("mouseleave",()=>{{if(version===state.version){{state.hovered.clear();draw();}}}});
      el.addEventListener("click",()=>{{
        if(version!==state.version)setVersion(version);
        for(const id of ids()){{state.pinned.has(id)?state.pinned.delete(id):state.pinned.add(id);}}
        draw();
      }});
    }});
    root.querySelector(".show-all").addEventListener("click",()=>{{state.pinned.clear(); state.hovered.clear(); draw();}});
    const omittedButton=root.querySelector(".show-omitted"); if(omittedButton) omittedButton.addEventListener("click",()=>{{state.showOmitted=!state.showOmitted; const count=(config.versionOmittedIds[state.version]||[]).length; omittedButton.textContent=state.showOmitted?`Hide ${{count}} omitted masks`:`Show ${{count}} omitted masks`; draw();}});
  }}
  </script>
</body>
</html>"""
    document = "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise
    return output_path
