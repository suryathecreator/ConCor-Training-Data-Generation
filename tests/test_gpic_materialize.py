from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from PIL import Image

from sam3_mask_captioning.campaign_manifest import initialize_campaign
from sam3_mask_captioning.gpic_materialize import extend_from_gpic
from sam3_mask_captioning.io_utils import read_jsonl


def _fixture_tar(path: Path, image_count: int) -> None:
    with tarfile.open(path, "w") as archive:
        for index in range(image_count):
            key = f"sample-{index:03d}"
            buffer = io.BytesIO()
            Image.new("RGB", (8, 8), (index, 20, 30)).save(buffer, format="JPEG")
            image_payload = buffer.getvalue()
            image_info = tarfile.TarInfo(f"{key}.jpg")
            image_info.size = len(image_payload)
            archive.addfile(image_info, io.BytesIO(image_payload))

            metadata_payload = json.dumps({"caption": f"caption {index}"}).encode()
            metadata_info = tarfile.TarInfo(f"{key}.json")
            metadata_info.size = len(metadata_payload)
            archive.addfile(metadata_info, io.BytesIO(metadata_payload))


def test_parallel_gpic_unit_writes_preserve_order_and_resume(tmp_path, monkeypatch):
    source_tar = tmp_path / "fixture.tar"
    _fixture_tar(source_tar, 20)
    campaign = tmp_path / "campaign"
    initialize_campaign(campaign, unit_size=3, seed=17)
    monkeypatch.setenv("BCC_MATERIALIZE_WRITE_WORKERS", "4")
    monkeypatch.setattr(
        "sam3_mask_captioning.gpic_materialize._candidate_tars",
        lambda *_args: ["train/fixture.tar"],
    )
    monkeypatch.setattr(
        "sam3_mask_captioning.gpic_materialize._download_tar",
        lambda *_args: source_tar,
    )

    first = extend_from_gpic(campaign, target_total=10, seed=17)
    assert first["source_count"] == 10
    assert first["unit_count"] == 4

    second = extend_from_gpic(campaign, add_images=2, seed=17)
    assert second["source_count"] == 12
    assert second["unit_count"] == 5

    rows = read_jsonl(campaign / "source_manifest.jsonl")
    assert [row["source_manifest_index"] for row in rows] == list(range(12))
    assert len({row["source_context"]["pair_key"] for row in rows}) == 12
    assert [
        json.loads((campaign / "units" / f"{index:06d}" / "unit.json").read_text())[
            "source_count"
        ]
        for index in range(5)
    ] == [3, 3, 3, 1, 2]
