"""SYSU-MM01 train and validation identity loading."""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from datasets.regdb import RegDBRecord

RGB_CAMERAS = (1, 2, 4, 5)
IR_CAMERAS = (3, 6)
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"
}


def _parse_id_file(path: Path) -> list[int]:
    """Read comma- or whitespace-separated identity lists under exp/."""
    if not path.is_file():
        raise FileNotFoundError(f"SYSU-MM01 identity file not found: {path}")
    tokens = re.split(r"[\s,]+", path.read_text(encoding="utf-8-sig").strip())
    if not tokens or not tokens[0]:
        raise ValueError(f"SYSU-MM01 identity file is empty: {path}")
    try:
        ids = [int(token) for token in tokens if token]
    except ValueError as exc:
        raise ValueError(f"Malformed SYSU-MM01 identity list: {path}") from exc
    if any(pid < 0 for pid in ids):
        raise ValueError(f"SYSU-MM01 identities must be non-negative: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate identities in SYSU-MM01 list: {path}")
    return ids


def _collect_records(
    root: Path, identities: list[int], cameras: tuple[int, ...], modality: int
) -> list[RegDBRecord]:
    missing_cameras = [
        root / f"cam{camera}"
        for camera in cameras
        if not (root / f"cam{camera}").is_dir()
    ]
    if missing_cameras:
        raise FileNotFoundError(
            "SYSU-MM01 camera directories are missing: "
            + ", ".join(str(path) for path in missing_cameras)
        )
    records: list[RegDBRecord] = []
    empty_identities: list[int] = []
    for pid in identities:
        identity_records: list[RegDBRecord] = []
        for camera in cameras:
            camera_dir = root / f"cam{camera}"
            candidates = (camera_dir / f"{pid:04d}", camera_dir / str(pid))
            directory = next((path for path in candidates if path.is_dir()), None)
            if directory is None:
                continue
            paths = sorted(
                path.resolve()
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            identity_records.extend(RegDBRecord(path, pid, modality) for path in paths)
        if not identity_records:
            empty_identities.append(pid)
        records.extend(identity_records)
    if empty_identities:
        preview = ", ".join(str(pid) for pid in empty_identities[:10])
        raise FileNotFoundError(
            f"SYSU-MM01 identities have no modality-{modality} images in "
            f"cameras {cameras}: {preview}"
        )
    return records


def load_train_split(root: str | Path) -> tuple[list[RegDBRecord], int]:
    """Combine train+validation identities and remap classification labels."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"SYSU-MM01 root directory not found: {root_path}")
    train_ids = _parse_id_file(root_path / "exp" / "train_id.txt")
    val_ids = _parse_id_file(root_path / "exp" / "val_id.txt")
    identities = sorted(set(train_ids) | set(val_ids))
    if len(identities) != len(train_ids) + len(val_ids):
        raise ValueError("SYSU-MM01 train_id.txt and val_id.txt overlap")
    visible = _collect_records(root_path, identities, RGB_CAMERAS, 0)
    thermal = _collect_records(root_path, identities, IR_CAMERAS, 1)
    mapping = {pid: index for index, pid in enumerate(identities)}
    records = [
        replace(record, pid=mapping[record.pid])
        for record in (*visible, *thermal)
    ]
    return records, len(mapping)
