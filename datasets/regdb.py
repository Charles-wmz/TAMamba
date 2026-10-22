"""RegDB training indices and RGB/thermal image loading."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RegDBRecord:
    path: Path
    pid: int
    modality: int  # 0: visible, 1: thermal


def _parse_index(root: Path, index_path: Path, modality: int) -> list[RegDBRecord]:
    """Read an index and verify that all referenced images exist."""
    if not index_path.is_file():
        raise FileNotFoundError(f"RegDB index not found: {index_path}")
    records: list[RegDBRecord] = []
    missing: list[Path] = []
    with index_path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                relative, label = line.rsplit(maxsplit=1)
                pid = int(label)
                if pid < 0:
                    raise ValueError("Identity labels must be non-negative")
            except ValueError as exc:
                raise ValueError(
                    f"Malformed RegDB entry at {index_path}:{line_no}: {line!r}"
                ) from exc
            path = root / Path(relative.replace("\\", "/"))
            if not path.is_file():
                missing.append(path)
            records.append(RegDBRecord(path, pid, modality))
    if not records:
        raise ValueError(f"No records found in {index_path}")
    if missing:
        preview = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} indexed images are missing (first entries):\n{preview}"
        )
    return records


def load_train_split(root: str | Path, trial: int) -> tuple[list[RegDBRecord], int]:
    """Load both training modalities and remap labels to consecutive indices."""
    if not 1 <= trial <= 10:
        raise ValueError(f"RegDB trial must be in [1, 10], got {trial}")
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"RegDB root directory not found: {root_path}")
    visible = _parse_index(
        root_path, root_path / "idx" / f"train_visible_{trial}.txt", 0
    )
    thermal = _parse_index(
        root_path, root_path / "idx" / f"train_thermal_{trial}.txt", 1
    )
    visible_ids = {record.pid for record in visible}
    thermal_ids = {record.pid for record in thermal}
    if visible_ids != thermal_ids:
        raise ValueError("Every RegDB training identity must have both modalities")
    mapping = {pid: index for index, pid in enumerate(sorted(visible_ids))}
    records = [
        replace(record, pid=mapping[record.pid])
        for record in (*visible, *thermal)
    ]
    return records, len(mapping)


class RegDBDataset(Dataset[tuple[Tensor, int, int, str]]):
    """Load training records from either supported dataset."""

    def __init__(
        self,
        records: list[RegDBRecord],
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        if not records:
            raise ValueError("The image dataset requires at least one record")
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int, str]:
        record = self.records[index]
        with Image.open(record.path) as image:
            # RGB conversion repeats a single thermal channel three times.
            tensor = self.transform(image.convert("RGB"))
        return tensor, record.pid, record.modality, str(record.path)
