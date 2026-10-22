"""Identity-balanced sampling with both image modalities in every batch."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from datasets.regdb import RegDBRecord


class CrossModalityIdentitySampler(Sampler[list[int]]):
    """Yield P identities with K visible and K thermal images per identity."""

    def __init__(
        self,
        records: list[RegDBRecord],
        identities_per_batch: int,
        instances_per_modality: int,
        seed: int = 42,
        paired: bool = True,
    ) -> None:
        if identities_per_batch <= 0 or instances_per_modality <= 0:
            raise ValueError("Sampler batch parameters must be positive")
        by_pid: dict[int, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
        for index, record in enumerate(records):
            if record.modality not in (0, 1):
                raise ValueError(f"Unknown image modality: {record.modality}")
            by_pid[record.pid][record.modality].append(index)
        self.pids = sorted(by_pid)
        if any(not modal[0] or not modal[1] for modal in by_pid.values()):
            raise ValueError("Every training identity must have both modalities")
        if len(self.pids) < identities_per_batch:
            raise ValueError(
                f"Need {identities_per_batch} cross-modal identities, "
                f"found {len(self.pids)}"
            )
        self.indices = by_pid
        self.identities_per_batch = identities_per_batch
        self.instances_per_modality = instances_per_modality
        self.seed = seed
        self.paired = paired
        self.epoch = 0
        self.pairs: dict[int, list[tuple[int, int]]] = {}
        if paired:
            for pid in self.pids:
                visible = {
                    records[index].path.name.replace("_v_", "_x_"): index
                    for index in by_pid[pid][0]
                }
                thermal = {
                    records[index].path.name.replace("_t_", "_x_"): index
                    for index in by_pid[pid][1]
                }
                self.pairs[pid] = [
                    (visible[key], thermal[key])
                    for key in sorted(visible.keys() & thermal.keys())
                ]
        total_pairs = sum(
            max(len(by_pid[pid][0]), len(by_pid[pid][1])) for pid in self.pids
        )
        self.num_batches = max(
            1, math.ceil(total_pairs / (identities_per_batch * instances_per_modality))
        )

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _sample(pool: list[int], count: int, generator: torch.Generator) -> list[int]:
        if len(pool) < count:
            selected = torch.randint(len(pool), (count,), generator=generator)
        else:
            selected = torch.randperm(len(pool), generator=generator)[:count]
        return [pool[int(index)] for index in selected]

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        for _ in range(self.num_batches):
            order = torch.randperm(len(self.pids), generator=generator)
            chosen = [self.pids[int(index)] for index in order[: self.identities_per_batch]]
            batch: list[int] = []
            for pid in chosen:
                if self.paired and self.pairs[pid]:
                    positions = self._sample(
                        list(range(len(self.pairs[pid]))),
                        self.instances_per_modality,
                        generator,
                    )
                    selected_pairs = [self.pairs[pid][position] for position in positions]
                    batch.extend(pair[0] for pair in selected_pairs)
                    batch.extend(pair[1] for pair in selected_pairs)
                else:
                    for modality in (0, 1):
                        batch.extend(
                            self._sample(
                                self.indices[pid][modality],
                                self.instances_per_modality,
                                generator,
                            )
                        )
            yield batch
