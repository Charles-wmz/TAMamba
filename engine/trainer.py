"""Training loop and paired augmentation for the Full TAMamba system."""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import RandomErasing
from tqdm.auto import tqdm

from losses.reid import CrossBatchMemory, reid_loss


def _cross_modal_pairs(
    labels: Tensor, modalities: Tensor
) -> list[tuple[Tensor, Tensor]]:
    pairs: list[tuple[Tensor, Tensor]] = []
    for pid in labels.unique():
        visible = torch.where(labels.eq(pid) & modalities.eq(0))[0]
        thermal = torch.where(labels.eq(pid) & modalities.eq(1))[0]
        pair_count = min(visible.numel(), thermal.numel())
        pairs.extend(zip(visible[:pair_count], thermal[:pair_count]))
    return pairs


def paired_geometric_augment(
    images: Tensor, labels: Tensor, modalities: Tensor
) -> Tensor:
    """Apply paired crop/flip transforms and independent random erasing."""
    height, width = images.shape[-2:]
    padded = F.pad(images, (10, 10, 10, 10), value=0.0)
    output = torch.empty_like(images)
    processed = torch.zeros(
        images.shape[0], dtype=torch.bool, device=images.device
    )

    def transform(index: Tensor, top: int, left: int, flip: bool) -> None:
        crop = padded[index, :, top : top + height, left : left + width]
        output[index] = crop.flip(-1) if flip else crop
        processed[index] = True

    for visible_index, thermal_index in _cross_modal_pairs(labels, modalities):
        top = int(torch.randint(21, (), device=images.device).item())
        left = int(torch.randint(21, (), device=images.device).item())
        flip = bool(torch.rand((), device=images.device) < 0.5)
        transform(visible_index, top, left, flip)
        transform(thermal_index, top, left, flip)
    for index in torch.where(~processed)[0]:
        top = int(torch.randint(21, (), device=images.device).item())
        left = int(torch.randint(21, (), device=images.device).item())
        flip = bool(torch.rand((), device=images.device) < 0.5)
        transform(index, top, left, flip)
    eraser = RandomErasing(p=0.5, value=0.0, inplace=False)
    return torch.stack([eraser(image) for image in output])


def cross_spectral_blend(
    images: Tensor, labels: Tensor, modalities: Tensor
) -> Tensor:
    """Blend paired spectra with probability 0.5 using a vertical mask."""
    blended = images.clone()
    height = images.shape[-2]
    rows = torch.linspace(-1.0, 1.0, height, device=images.device)
    for visible_index, thermal_index in _cross_modal_pairs(labels, modalities):
        if torch.rand((), device=images.device) >= 0.5:
            continue
        center = torch.empty((), device=images.device).uniform_(-0.35, 0.35)
        slope = torch.empty((), device=images.device).uniform_(3.0, 7.0)
        alpha = torch.sigmoid((rows - center) * slope)
        if torch.rand((), device=images.device) < 0.5:
            alpha = alpha.flip(0)
        alpha = alpha.view(1, height, 1)
        visible = images[visible_index]
        thermal = images[thermal_index]
        blended[visible_index] = alpha * visible + (1.0 - alpha) * thermal
        blended[thermal_index] = alpha * thermal + (1.0 - alpha) * visible
    return blended


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    epochs: int,
    xbm: CrossBatchMemory,
    spatial: bool,
) -> dict[str, float]:
    """Train one Full expert in FP32 with the fixed composite objective."""
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    steps = 0
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{epochs}",
        total=len(loader),
        dynamic_ncols=True,
        leave=True,
    )
    for step, (images, labels, modalities, _) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        modalities = modalities.to(device, non_blocking=True)
        images = paired_geometric_augment(images, labels, modalities)
        images = cross_spectral_blend(images, labels, modalities)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images, modalities, labels=labels)
        loss, values = reid_loss(
            outputs,
            labels,
            modalities,
            xbm=xbm,
            spatial=spatial,
            epoch=epoch,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at step {step}: {values}"
            )
        loss.backward()
        optimizer.step()
        steps += 1
        for key, value in values.items():
            totals[key] += value
        progress.set_postfix(
            loss=totals["loss"] / steps,
            xbm=totals["xbm"] / steps,
            mppr=totals["masked_part_reconstruction"] / steps,
        )
    if not steps:
        raise RuntimeError("Training loader produced no batches")
    return {key: value / steps for key, value in totals.items()}
