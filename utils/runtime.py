"""Reproducible training and fixed optimizer settings."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int = 42) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_optimizer(model: nn.Module) -> torch.optim.AdamW:
    backbone = list(model.visual.parameters()) + list(model.thermal_stem.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    task = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    return torch.optim.AdamW(
        [{"params": backbone, "lr": 2e-5}, {"params": task, "lr": 3e-4}],
        weight_decay=5e-4,
    )
