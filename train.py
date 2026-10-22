"""Train all four experts of the fixed Full TAMamba model."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.regdb import RegDBDataset, load_train_split as load_regdb
from datasets.sampler import CrossModalityIdentitySampler
from datasets.sysu import load_train_split as load_sysu
from datasets.transforms import build_train_transform
from engine.trainer import train_one_epoch
from losses.reid import CrossBatchMemory
from models.network import CMMPPRTAMambaReID, TAMambaReID
from utils.runtime import build_optimizer, seed_everything

ROOT = Path(__file__).resolve().parent
EXPERTS = (
    ("global", TAMambaReID, False),
    ("spatial", TAMambaReID, True),
    ("cmmppr_global", CMMPPRTAMambaReID, False),
    ("cmmppr_spatial", CMMPPRTAMambaReID, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("regdb", "sysu-mm01"), default="regdb")
    parser.add_argument("--data-root", type=Path, help="Dataset root; defaults to data/<dataset>.")
    parser.add_argument("--trial", type=int, default=1, help="RegDB training split, from 1 to 10.")
    parser.add_argument(
        "--pretrained", default="openai",
        help="OpenAI RN101 weights, or a local OpenCLIP RN101 checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--identities-per-batch", type=int, default=8)
    parser.add_argument("--instances-per-modality", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 1 <= args.trial <= 10 or (args.dataset == "sysu-mm01" and args.trial != 1):
        parser.error("Use trials 1-10 for RegDB; SYSU-MM01 has one training split (--trial 1).")
    if args.epochs < 1 or args.identities_per_batch < 2 or args.instances_per_modality < 1:
        parser.error("Require epochs >= 1, identities-per-batch >= 2 and instances-per-modality >= 1.")
    if args.workers < 0:
        parser.error("workers must be non-negative.")
    if args.pretrained != "openai":
        weight_path = Path(args.pretrained).expanduser().resolve()
        if not weight_path.is_file():
            parser.error(f"Pretrained RN101 checkpoint not found: {weight_path}")
        args.pretrained = str(weight_path)
    dataset_name = "RegDB" if args.dataset == "regdb" else "SYSU-MM01"
    args.data_root = (args.data_root or ROOT / "data" / dataset_name).expanduser().resolve()
    return args


def train_expert(args, records, num_classes, name, model_class, spatial, device) -> None:
    seed_everything()
    dataset = RegDBDataset(records, build_train_transform())
    sampler = CrossModalityIdentitySampler(
        records, args.identities_per_batch, args.instances_per_modality,
        seed=42, paired=args.dataset == "regdb",
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = model_class(num_classes, pretrained=args.pretrained).to(device)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    xbm = CrossBatchMemory().to(device)
    print(f"[{name}] Full model: {num_classes} identities, {len(records)} images, device={device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        losses = train_one_epoch(
            model, loader, optimizer, device,
            epoch=epoch, epochs=args.epochs, xbm=xbm, spatial=spatial,
        )
        scheduler.step()
        print(f"[{name}] Epoch {epoch}/{args.epochs}: loss={losses['loss']:.6f}", flush=True)
    print(f"[{name}] Training completed.", flush=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build or use --device cpu.")
    records, num_classes = (
        load_regdb(args.data_root, args.trial)
        if args.dataset == "regdb" else load_sysu(args.data_root)
    )
    for name, model_class, spatial in EXPERTS:
        train_expert(args, records, num_classes, name, model_class, spatial, device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("Full training completed for all four experts.", flush=True)


if __name__ == "__main__":
    main()
