"""Training functionality for fine-tuning CLIP."""

import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import config
from ..models.clip_model import ImgModel
from ..training.dataset import CoverAugDataset
from ..training.losses import NTXent
from ..utils.paths import detect_covers_root


def train_finetune(
    jsonl_path: str = None,
    root_dir: str = None,
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    workers: int = None,
    dim: int = None,
    amp: str = None,
    out_path: str = None,
    device: str = None
):
    """Fine-tune CLIP model using contrastive learning.

    Args:
        jsonl_path: Path to metadata JSONL file.
        root_dir: Root directory for images.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        workers: Number of data loader workers.
        dim: Output embedding dimension.
        amp: Automatic mixed precision mode ("none", "fp16", "bf16").
        out_path: Output path for saved model.
        device: Device for training.
    """
    # Use config defaults
    jsonl_path = jsonl_path or config.training.jsonl_path
    epochs = epochs or config.training.epochs
    batch_size = batch_size or config.training.batch_size
    lr = lr or config.training.lr
    workers = workers or config.training.workers
    dim = dim or config.training.dim
    amp = amp or config.training.amp
    out_path = out_path or config.training.out_path
    device = device or config.device
    root_dir = root_dir or detect_covers_root()

    # Load base CLIP model
    import open_clip
    base, _, _ = open_clip.create_model_and_transforms(
        config.model.clip_model, pretrained=config.model.clip_pretrained
    )

    # Determine input/output dimensions
    in_dim = getattr(base, "embed_dim", base.encode_image(torch.zeros(1, 3, 224, 224)).shape[-1])
    proj = nn.Identity() if dim == in_dim else nn.Linear(in_dim, dim)
    net = ImgModel(base, proj).to(device)

    # Freeze CLIP parameters except visual tower
    for p in base.parameters():
        p.requires_grad = False
    for p in base.visual.parameters():
        p.requires_grad = True
    for p in net.proj.parameters():
        p.requires_grad = True

    # Create dataset and dataloader
    ds = CoverAugDataset(jsonl_path, root_dir)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )

    # Optimizer and scheduler
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * max(1, len(dl))
    )

    # Loss function
    loss_fn = NTXent(temperature=config.training.temperature)

    # AMP setup
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = None
    if amp == "fp16":
        amp_dtype = torch.float16
    elif amp == "bf16":
        amp_dtype = torch.bfloat16
    use_scaler = (amp == "fp16" and device_type == "cuda")
    scaler = torch.amp.GradScaler(device=device_type, enabled=use_scaler)

    # Training loop
    net.train()
    for ep in range(epochs):
        pbar = tqdm(dl, desc=f"epoch {ep + 1}/{epochs}")
        for v1, v2 in pbar:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            # Choose autocast context
            autocast_ctx = nullcontext()
            if amp != "none":
                if device_type == "cuda":
                    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype)
                elif device_type == "cpu" and amp == "bf16":
                    autocast_ctx = torch.amp.autocast("cpu", dtype=torch.bfloat16)

            with autocast_ctx:
                z1 = net(v1)
                z2 = net(v2)
                loss = loss_fn(z1, z2)

            opt.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            pbar.set_postfix(loss=float(loss))

    # Save model
    torch.save(net.state_dict(), out_path)
    print(f"Saved model to {out_path}")


def main():
    """CLI entry point for training."""
    ap = argparse.ArgumentParser(description="Fine-tune CLIP for cover identification")
    ap.add_argument("--jsonl", default=config.training.jsonl_path,
                   help="Path to metadata.jsonl")
    ap.add_argument("--root", default=None,
                   help="Root dir for image files")
    ap.add_argument("--epochs", type=int, default=config.training.epochs,
                   help="Number of epochs")
    ap.add_argument("--batch-size", type=int, default=config.training.batch_size,
                   help="Batch size")
    ap.add_argument("--lr", type=float, default=config.training.lr,
                   help="Learning rate")
    ap.add_argument("--workers", type=int, default=config.training.workers,
                   help="Number of workers")
    ap.add_argument("--dim", type=int, default=config.training.dim,
                   help="Output embedding dimension")
    ap.add_argument("--amp", choices=["none", "fp16", "bf16"],
                   default=config.training.amp, help="Automatic mixed precision")
    ap.add_argument("--out", default=config.training.out_path,
                   help="Output path for model")
    ap.add_argument("--device", default=config.device,
                   help="Device for training")

    args = ap.parse_args()
    train_finetune(**vars(args))


if __name__ == "__main__":
    main()
