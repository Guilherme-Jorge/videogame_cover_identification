# train_finetune.py (CLI-enabled)
import math, random, json, os, numpy as np, torch
from contextlib import nullcontext
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageFilter
from tqdm import tqdm
import argparse
import open_clip


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class PerspectiveJitter:
    def __init__(self, max_warp=0.08):
        self.max_warp = max_warp

    def __call__(self, img):
        w, h = img.size
        dx, dy = self.max_warp * w, self.max_warp * h
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = [
            (np.random.uniform(-dx, dx), np.random.uniform(-dy, dy)),
            (w + np.random.uniform(-dx, dx), np.random.uniform(-dy, dy)),
            (w + np.random.uniform(-dx, dx), h + np.random.uniform(-dy, dy)),
            (np.random.uniform(-dx, dx), h + np.random.uniform(-dy, dy)),
        ]
        coeffs = _persp_coeffs(src, dst)
        return img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def _persp_coeffs(src, dst):
    A, B = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.extend(
            [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        )
        B.extend([u, v])
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return coeffs


def glare_overlay(img, alpha_range=(0.05, 0.2)):
    w, h = img.size
    overlay = Image.new("RGB", (w, h), (255, 255, 255))
    mask = (
        Image.linear_gradient("L")
        .resize((w, h))
        .rotate(np.random.uniform(0, 180), expand=False)
        .filter(ImageFilter.GaussianBlur(radius=np.random.uniform(20, 60)))
    )
    alpha = np.random.uniform(*alpha_range)
    return Image.composite(overlay, img, mask.point(lambda p: int(p * alpha)))


class CoverAugDataset(Dataset):
    def __init__(self, jsonl_path, root_dir, img_key="local_filename"):
        import os
        all_items = list(load_jsonl(jsonl_path))
        self.root = root_dir
        self.img_key = img_key
        valid, skipped_null, skipped_missing = [], 0, 0
        for it in all_items:
            p = it.get(img_key)
            if not p:
                skipped_null += 1
                continue
            full = p if os.path.isabs(p) else os.path.join(root_dir, p)
            if not os.path.exists(full):
                skipped_missing += 1
                continue
            valid.append(it)
        self.items = valid
        print(f"CoverAugDataset: {len(valid)} valid, "
              f"skipped {skipped_null} null paths, {skipped_missing} missing files.")
        self.base = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
            ]
        )
        self.color = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.aug = T.Compose(
            [
                T.Lambda(lambda im: PerspectiveJitter(0.10)(im)),
                T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.3),
                T.RandomApply([T.Lambda(glare_overlay)], p=0.2),
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.RandomRotation(degrees=3),
                T.RandomAdjustSharpness(1.5, p=0.3),
                T.RandomAutocontrast(p=0.3),
                T.Lambda(lambda im: im.convert("RGB")),
            ]
        )
        self.to_tensor = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        # self.img_key already set above

    def __len__(self):
        return len(self.items)

    def _load_img(self, path):
        full = path if os.path.isabs(path) else os.path.join(self.root, path)
        with Image.open(full) as im:
            return im.convert("RGB")

    def _view(self, im):
        im = self.base(im)
        im = self.color(im)
        im = self.aug(im)
        return self.to_tensor(im)

    def __getitem__(self, idx):
        p = self.items[idx][self.img_key]
        im = self._load_img(p)
        v1 = self._view(im)
        v2 = self._view(im)
        return v1, v2


class NTXent(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, z1, z2):
        z = torch.cat([z1, z2], dim=0)
        sim = (z @ z.t()) / self.t
        B = z1.size(0)
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)
        targets = torch.arange(B, device=z.device)
        targets = torch.cat([targets + B, targets])
        return nn.functional.cross_entropy(sim, targets)


class ImgModel(nn.Module):
    def __init__(self, clip_model, proj):
        super().__init__()
        self.clip = clip_model
        self.proj = proj

    def forward(self, x):
        feats = self.clip.encode_image(x)  # returns dim = base.embed_dim
        feats = self.proj(feats) if not isinstance(self.proj, nn.Identity) else feats
        feats = F.normalize(feats, dim=-1)
        return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to metadata.jsonl")
    ap.add_argument("--root", default=".", help="Root dir for image files")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--amp", choices=["none", "fp16", "bf16"], default="none")
    ap.add_argument("--out", default="cover_encoder.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k"
    )
    in_dim = getattr(base, "embed_dim", base.encode_image(torch.zeros(1,3,224,224)).shape[-1])
    proj = nn.Identity() if args.dim == in_dim else nn.Linear(in_dim, args.dim)
    net = ImgModel(base, proj).to(device)
    # Train only the visual tower (text tower unused)
    for p in base.parameters(): 
        p.requires_grad = False
    for p in base.visual.parameters():
        p.requires_grad = True
    for p in net.proj.parameters():
        p.requires_grad = True

    ds = CoverAugDataset(args.jsonl, args.root)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, len(dl))
    )
    loss_fn = NTXent(temperature=0.07)

    # AMP setup (new API)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = None
    if args.amp == "fp16":
        amp_dtype = torch.float16
    elif args.amp == "bf16":
        amp_dtype = torch.bfloat16
    use_scaler = (args.amp == "fp16" and device_type == "cuda")
    scaler = torch.amp.GradScaler(device=device_type, enabled=use_scaler)

    net.train()
    for ep in range(args.epochs):
        pbar = tqdm(dl, desc=f"epoch {ep + 1}/{args.epochs}")
        for v1, v2 in pbar:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            # Choose the right autocast context (or no-op if disabled)
            autocast_ctx = nullcontext()
            if args.amp != "none":
                if device_type == "cuda":
                    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype)
                elif device_type == "cpu" and args.amp == "bf16":
                    # CPU autocast supports only bfloat16
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

    torch.save(net.state_dict(), args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
