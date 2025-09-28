"""Dataset classes for training."""

import os

import numpy as np
import torch
import torchvision.transforms as t
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from ..utils.data import load_jsonl


class PerspectiveJitter:
    """Apply random perspective jitter to images."""

    def __init__(self, max_warp: float = 0.08):
        self.max_warp = max_warp

    def __call__(self, img: Image.Image) -> Image.Image:
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
    """Compute perspective transformation coefficients."""
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.extend([
            [x, y, 1, 0, 0, 0, -u * x, -u * y],
            [0, 0, 0, x, y, 1, -v * x, -v * y]
        ])
        b.extend([u, v])
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    coeffs, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    return coeffs


def glare_overlay(img: Image.Image, alpha_range: tuple = (0.05, 0.2)) -> Image.Image:
    """Add glare overlay to image."""
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
    """Dataset for contrastive learning on game covers with augmentations."""

    def __init__(
        self,
        jsonl_path: str,
        root_dir: str,
        img_key: str = "local_filename"
    ):
        """Initialize the dataset.

        Args:
            jsonl_path: Path to metadata JSONL file.
            root_dir: Root directory for image files.
            img_key: Key in metadata for image filename.
        """
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

        # Base preprocessing
        self.base = t.Compose([
            t.Resize(256, interpolation=t.InterpolationMode.BICUBIC),
            t.CenterCrop(224),
        ])

        # Color augmentation
        self.color = t.ColorJitter(0.4, 0.4, 0.4, 0.1)

        # Augmentation pipeline
        self.aug = t.Compose([
            t.Lambda(lambda im: PerspectiveJitter(0.10)(im)),
            t.RandomApply([t.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.3),
            t.RandomApply([t.Lambda(glare_overlay)], p=0.2),
            t.RandomResizedCrop(224, scale=(0.8, 1.0)),
            t.RandomRotation(degrees=3),
            t.RandomAdjustSharpness(1.5, p=0.3),
            t.RandomAutocontrast(p=0.3),
            t.Lambda(lambda im: im.convert("RGB")),
        ])

        # Final tensor conversion
        self.to_tensor = t.Compose([
            t.ToTensor(),
            t.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def _load_img(self, path: str) -> Image.Image:
        """Load image from path."""
        full = path if os.path.isabs(path) else os.path.join(self.root, path)
        with Image.open(full) as im:
            return im.convert("RGB")

    def _view(self, im: Image.Image) -> torch.Tensor:
        """Apply augmentations to create a view."""
        im = self.base(im)
        im = self.color(im)
        im = self.aug(im)
        return self.to_tensor(im)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a pair of augmented views for contrastive learning."""
        p = self.items[idx][self.img_key]
        im = self._load_img(p)
        v1 = self._view(im)
        v2 = self._view(im)
        return v1, v2
