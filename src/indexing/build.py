"""Index building functionality."""

import logging
import os
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

from ..config import config
from ..models.clip_model import load_encoder
from ..models.index import build_faiss, save_index
from ..utils.data import load_jsonl
from ..utils.paths import detect_covers_root

logger = logging.getLogger(__name__)


def embed_images(
    jsonl_path: str, root: Optional[str] = None, device: Optional[str] = None
) -> tuple[list[dict], torch.Tensor]:
    """Compute normalized embeddings for all images listed in a metadata JSONL.

    Args:
        jsonl_path: Path to metadata.jsonl with field `local_filename` per entry.
        root: Root directory to resolve relative `local_filename` paths.
        device: Device to use for computation.

    Returns:
        Tuple of (metadatas list, embeddings tensor of shape [N, D]).
    """
    device = device or config.device
    root = root or detect_covers_root()

    model, preprocess = load_encoder(device, config.model.weights_path)

    raw_metas = list(load_jsonl(jsonl_path))
    n = len(raw_metas)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        out_dim = int(model(dummy).shape[-1])

    valid_metas = []
    embs_list = []

    skipped_null = 0
    skipped_missing = 0

    logger.info("Embedding up to %d images to dimension %d", n, out_dim)

    for m in tqdm(raw_metas, desc="Embedding"):
        rel = m.get("local_filename")
        if not rel:
            skipped_null += 1
            continue
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            skipped_missing += 1
            continue
        with Image.open(path) as im:
            im = im.convert("RGB")
        x = preprocess(im).unsqueeze(0).to(device)
        with torch.no_grad():
            z = model(x)
        embs_list.append(z.squeeze(0).cpu())
        valid_metas.append(m)

    if skipped_null or skipped_missing:
        logger.warning(
            "Skipped %d entries with null paths and %d missing files", skipped_null, skipped_missing
        )

    if len(embs_list) == 0:
        return [], torch.zeros((0, out_dim), dtype=torch.float32, device="cpu")

    embs = torch.stack(embs_list, dim=0)
    return valid_metas, embs


def build_index(
    jsonl_path: Optional[str] = None,
    root: Optional[str] = None,
    use_gpu: Optional[bool] = None,
    device: Optional[str] = None,
    index_path: Optional[str] = None,
    npy_path: Optional[str] = None,
    meta_path: Optional[str] = None,
):
    """Build and persist the cover embedding index for retrieval.

    Args:
        jsonl_path: Path to metadata JSONL file.
        root: Root directory containing covers.
        use_gpu: Whether to use GPU for indexing.
        device: Device for embedding computation.
        index_path: Output path for FAISS index.
        npy_path: Output path for embeddings.
        meta_path: Output path for metadata.
    """
    covers_root = root or detect_covers_root()
    jsonl_path = jsonl_path or config.index.jsonl_path
    if not os.path.isabs(jsonl_path):
        jsonl_path = os.path.join(covers_root, jsonl_path)
    use_gpu = use_gpu if use_gpu is not None else config.index.use_gpu
    index_path = index_path or config.index.index_path
    if not os.path.isabs(index_path):
        index_path = os.path.join(covers_root, index_path)
    npy_path = npy_path or config.index.npy_path
    if not os.path.isabs(npy_path):
        npy_path = os.path.join(covers_root, npy_path)
    meta_path = meta_path or config.index.meta_path
    if not os.path.isabs(meta_path):
        meta_path = os.path.join(covers_root, meta_path)

    metas, embs = embed_images(jsonl_path, root=covers_root, device=device)

    # Convert to numpy for FAISS
    embs_np = embs.numpy().astype("float32")
    index = build_faiss(embs_np, use_gpu=use_gpu)

    save_index(index, embs_np, metas, index_path, npy_path, meta_path)
