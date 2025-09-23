import os
import json
import logging
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import faiss
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path):
    """Stream JSON lines from a file.

    Args:
        path: Path to a .jsonl file.

    Yields:
        Parsed JSON objects per non-empty line.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def detect_covers_root(preferred: str | None = None) -> str:
    """Detect repository-root for cover images.

    We expect metadata `local_filename` like `covers\\7346.jpg`. The returned
    directory should be the parent that contains the `covers/` folder so that
    `os.path.join(root, local_filename)` points to an actual image file.

    Args:
        preferred: Optional first candidate to try.

    Returns:
        Absolute path to a directory containing a `covers` subfolder.
    """
    candidates = []
    if preferred:
        candidates.append(preferred)
    env_root = os.environ.get("COVERS_ROOT")
    if env_root:
        candidates.append(env_root)
    here = os.path.abspath(os.path.dirname(__file__))
    candidates.extend(
        [
            here,
            os.path.abspath(os.path.join(here, "..")),
            os.path.abspath(os.path.join(here, "..", "igdb-cover-extraction")),
            os.path.abspath(os.path.join(here, "..", "..")),
        ]
    )
    for c in candidates:
        try:
            if os.path.isdir(os.path.join(c, "covers")):
                logger.info("Using covers root: %s", c)
                return c
        except Exception:
            continue
    logger.warning("Could not auto-detect covers root; defaulting to script dir: %s", here)
    return here

class ImgModel(nn.Module):
    """Image encoder wrapper that applies CLIP visual tower and a projection.

    This mirrors the training-time architecture in `train_finetune.py` so that
    saved weights load correctly.

    Args:
        clip_model: The full CLIP model from open_clip.
        projection: Linear projection or identity mapping to target dim.
    """

    def __init__(self, clip_model: nn.Module, projection: nn.Module):
        super().__init__()
        self.clip = clip_model
        self.proj = projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.clip.encode_image(x)
        feats = self.proj(feats) if not isinstance(self.proj, nn.Identity) else feats
        feats = F.normalize(feats, dim=-1)
        return feats


def _infer_clip_embed_dim(clip_model: nn.Module) -> int:
    """Infer the CLIP image embedding dimension robustly.

    Args:
        clip_model: The full CLIP model.

    Returns:
        Image embedding dimension as an integer.
    """
    embed_dim = getattr(clip_model, "embed_dim", None)
    if embed_dim is not None:
        return int(embed_dim)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        return int(clip_model.encode_image(dummy).shape[-1])


def load_encoder(device: str = "cuda"):
    """Load the image encoder and preprocessing pipeline.

    If `cover_encoder.pt` exists, loads the fine-tuned `ImgModel` weights that
    were produced by `train_finetune.py`. Otherwise returns a base CLIP visual
    model with identity projection.

    Args:
        device: Target device string (e.g., "cuda" or "cpu").

    Returns:
        Tuple of (encoder model in eval mode on device, PIL-to-tensor preprocess).
    """
    base, _, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k"
    )
    preprocess = preprocess_val

    weights_path = "cover_encoder.pt"
    in_dim = _infer_clip_embed_dim(base)

    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        proj_weight = state.get("proj.weight")
        if proj_weight is not None:
            out_dim = int(proj_weight.shape[0])
            proj = nn.Linear(in_dim, out_dim)
        else:
            proj = nn.Identity()
        model = ImgModel(base, proj)
        load_res = model.load_state_dict(state, strict=False)
        if hasattr(load_res, "missing_keys") and load_res.missing_keys:
            logger.warning("Missing keys when loading encoder: %s", load_res.missing_keys)
        if hasattr(load_res, "unexpected_keys") and load_res.unexpected_keys:
            logger.warning("Unexpected keys when loading encoder: %s", load_res.unexpected_keys)
        logger.info("Loaded fine-tuned encoder from %s", weights_path)
    else:
        logger.warning("Fine-tuned weights not found at %s; using base CLIP with identity projection", weights_path)
        model = ImgModel(base, nn.Identity())

    model.eval().to(device)
    return model, preprocess

def embed_images(jsonl_path: str, root: str = "."):
    """Compute normalized embeddings for all images listed in a metadata JSONL.

    Args:
        jsonl_path: Path to metadata.jsonl with field `local_filename` per entry.
        root: Root directory to resolve relative `local_filename` paths.

    Returns:
        Tuple of (metadatas list, embeddings ndarray of shape [N, D]).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_encoder(device)

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
        embs_list.append(z.squeeze(0).cpu().numpy().astype("float32"))
        valid_metas.append(m)

    if skipped_null or skipped_missing:
        logger.warning("Skipped %d entries with null paths and %d missing files", skipped_null, skipped_missing)

    if len(embs_list) == 0:
        return [], np.zeros((0, out_dim), dtype="float32")

    embs = np.stack(embs_list, axis=0)
    return valid_metas, embs

def build_faiss(embs: np.ndarray, use_gpu: bool = True):
    """Build an inner-product FAISS index for normalized embeddings.

    Args:
        embs: Array of shape [N, D], expected L2-normalized per row.
        use_gpu: Whether to accelerate indexing on GPU if available.

    Returns:
        A CPU FAISS index suitable for search and serialization.
    """
    d = int(embs.shape[1])
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index_cpu = faiss.IndexFlatIP(d)
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        index_gpu.add(embs)
        index = faiss.index_gpu_to_cpu(index_gpu)
    else:
        index = faiss.IndexFlatIP(d)
        index.add(embs)
    return index

def main():
    """Build and persist the cover embedding index for retrieval."""
    jsonl = "metadata.jsonl"
    covers_root = detect_covers_root()
    metas, embs = embed_images(jsonl, root=covers_root)
    index = build_faiss(embs, use_gpu=True)
    faiss.write_index(index, "covers.faiss")
    np.save("covers.npy", embs)
    with open("covers_meta.json", "w", encoding="utf-8") as f:
        json.dump(metas, f)
    logger.info("Saved covers.faiss, covers.npy, covers_meta.json")

if __name__ == "__main__":
    main()