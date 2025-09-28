"""Cover search functionality."""

import logging
import os
from typing import Any, Optional

import cv2
import torch

from ..config import config
from ..models.clip_model import load_encoder
from ..models.index import load_index
from ..utils.image import detect_and_rectify_cover, embed_image, sift_score
from ..utils.paths import detect_covers_root

logger = logging.getLogger(__name__)


def _search_once(
    index,
    metas: list,
    model,
    preprocess,
    query_bgr: cv2.Mat,
    topk: int,
    rerank_k: int
) -> dict[str, Any]:
    """Run a single retrieval pass and geometric re-ranking.

    Args:
        index: FAISS index.
        metas: Metadata list aligned with index entries.
        model: Image encoder model.
        preprocess: Preprocessing function for images.
        query_bgr: BGR image array used as query.
        topk: Number of candidates to retrieve from FAISS.
        rerank_k: Number of top candidates to rerank geometrically.

    Returns:
        Dictionary with cosine, geom_score, match metadata and alternatives.
    """
    q = embed_image(model, preprocess, query_bgr)
    d, indices = index.search(q, topk)
    cands = [(float(d[0][j]), int(indices[0][j]), metas[int(indices[0][j])]) for j in range(topk)]

    reranked = []
    covers_root = detect_covers_root()

    for sim, idx, meta in cands[:rerank_k]:
        rel = meta.get("local_filename")
        cand_path = os.path.join(covers_root, rel) if isinstance(rel, str) else None
        cand_bgr = None
        if isinstance(cand_path, str) and os.path.exists(cand_path):
            cand_bgr = cv2.imread(cand_path, cv2.IMREAD_COLOR)
        if cand_bgr is None:
            reranked.append((sim, 0.0, idx, meta))
            continue
        score, inliers = sift_score(query_bgr, cand_bgr)
        reranked.append((sim, score, idx, meta))

    reranked.sort(key=lambda t: (t[1], t[0]), reverse=True)
    best = reranked[0] if reranked else cands[0]

    sim = best[0]
    geo = best[1] if len(best) > 3 else 0.0
    meta = best[3] if len(best) > 3 else best[2]

    return {
        "cosine": float(sim),
        "geom_score": float(geo),
        "match": {
            "id": meta["id"],
            "name": meta["name"],
            "cover_id": meta["cover_id"],
            "cover_url": meta["cover_url"],
            "local_filename": meta["local_filename"],
        },
        "alternatives": [
            {"cosine": float(s), "id": m["id"], "name": m["name"]}
            for (s, _, _, m) in reranked[1:5]
        ],
    }


def search_cover(
    image_path: str,
    topk: Optional[int] = None,
    rerank_k: Optional[int] = None,
    accept: Optional[float] = None,
    device: Optional[str] = None
) -> dict[str, Any]:
    """Search for the best-matching game cover given an input image path.

    Args:
        image_path: Path to the user-provided photo.
        topk: Number of candidates to retrieve from FAISS.
        rerank_k: Number of top candidates to rerank geometrically.
        accept: Cosine similarity acceptance threshold.
        device: Device to use for computation.

    Returns:
        Result dictionary with scores, best match and alternatives.
    """
    # Use config defaults if not specified
    topk = topk or config.search.topk
    rerank_k = rerank_k or config.search.rerank_k
    accept = accept or config.search.accept_threshold
    device = device or config.device

    index, metas = load_index(config.index.index_path, config.index.meta_path)
    model, preprocess = load_encoder(device, config.model.weights_path)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=next(model.parameters()).device)
        out_dim = int(model(dummy).shape[-1])

    if getattr(index, "d", None) not in (None, out_dim):
        raise ValueError(f"Index dim {getattr(index, 'd', None)} does not match model dim {out_dim}")

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    crop, overlay, quad = detect_and_rectify_cover(bgr)

    try:
        cv2.imwrite("export_crop.png", crop)
        cv2.imwrite("export_debug.png", overlay)
        logger.info("Exported rectified crop to export_crop.png and debug overlay to export_debug.png")
    except Exception as e:
        logger.warning("Failed to export debug images: %s", e)

    res_crop = _search_once(index, metas, model, preprocess, crop, topk, rerank_k)
    res_full = _search_once(index, metas, model, preprocess, bgr, topk, rerank_k)

    def _score_key(r):
        return (r.get("geom_score", 0.0), r.get("cosine", 0.0))

    pick_crop = _score_key(res_crop) >= _score_key(res_full)
    chosen = res_crop if pick_crop else res_full
    chosen_variant = "crop" if pick_crop else "full"
    other = res_full if pick_crop else res_crop
    is_confident = (
        (chosen["cosine"] >= accept) or
        (chosen["geom_score"] >= config.search.geom_score_threshold)
    )

    result = {
        "confident": bool(is_confident),
        "strategy": chosen_variant,
        "cosine": float(chosen["cosine"]),
        "geom_score": float(chosen["geom_score"]),
        "match": chosen["match"],
        "alternatives": chosen["alternatives"],
        "other_variant": {
            "strategy": "full" if pick_crop else "crop",
            "cosine": float(other["cosine"]),
            "geom_score": float(other["geom_score"]),
        },
        "exports": {
            "crop": "export_crop.png",
            "debug": "export_debug.png",
            "quad": quad.tolist() if quad is not None else None,
        },
    }
    return result
