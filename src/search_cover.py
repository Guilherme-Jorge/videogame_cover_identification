import os
import json
import logging
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import faiss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_index():
    """Load FAISS index and associated metadata.

    Returns:
        Tuple (faiss.Index, list_of_meta_dicts)
    """
    index = faiss.read_index("data/covers.faiss")
    with open("data/covers_meta.json", "r", encoding="utf-8") as f:
        metas = json.load(f)
    return index, metas


def detect_covers_root(preferred: str | None = None) -> str:
    """Detect the directory that contains the `covers/` folder.

    This mirrors the logic in `build_index.py` to ensure consistency.
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
            os.path.abspath(os.path.join(here, "data"))
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
    """Wrapper for CLIP visual encoder plus projection.

    Args:
        clip_model: Full CLIP model from open_clip.
        projection: Linear layer or identity mapping to output dim.
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
    """Infer CLIP image embedding dimension robustly."""
    val = getattr(clip_model, "embed_dim", None)
    if val is not None:
        return int(val)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        return int(clip_model.encode_image(dummy).shape[-1])


def load_encoder(device: str = "cuda"):
    """Load fine-tuned image encoder and preprocess.

    If `cover_encoder.pt` is present, loads weights saved by training.
    Otherwise, returns base CLIP with identity projection.
    """
    base, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k"
    )

    in_dim = _infer_clip_embed_dim(base)
    weights_path = "data/cover_encoder.pt"

    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        proj_weight = state.get("proj.weight")
        proj = nn.Linear(in_dim, int(proj_weight.shape[0])) if proj_weight is not None else nn.Identity()
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


def detect_and_rectify_cover(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Detect and rectify a likely game cover in a handheld photo.

    The method searches for the largest convex quadrilateral with a portrait-like
    aspect ratio, then applies a perspective transform to obtain a rectified
    crop suitable for matching. If detection fails, the full image is returned.

    Args:
        bgr: Input image in BGR color space.

    Returns:
        Tuple ``(crop_bgr, debug_overlay_bgr, quad_points)`` where:
        - ``crop_bgr`` is the perspective-rectified cover crop (or the original
          image on failure),
        - ``debug_overlay_bgr`` is the original image with the detected
          quadrilateral drawn for visualization,
        - ``quad_points`` are the four corner points in image coordinates
          (ordered TL, TR, BR, BL) or ``None`` when not found.
    """
    overlay = bgr.copy()
    h_img, w_img = bgr.shape[:2]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
    edges = cv2.Canny(thr, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    def right_angle_score(rect_pts: np.ndarray) -> float:
        v0 = rect_pts[1] - rect_pts[0]
        v1 = rect_pts[2] - rect_pts[1]
        v2 = rect_pts[3] - rect_pts[2]
        v3 = rect_pts[0] - rect_pts[3]
        def angle_cos(u, v):
            denom = (np.linalg.norm(u) * np.linalg.norm(v)) + 1e-6
            return np.dot(u, v) / denom
        cos_vals = [abs(angle_cos(v0, v1)), abs(angle_cos(v1, v2)), abs(angle_cos(v2, v3)), abs(angle_cos(v3, v0))]
        return 1.0 - float(min(1.0, sum(cos_vals) / 4.0))

    EXPECT_AR = 1.48
    best = None
    best_score = -1.0

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    for c in cnts:
        if cv2.contourArea(c) < 0.01 * w_img * h_img:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            # fallback to minAreaRect to form a quad
            rect_box = cv2.boxPoints(cv2.minAreaRect(c))
            pts = rect_box.astype(np.float32)
        else:
            pts = approx.reshape(-1, 2).astype(np.float32)
        rect = _order_points(pts)
        width_top = np.linalg.norm(rect[0] - rect[1])
        width_bottom = np.linalg.norm(rect[3] - rect[2])
        height_left = np.linalg.norm(rect[0] - rect[3])
        height_right = np.linalg.norm(rect[1] - rect[2])
        width = max((width_top + width_bottom) * 0.5, 1.0)
        height = max((height_left + height_right) * 0.5, 1.0)
        ar = height / width
        if not (1.2 <= ar <= 2.2):
            continue
        area = width * height
        margin_to_border = min(rect[:, 0].min(), rect[:, 1].min(), w_img - rect[:, 0].max(), h_img - rect[:, 1].max())
        margin_norm = max(0.0, float(margin_to_border) / max(w_img, h_img))
        angle_score = right_angle_score(rect)
        ar_score = float(np.exp(-abs(ar - EXPECT_AR)))
        area_score = float(area / (w_img * h_img))
        total = 0.45 * ar_score + 0.35 * angle_score + 0.15 * area_score + 0.05 * margin_norm
        if total > best_score:
            best_score = total
            best = rect

    if best is None:
        cv2.putText(overlay, "No cover detected - using full image", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return bgr, overlay, None

    cv2.polylines(overlay, [best.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)

    w = int(max(np.linalg.norm(best[0] - best[1]), np.linalg.norm(best[2] - best[3])))
    h = int(max(np.linalg.norm(best[0] - best[3]), np.linalg.norm(best[1] - best[2])))
    w = max(w, 64)
    h = max(h, 64)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(best, dst)
    warped = cv2.warpPerspective(bgr, M, (w, h))
    return warped, overlay, best


def _order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def embed_image(model, preprocess, image_bgr):
    """Embed a BGR image into the encoder feature space."""
    im = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    x = preprocess(im).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        z = model(x)
    return z.cpu().numpy().astype("float32")


def sift_score(query_bgr, cand_bgr):
    sift = cv2.SIFT_create()
    qk, qd = sift.detectAndCompute(query_bgr, None)
    ck, cd = sift.detectAndCompute(cand_bgr, None)
    if qd is None or cd is None:
        return 0.0, 0
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(qd, cd, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return 0.0, len(good)
    src = np.float32([qk[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ck[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    score = inliers / max(len(good), 1)
    return score, inliers


def _search_once(index, metas, model, preprocess, query_bgr, topk: int, rerank_k: int):
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
    D, indices = index.search(q, topk)
    cands = [(float(D[0][j]), int(indices[0][j]), metas[int(indices[0][j])]) for j in range(topk)]

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


def search(image_path, topk=25, rerank_k=5, accept=0.25):
    """Search for the best-matching game cover given an input image path.

    Args:
        image_path: Path to the user-provided photo.
        topk: Number of candidates to retrieve from FAISS.
        rerank_k: Number of top candidates to rerank geometrically.
        accept: Cosine similarity acceptance threshold.

    Returns:
        Result dictionary with scores, best match and alternatives.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index, metas = load_index()
    model, preprocess = load_encoder(device)

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
    is_confident = (chosen["cosine"] >= accept) or (chosen["geom_score"] >= 0.3)

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


if __name__ == "__main__":
    import sys
    import json as pyjson

    res = search(sys.argv[1])
    logger.info("%s", pyjson.dumps(res, indent=2))
