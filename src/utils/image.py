"""Image processing utilities."""

import logging

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def embed_image(model, preprocess, image_bgr: np.ndarray) -> np.ndarray:
    """Embed a BGR image into the encoder feature space.

    Args:
        model: Image encoder model.
        preprocess: Preprocessing function for images.
        image_bgr: BGR image array.

    Returns:
        Normalized embedding vector as numpy array.
    """
    im = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    x = preprocess(im).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        z = model(x)
    return z.cpu().numpy().astype("float32")


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array of 4 points.

    Returns:
        Ordered points array.
    """
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


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

    expect_ar = 1.48
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
        ar_score = float(np.exp(-abs(ar - expect_ar)))
        area_score = float(area / (w_img * h_img))
        total = 0.45 * ar_score + 0.35 * angle_score + 0.15 * area_score + 0.05 * margin_norm
        if total > best_score:
            best_score = total
            best = rect

    if best is None:
        cv2.putText(overlay, "No cover detected - using full image", (12, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return bgr, overlay, None

    cv2.polylines(overlay, [best.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)

    w = int(max(np.linalg.norm(best[0] - best[1]), np.linalg.norm(best[2] - best[3])))
    h = int(max(np.linalg.norm(best[0] - best[3]), np.linalg.norm(best[1] - best[2])))
    w = max(w, 64)
    h = max(h, 64)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(best, dst)
    warped = cv2.warpPerspective(bgr, m, (w, h))
    return warped, overlay, best


def sift_score(query_bgr: np.ndarray, cand_bgr: np.ndarray) -> tuple[float, int]:
    """Compute SIFT-based geometric matching score between two images.

    Args:
        query_bgr: Query image in BGR format.
        cand_bgr: Candidate image in BGR format.

    Returns:
        Tuple of (score, num_inliers) where score is the ratio of inliers to good matches.
    """
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
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    score = inliers / max(len(good), 1)
    return score, inliers
