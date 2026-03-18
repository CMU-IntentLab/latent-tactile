"""
GelSight/tactile-specific evaluation metrics.

Background image = first frame of ground-truth video (no-contact baseline).
All metrics use background subtraction before processing.
"""

import numpy as np
import cv2

# Tactile metric names
TACTILE_METRIC_NAMES = [
    "contact_iou",
    "height_mae",
    "height_rmse",
    "gradient_mag_mae",
    "gradient_cos_sim",
    "flow_epe_mean",
    "flow_angular_error_deg",
]


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale uint8. Handles RGB/BGR (H,W,3) or already gray."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.max() <= 1.0:
        return (img * 255).astype(np.uint8)
    return img.astype(np.uint8)


def preprocess_with_background(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Subtract background and return grayscale diff. frame, background: (H,W,3) or (H,W)."""
    if frame.shape != background.shape:
        frame = cv2.resize(frame, (background.shape[1], background.shape[0]))
    diff = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY) if diff.ndim == 3 else diff
    return gray.astype(np.uint8)


def contact_area_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    background: np.ndarray,
    threshold: int = 30,
) -> float:
    """
    Contact Area IoU. Background subtract first, then threshold.
    pred, gt: (H,W,3) or (H,W). background: (H,W,3) or (H,W). Higher = better.
    """
    pred_g = preprocess_with_background(pred, background)
    gt_g = preprocess_with_background(gt, background)
    pred_mask = (pred_g > threshold).astype(np.uint8)
    gt_mask = (gt_g > threshold).astype(np.uint8)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(intersection / (union + 1e-8))


def estimate_surface_normals(gel_img: np.ndarray) -> np.ndarray:
    """
    Simplified photometric stereo for GelSight. gel_img: (H,W,3) RGB.
    Returns (H,W,3) normal map. Placeholder calibration matrix.
    """
    img = gel_img.astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]  # RGB
    A = np.array([
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    intensities = np.stack([r, g, b], axis=-1)
    normals = intensities @ np.linalg.pinv(A).T
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    return normals / norm


def normals_to_height(normals: np.ndarray) -> np.ndarray:
    """Integrate surface normals to height map (Frankot-Chellappa)."""
    dzdx = -normals[..., 0] / (normals[..., 2] + 1e-8)
    dzdy = -normals[..., 1] / (normals[..., 2] + 1e-8)
    H, W = dzdx.shape
    fx = np.fft.fftfreq(W)[np.newaxis, :]
    fy = np.fft.fftfreq(H)[:, np.newaxis]
    DZDX = np.fft.fft2(dzdx)
    DZDY = np.fft.fft2(dzdy)
    denom = (2j * np.pi * fx) ** 2 + (2j * np.pi * fy) ** 2
    denom[0, 0] = 1
    Z_freq = (2j * np.pi * fx * DZDX + 2j * np.pi * fy * DZDY) / (denom + 1e-8)
    Z_freq[0, 0] = 0
    return np.real(np.fft.ifft2(Z_freq)).astype(np.float64)


def height_map_error(pred_img: np.ndarray, gt_img: np.ndarray) -> dict[str, float]:
    """Height map MAE and RMSE. Lower = better."""
    pred_h = normals_to_height(estimate_surface_normals(pred_img))
    gt_h = normals_to_height(estimate_surface_normals(gt_img))
    pred_h -= pred_h.mean()
    gt_h -= gt_h.mean()
    diff = pred_h - gt_h
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return {"height_mae": mae, "height_rmse": rmse}


def gradient_field_similarity(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """pred, gt: grayscale float32 in [0,1]. Returns gradient_mag_mae, gradient_cos_sim."""
    def compute_gradients(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return gx, gy

    pred_gx, pred_gy = compute_gradients(pred)
    gt_gx, gt_gy = compute_gradients(gt)
    pred_mag = np.sqrt(pred_gx ** 2 + pred_gy ** 2)
    gt_mag = np.sqrt(gt_gx ** 2 + gt_gy ** 2)
    mag_mae = float(np.mean(np.abs(pred_mag - gt_mag)))

    pred_vec = np.stack([pred_gx, pred_gy], axis=-1)
    gt_vec = np.stack([gt_gx, gt_gy], axis=-1)
    dot = np.sum(pred_vec * gt_vec, axis=-1)
    norm_p = np.linalg.norm(pred_vec, axis=-1) + 1e-8
    norm_g = np.linalg.norm(gt_vec, axis=-1) + 1e-8
    cos_sim = dot / (norm_p * norm_g)
    contact_mask = gt_mag > gt_mag.mean() * 0.1
    mean_cos_sim = float(cos_sim[contact_mask].mean()) if contact_mask.any() else 0.0

    return {"gradient_mag_mae": mag_mae, "gradient_cos_sim": mean_cos_sim}


def gradient_similarity_rgb(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> dict[str, float]:
    """Compute gradient similarity per channel and average."""
    results = []
    for c in range(3):
        p = pred_rgb[..., c].astype(np.float64) / 255.0
        g = gt_rgb[..., c].astype(np.float64) / 255.0
        results.append(gradient_field_similarity(p, g))
    return {k: float(np.mean([r[k] for r in results])) for k in results[0]}


def compute_optical_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """frame1, frame2: grayscale uint8. Returns (H,W,2) flow."""
    if frame1.ndim == 3:
        frame1 = _to_grayscale(frame1)
    if frame2.ndim == 3:
        frame2 = _to_grayscale(frame2)
    flow = cv2.calcOpticalFlowFarneback(
        frame1, frame2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow


def optical_flow_epe(pred_frames: list[np.ndarray], gt_frames: list[np.ndarray], background: np.ndarray) -> dict[str, float]:
    """Mean End-Point Error. Lower = better."""
    pred_g = [preprocess_with_background(f, background) for f in pred_frames]
    gt_g = [preprocess_with_background(f, background) for f in gt_frames]
    epes = []
    for t in range(len(gt_g) - 1):
        flow_pred = compute_optical_flow(pred_g[t], pred_g[t + 1])
        flow_gt = compute_optical_flow(gt_g[t], gt_g[t + 1])
        epe = np.sqrt(np.sum((flow_pred - flow_gt) ** 2, axis=-1))
        epes.append(float(epe.mean()))
    if not epes:
        return {"flow_epe_mean": float("nan"), "flow_epe_std": float("nan")}
    return {"flow_epe_mean": float(np.mean(epes)), "flow_epe_std": float(np.std(epes))}


def optical_flow_angular_error(pred_frames: list[np.ndarray], gt_frames: list[np.ndarray], background: np.ndarray) -> float:
    """Angular error in degrees. Lower = better."""
    pred_g = [preprocess_with_background(f, background) for f in pred_frames]
    gt_g = [preprocess_with_background(f, background) for f in gt_frames]
    errors = []
    for t in range(len(gt_g) - 1):
        fp = compute_optical_flow(pred_g[t], pred_g[t + 1])
        fg = compute_optical_flow(gt_g[t], gt_g[t + 1])
        fp3 = np.dstack([fp, np.ones(fp.shape[:2])])
        fg3 = np.dstack([fg, np.ones(fg.shape[:2])])
        dot = np.sum(fp3 * fg3, axis=-1)
        normp = np.linalg.norm(fp3, axis=-1)
        normg = np.linalg.norm(fg3, axis=-1)
        cos_angle = np.clip(dot / (normp * normg + 1e-8), -1, 1)
        angle_deg = np.degrees(np.arccos(cos_angle))
        errors.append(float(angle_deg.mean()))
    return float(np.mean(errors)) if errors else float("nan")


def compute_tactile_metrics(
    gt_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    contact_threshold: int = 30,
) -> dict[str, float]:
    """
    Compute all GelSight metrics. Background = first GT frame.
    gt_frames, pred_frames: list of (H,W,3) RGB uint8.
    """
    if not gt_frames or not pred_frames:
        return {k: float("nan") for k in TACTILE_METRIC_NAMES}

    n = min(len(gt_frames), len(pred_frames))
    gt_frames = gt_frames[:n]
    pred_frames = pred_frames[:n]
    background = gt_frames[0]

    out = {k: float("nan") for k in TACTILE_METRIC_NAMES}

    # Per-frame: contact IoU, height error, gradient similarity
    contact_ious = []
    height_maes = []
    height_rmses = []
    grad_mag_maes = []
    grad_cos_sims = []

    for pred, gt in zip(pred_frames, gt_frames):
        gt_g = preprocess_with_background(gt, background)
        pred_g = preprocess_with_background(pred, background)
        contact_ious.append(contact_area_iou(pred, gt, background, contact_threshold))

        he = height_map_error(pred, gt)
        height_maes.append(he["height_mae"])
        height_rmses.append(he["height_rmse"])

        gs = gradient_similarity_rgb(pred, gt)
        grad_mag_maes.append(gs["gradient_mag_mae"])
        grad_cos_sims.append(gs["gradient_cos_sim"])

    out["contact_iou"] = float(np.mean(contact_ious))
    out["height_mae"] = float(np.mean(height_maes))
    out["height_rmse"] = float(np.mean(height_rmses))
    out["gradient_mag_mae"] = float(np.mean(grad_mag_maes))
    out["gradient_cos_sim"] = float(np.mean(grad_cos_sims))

    # Temporal: optical flow
    flow_epe = optical_flow_epe(pred_frames, gt_frames, background)
    out["flow_epe_mean"] = flow_epe["flow_epe_mean"]
    out["flow_angular_error_deg"] = optical_flow_angular_error(pred_frames, gt_frames, background)

    return out
