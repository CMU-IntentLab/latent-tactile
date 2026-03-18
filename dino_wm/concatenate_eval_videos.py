#!/usr/bin/env python3
"""
Concatenate eval videos horizontally across *_eval subfolders.

Finds all subfolders ending with "_eval", groups videos by filename (e.g. episode_0_traj_123_result.mp4),
and concatenates videos with the same name horizontally into a single output video.

Video format: each frame has top=GT, middle=prediction, bottom=difference (stacked vertically).
Computes metrics (MSE, MAE, PSNR, SSIM, LPIPS, FID, FVD) per video and per subfolder.

Usage:
  python concatenate_eval_videos.py --root_dir /path/to/runs --output_dir /path/to/concatenated
"""

import argparse
import json
import os
import tempfile

import cv2
import numpy as np

# Optional metrics dependencies
try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    from cleanfid import fid
    HAS_FID = True
except ImportError:
    HAS_FID = False

try:
    from cdfvd import fvd as cdfvd_module
    HAS_FVD_CD = True
except ImportError:
    HAS_FVD_CD = False

# I3D-based FVD, Inception FID, batch LPIPS (SAILOR-FM style)
try:
    from dino_wm.metrics import (
        compute_fvd as compute_fvd_i3d,
        compute_fid as compute_fid_tensor,
        compute_lpips_batch,
        frames_to_fvd_tensor,
        frames_to_image_tensor,
        HAS_INCEPTION,
        HAS_LPIPS as HAS_LPIPS_METRICS,
    )
    HAS_FVD_I3D = True
    HAS_FID_TENSOR = HAS_INCEPTION
except ImportError:
    try:
        from metrics import (
            compute_fvd as compute_fvd_i3d,
            compute_fid as compute_fid_tensor,
            compute_lpips_batch,
            frames_to_fvd_tensor,
            frames_to_image_tensor,
            HAS_INCEPTION,
            HAS_LPIPS as HAS_LPIPS_METRICS,
        )
        HAS_FVD_I3D = True
        HAS_FID_TENSOR = HAS_INCEPTION
    except ImportError:
        HAS_FVD_I3D = False
        HAS_FID_TENSOR = False
        compute_fvd_i3d = None
        compute_fid_tensor = None
        compute_lpips_batch = None
        frames_to_fvd_tensor = None
        frames_to_image_tensor = None
        HAS_LPIPS_METRICS = False

HAS_FVD = HAS_FVD_I3D or HAS_FVD_CD


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
METRIC_NAMES = ["mse", "mae", "psnr", "ssim", "lpips", "fid", "fvd"]


def extract_gt_pred_from_frames(frames: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Extract GT (top row) and pred (middle row) from each frame.
    Frame layout: top=GT, middle=pred, bottom=diff (stacked vertically).
    Returns (gt_frames, pred_frames) as lists of RGB uint8 (H, W, 3).
    """
    gt_list, pred_list = [], []
    for frame in frames:
        h, w = frame.shape[:2]
        if h < 3:
            continue
        row_h = h // 3
        gt = frame[:row_h]
        pred = frame[row_h : 2 * row_h]
        if frame.shape[-1] == 3 and frame.dtype == np.uint8:
            gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            pred_rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        else:
            gt_rgb = gt
            pred_rgb = pred
        
        gt_list.append(gt_rgb)
        pred_list.append(pred_rgb)
    return gt_list, pred_list


def compute_pixel_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute MSE, MAE, PSNR. Inputs uint8 [0,255] or float [0,1]."""
    gt_f = gt.astype(np.float64) / 255.0 if gt.max() > 1 else gt.astype(np.float64)
    pred_f = pred.astype(np.float64) / 255.0 if pred.max() > 1 else pred.astype(np.float64)
    mse = float(np.mean((gt_f - pred_f) ** 2))
    mae = float(np.mean(np.abs(gt_f - pred_f)))
    psnr = 10 * np.log10(1.0 / (mse + 1e-10)) if mse > 0 else float("inf")
    return {"mse": mse, "mae": mae, "psnr": psnr}


def compute_pixel_metrics_batch(gt_stack: np.ndarray, pred_stack: np.ndarray) -> dict[str, float]:
    """Vectorized MSE, MAE, PSNR. gt_stack, pred_stack: (N, H, W, C) uint8 or float."""
    if gt_stack.max() > 1:
        gt_f = gt_stack.astype(np.float64) / 255.0
        pred_f = pred_stack.astype(np.float64) / 255.0
    else:
        gt_f = gt_stack.astype(np.float64)
        pred_f = pred_stack.astype(np.float64)
    diff = gt_f - pred_f
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    psnr = 10 * np.log10(1.0 / (mse + 1e-10)) if mse > 0 else float("inf")
    return {"mse": mse, "mae": mae, "psnr": psnr}


def compute_ssim_metric(gt: np.ndarray, pred: np.ndarray) -> float:
    """SSIM between two images. Expects RGB uint8 or float."""
    if not HAS_SSIM:
        return float("nan")
    gt_f = gt.astype(np.float64) / 255.0 if gt.max() > 1 else gt.astype(np.float64)
    pred_f = pred.astype(np.float64) / 255.0 if pred.max() > 1 else pred.astype(np.float64)
    if gt_f.ndim == 3:
        return float(ssim_fn(gt_f, pred_f, channel_axis=2, data_range=1.0))
    return float(ssim_fn(gt_f, pred_f, data_range=1.0))


def compute_ssim_batch(gt_stack: np.ndarray, pred_stack: np.ndarray) -> float:
    """Mean SSIM over stacked images. gt_stack, pred_stack: (N, H, W, C)."""
    if not HAS_SSIM or len(gt_stack) == 0:
        return float("nan")
    if gt_stack.max() > 1:
        gt_f = gt_stack.astype(np.float64) / 255.0
        pred_f = pred_stack.astype(np.float64) / 255.0
    else:
        gt_f = gt_stack.astype(np.float64)
        pred_f = pred_stack.astype(np.float64)
    vals = [
        ssim_fn(gt_f[i], pred_f[i], channel_axis=2, data_range=1.0)
        for i in range(len(gt_f))
    ]
    return float(np.mean(vals))


def compute_lpips_metric(gt_frames: list[np.ndarray], pred_frames: list[np.ndarray], device: str = "cuda") -> float:
    """LPIPS between two sets of frames. Returns mean LPIPS over frames."""
    if not HAS_LPIPS or not HAS_TORCH:
        return float("nan")
    if not gt_frames or not pred_frames:
        return float("nan")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="vgg").to(dev)
    vals = []
    for gt, pred in zip(gt_frames, pred_frames):
        gt_t = torch.from_numpy(gt).float().to(dev) / 255.0
        pred_t = torch.from_numpy(pred).float().to(dev) / 255.0
        if gt_t.shape[-1] == 3:
            gt_t = gt_t.permute(2, 0, 1).unsqueeze(0)
            pred_t = pred_t.permute(2, 0, 1).unsqueeze(0)
        gt_t = gt_t * 2 - 1
        pred_t = pred_t * 2 - 1
        if gt_t.shape[-2:] != (224, 224):
            gt_t = nn.functional.interpolate(gt_t, size=(224, 224), mode="bilinear", align_corners=False)
            pred_t = nn.functional.interpolate(pred_t, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            vals.append(loss_fn(gt_t, pred_t).item())
    return float(np.mean(vals))


def compute_fid_metric(gt_frames: list[np.ndarray], pred_frames: list[np.ndarray]) -> float:
    """FID between two sets of images. Saves to temp dirs and uses clean-fid."""
    if not HAS_FID or not gt_frames or not pred_frames:
        return float("nan")
    try:
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            for i, img in enumerate(gt_frames):
                if img.shape[-1] == 3:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img
                cv2.imwrite(os.path.join(d1, f"{i:05d}.png"), img_bgr)
            for i, img in enumerate(pred_frames):
                if img.shape[-1] == 3:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img
                cv2.imwrite(os.path.join(d2, f"{i:05d}.png"), img_bgr)
            return float(fid.compute_fid(d1, d2))
    except Exception:
        return float("nan")


FVD_RESOLUTION = 256
FVD_NUM_FRAMES = 16


def _frames_to_fvd_array(
    frames: list[np.ndarray],
    resolution: int = FVD_RESOLUTION,
    num_frames: int = FVD_NUM_FRAMES,
) -> np.ndarray:
    """Convert frames to (1, T, H, W, C) uint8 for cd-fvd video_numpy path (avoids PyAV)."""
    if not frames:
        return np.zeros((1, 0, resolution, resolution, 3), dtype=np.uint8)
    n = min(num_frames, len(frames))
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    selected = [frames[i] for i in indices]
    h, w = selected[0].shape[:2]
    if (h, w) != (resolution, resolution):
        selected = [cv2.resize(f, (resolution, resolution), interpolation=cv2.INTER_LINEAR) for f in selected]
    arr = np.stack([f if f.shape[-1] == 3 else cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in selected], axis=0)
    if arr.max() <= 1:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr[np.newaxis, ...]  # (1, T, H, W, C)


def _frames_to_video_file(
    frames: list[np.ndarray],
    out_path: str,
    resolution: int = FVD_RESOLUTION,
    num_frames: int = FVD_NUM_FRAMES,
    fps: int = 8,
) -> None:
    """Save frames as video file, resized and subsampled for FVD."""
    if not frames:
        return
    n = min(num_frames, len(frames))
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    selected = [frames[i] for i in indices]
    h, w = selected[0].shape[:2]
    if (h, w) != (resolution, resolution):
        selected = [cv2.resize(f, (resolution, resolution), interpolation=cv2.INTER_LINEAR) for f in selected]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (resolution, resolution))
    for f in selected:
        if f.shape[-1] == 3:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        else:
            bgr = f
        writer.write(bgr)
    writer.release()


def _get_safe_device() -> str:
    """Return 'cuda' if available, else 'cpu'. Avoids CUDA init errors."""
    if not HAS_TORCH:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# Cache to avoid repeated CUDA checks
_safe_device_cache = None


def _get_device_for_metrics(device: str | None) -> str:
    """Get device for metrics; use provided or safe default. Falls back to cpu if cuda unavailable."""
    global _safe_device_cache
    if _safe_device_cache is None:
        _safe_device_cache = _get_safe_device()
    if device == "cuda" and _safe_device_cache != "cuda":
        return _safe_device_cache
    return device or _safe_device_cache


def compute_fvd_metric(
    gt_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    resolution: int = FVD_RESOLUTION,
    num_frames: int = FVD_NUM_FRAMES,
    device: str | None = None,
    fvd_ckpt_path: str | None = None,
    verbose: bool = False,
) -> float:
    """
    FVD for a single pair of GT and pred videos.
    Uses I3D (SAILOR-FM style) when available - no checkpoint needed.
    Falls back to cd-fvd (VideoMAE) if fvd_ckpt_path provided and I3D unavailable.
    Single-video FVD is undefined (returns nan) - need 2+ videos per distribution.
    """
    if not gt_frames or not pred_frames:
        return float("nan")
    n = min(len(gt_frames), len(pred_frames), num_frames)
    if n < 2:
        return float("nan")
    # Single-video: FVD undefined (covariance singular)
    return float("nan")


def compute_fvd_for_folder(
    gt_pred_pairs: list[tuple[list[np.ndarray], list[np.ndarray]]],
    resolution: int = FVD_RESOLUTION,
    num_frames: int = FVD_NUM_FRAMES,
    device: str | None = None,
    fvd_ckpt_path: str | None = None,
    verbose: bool = True,
) -> float:
    """
    Compute FVD between GT and pred video sets (SAILOR-FM style).
    Prefers I3D (no checkpoint). Falls back to cd-fvd (VideoMAE) if fvd_ckpt_path provided.
    """
    if len(gt_pred_pairs) < 2:
        return float("nan")
    dev = _get_device_for_metrics(device)

    # I3D path (SAILOR-FM style) - no checkpoint needed, expects 224x224
    if HAS_FVD_I3D and HAS_TORCH:
        i3d_res = 224
        gt_tensors = []
        pred_tensors = []
        for gt_frames, pred_frames in gt_pred_pairs:
            if not gt_frames or not pred_frames:
                continue
            gt_tensors.append(frames_to_fvd_tensor(gt_frames, num_frames, i3d_res))
            pred_tensors.append(frames_to_fvd_tensor(pred_frames, num_frames, i3d_res))
        if len(gt_tensors) < 2 or len(pred_tensors) < 2:
            return float("nan")
        try:
            real = torch.cat(gt_tensors, dim=0)
            fake = torch.cat(pred_tensors, dim=0)
            return compute_fvd_i3d(real, fake, batch_size=8, device=dev, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"\n    FVD (I3D) failed: {e}")
            # Fall through to cd-fvd if available

    # cd-fvd (VideoMAE) fallback
    if HAS_FVD_CD and fvd_ckpt_path and os.path.isfile(fvd_ckpt_path):
        try:
            gt_arrays = []
            pred_arrays = []
            for gt_frames, pred_frames in gt_pred_pairs:
                if not gt_frames or not pred_frames:
                    continue
                gt_arrays.append(_frames_to_fvd_array(gt_frames, resolution, num_frames))
                pred_arrays.append(_frames_to_fvd_array(pred_frames, resolution, num_frames))
            if len(gt_arrays) < 2 or len(pred_arrays) < 2:
                return float("nan")
            gt_stack = np.concatenate(gt_arrays, axis=0)
            pred_stack = np.concatenate(pred_arrays, axis=0)
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f_gt, tempfile.NamedTemporaryFile(
                suffix=".npy", delete=False
            ) as f_pred:
                np.save(f_gt.name, gt_stack)
                np.save(f_pred.name, pred_stack)
                gt_path, pred_path = f_gt.name, f_pred.name
            try:
                evaluator = cdfvd_module.cdfvd("videomae", ckpt_path=fvd_ckpt_path, device=dev)
                gt_loader = evaluator.load_videos(gt_path)
                pred_loader = evaluator.load_videos(pred_path)
                evaluator.compute_real_stats(gt_loader)
                evaluator.compute_fake_stats(pred_loader)
                return float(evaluator.compute_fvd_from_stats())
            finally:
                for p in (gt_path, pred_path):
                    if os.path.isfile(p):
                        os.unlink(p)
        except Exception as e:
            if verbose:
                print(f"\n    FVD (cd-fvd) failed: {e}")
            return float("nan")

    return float("nan")


def compute_video_metrics(
    gt_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    compute_fid_lpips: bool = True,
    device: str = "cuda",
    fvd_ckpt_path: str | None = None,
    metric_max_frames: int | None = 64,
) -> dict[str, float]:
    """
    Compute all metrics for a single video (optimized with batching).
    metric_max_frames: subsample to this many frames for LPIPS/FID (faster).
    """
    if not gt_frames or not pred_frames:
        return {k: float("nan") for k in METRIC_NAMES}

    n = min(len(gt_frames), len(pred_frames))
    gt_frames = gt_frames[:n]
    pred_frames = pred_frames[:n]
    dev = _get_device_for_metrics(device)

    # Vectorized pixel metrics (fast)
    gt_stack = np.stack(gt_frames, axis=0)
    pred_stack = np.stack(pred_frames, axis=0)
    pm = compute_pixel_metrics_batch(gt_stack, pred_stack)
    ssim = compute_ssim_batch(gt_stack, pred_stack)
    out = {**pm, "ssim": ssim}

    if compute_fid_lpips:
        # Prefer tensor-based LPIPS/FID (no disk I/O, batched)
        if HAS_TORCH and HAS_LPIPS_METRICS and frames_to_image_tensor is not None and compute_lpips_batch is not None:
            n_frames = min(len(gt_frames), metric_max_frames) if metric_max_frames else len(gt_frames)
            gt_t = frames_to_image_tensor(gt_frames, resolution=224, max_frames=n_frames)
            pred_t = frames_to_image_tensor(pred_frames, resolution=224, max_frames=n_frames)
            if len(gt_t) > 0:
                out["lpips"] = compute_lpips_batch(gt_t, pred_t, batch_size=64, device=dev)
            else:
                out["lpips"] = float("nan")
        else:
            out["lpips"] = compute_lpips_metric(gt_frames, pred_frames, device)

        if HAS_FID_TENSOR and frames_to_image_tensor is not None and compute_fid_tensor is not None:
            n_frames = min(len(gt_frames), metric_max_frames) if metric_max_frames else len(gt_frames)
            gt_t = frames_to_image_tensor(gt_frames, resolution=299, max_frames=n_frames)
            pred_t = frames_to_image_tensor(pred_frames, resolution=299, max_frames=n_frames)
            if len(gt_t) >= 2:
                out["fid"] = compute_fid_tensor(gt_t, pred_t, batch_size=64, device=dev)
            else:
                out["fid"] = float("nan")
        else:
            out["fid"] = compute_fid_metric(gt_frames, pred_frames)
        out["fvd"] = compute_fvd_metric(gt_frames, pred_frames, device=dev, fvd_ckpt_path=fvd_ckpt_path)
    else:
        out["lpips"] = out["fid"] = out["fvd"] = float("nan")

    return out


def compute_all_metrics(
    eval_folders: list[str],
    compute_fid_lpips: bool = True,
    device: str = "cuda",
    fvd_ckpt_path: str | None = None,
    metric_max_frames: int | None = 64,
) -> dict:
    """
    Compute metrics for all videos in all _eval folders.
    Returns: {
        "per_video": {folder_name: {video_filename: {mse, mae, ...}}},
        "per_folder_mean": {folder_name: {mse, mae, ...}},
    }
    FVD is computed per-folder (requires multiple videos); per-video FVD is nan.
    """
    per_video = {}
    folder_gt_pred_pairs = {}
    for folder in eval_folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        print(f"  Metrics: {folder_name}...", end=" ", flush=True)
        per_video[folder_name] = {}
        folder_gt_pred_pairs[folder_name] = []
        video_files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]
        for fname in sorted(video_files):
            path = os.path.join(folder, fname)
            if not os.path.isfile(path):
                continue
            try:
                frames = get_video_frames(path)
                gt_frames, pred_frames = extract_gt_pred_from_frames(frames)
                if not gt_frames or not pred_frames:
                    per_video[folder_name][fname] = {k: float("nan") for k in METRIC_NAMES}
                else:
                    m = compute_video_metrics(
                        gt_frames, pred_frames,
                        compute_fid_lpips=compute_fid_lpips,
                        device=device,
                        fvd_ckpt_path=fvd_ckpt_path,
                        metric_max_frames=metric_max_frames,
                    )
                    per_video[folder_name][fname] = m
                    folder_gt_pred_pairs[folder_name].append((gt_frames, pred_frames))
            except Exception as e:
                per_video[folder_name][fname] = {k: float("nan") for k in METRIC_NAMES}
                print(f"\n    Warning: failed {fname}: {e}")
        print(f"{len(per_video[folder_name])} videos")

    # Aggregate mean per folder (FVD computed separately per-folder)
    per_folder_mean = {}
    for folder_name, videos in per_video.items():
        if not videos:
            per_folder_mean[folder_name] = {k: float("nan") for k in METRIC_NAMES}
            continue
        means = {}
        for k in METRIC_NAMES:
            if k == "fvd":
                continue
            vals = [
                v[k] for v in videos.values()
                if isinstance(v[k], (int, float)) and np.isfinite(v[k])
            ]
            means[k] = float(np.mean(vals)) if vals else float("nan")
        # FVD: compute per-folder (I3D or cd-fvd; requires 2+ videos)
        if compute_fid_lpips and HAS_FVD:
            pairs = folder_gt_pred_pairs.get(folder_name, [])
            if len(pairs) >= 2:
                print(f"    Computing FVD for {folder_name} ({len(pairs)} videos)...", end=" ", flush=True)
                means["fvd"] = compute_fvd_for_folder(
                    pairs, device=device, fvd_ckpt_path=fvd_ckpt_path, verbose=False
                )
                print("done")
            else:
                means["fvd"] = float("nan")
        else:
            means["fvd"] = float("nan")
        per_folder_mean[folder_name] = means

    return {"per_video": per_video, "per_folder_mean": per_folder_mean}


def print_metrics_summary(metrics_data: dict) -> None:
    """Print metrics summary to console."""
    per_folder = metrics_data.get("per_folder_mean", {})
    print("\n" + "=" * 80)
    print("METRICS SUMMARY (mean per subfolder)")
    print("=" * 80)
    for folder_name in sorted(per_folder.keys()):
        m = per_folder[folder_name]
        print(f"\n{folder_name}:")
        for k in METRIC_NAMES:
            v = m.get(k, float("nan"))
            if isinstance(v, float) and np.isnan(v):
                print(f"  {k}: N/A")
            else:
                print(f"  {k}: {v:.4f}")
    print()


def find_eval_folders(root_dir: str) -> list[str]:
    """Return paths to subdirectories ending with '_eval'."""
    eval_folders = []
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path) and name.endswith("_eval"):
            eval_folders.append(path)
    return sorted(eval_folders)


def collect_videos_by_name(eval_folders: list[str]) -> dict[str, list[tuple[str, str]]]:
    """
    Group video paths by filename across all _eval folders.
    Returns: {filename: [(path, folder_name), ...]} where folder_name is the _eval subfolder basename.
    """
    by_name = {}
    for folder in eval_folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        for fname in os.listdir(folder):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VIDEO_EXTENSIONS:
                continue
            path = os.path.join(folder, fname)
            if not os.path.isfile(path):
                continue
            if fname not in by_name:
                by_name[fname] = []
            by_name[fname].append((path, folder_name))
    return by_name


def get_video_frames(path: str) -> list[np.ndarray]:
    """Load all frames from a video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_video_props(path: str) -> tuple[int, int, float]:
    """Return (width, height, fps) for a video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    cap.release()
    return w, h, fps


def make_header_with_label(width: int, label: str, header_height: int = 48) -> np.ndarray:
    """Create a white header bar with the folder name centered."""
    header = np.ones((header_height, width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(1.0, (width / 200) * 0.5)
    thickness = max(1, int(2 * font_scale))
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    x = (width - tw) // 2
    y = (header_height + th) // 2
    cv2.putText(header, label, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return header


def concatenate_videos_horizontally(
    path_label_pairs: list[tuple[str, str]],
    output_path: str,
    fps: float | None = None,
    min_length: bool = True,
    header_height: int = 48,
) -> None:
    """
    Concatenate videos side by side with a white header above each showing the folder name.

    Args:
        path_label_pairs: List of (video_path, folder_name) tuples (order preserved left-to-right).
        output_path: Output video path.
        fps: FPS for output (default: 1/3 of first video's FPS).
        min_length: If True, use min frame count across videos; else pad shorter videos to max.
        header_height: Height of the white label bar in pixels.
    """
    if not path_label_pairs:
        return

    paths = [p for p, _ in path_label_pairs]
    labels = [lbl for _, lbl in path_label_pairs]

    frames_list = []
    for p in paths:
        frames = get_video_frames(p)
        frames_list.append(frames)

    n_frames = min(len(f) for f in frames_list) if min_length else max(len(f) for f in frames_list)
    h_max = max(f[0].shape[0] for f in frames_list)
    w_first, h_first, fps_first = get_video_props(paths[0])
    out_fps = fps if fps is not None else fps_first / 3

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w = 0
    out_h = header_height + h_max

    for frame_idx in range(n_frames):
        row_parts = []
        for vid_idx, (frames, label) in enumerate(zip(frames_list, labels)):
            if frame_idx < len(frames):
                frame = frames[frame_idx]
            else:
                frame = frames[-1].copy() if frames else np.zeros((h_max, w_first, 3), dtype=np.uint8)
            h, w = frame.shape[:2]
            if h != h_max:
                frame = cv2.resize(frame, (w, h_max), interpolation=cv2.INTER_LINEAR)
            header = make_header_with_label(w, label, header_height)
            block = np.concatenate([header, frame], axis=0)
            row_parts.append(block)
        out_frame = np.concatenate(row_parts, axis=1)
        if frame_idx == 0:
            out_w = out_frame.shape[1]
            writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))
        writer.write(out_frame)

    writer.release()
    print(f"  Saved: {output_path} ({n_frames} frames, {out_w}x{out_h})")


def _run_fvd_only(eval_folders: list[str], fvd_ckpt_path: str | None = None) -> None:
    """Compute FVD only per folder (for testing FVD separately)."""
    if not HAS_FVD:
        raise SystemExit("FVD not available. Install: pip install torch (I3D) or cdfvd (VideoMAE)")
    print("\nFVD-only mode (I3D by default, cd-fvd if --fvd_ckpt_path provided)")
    print("=" * 60)
    for folder in eval_folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        pairs = []
        video_files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]
        for fname in sorted(video_files):
            path = os.path.join(folder, fname)
            if not os.path.isfile(path):
                continue
            try:
                frames = get_video_frames(path)
                gt_frames, pred_frames = extract_gt_pred_from_frames(frames)
                if gt_frames and pred_frames:
                    pairs.append((gt_frames, pred_frames))
            except Exception as e:
                print(f"  Warning: failed {fname}: {e}")
        if len(pairs) < 2:
            print(f"{folder_name}: N/A (need 2+ videos, got {len(pairs)})")
            continue
        print(f"{folder_name} ({len(pairs)} videos)...", end=" ", flush=True)
        fvd = compute_fvd_for_folder(pairs, fvd_ckpt_path=fvd_ckpt_path, verbose=True)
        if np.isnan(fvd):
            print("FAILED (nan)")
        else:
            print(f"FVD = {fvd:.4f}")
    print("=" * 60)
    print("Done.")


def main():
    p = argparse.ArgumentParser(
        description="Concatenate eval videos horizontally across *_eval subfolders"
    )
    p.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing *_eval subfolders",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for concatenated videos (default: root_dir/concatenated)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video FPS (default: 1/3 of first video's FPS)",
    )
    p.add_argument(
        "--min_length",
        action="store_true",
        default=True,
        help="Use min frame count across videos (default)",
    )
    p.add_argument(
        "--max_length",
        action="store_true",
        help="Pad shorter videos to max length instead of trimming",
    )
    p.add_argument(
        "--header_height",
        type=int,
        default=48,
        help="Height of the white label bar above each video in pixels (default 48)",
    )
    p.add_argument(
        "--metrics",
        action="store_true",
        help="Compute metrics (MSE, MAE, PSNR, SSIM, LPIPS, FID, FVD) per video and per subfolder",
    )
    p.add_argument(
        "--metrics_output",
        type=str,
        default=None,
        help="JSON file to save metrics (default: output_dir/metrics.json when --metrics)",
    )
    p.add_argument(
        "--no_fid_lpips",
        action="store_true",
        help="Skip LPIPS and FID (faster; LPIPS/FID require torch/lpips/clean-fid)",
    )
    p.add_argument(
        "--fvd_ckpt_path",
        type=str,
        default=None,
        help="Path to VideoMAE checkpoint for FVD (optional; I3D used by default). Download from: "
        "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/mae-g/vit_g_hybrid_pt_1200e_ssv2_ft.pth",
    )
    p.add_argument(
        "--fvd_only",
        action="store_true",
        help="Compute FVD only (skip concatenation and other metrics). Use to test FVD separately.",
    )
    p.add_argument(
        "--metric_max_frames",
        type=int,
        default=64,
        help="Max frames for LPIPS/FID (subsample for speed). 0 = use all. Default 64.",
    )
    args = p.parse_args()

    root = os.path.abspath(args.root_dir)
    if not os.path.isdir(root):
        raise SystemExit(f"Root directory not found: {root}")

    eval_folders = find_eval_folders(root)
    if not eval_folders:
        raise SystemExit(f"No *_eval subfolders found in {root}")

    print(f"Found {len(eval_folders)} _eval folders:")
    for f in eval_folders:
        print(f"  {f}")

    # FVD-only mode: compute FVD per folder and exit
    if args.fvd_only:
        _run_fvd_only(eval_folders, fvd_ckpt_path=args.fvd_ckpt_path)
        return

    by_name = collect_videos_by_name(eval_folders)
    to_concat = {k: sorted(v, key=lambda x: x[0]) for k, v in by_name.items() if v}

    if not to_concat:
        raise SystemExit("No videos found in _eval folders")

    out_dir = args.output_dir or os.path.join(root, "concatenated")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput directory: {out_dir}\n")

    min_len = not args.max_length
    for fname, path_label_pairs in sorted(to_concat.items()):
        base, ext = os.path.splitext(fname)
        out_path = os.path.join(out_dir, f"{base}_concat{ext}")
        print(f"Concatenating {len(path_label_pairs)} videos -> {fname}")
        concatenate_videos_horizontally(
            path_label_pairs,
            out_path,
            fps=args.fps,
            min_length=min_len,
            header_height=args.header_height,
        )

    # Metrics computation
    if args.metrics:
        missing = []
        if not HAS_SSIM:
            missing.append("scikit-image")
        if not HAS_LPIPS:
            missing.append("lpips")
        if not HAS_FID:
            missing.append("clean-fid")
        if not HAS_FVD:
            missing.append("cd-fvd")
        if missing:
            print(f"\nNote: For full metrics: pip install {' '.join(missing)}")
        if not HAS_FVD:
            print("Note: FVD requires torch (I3D) or cdfvd+--fvd_ckpt_path (VideoMAE)")
        print("Computing metrics (MSE, MAE, PSNR, SSIM, LPIPS, FID, FVD)...")
        metrics_data = compute_all_metrics(
            eval_folders,
            compute_fid_lpips=not args.no_fid_lpips,
            fvd_ckpt_path=args.fvd_ckpt_path,
            metric_max_frames=args.metric_max_frames or None,
        )
        metrics_path = args.metrics_output or os.path.join(out_dir, "metrics.json")

        def _nan_inf_to_none(obj):
            if isinstance(obj, dict):
                return {k: _nan_inf_to_none(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_nan_inf_to_none(v) for v in obj]
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            return obj

        with open(metrics_path, "w") as f:
            json.dump(_nan_inf_to_none(metrics_data), f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")
        print_metrics_summary(metrics_data)

    print(f"\nDone. {len(to_concat)} concatenated videos saved to {out_dir}")


if __name__ == "__main__":
    main()
