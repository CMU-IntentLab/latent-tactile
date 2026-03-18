"""
Compute tactile-text cosine similarity for reward computation.

Two modes:
  1. video: Encode tactile videos through AnyTouch, compute similarity with text.
     Videos are tactile-only (no cropping needed).
  2. embeddings: Use pre-computed patch embeddings directly. Mean over patches,
     no encoder needed. Compatible with eval_dino_wm_tactile.py --save_embeds output.

Based on AnyTouch2/scripts/infer_touch_text_similarity.py.

Usage:
  # From tactile videos (encode with AnyTouch)
  python compute_reward_tactile.py --mode video --video_path path/to/video.mp4 \\
    --text "wiping the board" --text "no contact" --checkpoint /path/to/AnyTouch2/checkpoints/checkpoint-4frames.pth

  # From saved embeddings (mean over patches, no encoder)
  python compute_reward_tactile.py --mode embeddings --embeddings_path eval_embeds/*_pred_embeds.npz \\
    --text "wiping the board" --checkpoint /path/to/AnyTouch2/checkpoints/checkpoint-4frames.pth
"""

import argparse
import copy
import glob
import os
import re

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
OFFSET = 130.0 / 255.0

SENSOR_NAME_TO_ID = {
    "gelsight": 0,
    "digit": 1,
    "gelslim": 2,
    "gelsight_mini": 3,
    "duragel": 4,
    "dm": 5,
    "universal": -1,
}


# ── Video / frame loading (no crop – tactile-only videos) ─────────────────────

def load_frames_from_video(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_frames_from_directory(dir_path: str) -> list[np.ndarray]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(dir_path, ext)))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No image files in {dir_path}")
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


def resolve_video_sources(path: str):
    """Resolve to list of (path, name) for videos or frame dirs."""
    if os.path.isdir(path):
        # Directory: could be frames of one sequence, or subdirs with videos
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if subdirs:
            # Assume subdirs are frame dirs or contain videos
            return [(os.path.join(path, d), d) for d in sorted(subdirs)]
        # Single dir of frames
        return [(path, os.path.basename(path.rstrip("/")))]
    # Glob or single file
    matches = sorted(glob.glob(path))
    if not matches:
        raise FileNotFoundError(f"No files matched: {path}")
    return [(m, os.path.splitext(os.path.basename(m))[0]) for m in matches]


# ── Preprocessing (from AnyTouch, no crop) ──────────────────────────────────

def preprocess_frames(
    frames: list[np.ndarray],
    bg_image: np.ndarray | None,
    num_frames: int,
    frame_stride: int,
) -> list[torch.Tensor]:
    to_tensor = transforms.ToTensor()
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224), antialias=True),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    if bg_image is None:
        bg_image = frames[0]

    bg_tensor = to_tensor(Image.fromarray(bg_image).convert("RGB"))
    subsampled = frames[::frame_stride]

    windows = []
    for start in range(0, len(subsampled) - num_frames + 1):
        window_frames = subsampled[start : start + num_frames]
        tensors = [to_tensor(Image.fromarray(f).convert("RGB")) for f in window_frames]
        stacked = torch.stack(tensors, dim=0)
        imgs = stacked - bg_tensor.unsqueeze(0) + OFFSET
        imgs = torch.clamp(imgs, 0.0, 1.0)
        imgs = transform(imgs)
        windows.append(imgs)

    return windows


# ── Model loading (from AnyTouch2) ───────────────────────────────────────────

def load_tactile_model_and_clip(checkpoint: str, sensor: str, num_frames: int, stride: int, device):
    from transformers import AutoConfig, CLIPConfig, CLIPModel, CLIPTokenizer
    from model.tactile_mae import TactileVideoMAE

    ckpt = torch.load(checkpoint, map_location="cpu")
    config_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
    clip_dir = os.path.join(config_dir, "CLIP-B-16")
    if not os.path.isdir(clip_dir):
        # Fallback: use CLIP-B-16 from the anytouch package
        import model.tactile_mae as tactile_mod
        pkg_root = os.path.dirname(os.path.dirname(tactile_mod.__file__))
        clip_dir = os.path.join(pkg_root, "CLIP-B-16")

    # Tactile model
    config = AutoConfig.from_pretrained(os.path.join(clip_dir, "config.json"))
    sensor_id = SENSOR_NAME_TO_ID.get(sensor, SENSOR_NAME_TO_ID["digit"])
    fake_args = argparse.Namespace(
        model="anytouch",
        model_size="base",
        dataset="material",
        pooling="global",
        mask_ratio=0,
        stride=stride,
        num_frames=num_frames,
        load_from_clip=False,
    )
    tactile_model = TactileVideoMAE(fake_args, config, num_frames, 1)
    new_ckpt = {}
    for key, val in ckpt.items():
        if "touch_mae_model" in key and "decoder" not in key and "mask_token" not in key:
            new_ckpt[key.replace("touch_mae_model.", "")] = copy.deepcopy(val)
    if not new_ckpt:
        for key, val in ckpt.get("model", ckpt).items():
            if key.startswith("tactile_model."):
                new_ckpt[key.replace("tactile_model.", "")] = copy.deepcopy(val)
    for k, v in tactile_model.named_parameters():
        if k not in new_ckpt:
            new_ckpt[k] = v
    tactile_model.load_state_dict(new_ckpt, strict=True)
    tactile_model = tactile_model.to(device).eval()

    # CLIP text encoder
    config = CLIPConfig.from_pretrained(os.path.join(clip_dir, "config.json"))
    tokenizer = CLIPTokenizer.from_pretrained(clip_dir)
    clip_model = CLIPModel(config)
    TEXT_PREFIXES = ["text_model.", "text_projection."]
    WRAPPER_PREFIXES = ["", "clip_model.", "hf_clip_model.", "clip."]
    text_state = {}
    for key, val in ckpt.items():
        for wp in WRAPPER_PREFIXES:
            for tp in TEXT_PREFIXES:
                full = wp + tp
                if key.startswith(full):
                    clean_key = key[len(wp):]
                    text_state[clean_key] = val
    if text_state:
        clip_model.load_state_dict(text_state, strict=False)
    clip_model = clip_model.to(device).eval()

    return tactile_model, clip_model, tokenizer, sensor_id


@torch.no_grad()
def extract_tactile_embeddings(
    model, windows: list[torch.Tensor], sensor_id: int, device, aggregate: str = "mean_patch", batch_size: int = 32
) -> np.ndarray:
    """Returns (N, 512) L2-normalized tactile embeddings."""
    all_embeds = []
    for i in range(0, len(windows), batch_size):
        batch = torch.stack(windows[i : i + batch_size]).to(device)
        sensors = torch.full((batch.shape[0],), sensor_id, dtype=torch.long, device=device)
        outputs = model(batch, sensor_type=sensors, probe=False, get_cls=False)

        if aggregate == "cls":
            emb = outputs[:, 0, :]
        elif aggregate == "mean_patch":
            emb = outputs[:, 6:, :].mean(dim=1)
        elif aggregate == "max_patch":
            emb = outputs[:, 6:, :].max(dim=1).values
        elif aggregate == "min_patch":
            emb = outputs[:, 6:, :].min(dim=1).values
        else:
            raise ValueError(f"Invalid aggregate: {aggregate}")

        emb = F.normalize(emb, dim=-1)
        all_embeds.append(emb.cpu().float().numpy())
    return np.concatenate(all_embeds, axis=0)


@torch.no_grad()
def extract_text_embeddings(clip_model, tokenizer, texts: list[str], device) -> np.ndarray:
    """Returns (T, 512) L2-normalized text embeddings."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    text_emb = clip_model.get_text_features(**inputs)
    text_emb = F.normalize(text_emb, dim=-1)
    return text_emb.cpu().float().numpy()


# ── Embeddings mode: mean over patches ──────────────────────────────────────

def load_embeddings_from_npz(
    path: str,
    camera_key: str = "camera_2",
    skip_tokens: int = 0,
) -> np.ndarray:
    """
    Load patch embeddings from .npz (e.g. from eval_dino_wm_tactile --save_embeds).
    Pred/GT embeds are patch embeddings (T, N_patches, D). We mean over the patch
    dimension and L2-normalize for CLIP text similarity (512-dim space).
    Returns (T, D).
    """
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    if camera_key in keys:
        emb = data[camera_key]
    elif len(keys) == 1:
        emb = data[keys[0]]
    else:
        raise ValueError(f"Multiple keys {keys}; specify --camera_key (e.g. camera_2)")

    emb = emb.astype(np.float32)

    if emb.ndim == 2:
        # Already (T, D) – pre-aggregated
        pass
    elif emb.ndim == 3:
        # (T, N_patches, D) – patch embeddings: mean over patch dimension
        if skip_tokens > 0:
            emb = emb[:, skip_tokens:, :]  # e.g. skip cls + sensor tokens
        emb = emb.mean(axis=1)  # (T, D)
    else:
        raise ValueError(f"Expected 2D or 3D embeddings, got shape {emb.shape}")

    # L2 normalize (required for cosine similarity with CLIP text)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    emb = emb / norms
    return emb


def resolve_embedding_sources(path: str):
    """Resolve to list of .npz paths."""
    if os.path.isfile(path) and path.endswith(".npz"):
        return [(path, os.path.splitext(os.path.basename(path))[0])]
    matches = sorted(glob.glob(path))
    npz_matches = [m for m in matches if m.endswith(".npz")]
    if not npz_matches:
        raise FileNotFoundError(f"No .npz files matched: {path}")
    return [(m, os.path.splitext(os.path.basename(m))[0]) for m in npz_matches]


def infer_paired_gt_path(pred_path: str) -> str | None:
    """Given path to *_pred_embeds.npz, return path to *_gt_embeds.npz if it exists."""
    if "_pred_embeds.npz" not in pred_path:
        return None
    gt_path = pred_path.replace("_pred_embeds.npz", "_gt_embeds.npz")
    return gt_path if os.path.isfile(gt_path) else None


def infer_eval_video_path(npz_path: str, video_dir: str) -> str | None:
    """
    Given path to *_pred_embeds.npz or *_gt_embeds.npz, return path to the
    corresponding video saved by eval_dino_wm_tactile: episode_X_traj_Y_result.mp4
    """
    base = os.path.basename(npz_path)
    # episode_0_traj_123_pred_embeds.npz -> episode_0_traj_123_result.mp4
    for suffix in ("_pred_embeds.npz", "_gt_embeds.npz"):
        if base.endswith(suffix):
            name_short = base[: -len(suffix)]
            video_path = os.path.join(video_dir, f"{name_short}_result.mp4")
            return video_path if os.path.isfile(video_path) else None
    return None


def parse_trajectory_id_from_filename(npz_path: str) -> str | None:
    """
    Extract trajectory_id from npz filename.
    e.g. episode_0_traj_trajectory_0_pred_embeds.npz -> trajectory_0
         episode_0_traj_123_pred_embeds.npz -> 123 (or trajectory_123 if that's the key)
    """
    base = os.path.basename(npz_path)
    m = re.match(r"episode_\d+_traj_(.+?)_(?:pred|gt)_embeds\.npz", base)
    if m:
        return m.group(1)
    return None


def load_tactile_frames_from_hdf5(
    hdf5_path: str,
    traj_id: str,
    num_frames: int,
    start_idx: int = 0,
    resize_to: tuple[int, int] = (320, 240),
) -> list[np.ndarray]:
    """
    Load camera_2 (tactile) images from consolidated HDF5 for MarkerTracker.
    Returns RGB frames resized to (320, 240) for gsmini compatibility.
    """
    with h5py.File(hdf5_path, "r") as hf:
        if traj_id not in hf:
            raise KeyError(f"trajectory {traj_id} not found in {hdf5_path}")
        traj = hf[traj_id]
        tactile = np.array(traj["camera_2"][start_idx : start_idx + num_frames])

    frames = []
    for i in range(len(tactile)):
        img = np.asarray(tactile[i])
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.0:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
        frames.append(img)
    return frames


def load_rgb_frames_from_hdf5(
    hdf5_path: str,
    traj_id: str,
    num_frames: int,
    start_idx: int = 0,
) -> list[np.ndarray]:
    """
    Load camera_0 and camera_1 RGB images from consolidated HDF5.
    Returns list of frames: [camera_0 | camera_1 | black] horizontally concatenated.
    """
    with h5py.File(hdf5_path, "r") as hf:
        if traj_id not in hf:
            raise KeyError(f"trajectory {traj_id} not found in {hdf5_path}")
        traj = hf[traj_id]
        cam0 = np.array(traj["camera_0"][start_idx : start_idx + num_frames])
        cam1 = np.array(traj["camera_1"][start_idx : start_idx + num_frames])

    frames = []
    for i in range(len(cam0)):
        c0 = np.asarray(cam0[i])
        c1 = np.asarray(cam1[i])
        if c0.ndim == 2:
            c0 = np.stack([c0] * 3, axis=-1)
        if c1.ndim == 2:
            c1 = np.stack([c1] * 3, axis=-1)
        if c0.ndim == 3 and c0.shape[0] == 3:
            c0 = np.transpose(c0, (1, 2, 0))
        if c1.ndim == 3 and c1.shape[0] == 3:
            c1 = np.transpose(c1, (1, 2, 0))
        if c0.max() <= 1.0:
            c0 = (np.clip(c0, 0, 1) * 255).astype(np.uint8)
        if c1.max() <= 1.0:
            c1 = (np.clip(c1, 0, 1) * 255).astype(np.uint8)
        # Resize to same height if needed
        h, w = c0.shape[:2]
        if c0.shape != c1.shape:
            c1 = cv2.resize(c1, (w, h), interpolation=cv2.INTER_AREA)
        black = np.zeros((h, w, 3), dtype=np.uint8)
        row = np.concatenate([c0, c1, black], axis=1)
        frames.append(np.ascontiguousarray(row))
    return frames


# ── Plotting and video (like plot_combined_similarity.py) ───────────────────

def _fig_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    return np.asarray(buf)[:, :, :3].copy()


def plot_similarity(
    sim_matrix: np.ndarray,
    texts: list[str],
    output_path: str,
    title: str = "Tactile-Text Cosine Similarity",
):
    """Plot similarity curves. sim_matrix: (N, N_texts)."""
    fig, ax = plt.subplots(figsize=(max(8, len(texts) * 2), 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(texts)))
    for j, (text, color) in enumerate(zip(texts, colors)):
        short_label = text[:40] + ("..." if len(text) > 40 else "")
        ax.plot(sim_matrix[:, j], label=short_label, color=color, linewidth=1.5)
    ax.set_xlabel("Frame / window index")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=min(len(texts), 3), framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {output_path}")


def plot_similarity_gt_pred(
    sim_gt: np.ndarray,
    sim_pred: np.ndarray,
    texts: list[str],
    output_path: str,
    title: str = "Tactile-Text Similarity (GT vs Pred)",
):
    """Plot both GT and pred similarity curves. sim_gt, sim_pred: (N, N_texts)."""
    n_steps = max(sim_gt.shape[0], sim_pred.shape[0])
    fig, ax = plt.subplots(figsize=(max(8, len(texts) * 2), 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(texts)))

    for j, (text, color) in enumerate(zip(texts, colors)):
        short_label = text[:35] + ("..." if len(text) > 35 else "")
        if sim_gt.shape[0] > 0:
            ax.plot(
                np.arange(sim_gt.shape[0]),
                sim_gt[:, j],
                label=f"GT: {short_label}",
                color=color,
                linewidth=1.5,
                linestyle="-",
            )
        if sim_pred.shape[0] > 0:
            ax.plot(
                np.arange(sim_pred.shape[0]),
                sim_pred[:, j],
                label=f"Pred: {short_label}",
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.8,
            )
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(n_steps - 1, 1))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {output_path}")


def load_video_frames(video_path: str) -> list[np.ndarray]:
    """Load frames from video file or directory of images."""
    if os.path.isdir(video_path):
        frames = load_frames_from_directory(video_path)
    else:
        frames = load_frames_from_video(video_path)
    return frames


LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


def _draw_motion_arrows(frame_bgr, init_markers, motion, arrow_scale=1.0, magnitude_coloring=True):
    """Draw displacement arrows on frame (BGR)."""
    img = frame_bgr.copy()
    magnitudes = np.linalg.norm(motion, axis=1)
    max_mag = magnitudes.max() if magnitudes.max() > 0 else 1.0
    for i in range(len(init_markers)):
        dx, dy = motion[i]
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            continue
        sx, sy = int(init_markers[i, 0]), int(init_markers[i, 1])
        ex = int(np.clip(sx + dx * arrow_scale, 0, img.shape[1] - 1))
        ey = int(np.clip(sy + dy * arrow_scale, 0, img.shape[0] - 1))
        if magnitude_coloring:
            ratio = magnitudes[i] / max_mag
            c = (int(255 * (1 - ratio)), 0, int(255 * ratio))
        else:
            c = (0, 255, 255)
        cv2.arrowedLine(img, (sx, sy), (ex, ey), c, 2, tipLength=0.3)
    return img


def _draw_marker_trace(frame_bgr, trace_points: list[np.ndarray], color=(0, 255, 0), thickness=1):
    """Draw marker trajectory traces (polylines) on frame (BGR). trace_points: list of (N, 2) arrays, one per marker."""
    img = frame_bgr.copy()
    for pts in trace_points:
        if len(pts) < 2:
            continue
        pts_int = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts_int], isClosed=False, color=color, thickness=thickness)
    return img


def compute_tactile_motion_frames(
    tactile_frames_rgb: list[np.ndarray],
    arrow_scale: float = 3.0,
    magnitude_coloring: bool = True,
    draw_trace: bool = True,
) -> list[np.ndarray] | None:
    """Run marker tracking on tactile frames. Returns arrow+trace frames or None if gsmini unavailable."""
    try:
        from gsmini.markertracker import MarkerTracker
    except ImportError:
        print("  Warning: gsmini not available — skipping motion arrows")
        return None
    if not tactile_frames_rgb:
        return None
    first_bgr = cv2.cvtColor(tactile_frames_rgb[0], cv2.COLOR_RGB2BGR)
    try:
        mtracker = MarkerTracker(np.float32(first_bgr) / 255.0)
    except (IndexError, ValueError) as e:
        print(f"  Warning: MarkerTracker failed: {e}")
        return None
    Ox = mtracker.initial_marker_center[:, 1]
    Oy = mtracker.initial_marker_center[:, 0]
    init_markers = np.array((Ox, Oy), np.float32).T.reshape((-1, 2))
    nct = len(mtracker.initial_marker_center)
    p0 = np.array((Ox, Oy), np.float32).T.reshape((-1, 1, 2))
    old_gray = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)
    n_markers = init_markers.shape[0]
    traces = [[pt.tolist()] for pt in init_markers]
    arrow_frames = [tactile_frames_rgb[0].copy()]
    for idx in range(1, len(tactile_frames_rgb)):
        frame_bgr = cv2.cvtColor(tactile_frames_rgb[idx], cv2.COLOR_RGB2BGR)
        new_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **LK_PARAMS)
        if np.sum(st) >= nct:
            p0 = p1.reshape(-1, 1, 2)
            old_gray = new_gray
        markers = np.squeeze(p0)
        for i in range(n_markers):
            traces[i].append(markers[i].tolist())
        motion = markers - init_markers
        vis_bgr = _draw_motion_arrows(frame_bgr, init_markers, motion, arrow_scale, magnitude_coloring)
        if draw_trace:
            trace_pts = [np.array(t, dtype=np.float32) for t in traces]
            vis_bgr = _draw_marker_trace(vis_bgr, trace_pts, color=(0, 255, 0), thickness=1)
        arrow_frames.append(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB))
    return arrow_frames


# Match AnyTouch2: tactile for MarkerTracker should be 320x240 (GelSight marker detection)
TACTILE_MOTION_RESIZE = (320, 240)  # (width, height)


def _normalize_tactile_for_markertracker(img: np.ndarray) -> np.ndarray:
    """
    Normalize tactile image to match HDF5 format expected by MarkerTracker.
    Ensures (H, W, 3) uint8 [0, 255]. Resize to 320x240 for gsmini compatibility.
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.max() <= 1.0:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.resize(img, TACTILE_MOTION_RESIZE, interpolation=cv2.INTER_AREA)
    return img


def extract_tactile_from_horizontal_frames(
    frames: list[np.ndarray],
    panel: str = "gt",
) -> list[np.ndarray]:
    """
    Extract tactile (camera_2) from [gt|pred|diff] horizontal frames.
    panel: 'gt' = left panel (real markers for motion tracking), 'pred' = middle panel.
     Normalize to match HDF5 format for MarkerTracker.
    """
    result = []
  
    for frame in frames:
        h, w = frame.shape[:2]
        panel_w = w // 3
       
        if panel == "gt":
            target_panel = frame[:, 0:panel_w]
        elif panel == "pred":
            target_panel = frame[:, panel_w : 2 * panel_w]
        else:
            target_panel = frame[:, 2 * panel_w :]
        tactile = target_panel
        tactile = _normalize_tactile_for_markertracker(tactile)
        result.append(tactile)
    return result


def build_motion_row_frames(
    gt_tactile_frames: list[np.ndarray],
    pred_tactile_frames: list[np.ndarray] | None = None,
    arrow_scale: float = 3.0,
) -> list[np.ndarray] | None:
    """
    Compute motion arrows + marker trace on tactile frames.
    Returns [gt_motion | pred_motion | black] per timestep.
    If pred_tactile_frames is None or pred motion fails, pred slot is black.
    """
    gt_frames = compute_tactile_motion_frames(
        gt_tactile_frames, arrow_scale=arrow_scale, draw_trace=True
    )
    if gt_frames is None:
        return None
    h, w = gt_frames[0].shape[:2]
    black = np.zeros((h, w, 3), dtype=np.uint8)
    pred_frames = None
    if pred_tactile_frames is not None and len(pred_tactile_frames) == len(gt_tactile_frames):
        pred_frames = compute_tactile_motion_frames(
            pred_tactile_frames, arrow_scale=arrow_scale, draw_trace=True
        )
    result = []
    for i, gf in enumerate(gt_frames):
        if pred_frames is not None:
            pf = pred_frames[i]
            row = np.concatenate([gf, pf, black], axis=1)
        else:
            row = np.concatenate([gf, black, black], axis=1)
        result.append(row)
    return result


def load_frames_from_eval_video(
    video_path: str,
    extract: str = "full",
) -> list[np.ndarray]:
    """
    Load video saved by eval_dino_wm_tactile: each frame is vertically stacked
    [gt_row | pred_row | diff_row]. Each row has cameras concatenated horizontally.
    Returns list of frames for display or encoding.
    extract: 'full' = full frame (gt|pred|diff vertical), 'horizontal' = [gt|pred|diff] side by side,
             'pred' = middle row only, 'tactile' = tactile (camera_2, right 1/3) from pred row.
    """
    raw_frames = load_frames_from_video(video_path) if not os.path.isdir(video_path) else load_frames_from_directory(video_path)
    result = []
    for frame in raw_frames:
        h, w = frame.shape[:2]
        row_h = h // 3
        gt_row = frame[0:row_h]
        pred_row = frame[row_h : 2 * row_h]
        diff_row = frame[2 * row_h : 3 * row_h]

        if extract == "full":
            result.append(frame)
        elif extract == "horizontal":
            # [gt | pred | diff] concatenated horizontally
            result.append(np.concatenate([gt_row, pred_row, diff_row], axis=1))
        elif extract == "pred":
            result.append(pred_row)
        elif extract == "tactile":
            # camera_2 is rightmost 1/3 of each row
            tactile = pred_row[:, 2 * w // 3 :, :]
            result.append(tactile)
        elif extract == "gt_tactile":
            tactile = gt_row[:, 2 * w // 3 :, :]
            result.append(tactile)
        elif extract == "pred_tactile":
            tactile = pred_row[:, 2 * w // 3 :, :]
            result.append(tactile)
        else:
            raise ValueError(f"extract must be 'full', 'horizontal', 'pred', 'tactile', 'gt_tactile', or 'pred_tactile', got {extract}")
    return result


def generate_result_video(
    sim_matrix: np.ndarray,
    texts: list[str],
    video_frames: list[np.ndarray],
    output_path: str,
    fps: float = 15.0,
    title: str = "Tactile-Text Cosine Similarity",
    sim_gt: np.ndarray | None = None,
    rgb_frames: list[np.ndarray] | None = None,
    motion_row_frames: list[np.ndarray] | None = None,
):
    """
    Render video: top = animated similarity curve, middle = tactile [gt|pred|diff],
    optional motion row [gt_motion+trace|pred_motion+trace|black], bottom = RGB [camera_0|camera_1|black].
    """
    n_steps = sim_matrix.shape[0]
    n_texts = sim_matrix.shape[1]

    sim_min = sim_matrix.min() - 0.02
    sim_max = sim_matrix.max() + 0.02
    if sim_gt is not None:
        sim_min = min(sim_min, sim_gt.min() - 0.02)
        sim_max = max(sim_max, sim_gt.max() + 0.02)

    # Same color scheme as plot_similarity / plot_similarity_gt_pred
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(texts), 3)))

    fh, fw = video_frames[0].shape[:2]
    dpi = 120
    plot_w_px = max(fw, 768)
    plot_h_inches = 5.0
    plot_w_inches = plot_w_px / dpi
    plot_h_px = int(plot_h_inches * dpi)
    tactile_display_h = int(plot_w_px * fh / fw)
    frame_display_h = tactile_display_h
    if motion_row_frames:
        frame_display_h += tactile_display_h
    if rgb_frames:
        frame_display_h += tactile_display_h
    video_w = plot_w_px
    video_h = plot_h_px + frame_display_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))
    print(f"  Rendering result video ({n_steps} frames, {video_w}x{video_h}) ...")

    for t in range(n_steps):
        fig, ax = plt.subplots(figsize=(plot_w_inches, plot_h_inches), dpi=dpi)
        for j in range(n_texts):
            short_label = texts[j][:50] + ("..." if len(texts[j]) > 50 else "")
            c = colors[j % len(colors)]
            if sim_gt is not None and sim_gt.shape[0] > 0:
                max_t_gt = min(t + 1, sim_gt.shape[0])
                ax.plot(
                    np.arange(max_t_gt),
                    sim_gt[:max_t_gt, j],
                    color=c,
                    linewidth=1.5,
                    linestyle="-",
                    label=f"GT: {short_label}",
                )
            max_t = min(t + 1, sim_matrix.shape[0])
            ax.plot(
                np.arange(max_t),
                sim_matrix[:max_t, j],
                color=c,
                linewidth=1.5,
                linestyle="--" if sim_gt is not None else "-",
                label=f"Pred: {short_label}" if sim_gt is not None else short_label,
            )
            if max_t > 0:
                ax.scatter([t], [sim_matrix[t, j]], color=c, s=25, zorder=5)

        ax.set_xlim(0, max(n_steps - 1, 1))
        ax.set_ylim(sim_min, sim_max)
        ax.set_xlabel("Frame index", fontsize=9)
        ax.set_ylabel("Cosine similarity", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=7, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1])

        plot_img = _fig_to_array(fig)
        plt.close(fig)
        plot_img = cv2.resize(plot_img, (plot_w_px, plot_h_px), interpolation=cv2.INTER_AREA)

        frame_idx = min(t, len(video_frames) - 1)
        tactile_frame = video_frames[frame_idx]
        tactile_frame = cv2.resize(tactile_frame, (video_w, tactile_display_h), interpolation=cv2.INTER_AREA)
        combined = np.vstack([plot_img, tactile_frame])

        if motion_row_frames:
            motion_idx = min(t, len(motion_row_frames) - 1)
            motion_frame = motion_row_frames[motion_idx]
            motion_frame = cv2.resize(motion_frame, (video_w, tactile_display_h), interpolation=cv2.INTER_AREA)
            combined = np.vstack([combined, motion_frame])

        if rgb_frames:
            rgb_idx = min(t, len(rgb_frames) - 1)
            rgb_frame = rgb_frames[rgb_idx]
            rgb_frame = cv2.resize(rgb_frame, (video_w, tactile_display_h), interpolation=cv2.INTER_AREA)
            combined = np.vstack([combined, rgb_frame])

        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"  Video saved to {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def run_video_mode(args):
    """Encode tactile videos through AnyTouch, compute similarity with text."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tactile_model, clip_model, tokenizer, sensor_id = load_tactile_model_and_clip(
        args.checkpoint, args.sensor, args.num_frames, args.stride, device
    )

    texts = args.text
    text_embeds = extract_text_embeddings(clip_model, tokenizer, texts, device)
    print(f"Text embeddings: {text_embeds.shape}")

    sources = resolve_video_sources(args.video_path)
    os.makedirs(args.output_dir, exist_ok=True)

    for src_path, name in sources:
        print(f"\nProcessing: {src_path}")
        if os.path.isdir(src_path):
            frames = load_frames_from_directory(src_path)
        else:
            frames = load_frames_from_video(src_path)

        if len(frames) < args.num_frames:
            print(f"  Skip: need >= {args.num_frames} frames, got {len(frames)}")
            continue

        windows = preprocess_frames(frames, None, args.num_frames, args.frame_stride)
        touch_embeds = extract_tactile_embeddings(
            tactile_model, windows, sensor_id, device, args.aggregate
        )
        sim_matrix = touch_embeds @ text_embeds.T
        print(f"  Touch embeds: {touch_embeds.shape}, similarity: {sim_matrix.shape}")

        mean_sim = sim_matrix.mean(axis=0)
        print("  Mean similarity per text:")
        for j, t in enumerate(texts):
            print(f"    [{j}] {t}: {mean_sim[j]:.4f}")

        # Frames per window (last frame of each window) for video
        subsampled = frames[::args.frame_stride]
        n_windows = len(windows)
        frames_per_window = []
        for t in range(n_windows):
            last_idx = min((t + args.num_frames - 1) * args.frame_stride, len(frames) - 1)
            frames_per_window.append(frames[last_idx])

        if args.save_csv:
            import csv
            csv_path = os.path.join(args.output_dir, f"{name}_similarity.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["window_idx"] + [f'"{t}"' for t in texts])
                for i, row in enumerate(sim_matrix):
                    w.writerow([i] + list(row))
            print(f"  Saved {csv_path}")

        if args.save_plot:
            plot_path = os.path.join(args.output_dir, f"{name}_similarity.png")
            plot_similarity(sim_matrix, texts, plot_path, title=f"Tactile-Text Similarity — {name}")

        if args.save_video:
            video_out = os.path.join(args.output_dir, f"{name}_result.mp4")
            generate_result_video(
                sim_matrix, texts, frames_per_window,
                output_path=video_out,
                fps=args.video_fps,
                title=f"Tactile-Text Similarity — {name}",
            )


def run_embeddings_mode(args):
    """Use pre-computed embeddings, mean over patches, no encoder."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load CLIP text encoder only (for text embeddings)
    from transformers import CLIPConfig, CLIPModel, CLIPTokenizer

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    clip_dir = os.path.join(config_dir, "CLIP-B-16")
    if not os.path.isdir(clip_dir):
        import model.tactile_mae as tactile_mod
        pkg_root = os.path.dirname(os.path.dirname(tactile_mod.__file__))
        clip_dir = os.path.join(pkg_root, "CLIP-B-16")

    config = CLIPConfig.from_pretrained(os.path.join(clip_dir, "config.json"))
    tokenizer = CLIPTokenizer.from_pretrained(clip_dir)
    clip_model = CLIPModel(config)
    TEXT_PREFIXES = ["text_model.", "text_projection."]
    WRAPPER_PREFIXES = ["", "clip_model.", "hf_clip_model.", "clip."]
    text_state = {}
    for key, val in ckpt.items():
        for wp in WRAPPER_PREFIXES:
            for tp in TEXT_PREFIXES:
                full = wp + tp
                if key.startswith(full):
                    text_state[key[len(wp):]] = val
    if text_state:
        clip_model.load_state_dict(text_state, strict=False)
    clip_model = clip_model.to(device).eval()

    texts = args.text
    text_embeds = extract_text_embeddings(clip_model, tokenizer, texts, device)
    print(f"Text embeddings: {text_embeds.shape}")

    sources = resolve_embedding_sources(args.embeddings_path)
    os.makedirs(args.output_dir, exist_ok=True)

    for npz_path, name in sources:
        print(f"\nProcessing: {npz_path}")
        pred_embeds = load_embeddings_from_npz(
            npz_path, args.camera_key, skip_tokens=args.skip_tokens
        )
        if pred_embeds.shape[-1] != text_embeds.shape[-1]:
            raise ValueError(
                f"Embedding dim {pred_embeds.shape[-1]} != text dim {text_embeds.shape[-1]}. "
                "Use camera_2 (tactile, 512-dim) for CLIP text similarity."
            )
        sim_pred = pred_embeds @ text_embeds.T
        print(f"  Pred embeds: {pred_embeds.shape}, similarity: {sim_pred.shape}")

        gt_path = infer_paired_gt_path(npz_path)
        sim_gt = None
        if gt_path:
            gt_embeds = load_embeddings_from_npz(
                gt_path, args.camera_key, skip_tokens=args.skip_tokens
            )
            sim_gt = gt_embeds @ text_embeds.T
            print(f"  GT embeds: {gt_embeds.shape} (paired)")

        mean_sim = sim_pred.mean(axis=0)
        print("  Mean similarity per text (pred):")
        for j, t in enumerate(texts):
            print(f"    [{j}] {t}: {mean_sim[j]:.4f}")

        if args.save_csv:
            import csv
            csv_path = os.path.join(args.output_dir, f"{name}_similarity.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                cols = ["frame_idx"]
                for txt in texts:
                    cols.append(f'"{txt}" (pred)')
                if sim_gt is not None:
                    for txt in texts:
                        cols.append(f'"{txt}" (gt)')
                w.writerow(cols)
                n_rows = max(sim_pred.shape[0], sim_gt.shape[0] if sim_gt is not None else 0)
                for i in range(n_rows):
                    row = [i]
                    if i < sim_pred.shape[0]:
                        row.extend(sim_pred[i].tolist())
                    else:
                        row.extend([""] * len(texts))
                    if sim_gt is not None:
                        if i < sim_gt.shape[0]:
                            row.extend(sim_gt[i].tolist())
                        else:
                            row.extend([""] * len(texts))
                    w.writerow(row)
            print(f"  Saved {csv_path}")

        if args.save_plot:
            plot_path = os.path.join(args.output_dir, f"{name}_similarity.png")
            if sim_gt is not None:
                plot_similarity_gt_pred(
                    sim_gt, sim_pred, texts, plot_path,
                    title=f"Tactile-Text Similarity (GT vs Pred) — {name}",
                )
            else:
                plot_similarity(
                    sim_pred, texts, plot_path,
                    title=f"Tactile-Text Similarity — {name}",
                )

        if args.save_video and (args.video_path or args.video_dir):
            if args.video_dir:
                video_dir = args.video_dir
            elif args.video_path and os.path.isdir(args.video_path):
                video_dir = args.video_path
            elif args.video_path:
                video_dir = os.path.dirname(os.path.abspath(args.video_path)) or "."
            else:
                video_dir = "."

            # Match video to npz: episode_X_traj_Y_result.mp4 (from eval_dino_wm_tactile)
            video_path = infer_eval_video_path(npz_path, video_dir)
            if video_path is None and args.video_path and os.path.isfile(args.video_path):
                video_path = args.video_path
            if video_path is None:
                print(f"  Warning: no video found for {npz_path} in {video_dir}, skipping video")
            else:
                video_frames = (
                    load_frames_from_eval_video(video_path, extract="horizontal")
                    if args.eval_video_format
                    else load_video_frames(video_path)
                )
            
                if len(video_frames) > 0:
                    # Align frames to sim length (linear mapping)
                    n_steps = sim_pred.shape[0]
                    frames_for_video = []
                    for t in range(n_steps):
                        idx = min(int(t * len(video_frames) / max(n_steps, 1)), len(video_frames) - 1)
                        frames_for_video.append(video_frames[idx])

                    motion_row_frames = None
                    if args.eval_video_format:
                        gt_tactile_frames = extract_tactile_from_horizontal_frames(
                            frames_for_video, panel="gt"
                        )
                        pred_tactile_frames = extract_tactile_from_horizontal_frames(
                            frames_for_video, panel="pred"
                        )
                        
                        motion_row_frames = build_motion_row_frames(
                            gt_tactile_frames,
                            pred_tactile_frames,
                            arrow_scale=3.0,
                        )

                    rgb_frames = None
                    if args.hdf5_path:
                        traj_id = parse_trajectory_id_from_filename(npz_path)
                        if traj_id:
                            try:
                                rgb_frames = load_rgb_frames_from_hdf5(
                                    args.hdf5_path,
                                    traj_id,
                                    num_frames=n_steps,
                                    start_idx=args.hdf5_start_idx,
                                )
                                if len(rgb_frames) != n_steps:
                                    rgb_frames = [
                                        rgb_frames[min(int(t * len(rgb_frames) / max(n_steps, 1)), len(rgb_frames) - 1)]
                                        for t in range(n_steps)
                                    ]
                            except (KeyError, OSError) as e:
                                print(f"  Warning: could not load RGB from HDF5: {e}")
                        else:
                            print(f"  Warning: could not parse trajectory_id from {npz_path}")

                    video_out = os.path.join(args.output_dir, f"{name}_result.mp4")
                    generate_result_video(
                        sim_pred, texts, frames_for_video,
                        output_path=video_out,
                        fps=args.video_fps,
                        title=f"Tactile-Text Similarity — {name}",
                        sim_gt=sim_gt,
                        rgb_frames=rgb_frames,
                        motion_row_frames=motion_row_frames,
                    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute tactile-text similarity for reward (video or embeddings mode)"
    )
    p.add_argument("--mode", type=str, choices=["video", "embeddings"], required=True,
                   help="video: encode tactile videos; embeddings: use pre-computed patch embeds (mean over patches)")
    p.add_argument("--text", type=str, action="append", required=True,
                   help="Text sentence(s) to compare. Repeat for multiple.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="AnyTouch checkpoint (for CLIP text encoder in both modes; tactile encoder in video mode)")
    p.add_argument("--output_dir", type=str, default="results/touch_text_similarity")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_csv", action="store_true", default=True)
    p.add_argument("--save_plot", action="store_true", default=True,
                   help="Save similarity plot (GT vs Pred when paired)")
    p.add_argument("--save_video", action="store_true", default=False,
                   help="Generate result video with plot + tactile frames")
    p.add_argument("--video_fps", type=float, default=15.0)

    # Video mode
    p.add_argument("--video_path", type=str, default=None,
                   help="Video file, directory of frames, or glob (tactile-only, no crop)")
    p.add_argument("--sensor", type=str, default="digit", choices=list(SENSOR_NAME_TO_ID.keys()))
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--aggregate", type=str, default="mean_patch",
                   choices=["cls", "mean_patch", "max_patch", "min_patch"])

    # Embeddings mode
    p.add_argument("--embeddings_path", type=str, default=None,
                   help="Path to .npz file or glob of .npz (e.g. from eval_dino_wm_tactile --save_embeds)")
    p.add_argument("--camera_key", type=str, default="camera_2",
                   help="Key in .npz for tactile embeddings (default: camera_2)")
    p.add_argument("--skip_tokens", type=int, default=0,
                   help="Skip first N tokens before mean (e.g. 6 for cls+sensor in some formats)")

    p.add_argument("--video_dir", type=str, default=None,
                   help="Dir with per-episode videos: {video_dir}/{name}.mp4 (embeddings mode)")
    p.add_argument("--eval_video_format", action="store_true",
                   help="Video is from eval_dino_wm_tactile (gt|pred|diff stacked vertically). Use when attaching eval output videos.")
    p.add_argument("--hdf5_path", type=str, default=None,
                   help="Consolidated HDF5 path for RGB row and tactile motion (camera_0/1 for RGB, camera_2 for motion arrows). Use this when video compression corrupts tactile markers.")
    p.add_argument("--hdf5_start_idx", type=int, default=0,
                   help="Start index in HDF5 trajectory for RGB frames (default 0)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "video":
        if not args.video_path:
            raise ValueError("--video_path required for video mode")
        run_video_mode(args)
    else:
        if not args.embeddings_path:
            raise ValueError("--embeddings_path required for embeddings mode")
        run_embeddings_mode(args)

    print("\nDone.")
