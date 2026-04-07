"""
Tactile dataset for K-step dynamics model (LPB-style or dense consecutive windows).

temporal_mode "lpb" (default):
- Subsample observations every frameskip steps; "Frame" i = raw step start + i*frameskip.
- num_hist + num_pred subsampled observation frames; actions: (num_hist+num_pred)*frameskip raw,
  collapsed per slot to (frameskip * action_dim).

temporal_mode "consecutive":
- History embeddings at raw indices (t - num_hist + 1, ..., t).
- Supervision embeddings at raw indices (t + k * frameskip) for k = 1, ..., num_pred
  (same spacing as LPB along the future: last target at t + num_pred * frameskip).
- Actions: (num_hist + num_pred) slots of frameskip each — history: a[h_j : h_j + frameskip];
  prediction slot k: a[t + k * frameskip : t + (k + 1) * frameskip] for k = 0..num_pred-1.
"""

import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from tactile_dataset import CAMERA_CONFIG


PATCH_GRID_SIZE = 196  # 14x14, required for collate (all samples must have same shape)


def _ensure_patch_count(embd: np.ndarray, target_n: int = PATCH_GRID_SIZE) -> np.ndarray:
    """Pad or crop embeddings to target_n patches for consistent collate."""
    T, N, D = embd.shape
    if N == target_n:
        return embd.copy()
    if N > target_n:
        return embd[:, :target_n, :].copy()
    pad = np.zeros((T, target_n - N, D), dtype=embd.dtype)
    return np.concatenate([embd, pad], axis=1)


def _consecutive_anchor_bounds(traj_len: int, num_hist: int, num_pred: int, frameskip: int):
    """
    Valid last-history index t: hist [t-nh+1..t], targets [t+fs, t+2fs, ..., t+np*fs].
    Returns (t_min, t_max) inclusive.
    """
    t_min = num_hist - 1
    t_max = traj_len - 1 - num_pred * frameskip
    t_max = min(t_max, traj_len - frameskip + num_hist - 1)
    return t_min, t_max


class TactileKStepDataset(Dataset):
    """
    Tactile HDF5 with num_hist, num_pred, frameskip.

    temporal_mode "lpb": subsampled observation grid (see module docstring).
    temporal_mode "consecutive": dense history, LPB-spaced future targets (see module docstring).

    Embeddings are normalized to PATCH_GRID_SIZE (196) patches for consistent batching.
    """

    def __init__(
        self,
        hdf5_path: str,
        cameras: list[str],
        num_hist: int = 1,
        num_pred: int = 1,
        frameskip: int = 8,
        split: str = "train",
        num_test: int = 100,
        is_consolidated: bool = True,
        seed: int = 42,
        action_dim: int = 7,
        temporal_mode: str = "lpb",
    ):
        self.hdf5_path = hdf5_path
        self.cameras = cameras
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.frameskip = frameskip
        self.num_frames = num_hist + num_pred
        self.split = split
        self.num_test = num_test
        self.is_consolidated = is_consolidated
        self.original_action_dim = action_dim
        self.action_dim = action_dim * frameskip
        self._hf = None
        if temporal_mode not in ("lpb", "consecutive"):
            raise ValueError('temporal_mode must be "lpb" or "consecutive"')
        self.temporal_mode = temporal_mode

        for cam in cameras:
            if cam not in CAMERA_CONFIG:
                raise ValueError(f"Unknown camera: {cam}")

        if is_consolidated:
            with h5py.File(hdf5_path, "r") as hf:
                self.trajectory_ids = [k for k in hf.keys() if k.startswith("trajectory_")]
            self.trajectory_ids.sort()
        else:
            all_files = [
                os.path.join(hdf5_path, f)
                for f in os.listdir(hdf5_path)
                if f.endswith(".hdf5")
            ]
            all_files.sort()
            self.trajectory_ids = all_files

        rng = random.Random(seed)
        shuffled = self.trajectory_ids.copy()
        rng.shuffle(shuffled)
        if split == "train":
            self.trajectory_ids = shuffled[num_test:]
        elif split == "test":
            self.trajectory_ids = shuffled[:num_test]
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.slice_indices = []
        if self.temporal_mode == "lpb":
            min_length = self.num_frames * self.frameskip
            if is_consolidated:
                with h5py.File(hdf5_path, "r") as hf:
                    for traj_id in self.trajectory_ids:
                        traj = hf[traj_id]
                        traj_len = traj["actions"].shape[0]
                        anchor_end = traj_len - min_length + 1
                        if anchor_end > 0:
                            for start in range(0, anchor_end):
                                self.slice_indices.append((traj_id, start))
            else:
                for filepath in self.trajectory_ids:
                    with h5py.File(filepath, "r") as hf:
                        dg = hf["data"] if "data" in hf else hf
                        traj_len = dg["actions"].shape[0]
                        anchor_end = traj_len - min_length + 1
                        if anchor_end > 0:
                            for start in range(0, anchor_end):
                                self.slice_indices.append((filepath, start))
        else:
            if is_consolidated:
                with h5py.File(hdf5_path, "r") as hf:
                    for traj_id in self.trajectory_ids:
                        traj = hf[traj_id]
                        traj_len = traj["actions"].shape[0]
                        t_min, t_max = _consecutive_anchor_bounds(
                            traj_len, self.num_hist, self.num_pred, self.frameskip
                        )
                        if t_max >= t_min:
                            for t in range(t_min, t_max + 1):
                                self.slice_indices.append((traj_id, t))
            else:
                for filepath in self.trajectory_ids:
                    with h5py.File(filepath, "r") as hf:
                        dg = hf["data"] if "data" in hf else hf
                        traj_len = dg["actions"].shape[0]
                        t_min, t_max = _consecutive_anchor_bounds(
                            traj_len, self.num_hist, self.num_pred, self.frameskip
                        )
                        if t_max >= t_min:
                            for t in range(t_min, t_max + 1):
                                self.slice_indices.append((filepath, t))

    def __len__(self):
        return len(self.slice_indices)

    def _get_trajectory(self, hf, traj_id: str):
        if self.is_consolidated:
            return hf[traj_id]
        with h5py.File(traj_id, "r") as f:
            return f["data"] if "data" in f else f

    def __getitem__(self, idx):
        traj_id, anchor = self.slice_indices[idx]

        file_path = self.hdf5_path if self.is_consolidated else traj_id
        if self.is_consolidated and self._hf is not None:
            hf = self._hf
            traj = hf[traj_id]
        else:
            hf = h5py.File(file_path, "r")
            traj = hf[traj_id] if self.is_consolidated else (hf["data"] if "data" in hf else hf)

        if self.temporal_mode == "lpb":
            start_idx = anchor
            end_idx = start_idx + self.num_frames * self.frameskip
            obs_indices = list(range(start_idx, end_idx, self.frameskip))
            actions = np.array(traj["actions"][start_idx:end_idx], dtype=np.float32).copy()
            pad_val = np.array(traj["actions"][obs_indices[-1] - 1], dtype=np.float32)
            actions[-self.frameskip:] = pad_val
        else:
            t_last = anchor
            hist_idx = list(range(t_last - self.num_hist + 1, t_last + 1))
            fs = self.frameskip
            tgt_idx = [t_last + k * fs for k in range(1, self.num_pred + 1)]
            obs_indices = hist_idx + tgt_idx
            blocks = []
            for j in range(self.num_hist):
                h = hist_idx[j]
                blocks.append(np.array(traj["actions"][h : h + fs], dtype=np.float32))
            for k in range(self.num_pred):
                a0 = t_last + k * fs
                blocks.append(np.array(traj["actions"][a0 : a0 + fs], dtype=np.float32))
            actions = np.concatenate(blocks, axis=0)

        out = {}
        for cam in self.cameras:
            cfg = CAMERA_CONFIG[cam]
            embd = np.array(traj[cfg["embd_key"]][obs_indices], dtype=np.float32)
            embd = _ensure_patch_count(embd)
            emb_tensor = torch.from_numpy(np.ascontiguousarray(embd)).float().clone()
            out[f"{cam}_embd"] = emb_tensor

        states = np.array(traj["states"][obs_indices], dtype=np.float32).copy()
        if states.shape[-1] > 8:
            states = states[..., :8]
        elif states.shape[-1] < 8:
            pad = np.zeros((*states.shape[:-1], 8 - states.shape[-1]), dtype=np.float32)
            states = np.concatenate([states, pad], axis=-1)
        out["state"] = torch.from_numpy(np.ascontiguousarray(states)).float().clone()

        if actions.shape[-1] > self.original_action_dim:
            actions = actions[..., : self.original_action_dim]
        out["action"] = torch.from_numpy(np.ascontiguousarray(actions)).float().clone()

        if not self.is_consolidated:
            hf.close()

        return out


def load_full_episode_subsampled(
    hdf5_path: str,
    traj_id: str,
    cameras: list[str],
    frameskip: int,
    is_consolidated: bool,
    load_images: bool = True,
) -> dict:
    """
    Load full episode subsampled by frameskip (obs every frameskip steps).
    Returns dict with {cam}_embd, states, actions_full, obs_indices.
    Optionally {cam}_image at subsampled indices.
    """
    from tactile_dataset import resize_image_to_224

    file_path = hdf5_path if is_consolidated else traj_id
    with h5py.File(file_path, "r") as hf:
        traj = hf[traj_id] if is_consolidated else (hf["data"] if "data" in hf else hf)
        traj_len = traj["actions"].shape[0]
        obs_indices = list(range(0, traj_len, frameskip))
        if not obs_indices:
            return {}

        out = {}
        for cam in cameras:
            cfg = CAMERA_CONFIG[cam]
            out[f"{cam}_embd"] = np.array(traj[cfg["embd_key"]][obs_indices], dtype=np.float32)
            if load_images and cfg.get("image_key"):
                img_key = cfg["image_key"]
                if img_key in traj:
                    img = np.array(traj[img_key][obs_indices])
                    if img.ndim == 3:
                        img = img[np.newaxis, ...]
                    resized = [resize_image_to_224(img[t]) for t in range(img.shape[0])]
                    img = np.stack(resized).astype(np.float32)
                    if img.max() > 1.0:
                        img = img / 255.0
                    out[f"{cam}_image"] = img
        out["states"] = np.array(traj["states"][obs_indices], dtype=np.float32)
        actions_full = np.array(traj["actions"][:traj_len], dtype=np.float32)
        if actions_full.shape[-1] > 7:
            actions_full = actions_full[..., :7]
        out["actions_full"] = actions_full
        out["obs_indices"] = np.array(obs_indices)
        out["frameskip"] = frameskip
    return out
