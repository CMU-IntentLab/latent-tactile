"""
Dataset for tactile HDF5 format produced by hdf5_to_dataset_tactile.py.

Each trajectory has:
- camera_0, camera_1, camera_2: images (H, W, 3)
- cam_0_patch_embd, cam_1_patch_embd, cam_tactile_patch_embd: patch embeddings (T, num_patches, emb_dim)
- cam_0_cls_embd, cam_1_cls_embd, cam_tactile_cls_embd: CLS embeddings (T, emb_dim)
- states, actions
"""

import json
import os
import random

import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
from torchvision import transforms


# Mapping from camera key to embedding and image keys
CAMERA_CONFIG = {
    "camera_0": {
        "embd_key": "cam_0_patch_embd",
        "image_key": "camera_0",
    },
    "camera_1": {
        "embd_key": "cam_1_patch_embd",
        "image_key": "camera_1",
    },
    "camera_2": {
        "embd_key": "cam_tactile_patch_embd",
        "image_key": "camera_2",
    },
}


def resize_image_to_224(img: np.ndarray) -> np.ndarray:
    """Resize image to 224x224. Handles both HWC and CHW."""
    from PIL import Image
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(pil).numpy().transpose(1, 2, 0)  # back to HWC for consistency


class TactileTrajectoryDataset(Dataset):
    """
    Dataset for tactile HDF5 format (consolidated or per-file).

    Supports loading specific cameras. Each sample is a single timestep from a trajectory.
    """

    def __init__(
        self,
        hdf5_path: str,
        cameras: list[str],
        segment_length: int = 1,
        split: str = "train",
        num_test: int = 100,
        is_consolidated: bool = True,
        seed: int = 42,
        resize_to_224: bool = True,
        segment_sampling: str = "uniform",
        gripper_state_idx: int = -1,
        gripper_change_weight: float = 5.0,
        weight_source: str = "gripper_state",
        gripper_change_json_path: str | None = None,
    ):
        """
        Args:
            hdf5_path: Path to consolidated HDF5 file, or directory containing .hdf5 files.
            cameras: List of camera keys to load, e.g. ["camera_0", "camera_1"] or ["camera_2"].
            segment_length: Number of timesteps per segment (default 1 for single-frame).
            split: "train" or "test".
            num_test: Number of trajectories for test split.
            is_consolidated: If True, hdf5_path is a single file. If False, it's a directory.
            seed: Random seed for train/test split shuffle (for reproducibility).
            resize_to_224: If True, resize images to 224x224. Default False keeps original size.
            segment_sampling: "uniform" (default) or "weighted". Weighted samples more from
                segments where gripper state range (max-min) > 0.01.
            gripper_action_idx: Action dimension for gripper (default -1 = last).
            gripper_state_idx: State dimension for gripper (default -1 = last).
            gripper_change_weight: When weighted, segments with gripper change get this weight
                vs 1.0 for others (default 5.0 = ~5x more likely to sample).
            weight_source: "gripper_state" (default) = compute from states max-min > 0.01;
                "gripper_change_json" = load indices from JSON and weight segments overlapping
                [index_0±2] or [index_1±2].
            gripper_change_json_path: Path to gripper_change.json (required when weight_source=
                "gripper_change_json"). JSON format: [{"traj_id": "...", "high_diff_idxs": [i0, i1]}, ...]
        """
        self.hdf5_path = hdf5_path
        self.cameras = cameras
        self.segment_length = segment_length
        self.split = split
        self.num_test = num_test
        self.is_consolidated = is_consolidated
        self.resize_to_224 = resize_to_224
        self.segment_sampling = segment_sampling
        self.gripper_state_idx = gripper_state_idx
        self.gripper_change_weight = gripper_change_weight
        self.weight_source = weight_source
        self.gripper_change_json_path = gripper_change_json_path
        self._hf = None  # Lazy-open persistent handle for consolidated mode

        for cam in cameras:
            if cam not in CAMERA_CONFIG:
                raise ValueError(f"Unknown camera: {cam}. Choose from {list(CAMERA_CONFIG.keys())}")

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
        if is_consolidated:
            with h5py.File(hdf5_path, "r") as hf:
                for traj_id in self.trajectory_ids:
                    traj = hf[traj_id]
                    traj_len = traj["actions"].shape[0]
                    for start in range(0, traj_len - segment_length + 1, 1):
                        self.slice_indices.append((traj_id, start))
        else:
            for filepath in self.trajectory_ids:
                with h5py.File(filepath, "r") as hf:
                    dg = hf["data"] if "data" in hf else hf
                    traj_len = dg["actions"].shape[0]
                    for start in range(0, traj_len - segment_length + 1, 1):
                        self.slice_indices.append((filepath, start))

        self.segment_weights = None
        if segment_sampling == "weighted":
            self._compute_segment_weights()

    def _get_traj_id_for_lookup(self, traj_id_or_path: str) -> str:
        """Return traj_id for JSON lookup. For consolidated, it's traj_id; for non-consolidated, derive from path."""
        if self.is_consolidated:
            return traj_id_or_path
        return os.path.splitext(os.path.basename(traj_id_or_path))[0]

    def _compute_segment_weights(self):
        """Dispatch to gripper_state or gripper_change_json weight computation."""
        if self.weight_source == "gripper_change_json":
            self._compute_segment_weights_from_json()
        else:
            self._compute_segment_weights_gripper_state()

    def _compute_segment_weights_gripper_state(self):
        """Compute weights: gripper_change_weight for segments with max-min > 0.01, else 1.0."""
        weights = []

        def _gripper_change(traj, start: int, end: int) -> float:
            """Weight = gripper_change_weight if max-min > 0.01, else 1.0."""
            threshold = 0.01
            if "states" in traj:
                states = np.array(traj["states"][start:end], dtype=np.float32)
                if states.size > 0 and abs(self.gripper_state_idx) <= states.shape[-1]:
                    gripper = states[:, self.gripper_state_idx]
                    diff = float(np.max(gripper) - np.min(gripper))
                    if diff > threshold:
                        return self.gripper_change_weight
            return 1.0

        if self.is_consolidated:
            with h5py.File(self.hdf5_path, "r") as hf:
                for traj_id, start in self.slice_indices:
                    traj = hf[traj_id]
                    end = start + self.segment_length
                    weights.append(_gripper_change(traj, start, end))
        else:
            for filepath, start in self.slice_indices:
                with h5py.File(filepath, "r") as hf:
                    traj = hf["data"] if "data" in hf else hf
                    end = start + self.segment_length
                    weights.append(_gripper_change(traj, start, end))

        self.segment_weights = np.array(weights, dtype=np.float64)
        n_weighted = int(np.sum(self.segment_weights > 1.0))
        n_total = len(self.segment_weights)
        print(
            f"[TactileTrajectoryDataset] weighted sampling (gripper_state): {n_weighted}/{n_total} segments "
            f"({100 * n_weighted / n_total:.1f}%) have gripper change (max-min > 0.01)"
        )

    def _compute_segment_weights_from_json(self):
        """Compute weights from gripper_change.json: weight segments overlapping [index_0±2] or [index_1±2]."""
        if not self.gripper_change_json_path or not os.path.isfile(self.gripper_change_json_path):
            raise FileNotFoundError(
                f"gripper_change_json_path must point to existing file when weight_source=gripper_change_json, "
                f"got {self.gripper_change_json_path}"
            )
        with open(self.gripper_change_json_path) as f:
            data = json.load(f)
        # Build lookup: traj_id -> (index_0, index_1) from high_diff_idxs
        traj_to_indices: dict[str, tuple[int, int]] = {}
        for item in data:
            tid = item["traj_id"]
            idxs = item.get("high_diff_idxs", [])
            if len(idxs) >= 2:
                traj_to_indices[tid] = (int(idxs[0]), int(idxs[1]))
            elif len(idxs) == 1:
                traj_to_indices[tid] = (int(idxs[0]), int(idxs[0]))

        def _overlaps(segment_start: int, segment_end: int, center: int) -> bool:
            """Check if [center-2, center+2] overlaps with [segment_start, segment_end)."""
            lo, hi = center - 2, center + 2
            return segment_start <= hi and lo <= segment_end - 1

        weights = []
        for traj_id_or_path, start in self.slice_indices:
            traj_id = self._get_traj_id_for_lookup(traj_id_or_path)
            segment_end = start + self.segment_length
            indices = traj_to_indices.get(traj_id)
            if indices is None:
                weights.append(1.0)
                continue
            index_0, index_1 = indices
            if _overlaps(start, segment_end, index_0) or _overlaps(start, segment_end, index_1):
                weights.append(self.gripper_change_weight)
            else:
                weights.append(1.0)

        self.segment_weights = np.array(weights, dtype=np.float64)
        n_weighted = int(np.sum(self.segment_weights > 1.0))
        n_total = len(self.segment_weights)
        print(
            f"[TactileTrajectoryDataset] weighted sampling (gripper_change_json): {n_weighted}/{n_total} segments "
            f"({100 * n_weighted / n_total:.1f}%) overlap [index_0±2] or [index_1±2]"
        )

    def get_sampler(self, num_samples: int | None = None):
        """
        Return a sampler for use with DataLoader when segment_sampling="weighted".
        Returns None for uniform sampling (DataLoader will use shuffle=True).
        """
        if self.segment_sampling != "weighted" or self.segment_weights is None:
            return None
        from torch.utils.data import WeightedRandomSampler
        n = len(self.slice_indices)
        weights = torch.from_numpy(self.segment_weights)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples if num_samples is not None else n,
            replacement=True,
        )

    def __len__(self):
        return len(self.slice_indices)

    def _get_trajectory(self, traj_id):
        if self.is_consolidated:
            return h5py.File(self.hdf5_path, "r")[traj_id]
        f = h5py.File(traj_id, "r")
        return f["data"] if "data" in f else f

    def _get_hf(self, file_path: str):
        """Return open HDF5 file handle. Reuses handle for consolidated mode."""
        if self.is_consolidated:
            if self._hf is None:
                self._hf = h5py.File(file_path, "r")
            return self._hf
        return None

    def __getitem__(self, idx):
        traj_id, start_idx = self.slice_indices[idx]
        end_idx = start_idx + self.segment_length

        file_path = self.hdf5_path if self.is_consolidated else traj_id
        hf = self._get_hf(file_path)
        if hf is not None:
            traj = hf[traj_id]
        else:
            hf = h5py.File(file_path, "r")
            traj = hf["data"] if "data" in hf else hf

        try:
            out = {}
            for cam in self.cameras:
                cfg = CAMERA_CONFIG[cam]
                embd = np.array(traj[cfg["embd_key"]][start_idx:end_idx], dtype=np.float32)
                img = np.array(traj[cfg["image_key"]][start_idx:end_idx])

                if img.ndim == 3:
                    img = img[np.newaxis, ...]
                if self.resize_to_224 and (img.shape[2] != 224 or img.shape[3] != 224):
                    resized = []
                    for t in range(img.shape[0]):
                        resized.append(resize_image_to_224(img[t]))
                    img = np.stack(resized)

                img = np.asarray(img, dtype=np.float32)
                if img.max() > 1.0:
                    img = img / 255.0

                out[f"{cam}_embd"] = torch.tensor(embd, dtype=torch.float32)
                out[f"{cam}_image"] = torch.tensor(img, dtype=torch.float32)

            if "actions" in traj:
                out["action"] = torch.tensor(
                    traj["actions"][start_idx:end_idx], dtype=torch.float32
                )
            if "states" in traj:
                out["state"] = torch.tensor(
                    traj["states"][start_idx:end_idx], dtype=torch.float32
                )

            return out
        finally:
            if not self.is_consolidated:
                hf.close()


def load_full_episode(
    hdf5_path: str,
    traj_id: str,
    cameras: list[str],
    is_consolidated: bool,
    resize_to_224: bool = True,
) -> dict:
    """
    Load a full episode (all timesteps) for the given trajectory.
    Returns dict with {cam}_embd, {cam}_image for each camera (numpy arrays).
    """
    file_path = hdf5_path if is_consolidated else traj_id
    with h5py.File(file_path, "r") as hf:
        traj = hf[traj_id] if is_consolidated else (hf["data"] if "data" in hf else hf)
        traj_len = traj["actions"].shape[0]

        out = {}
        for cam in cameras:
            cfg = CAMERA_CONFIG[cam]
            embd = np.array(traj[cfg["embd_key"]][:traj_len], dtype=np.float32)
            img = np.array(traj[cfg["image_key"]][:traj_len])

            if img.ndim == 3:
                img = img[np.newaxis, ...]
            if resize_to_224 and (img.shape[2] != 224 or img.shape[3] != 224):
                resized = []
                for t in range(img.shape[0]):
                    resized.append(resize_image_to_224(img[t]))
                img = np.stack(resized)

            img = np.asarray(img, dtype=np.float32)
            if img.max() > 1.0:
                img = img / 255.0

            out[f"{cam}_embd"] = embd
            out[f"{cam}_image"] = img

        if "states" in traj:
            out["states"] = np.array(traj["states"][:traj_len], dtype=np.float32)
        if "actions" in traj:
            out["actions"] = np.array(traj["actions"][:traj_len], dtype=np.float32)

    return out
