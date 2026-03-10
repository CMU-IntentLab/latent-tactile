"""
Dataset for tactile HDF5 format produced by hdf5_to_dataset_tactile.py.

Each trajectory has:
- camera_0, camera_1, camera_2: images (H, W, 3)
- cam_0_patch_embd, cam_1_patch_embd, cam_tactile_patch_embd: patch embeddings (T, num_patches, emb_dim)
- cam_0_cls_embd, cam_1_cls_embd, cam_tactile_cls_embd: CLS embeddings (T, emb_dim)
- states, actions
"""

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
        """
        self.hdf5_path = hdf5_path
        self.cameras = cameras
        self.segment_length = segment_length
        self.split = split
        self.num_test = num_test
        self.is_consolidated = is_consolidated
        self.resize_to_224 = resize_to_224

        for cam in cameras:
            if cam not in CAMERA_CONFIG:
                raise ValueError(f"Unknown camera: {cam}. Choose from {list(CAMERA_CONFIG.keys())}")

        if is_consolidated:
            with h5py.File(hdf5_path, "r") as hf:
                self.trajectory_ids = [k for k in hf.keys() if k.startswith("trajectory_")]
            self.trajectory_ids.sort()
        else:
            import os
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

    def __len__(self):
        return len(self.slice_indices)

    def _get_trajectory(self, traj_id):
        if self.is_consolidated:
            return h5py.File(self.hdf5_path, "r")[traj_id]
        f = h5py.File(traj_id, "r")
        return f["data"] if "data" in f else f

    def __getitem__(self, idx):
        traj_id, start_idx = self.slice_indices[idx]
        end_idx = start_idx + self.segment_length

        file_path = self.hdf5_path if self.is_consolidated else traj_id
        with h5py.File(file_path, "r") as hf:
            traj = hf[traj_id] if self.is_consolidated else (hf["data"] if "data" in hf else hf)

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
