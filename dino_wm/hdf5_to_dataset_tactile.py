"""
HDF5 trajectories → embeddings and optional consolidation.

Streams (two modalities, two HDF5 keys):
- ``camera_0``: RGB (e.g. ZED Mini) → DINOv3 ViT-S/16 CLS + patch tokens.
- ``camera_1``: GelSight (or other) RGB video → AnyTouch TactileVideoMAE on a sliding
  window of the last ``num_tactile_frames`` frames per timestep (CLS + patch embeddings).

The AnyTouch2 code (``model.tactile_mae``) must be importable on ``PYTHONPATH``.
"""

import argparse
import copy
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as TF
import json
# Tactile preprocessing constants (from AnyTouch infer_touch_text_similarity.py)
OFFSET = 130.0 / 255.0
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

SENSOR_NAME_TO_ID = {
    "gelsight": 0,
    "digit": 1,
    "gelslim": 2,
    "gelsight_mini": 3,
    "duragel": 4,
    "dm": 5,
    "universal": -1,
}



DINO_crop = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.1)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Wrist cam transforms (camera_0)
resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
DINO_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


def resize_images_to_224(images, key):
    """Resize a trajectory of images to 224×224 for consolidation.

    ``camera_0`` uses the same ToTensor pipeline as ``resize_transform``; ``camera_1``
    (tactile) uses a plain PIL resize. ``key`` must be ``"camera_0"`` or ``"camera_1"``.
    """
    resized = []
    for i in range(len(images)):
        img = images[i]
        if key == "camera_0":  # wrist camera
            img_tensor = resize_transform(img.astype(np.uint8))
        else:  # camera_1 tactile - simple resize
            img_tensor = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])(img.astype(np.uint8))
        resized.append(img_tensor.numpy().transpose(1, 2, 0))
    return np.stack(resized)


def eef_pose_to_state(T, gripper):
    """End-effector pose matrix and scalar gripper → 8-D state (xyz, quat xyzw, gripper)."""
    x, y, z = T[:3, 3]
    rotation_matrix = T[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()
    return np.concatenate(([x, y, z], quat, [gripper]))


# ── Tactile model loading (from AnyTouch) ───────────────────────────────────

def _load_tactile_model(checkpoint_path, sensor, num_frames=4, stride=2, device="cuda"):
    """Load AnyTouch ``TactileVideoMAE`` weights from ``checkpoint_path``."""
    # if anytouch_path not in sys.path:
    #     sys.path.insert(0, anytouch_path)

    from transformers import AutoConfig
    from model.tactile_mae import TactileVideoMAE
    
    ## check the parent directory of the checkpoint path (string)
    config_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    config = AutoConfig.from_pretrained(os.path.join(config_dir, "CLIP-B-16", "config.json"))
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

    model = TactileVideoMAE(fake_args, config, num_frames, 1)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    new_ckpt = {}
    for key, val in ckpt.items():
        if "touch_mae_model" in key and "decoder" not in key and "mask_token" not in key:
            new_ckpt[key.replace("touch_mae_model.", "")] = copy.deepcopy(val)

    if not new_ckpt:
        for key, val in ckpt.get("model", ckpt).items():
            if key.startswith("tactile_model."):
                new_ckpt[key.replace("tactile_model.", "")] = copy.deepcopy(val)

    for k, v in model.named_parameters():
        if k not in new_ckpt:
            new_ckpt[k] = v

    model.load_state_dict(new_ckpt, strict=True)
    return model.to(device).eval(), sensor_id


# ── Tactile preprocessing (from AnyTouch infer_touch_text_similarity) ─────────

def preprocess_tactile_window(
    window_frames: list[np.ndarray],
    bg_image: np.ndarray,
) -> torch.Tensor:
    """
    Preprocess one tactile window for AnyTouch (background subtraction + CLIP norm).

    Args:
        window_frames: ``num_frames`` RGB arrays ``(H, W, 3)`` uint8.
        bg_image: Background (no-contact) frame, same layout.

    Returns:
        Tensor of shape ``(len(window_frames), 3, 224, 224)``.
    """
    to_tensor = transforms.ToTensor()
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224), antialias=True),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    bg_tensor = to_tensor(Image.fromarray(bg_image).convert("RGB"))
    tensors = [to_tensor(Image.fromarray(f).convert("RGB")) for f in window_frames]
    stacked = torch.stack(tensors, dim=0)
    imgs = stacked - bg_tensor.unsqueeze(0) + OFFSET
    imgs = torch.clamp(imgs, 0.0, 1.0)
    imgs = transform(imgs)
    return imgs  # (4, 3, 224, 224)


def build_tactile_windows_per_timestep(
    tactile_frames: np.ndarray,
    num_frames: int = 4,
) -> list[torch.Tensor]:
    """
    For each timestep ``t``, take the last ``num_frames`` tactile frames (pad with frame 0).

    Args:
        tactile_frames: ``(T, H, W, C)`` uint8.
        num_frames: Window length (default 4).

    Returns:
        List of length ``T``, each element a ``(num_frames, 3, 224, 224)`` tensor.
    """
    T = tactile_frames.shape[0]
    bg_image = tactile_frames[0]

    windows = []
    for t in range(T):
        # Last num_frames: [t-3, t-2, t-1, t], pad with frame 0 when needed
        indices = [max(0, t - num_frames + 1 + i) for i in range(num_frames)]
        window_frames = [tactile_frames[i] for i in indices]
        window_tensor = preprocess_tactile_window(window_frames, bg_image)
        windows.append(window_tensor)

    return windows


@torch.no_grad()
def extract_tactile_embeddings_batch(
    model,
    windows: list[torch.Tensor],
    sensor_id: int,
    device: torch.device,
    aggregate: str = "cls",
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalized CLS and patch embeddings for each window.

    Returns:
        ``(cls_emb, patch_emb)`` with shapes ``(N, D)`` and ``(N, P, D)`` (model-dependent ``D``).
    """
    all_cls_embeds = []
    all_patch_embeds = []
    for i in range(0, len(windows), batch_size):
        batch = torch.stack(windows[i : i + batch_size]).to(device)  # (B, T, C, H, W)
        sensors = torch.full((batch.shape[0],), sensor_id, dtype=torch.long, device=device)

        outputs = model(batch, sensor_type=sensors, probe=False, get_cls=False)

        # if aggregate == "cls":
        cls_emb = outputs[:, 0, :]
        # else:
        patch_emb = outputs[:, 6:, :]

        cls_emb = F.normalize(cls_emb, dim=-1)
        patch_emb = F.normalize(patch_emb, dim=-1)
        all_cls_embeds.append(cls_emb.cpu().float().numpy())
        all_patch_embeds.append(patch_emb.cpu().float().numpy())

    return np.concatenate(all_cls_embeds, axis=0), np.concatenate(all_patch_embeds, axis=0)


# ── Main preprocessing ──────────────────────────────────────────────────────

def preprocess(
    demo_paths: list[str],
    checkpoint_path: str,
    sensor: str = "digit",
    num_tactile_frames: int = 4,
    device: str = "cuda:0",
    output_json_file: str = None,
):
    """Encode each trajectory HDF5: DINO on ``camera_0``, AnyTouch on ``camera_1``, write stats JSON."""
    # dino = torch.hub.load('/home/yilin/Projects/flow_policy/git-packages/dinov3', 
    # 'dinov3_vitb16', source='local', weights = '/home/yilin/.cache/torch/hub/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth').to(device)
    dino = torch.hub.load('/home/yilin/Projects/flow_policy/git-packages/dinov3',
     'dinov3_vits16', source='local', weights = '/home/yilin/Projects/latent-tactile/dinov3/dinov3-vits16-pretrain-lvd1689m/dinov3_vits16_pretrain_lvd1689m-08c60483.pth').to(device)
    tactile_model, sensor_id = _load_tactile_model(
    checkpoint_path, sensor,
        num_frames=num_tactile_frames, device=device
    )

    hdf5_files = []
    for demo_path in demo_paths:
        hdf5_files.extend(
            os.path.join(demo_path, f) for f in os.listdir(demo_path) if f.endswith('.hdf5')
        )

    all_acs = []
    transitions = 0
    all_states = []
    for i, hdf5_file in tqdm(enumerate(hdf5_files), desc="Loading expert data",
                              total=len(hdf5_files), ncols=0, leave=False):
        with h5py.File(hdf5_file, "r+") as f:
            if 'data' not in f:
                data_group = f
            else:
                data_group = f["data"]
            if "camera_0" not in data_group:
                print(f"Skipping {hdf5_file} due to missing camera_0.")
                continue
            if "camera_1" not in data_group:
                print(f"Skipping {hdf5_file} due to missing camera_1 (tactile).")
                continue

            # Truncation: max 400 steps; label=='4' → max 80 steps
            label = str(data_group.attrs.get("label", ""))
            max_len = 80 if label == "4" else 400
            traj_len = data_group["actions"].shape[0]
            T = min(traj_len, max_len)

            actions = data_group["actions"][:T]
            cam_0 = data_group["camera_0"][:T]
            cam_tactile = data_group["camera_1"][:T]

            ee_states = data_group["ee_states"][:T]
            gripper_states = data_group["gripper_states"][:T]

            # Check for partially saved embeddings (storage full during save) and delete so we recompute
            expected_len = actions.shape[0]
            embd_keys = [
                "cam_0_cls_embd", "cam_0_patch_embd",
                "cam_tactile_cls_embd", "cam_tactile_patch_embd",
            ]
            for key in embd_keys:
                if key in data_group and data_group[key].shape[0] != expected_len:
                    print(f"Rewriting {key} in {hdf5_file}: embd len {data_group[key].shape[0]} != actions {expected_len}")
                    del data_group[key]

            cam_0_patch_embds = []
            cam_0_cls_embds = []
            cam_tactile_cls_embds = []
            cam_tactile_patch_embds = []
            states = []

            # DINO on RGB (camera_0) only
            for t in range(actions.shape[0]):
                if "cam_0_patch_embd" not in data_group:
                    rs_img = cam_0[t]
                    img_PIL = Image.fromarray(np.uint8(rs_img)).convert('RGB')
                    img_tensor = DINO_transform(img_PIL).to(device)
                    with torch.no_grad():
                        cls_emb = dino.forward_features(img_tensor.unsqueeze(0))['x_norm_clstoken'].squeeze().cpu().numpy()
                        patch_emb = dino.forward_features(img_tensor.unsqueeze(0))['x_norm_patchtokens'].squeeze().cpu().numpy()
                    cam_0_cls_embds.append(cls_emb)
                    cam_0_patch_embds.append(patch_emb)

                ee_state = eef_pose_to_state(ee_states[t].reshape(4, 4).T, gripper_states[t])
                states.append(ee_state)

            # Tactile embeddings: last 4 frames per timestep
            if "cam_tactile_patch_embd" not in data_group:
                tactile_windows = build_tactile_windows_per_timestep(
                    cam_tactile, num_frames=num_tactile_frames
                )
                cam_tactile_cls_embds, cam_tactile_patch_embds = extract_tactile_embeddings_batch(
                    tactile_model, tactile_windows, sensor_id, torch.device(device)
                )

            # Save to HDF5
            if "cam_0_cls_embd" not in data_group:
                data_group.create_dataset("cam_0_cls_embd", data=np.stack(cam_0_cls_embds))
            if "cam_0_patch_embd" not in data_group:
                data_group.create_dataset("cam_0_patch_embd", data=np.stack(cam_0_patch_embds))
            if "cam_tactile_cls_embd" not in data_group:
                data_group.create_dataset("cam_tactile_cls_embd", data=np.stack(cam_tactile_cls_embds))
            if "cam_tactile_patch_embd" not in data_group:
                data_group.create_dataset("cam_tactile_patch_embd", data=np.stack(cam_tactile_patch_embds))
            if "states" not in data_group:
                data_group.create_dataset("states", data=np.stack(states))

            all_acs.extend(actions)
            all_states.extend(states)
            transitions += len(actions)

    all_acs = np.array(all_acs)
    all_states = np.array(all_states)
    assert all_states.shape[0] == all_acs.shape[0], "Number of states and actions must match"
    print('max', np.max(all_acs, axis=0))
    print('min', np.min(all_acs, axis=0))
    print('total transitions:', transitions)
    max_acs, min_acs, total_transitions, num_trajs = np.max(all_acs, axis=0), np.min(all_acs, axis=0), transitions, i+1
    mean_acs = np.mean(all_acs, axis=0)
    std_acs = np.std(all_acs, axis=0)
    info = {'max_acs': max_acs.tolist(), 'min_acs': min_acs.tolist(), 'mean_acs': mean_acs.tolist(), 'std_acs':std_acs.tolist(), 'num_transitions': int(total_transitions), 'num_trajectories': int(num_trajs)}
    ## also computing the states stats
    max_states, min_states, total_transitions, num_trajs = np.max(all_states, axis=0), np.min(all_states, axis=0), transitions, i+1
    mean_states = np.mean(all_states, axis=0)
    std_states = np.std(all_states, axis=0)
    info['max_states'] = max_states.tolist()
    info['min_states'] = min_states.tolist()
    info['mean_states'] = mean_states.tolist()
    info['std_states'] = std_states.tolist()
    with open(output_json_file, "w") as f:
        json.dump(info, f)  # indent=4 makes it pretty-printed



def compute_norm_stats_only(
    consolidated_hdf5_path: str,
    output_json_file: str,
):
    """Aggregate min/max/mean/std for actions and EE states across a consolidated HDF5."""
    with h5py.File(consolidated_hdf5_path, "r") as hf:
        traj_ids = [k for k in hf.keys() if k.startswith("trajectory_")]
    traj_ids.sort()

    all_acs = []
    all_states = []
    transitions = 0
    num_trajs = 0
    with h5py.File(consolidated_hdf5_path, "r") as hf:
        for traj_id in tqdm(traj_ids, desc="Computing norm stats", ncols=0):
            traj = hf[traj_id]
            if "actions" not in traj:
                print(f"Skipping {traj_id}: no actions")
                continue
            if "ee_states" not in traj or "gripper_states" not in traj:
                print(f"Skipping {traj_id}: no ee_states or gripper_states")
                continue

            actions = traj["actions"][...]
            ee_states = traj["ee_states"][...]
            gripper_states = traj["gripper_states"][...]
            T = len(actions)

            for t in range(T):
                ee_state = eef_pose_to_state(ee_states[t].reshape(4, 4).T, gripper_states[t])
                all_states.append(ee_state)
            all_acs.extend(actions)
            transitions += T
            num_trajs += 1

    if not all_acs or not all_states:
        raise ValueError("No valid trajectories found; all files were skipped or empty")

    all_acs = np.array(all_acs)
    all_states = np.array(all_states)
    assert all_states.shape[0] == all_acs.shape[0], "Number of states and actions must match"

    print("max", np.max(all_acs, axis=0))
    print("min", np.min(all_acs, axis=0))
    print("total transitions:", transitions)

    max_acs = np.max(all_acs, axis=0)
    min_acs = np.min(all_acs, axis=0)
    mean_acs = np.mean(all_acs, axis=0)
    std_acs = np.std(all_acs, axis=0)
    max_states = np.max(all_states, axis=0)
    min_states = np.min(all_states, axis=0)
    mean_states = np.mean(all_states, axis=0)
    std_states = np.std(all_states, axis=0)

    info = {
        "max_acs": max_acs.tolist(),
        "min_acs": min_acs.tolist(),
        "mean_acs": mean_acs.tolist(),
        "std_acs": std_acs.tolist(),
        "max_states": max_states.tolist(),
        "min_states": min_states.tolist(),
        "mean_states": mean_states.tolist(),
        "std_states": std_states.tolist(),
        "num_transitions": int(transitions),
        "num_trajectories": num_trajs,
    }
    with open(output_json_file, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Norm stats saved to {output_json_file}")


def convert_hdf5_to_consolidated_hdf5(hdf5_dirs: list[str], output_hdf5_file: str):
    """Merge per-file trajectory HDF5s into one file under ``trajectory_*`` groups.

    Applies the same step cap as training (400, or 80 when label ``"4"``) and resizes
    ``camera_0`` / ``camera_1`` frames to 224×224.
    """
    # Collect (dir, filename) pairs, then sort by (dir, filename) for deterministic order
    all_files = []
    for hdf5_dir in hdf5_dirs:
        for f in os.listdir(hdf5_dir):
            if f.endswith(".hdf5"):
                all_files.append((hdf5_dir, f))
    all_files.sort(key=lambda x: (x[0], x[1]))

    with h5py.File(output_hdf5_file, "w") as hf_out:
        for i, (hdf5_dir, hdf5_file) in enumerate(all_files):
            file_path = os.path.join(hdf5_dir, hdf5_file)
            with h5py.File(file_path, "r") as hf_in:
                group = hf_out.create_group(f"trajectory_{i}")

                if "data" in hf_in:
                    data_group = hf_in["data"]
                    if "config" in data_group.attrs:
                        group.attrs["config"] = data_group.attrs["config"]

                    # Truncation: max 400 steps; label=='4' → max 80 steps
                    label = str(data_group.attrs.get("label", ""))
                    max_len = 80 if label == "4" else 400

                    for key in data_group.keys():
                        data = data_group[key][...]
                        # Truncate trajectory dimension (first axis)
                        if data.ndim >= 1 and data.shape[0] > max_len:
                            data = data[:max_len]
                        if key in ("camera_0", "camera_1"):
                            data = resize_images_to_224(data, key)
                        group.create_dataset(key, data=data)
                else:
                    for key in hf_in.keys():
                        hf_in.copy(hf_in[key], group)

            print(f"Copied {hdf5_file} → trajectory_{i}")


def parse_args():
    p = argparse.ArgumentParser(description="HDF5 to dataset with DINO (camera_0) + AnyTouch (camera_1)")
    p.add_argument("--hdf5_dir", type=str, nargs="+", default=None,
                  help="Directory (or directories) with trajectory HDF5 files (required unless --norm_stats_only)")
    p.add_argument("--input_hdf5", type=str, default=None,
                  help="Consolidated HDF5 path (required for --norm_stats_only)")
    p.add_argument("--output_hdf5", type=str, default=None,
                  help="Output consolidated HDF5 path (default: first hdf5_dir/consolidated.h5)")
    p.add_argument("--output_json", type=str, default=None,
                  help="Output JSON path for norm stats")
    # p.add_argument("--anytouch_path", type=str, default=None,
    #               help="Path to AnyTouch2 repo (default: $ANYTOUCH_PATH or ../AnyTouch2)")
    p.add_argument("--checkpoint", type=str, default=None,
                  help="Path to AnyTouch checkpoint (required unless --norm_stats_only)")
    p.add_argument("--sensor", type=str, default="gelsight_mini",
                  choices=list(SENSOR_NAME_TO_ID.keys()))
    p.add_argument("--num_tactile_frames", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip_consolidate", action="store_true",
                  help="Skip consolidation step")
    p.add_argument("--norm_stats_only", action="store_true",
                  help="Only compute action/state norm stats; skip embedding encoding and consolidation")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.norm_stats_only:
        if not args.input_hdf5:
            raise ValueError("--input_hdf5 required for --norm_stats_only (path to consolidated HDF5)")
        output_json = args.output_json or os.path.join(
            os.path.dirname(os.path.abspath(args.input_hdf5)), "norm_stats.json"
        )
        compute_norm_stats_only(args.input_hdf5, output_json)
    else:
        if not args.hdf5_dir:
            raise ValueError("--hdf5_dir required (omit --norm_stats_only to run full pipeline)")
        if not args.checkpoint:
            raise ValueError("--checkpoint required unless --norm_stats_only")

        output_json = args.output_json or os.path.join(args.hdf5_dir[0], "norm_stats.json")
        preprocess(
            args.hdf5_dir,
            checkpoint_path=args.checkpoint,
            sensor=args.sensor,
            num_tactile_frames=args.num_tactile_frames,
            device=args.device,
            output_json_file=output_json,
        )

        if not args.skip_consolidate:
            output_path = args.output_hdf5 or os.path.join(args.hdf5_dir[0], "consolidated.h5")
            convert_hdf5_to_consolidated_hdf5(args.hdf5_dir, output_path)
