"""
Evaluate a trained DINO world model on the tactile dataset.

Loads full episodes, runs autoregressive rollout with the world model (conditioned on
images/embeddings, states, actions), decodes predicted latents to images, and reports
MSE loss. Optionally saves gt/pred/diff videos to wandb.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from tactile_dataset import TactileTrajectoryDataset, CAMERA_CONFIG, load_full_episode
from dino_models import TactileVideoTransformer, normalize_acs
from dino_decoder import VQVAE

# DINOv3 ViT-B/16: 196 patches (14x14), 768 dim. AnyTouch: 512 dim.
CAMERA_EMB_DIMS = {"camera_0": 768, "camera_1": 768, "camera_2": 512}
PATCH_SIDE_DINOV3 = 14


def ensure_patch_grid(embd: torch.Tensor, target_side: int) -> torch.Tensor:
    """Reshape or interpolate patch embeddings to target_side x target_side grid."""
    B, T, N, D = embd.shape
    side = int(N ** 0.5)
    if side * side != N:
        target_n = target_side * target_side
        if N >= target_n:
            embd = embd[:, :, :target_n, :]
        else:
            pad = torch.zeros(B, T, target_n - N, D, device=embd.device, dtype=embd.dtype)
            embd = torch.cat([embd, pad], dim=2)
        N = target_n
        side = target_side

    if side != target_side:
        embd = rearrange(embd, "b t (h w) d -> (b t) d h w", h=side, w=side)
        embd = nn.functional.interpolate(
            embd, size=(target_side, target_side), mode="bilinear", align_corners=False
        )
        embd = rearrange(embd, "(b t) d h w -> b t (h w) d", h=target_side, w=target_side)
    return embd


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove _orig_mod. prefix from keys when checkpoint was saved with torch.compile."""
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k.replace(prefix, "", 1): v for k, v in state_dict.items()}


def _find_decoder_path(decoder_dir: str, base_name: str):
    """Try base, _adv, _last suffixes."""
    for suffix in ("", "_adv", "_last"):
        path = os.path.join(decoder_dir, f"{base_name}{suffix}.pth")
        if os.path.isfile(path):
            return path
    return None


def load_decoders(decoder_dir: str, predict_cameras: list, device: str) -> dict:
    """Load decoders for each camera. Vision cams share decoder_camera_0_camera_1.pth."""
    decoders = {}
    vision_path = _find_decoder_path(decoder_dir, "decoder_camera_0_camera_1")
    tactile_path = _find_decoder_path(decoder_dir, "decoder_camera_2")
    if vision_path:
        dec = VQVAE(emb_dim=768).to(device)
        dec.load_state_dict(torch.load(vision_path, map_location=device))
        dec.eval()
        for c in predict_cameras:
            if CAMERA_EMB_DIMS.get(c, 768) == 768:
                decoders[c] = dec
    if tactile_path and "camera_2" in predict_cameras:
        dec = VQVAE(emb_dim=512).to(device)
        dec.load_state_dict(torch.load(tactile_path, map_location=device))
        dec.eval()
        decoders["camera_2"] = dec
    return decoders


def create_episode_videos(
    transition,
    decoders: dict,
    eval_ds: TactileTrajectoryDataset,
    predict_cameras: list,
    condition_cameras: list,
    args,
    device: str,
) -> dict:
    """
    Load full episodes, run autoregressive rollout, decode to images, create gt/pred/diff videos.
    Returns dict of wandb.Video for each episode.
    """
    if not decoders:
        return {}
    videos = {}
    traj_ids = eval_ds.trajectory_ids[: args.num_episode_videos]

    for ep_idx, traj_id in enumerate(tqdm(traj_ids, desc="Episode videos")):
        ep_data = load_full_episode(
            args.hdf5_path,
            traj_id,
            predict_cameras + [c for c in condition_cameras if c not in predict_cameras],
            is_consolidated=not args.no_consolidated,
        )
        if "states" not in ep_data or "actions" not in ep_data:
            continue

        T_total = ep_data[f"{predict_cameras[0]}_embd"].shape[0]
        T = min(T_total, args.max_episode_len)
        H = args.segment_length - 1
        if T <= H:
            continue

        # Load GT images
        all_gt_imgs = []
        for cam in predict_cameras:
            gt_img = np.asarray(ep_data[f"{cam}_image"][:T], dtype=np.float32)
            if gt_img.ndim == 3:
                gt_img = np.stack([gt_img] * 3, axis=-1)
            if gt_img.max() > 1.0:
                gt_img = gt_img / 255.0
            all_gt_imgs.append(np.clip(gt_img, 0, 1))

        # Build initial inputs
        states = torch.tensor(ep_data["states"][:T], dtype=torch.float32, device=device).unsqueeze(0)
        actions = torch.tensor(ep_data["actions"][:T], dtype=torch.float32, device=device).unsqueeze(0)
        if actions.shape[-1] > 7:
            actions = actions[..., :7]
        actions = normalize_acs(actions, device)

        inputs = {}
        for c in condition_cameras:
            emb = torch.tensor(ep_data[f"{c}_embd"][:T], dtype=torch.float32, device=device).unsqueeze(0)
            inputs[c] = ensure_patch_grid(emb, PATCH_SIDE_DINOV3)[:, :H]
        inp_states = states[:, :H]
        inp_acs = actions[:, :H]

        pred_imgs = [[all_gt_imgs[i][t].copy() for t in range(T)] for i in range(len(predict_cameras))]

        # Autoregressive rollout
        with torch.no_grad():
            for k in range(T - H):
                preds, pred_state = transition(inputs, inp_states, inp_acs)
                next_embds = {c: preds[c][:, -1:] for c in predict_cameras}
                next_state = pred_state[:, -1:]

                for i, cam in enumerate(predict_cameras):
                    if cam in decoders:
                        emb = next_embds[cam]
                        pred_img, _ = decoders[cam](emb)
                        pred_img = rearrange(pred_img, "(b t) c h w -> b t c h w", t=1)
                        pred_img = pred_img[0, 0].cpu().numpy().transpose(1, 2, 0)
                        pred_imgs[i][H + k] = np.clip(pred_img, 0, 1)

                for c in condition_cameras:
                    if c in predict_cameras:
                        inputs[c] = torch.cat([inputs[c][:, 1:], next_embds[c]], dim=1)
                    else:
                        next_gt = torch.tensor(
                            ep_data[f"{c}_embd"][H + k], dtype=torch.float32, device=device
                        ).unsqueeze(0).unsqueeze(0)
                        inputs[c] = torch.cat([inputs[c][:, 1:], ensure_patch_grid(next_gt, PATCH_SIDE_DINOV3)], dim=1)
                inp_states = torch.cat([inp_states[:, 1:], next_state], dim=1)
                if H + k + 1 < actions.shape[1]:
                    inp_acs = torch.cat([inp_acs[:, 1:], actions[:, H + k : H + k + 1]], dim=1)

        # Build video: gt | pred | diff per frame, cameras concatenated horizontally
        frames = []
        for t in range(T):
            gt_row = np.concatenate([all_gt_imgs[i][t] for i in range(len(predict_cameras))], axis=1)
            pred_row = np.concatenate([pred_imgs[i][t] for i in range(len(predict_cameras))], axis=1)
            diff_row = np.abs(gt_row.astype(np.float32) - pred_row.astype(np.float32))
            if diff_row.shape[-1] == 1:
                diff_row = np.repeat(diff_row, 3, axis=-1)
            frame = np.vstack([gt_row, pred_row, diff_row])
            frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

        video = np.stack(frames, axis=0)
        video = np.transpose(video, (0, 3, 1, 2))  # (T, C, H, W)
        videos[f"episode_{ep_idx}"] = wandb.Video(video, fps=args.video_fps, format="mp4")

    return videos


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate trained DINO world model on tactile dataset (autoregressive rollout + decoder)"
    )
    p.add_argument(
        "--wm_checkpoint",
        type=str,
        required=True,
        help="Path to world model checkpoint (.pth)",
    )
    p.add_argument(
        "--decoder_dir",
        type=str,
        required=True,
        help="Directory with decoder checkpoints (decoder_camera_0_camera_1.pth, decoder_camera_2.pth)",
    )
    p.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="Path to consolidated HDF5 or directory with .hdf5 files",
    )
    p.add_argument(
        "--cameras",
        type=str,
        default="camera_0,camera_1,camera_2",
        help="Comma-separated cameras (must match world model training)",
    )
    p.add_argument(
        "--condition_cameras",
        type=str,
        default=None,
        help="Cameras to condition on (default: same as --cameras)",
    )
    p.add_argument(
        "--predict_cameras",
        type=str,
        default=None,
        help="Cameras to predict (default: same as --cameras)",
    )
    p.add_argument(
        "--segment_length",
        type=int,
        default=4,
        help="Segment length used during world model training (BL)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    p.add_argument(
        "--num_test",
        type=int,
        default=100,
        help="Number of trajectories for test split",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="If set, save sample gt/pred image pairs",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=8,
    )
    p.add_argument(
        "--no_consolidated",
        action="store_true",
        help="hdf5_path is a directory of individual .hdf5 files",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Log results and videos to Weights & Biases",
    )
    p.add_argument(
        "--wandb_project",
        type=str,
        default="dino-wm-tactile",
    )
    p.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
    )
    p.add_argument(
        "--num_episode_videos",
        type=int,
        default=3,
        help="Number of full episodes to log as gt/pred/diff videos",
    )
    p.add_argument(
        "--max_episode_len",
        type=int,
        default=100,
        help="Max timesteps per episode video",
    )
    p.add_argument(
        "--video_fps",
        type=int,
        default=10,
    )
    return p.parse_args()


def main():
    args = parse_args()

    # VQVAE Quantize needs dist_fn for training; no-op for eval
    import dino_decoder as _dino_dec_mod
    if not hasattr(_dino_dec_mod, "dist_fn"):
        _dino_dec_mod.dist_fn = type("DistFn", (), {"all_reduce": lambda x: None})()

    cameras = [c.strip() for c in args.cameras.split(",")]
    condition_cameras = cameras if not args.condition_cameras else [c.strip() for c in args.condition_cameras.split(",")]
    predict_cameras = cameras if not args.predict_cameras else [c.strip() for c in args.predict_cameras.split(",")]

    for c in cameras + condition_cameras + predict_cameras:
        if c not in CAMERA_CONFIG:
            raise ValueError(f"Unknown camera: {c}. Choose from {list(CAMERA_CONFIG.keys())}")

    device = torch.device(args.device)
    H = args.segment_length - 1

    # Load world model
    camera_dims = {c: CAMERA_EMB_DIMS.get(c, 768) for c in cameras}
    transition = TactileVideoTransformer(
        cameras=cameras,
        camera_dims=camera_dims,
        condition_cameras=condition_cameras,
        predict_cameras=predict_cameras,
        common_dim=384,
        ac_dim=10,
        state_dim=8,
        patches_per_frame=PATCH_SIDE_DINOV3 * PATCH_SIDE_DINOV3,
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=H,
        dropout=0.1,
    ).to(device)
    wm_state = torch.load(args.wm_checkpoint, map_location=device)
    transition.load_state_dict(_strip_compile_prefix(wm_state))
    transition.eval()

    # Load decoders
    decoders = load_decoders(args.decoder_dir, predict_cameras, device)
    if not decoders:
        raise ValueError(f"No decoders found in {args.decoder_dir}")
    print(f"Loaded decoders for: {list(decoders.keys())}")

    is_consolidated = not args.no_consolidated
    eval_ds = TactileTrajectoryDataset(
        args.hdf5_path,
        cameras=cameras,
        segment_length=args.segment_length,
        split="test",
        num_test=args.num_test,
        is_consolidated=is_consolidated,
        seed=args.seed,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    total_loss = 0.0
    n_batches = 0
    samples_saved = 0
    wandb_images = {} if (args.wandb and HAS_WANDB) else None

    if args.wandb and HAS_WANDB:
        run_name = args.wandb_run_name or f"eval_wm_{'+'.join(cameras)}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    with torch.no_grad():
        for data in tqdm(eval_loader, desc="Evaluating"):
            # Build inputs: (B, H, N, D) for each condition camera
            cond_inputs = {}
            for cam in condition_cameras:
                embd = data[f"{cam}_embd"].to(device)
                embd = ensure_patch_grid(embd, PATCH_SIDE_DINOV3)
                cond_inputs[cam] = embd[:, :-1]

            states = data["state"].to(device)[:, :-1]
            acs_raw = data["action"].to(device)
            if acs_raw.shape[-1] > 7:
                acs_raw = acs_raw[..., :7]
            acs = normalize_acs(acs_raw, device)[:, :-1]

            preds, pred_state = transition(cond_inputs, states, acs)

            # Decode predicted next-frame embeddings and compare to GT
            batch_loss = 0.0
            preds_per_cam = {}
            targets_per_cam = {}
            for cam in predict_cameras:
                if cam not in decoders:
                    continue
                pred_embd = preds[cam][:, -1:]  # (B, 1, N, D)
                target_embd = data[f"{cam}_embd"].to(device)[:, 1:2]
                target_img = data[f"{cam}_image"].to(device)[:, 1:2]
                target_img = target_img.permute(0, 1, 4, 2, 3)
                if target_img.shape[2] != 3:
                    target_img = target_img[:, :, :3]

                pred_img, _ = decoders[cam](pred_embd)
                pred_img = rearrange(pred_img, "(b t) c h w -> b t c h w", t=1)
                batch_loss += nn.MSELoss()(pred_img, target_img).item()
                preds_per_cam[cam] = pred_img
                targets_per_cam[cam] = target_img

            batch_loss /= len([c for c in predict_cameras if c in decoders])
            total_loss += batch_loss
            n_batches += 1

            if (args.output_dir or wandb_images is not None) and samples_saved < args.num_samples:
                B = pred_img.shape[0]
                if args.output_dir:
                    os.makedirs(args.output_dir, exist_ok=True)
                for i in range(min(B, args.num_samples - samples_saved)):
                    for cam in predict_cameras:
                        if cam not in preds_per_cam:
                            continue
                        gt = targets_per_cam[cam][i, 0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                        pred_img_np = preds_per_cam[cam][i, 0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                        if args.output_dir:
                            gt_path = os.path.join(args.output_dir, f"sample_{samples_saved}_{cam}_gt.png")
                            pred_path = os.path.join(args.output_dir, f"sample_{samples_saved}_{cam}_pred.png")
                            Image.fromarray((gt * 255).astype(np.uint8)).save(gt_path)
                            Image.fromarray((pred_img_np * 255).astype(np.uint8)).save(pred_path)
                        if wandb_images is not None and samples_saved == 0:
                            wandb_images[f"eval_{cam}_gt"] = wandb.Image(gt, caption=f"{cam} ground truth")
                            wandb_images[f"eval_{cam}_pred"] = wandb.Image(pred_img_np, caption=f"{cam} predicted")
                    samples_saved += 1
                    if samples_saved >= args.num_samples:
                        break

    mean_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"\nTest MSE Loss (1-step pred, decoded): {mean_loss:.6f}")
    if args.output_dir:
        print(f"Saved {samples_saved} sample images to {args.output_dir}")

    # Episode rollout videos
    if args.wandb and HAS_WANDB and args.num_episode_videos > 0:
        episode_videos = create_episode_videos(
            transition=transition,
            decoders=decoders,
            eval_ds=eval_ds,
            predict_cameras=predict_cameras,
            condition_cameras=condition_cameras,
            args=args,
            device=device,
        )
        wandb.log({
            "eval_mse_loss": mean_loss,
            **(wandb_images or {}),
            **episode_videos,
        })
    elif args.wandb and HAS_WANDB:
        wandb.log({"eval_mse_loss": mean_loss, **(wandb_images or {})})


if __name__ == "__main__":
    main()
