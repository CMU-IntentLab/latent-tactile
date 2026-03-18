"""
Evaluate a trained DINO world model on the tactile dataset.

Loads full episodes, runs autoregressive rollout with the world model (conditioned on
images/embeddings, states, actions), decodes predicted latents to images, and reports
MSE loss. Optionally saves gt/pred/diff videos to wandb and/or gt/pred embeddings to disk.
"""

import argparse
import os
from pathlib import Path

import cv2
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
from dino_models import TactileVideoTransformer, NormStats
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
    norm_stats: NormStats,
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
        start_idx = 0
        if args.start_after_gripper_close:
            raw_actions = np.asarray(ep_data["actions"][:T_total])
            if raw_actions.shape[-1] <= args.gripper_idx:
                continue
            gripper_vals = raw_actions[:, args.gripper_idx]
            closed_mask = gripper_vals >= args.gripper_closed_threshold
            closed_indices = np.where(closed_mask)[0]
            if len(closed_indices) == 0:
                continue
            start_idx = int(closed_indices[0])
            print(f"Start index: {start_idx}")
            if T_total - start_idx < args.segment_length:
                continue
        T = min(T_total - start_idx, args.max_episode_len)
        H = args.segment_length - 1
        if T <= H:
            continue

        # Load GT images (sliced from start_idx)
        all_gt_imgs = []
        for cam in predict_cameras:
            gt_img = np.asarray(ep_data[f"{cam}_image"][start_idx : start_idx + T], dtype=np.float32)
            if gt_img.ndim == 3:
                gt_img = np.stack([gt_img] * 3, axis=-1)
            if gt_img.max() > 1.0:
                gt_img = gt_img / 255.0
            all_gt_imgs.append(np.clip(gt_img, 0, 1))

        # Build full episode tensors (sliced from start_idx)
        states = torch.tensor(ep_data["states"][start_idx : start_idx + T], dtype=torch.float32, device=device).unsqueeze(0)
        if args.normalize_states:
            states = norm_stats.normalize_states(states)
        actions = torch.tensor(ep_data["actions"][start_idx : start_idx + T], dtype=torch.float32, device=device).unsqueeze(0)
        if actions.shape[-1] > 7:
            actions = actions[..., :7]
        actions = norm_stats.normalize_acs(actions)

        # Pre-load all embeddings for condition cameras (for GT resets)
        all_embds = {}
        for c in condition_cameras:
            emb = torch.tensor(ep_data[f"{c}_embd"][start_idx : start_idx + T], dtype=torch.float32, device=device).unsqueeze(0)
            all_embds[c] = ensure_patch_grid(emb, PATCH_SIDE_DINOV3)

        # Pre-load GT embeddings for predict cameras (for save_embeds)
        gt_embds_per_cam = {}
        if args.save_embeds:
            for c in predict_cameras:
                emb = torch.tensor(ep_data[f"{c}_embd"][start_idx : start_idx + T], dtype=torch.float32, device=device)
                emb = ensure_patch_grid(emb.unsqueeze(0), PATCH_SIDE_DINOV3).squeeze(0)
                gt_embds_per_cam[c] = emb.cpu().numpy()

        pred_imgs = [[all_gt_imgs[i][t].copy() for t in range(T)] for i in range(len(predict_cameras))]
        pred_embds_per_cam = {c: [] for c in predict_cameras} if args.save_embeds else None
        if pred_embds_per_cam is not None:
            # Pre-fill with GT for first H frames (context window, not predicted)
            for c in predict_cameras:
                for t in range(H):
                    pred_embds_per_cam[c].append(gt_embds_per_cam[c][t])

        # Chunked autoregressive rollout: reset to GT state/frames every chunk
        # First chunk: 8 steps. Subsequent chunks: 8-16 steps (configurable).
        chunk_size = args.chunk_size
        

        with torch.no_grad():
            chunk_start = 0
            chunk_idx = 0
            while chunk_start + H < T:
               
                chunk_size = min(chunk_size, T - chunk_start - H)  # don't exceed episode
                if chunk_size <= 0:
                    break

                # Reset context from GT at chunk_start
                inputs = {}
                for c in condition_cameras:
                    inputs[c] = all_embds[c][:, chunk_start : chunk_start + H]
                inp_states = states[:, chunk_start : chunk_start + H]
                inp_acs = actions[:, chunk_start : chunk_start + H]

                # Predict chunk_size frames autoregressively
                for k in range(chunk_size):
                    preds, pred_state = transition(inputs, inp_states, inp_acs)
                    next_embds = {c: preds[c][:, -1:] for c in predict_cameras}
                    next_state = pred_state[:, -1:]

                    t_pred = chunk_start + H + k
                    for i, cam in enumerate(predict_cameras):
                        if cam in decoders:
                            emb = next_embds[cam]
                            pred_img, _ = decoders[cam](emb)
                            pred_img = rearrange(pred_img, "(b t) c h w -> b t c h w", t=1)
                            pred_img = pred_img[0, 0].cpu().numpy().transpose(1, 2, 0)
                            pred_imgs[i][t_pred] = np.clip(pred_img, 0, 1)
                        if pred_embds_per_cam is not None and cam in next_embds:
                            pred_embds_per_cam[cam].append(next_embds[cam][0, 0].cpu().numpy())

                    # Roll context window forward
                    for c in condition_cameras:
                        if c in predict_cameras:
                            inputs[c] = torch.cat([inputs[c][:, 1:], next_embds[c]], dim=1)
                        else:
                            next_gt = all_embds[c][:, chunk_start + H + k : chunk_start + H + k + 1]
                            inputs[c] = torch.cat([inputs[c][:, 1:], next_gt], dim=1)
                    inp_states = torch.cat([inp_states[:, 1:], next_state], dim=1)
                    inp_acs = torch.cat([inp_acs[:, 1:], actions[:, chunk_start + k + 1 : chunk_start + k + 2]], dim=1)

                chunk_start += chunk_size
                chunk_idx += 1

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
        if args.wandb and HAS_WANDB:
            video_wandb = np.transpose(video, (0, 3, 1, 2))  # (T, C, H, W)
            videos[f"episode_{ep_idx}"] = wandb.Video(video_wandb, fps=args.video_fps, format="mp4")

        # Save video to disk (same folder as embeds)
        save_dir = args.output_dir or "./eval_embeds"
        os.makedirs(save_dir, exist_ok=True)
        video_path = os.path.join(save_dir, f"episode_{ep_idx}_traj_{traj_id}_result.mp4")
        T_v, H_v, W_v, C_v = video.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, args.video_fps, (W_v, H_v))
        for t in range(T_v):
            writer.write(cv2.cvtColor(video[t], cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved video: {video_path}")

        # Save GT and pred embeddings to disk
        if args.save_embeds and pred_embds_per_cam is not None:
            save_dir = args.output_dir or "./eval_embeds"
            os.makedirs(save_dir, exist_ok=True)
            gt_dict = {c: gt_embds_per_cam[c] for c in predict_cameras}
            pred_dict = {
                c: np.stack(pred_embds_per_cam[c], axis=0) for c in predict_cameras
                if len(pred_embds_per_cam[c]) > 0
            }
            gt_path = os.path.join(save_dir, f"episode_{ep_idx}_traj_{traj_id}_gt_embeds.npz")
            pred_path = os.path.join(save_dir, f"episode_{ep_idx}_traj_{traj_id}_pred_embeds.npz")
            np.savez(gt_path, **gt_dict)
            np.savez(pred_path, **pred_dict)
            print(f"Saved embeds: {gt_path}, {pred_path}")

    return videos


def parse_args():
    import sys
    from config import load_config

    config_path = str(Path(__file__).parent / "configs" / "default.yaml")
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
    cfg = load_config("eval_wm", config_path)

    p = argparse.ArgumentParser(
        description="Evaluate DINO world model (see configs/default.yaml)"
    )
    p.add_argument("--config", type=str, default=config_path)
    p.add_argument("--wm_checkpoint", type=str, required=True)
    p.add_argument("--decoder_dir", type=str, required=True)
    p.add_argument("--hdf5_path", type=str, required=True)
    p.add_argument("--norm_stats_json", type=str, default=None)
    p.add_argument("--cameras", type=str, default=cfg.get("cameras", "camera_0,camera_1,camera_2"))
    p.add_argument("--condition_cameras", type=str, default=None)
    p.add_argument("--predict_cameras", type=str, default=None)
    p.add_argument("--segment_length", type=int, default=cfg.get("segment_length", 4))
    p.add_argument("--batch_size", type=int, default=cfg.get("batch_size", 16))
    p.add_argument("--num_test", type=int, default=cfg.get("num_test", 100))
    p.add_argument("--device", type=str, default=cfg.get("device", "cuda:0"))
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--num_samples", type=int, default=cfg.get("num_samples", 8))
    p.add_argument("--no_consolidated", action="store_true")
    p.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=cfg.get("wandb_project", "dino-wm-tactile"))
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--num_episode_videos", type=int, default=cfg.get("num_episode_videos", 3))
    p.add_argument("--max_episode_len", type=int, default=cfg.get("max_episode_len", 100))
    p.add_argument("--video_fps", type=int, default=cfg.get("video_fps", 10))
    p.add_argument("--start_after_gripper_close", action="store_true")
    p.add_argument("--gripper_idx", type=int, default=cfg.get("gripper_idx", 6))
    p.add_argument("--gripper_closed_threshold", type=float, default=cfg.get("gripper_closed_threshold", -0.5))
    p.add_argument("--chunk_size", type=int, default=cfg.get("chunk_size", 8))
    p.add_argument("--save_embeds", action="store_true")
    p.add_argument("--normalize_states", action="store_true", help="Use when model was trained with --normalize_states")
    p.set_defaults(normalize_states=cfg.get("normalize_states", False))

    args = p.parse_args()
    hdf5_dir = args.hdf5_path if os.path.isdir(args.hdf5_path) else os.path.dirname(os.path.abspath(args.hdf5_path))
    args.norm_stats_path = args.norm_stats_json or os.path.join(hdf5_dir, "norm_stats.json")
    return args


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
    norm_stats = NormStats(args.norm_stats_path, str(args.device))

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
        from config import get_wandb_config
        run_name = args.wandb_run_name or f"eval_wm_{'+'.join(cameras)}"
        wandb.init(project=args.wandb_project, name=run_name, config=get_wandb_config("eval_wm", args))

    # with torch.no_grad():
    #     for data in tqdm(eval_loader, desc="Evaluating"):
    #         # Build inputs: (B, H, N, D) for each condition camera
    #         cond_inputs = {}
    #         for cam in condition_cameras:
    #             embd = data[f"{cam}_embd"].to(device)
    #             embd = ensure_patch_grid(embd, PATCH_SIDE_DINOV3)
    #             cond_inputs[cam] = embd[:, :-1]

    #         states = data["state"].to(device)[:, :-1]
    #         acs_raw = data["action"].to(device)
    #         if acs_raw.shape[-1] > 7:
    #             acs_raw = acs_raw[..., :7]
    #         acs = normalize_acs(acs_raw, device)[:, :-1]

    #         preds, pred_state = transition(cond_inputs, states, acs)

    #         # Decode predicted next-frame embeddings and compare to GT
    #         batch_loss = 0.0
    #         preds_per_cam = {}
    #         targets_per_cam = {}
    #         for cam in predict_cameras:
    #             if cam not in decoders:
    #                 continue
    #             pred_embd = preds[cam][:, -1:]  # (B, 1, N, D)
    #             target_embd = data[f"{cam}_embd"].to(device)[:, 1:2]
    #             target_img = data[f"{cam}_image"].to(device)[:, 1:2]
    #             target_img = target_img.permute(0, 1, 4, 2, 3)
    #             if target_img.shape[2] != 3:
    #                 target_img = target_img[:, :, :3]

    #             pred_img, _ = decoders[cam](pred_embd)
    #             pred_img = rearrange(pred_img, "(b t) c h w -> b t c h w", t=1)
    #             batch_loss += nn.MSELoss()(pred_img, target_img).item()
    #             preds_per_cam[cam] = pred_img
    #             targets_per_cam[cam] = target_img

    #         batch_loss /= len([c for c in predict_cameras if c in decoders])
    #         total_loss += batch_loss
    #         n_batches += 1

    #         if (args.output_dir or wandb_images is not None) and samples_saved < args.num_samples:
    #             B = pred_img.shape[0]
    #             if args.output_dir:
    #                 os.makedirs(args.output_dir, exist_ok=True)
    #             for i in range(min(B, args.num_samples - samples_saved)):
    #                 for cam in predict_cameras:
    #                     if cam not in preds_per_cam:
    #                         continue
    #                     gt = targets_per_cam[cam][i, 0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    #                     pred_img_np = preds_per_cam[cam][i, 0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    #                     if args.output_dir:
    #                         gt_path = os.path.join(args.output_dir, f"sample_{samples_saved}_{cam}_gt.png")
    #                         pred_path = os.path.join(args.output_dir, f"sample_{samples_saved}_{cam}_pred.png")
    #                         Image.fromarray((gt * 255).astype(np.uint8)).save(gt_path)
    #                         Image.fromarray((pred_img_np * 255).astype(np.uint8)).save(pred_path)
    #                     if wandb_images is not None and samples_saved == 0:
    #                         wandb_images[f"eval_{cam}_gt"] = wandb.Image(gt, caption=f"{cam} ground truth")
    #                         wandb_images[f"eval_{cam}_pred"] = wandb.Image(pred_img_np, caption=f"{cam} predicted")
    #                 samples_saved += 1
    #                 if samples_saved >= args.num_samples:
    #                     break

    # mean_loss = total_loss / n_batches if n_batches > 0 else 0.0
    # print(f"\nTest MSE Loss (1-step pred, decoded): {mean_loss:.6f}")
    # if args.output_dir:
    #     print(f"Saved {samples_saved} sample images to {args.output_dir}")

    # Episode rollout videos (and optionally save embeddings)
    if (args.wandb and HAS_WANDB and args.num_episode_videos > 0) or args.save_embeds:
        num_episodes = max(args.num_episode_videos, 1) if args.save_embeds else args.num_episode_videos
        orig_num = args.num_episode_videos
        args.num_episode_videos = num_episodes
        episode_videos = create_episode_videos(
            transition=transition,
            decoders=decoders,
            eval_ds=eval_ds,
            predict_cameras=predict_cameras,
            condition_cameras=condition_cameras,
            args=args,
            device=device,
            norm_stats=norm_stats,
        )
        args.num_episode_videos = orig_num
        if args.wandb and HAS_WANDB and episode_videos:
            wandb.log({
                # "eval_mse_loss": mean_loss,
                # **(wandb_images or {}),
                **episode_videos,
            })
    elif args.wandb and HAS_WANDB:
        wandb.log({**(wandb_images or {})})


if __name__ == "__main__":
    main()
