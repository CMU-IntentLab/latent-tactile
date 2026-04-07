"""
Evaluate a trained DINO decoder on the tactile dataset.

Loads a checkpoint and runs evaluation on the test split, reporting MSE loss
and optionally saving decoded image samples.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
import torch.nn as nn
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from tactile_dataset import TactileTrajectoryDataset, CAMERA_CONFIG, load_full_episode
from dino_decoder import VQVAE


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


def _create_episode_videos(
    decoder,
    eval_ds: TactileTrajectoryDataset,
    cameras: list[str],
    args,
    device,
    ensure_patch_grid,
):
    """Load full episodes, run decoder, create 3-row (gt/pred/diff) videos, return wandb dict."""
    videos = {}
    traj_ids = eval_ds.trajectory_ids[: args.num_episode_videos]

    for ep_idx, traj_id in enumerate(tqdm(traj_ids, desc="Episode videos")):
        ep_data = load_full_episode(
            args.hdf5_path,
            traj_id,
            cameras,
            is_consolidated=not args.no_consolidated,
        )

        T = ep_data[f"{cameras[0]}_embd"].shape[0]
        T = min(T, args.max_episode_len)

        all_gt = []
        all_pred = []
        for cam in cameras:
            embd = torch.tensor(ep_data[f"{cam}_embd"][:T], dtype=torch.float32, device=device)
            gt = ep_data[f"{cam}_image"][:T]  # (T, H, W, C)
            embd = embd.unsqueeze(0)  # (1, T, N, D)
            embd = ensure_patch_grid(embd, target_side=args.patch_side)
            with torch.no_grad():
                pred, _ = decoder(embd)
            pred = rearrange(pred, "(b t) c h w -> b t c h w", b=1, t=T)
            pred = pred[0].cpu().numpy().transpose(0, 2, 3, 1)  # (T, H, W, C)
            gt = np.asarray(gt, dtype=np.float32)
            if gt.ndim == 3:
                gt = np.stack([gt] * 3, axis=-1)  # (T, H, W) -> (T, H, W, 3)
            all_gt.append(gt)
            all_pred.append(pred)

        for cam_idx, cam in enumerate(cameras):
            gt_cam = np.clip(all_gt[cam_idx], 0, 1)
            pred_cam = np.clip(all_pred[cam_idx], 0, 1)
            diff = np.abs(gt_cam - pred_cam)
            diff = np.repeat(diff, 3, axis=-1) if diff.shape[-1] == 1 else diff

            # Stack rows: gt, pred, diff -> (3*H, W, 3) per frame
            # wandb.Video expects (T, C, H, W) or (T, C, height, width), uint8 0-255
            frames = []
            for t in range(T):
                row1 = gt_cam[t]
                row2 = pred_cam[t]
                row3 = diff[t]
                frame = np.vstack([row1, row2, row3])  # (3*H, W, 3)
                frames.append((frame * 255).astype(np.uint8))
            video = np.stack(frames, axis=0)  # (T, 3*H, W, 3)
            video = np.transpose(video, (0, 3, 1, 2))  # (T, C, H, W) for wandb
            key = f"episode_{ep_idx}_{cam}"
            videos[key] = wandb.Video(video, fps=args.video_fps, format="mp4")

    return videos


def infer_emb_dim_from_checkpoint(checkpoint_path: str) -> int:
    """Infer emb_dim from the quantize_b.embed buffer shape in the checkpoint."""
    state = torch.load(checkpoint_path, map_location="cpu")
    if "quantize_b.embed" in state:
        return int(state["quantize_b.embed"].shape[0])
    raise ValueError(
        f"Cannot infer emb_dim from {checkpoint_path}. "
        "Pass --emb_dim explicitly."
    )


def parse_args():
    import sys
    from config import load_config

    config_path = str(Path(__file__).parent / "configs" / "default.yaml")
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
    cfg = load_config("eval_decoder", config_path)

    p = argparse.ArgumentParser(
        description="Evaluate DINO decoder (see configs/default.yaml)"
    )
    p.add_argument("--config", type=str, default=config_path)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--hdf5_path", type=str, required=True)
    p.add_argument("--cameras", type=str, required=True, help="Cameras this decoder was trained on, e.g. 'camera_0' or 'camera_1'")
    p.add_argument("--emb_dim", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=cfg.get("batch_size", 64))
    p.add_argument("--segment_length", type=int, default=cfg.get("segment_length", 1))
    p.add_argument("--num_test", type=int, default=cfg.get("num_test", 100))
    p.add_argument("--patch_side", type=int, default=cfg.get("patch_side", 14))
    p.add_argument("--device", type=str, default=cfg.get("device", "cuda:0"))
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--num_samples", type=int, default=cfg.get("num_samples", 8))
    p.add_argument("--no_consolidated", action="store_true")
    p.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=cfg.get("wandb_project", "dino-decoder-tactile"))
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--num_episode_videos", type=int, default=cfg.get("num_episode_videos", 3))
    p.add_argument("--max_episode_len", type=int, default=cfg.get("max_episode_len", 100))
    p.add_argument("--video_fps", type=int, default=cfg.get("video_fps", 10))
    return p.parse_args()


def main():
    args = parse_args()

    # Provide no-op dist_fn for VQVAE
    import dino_decoder as _dino_dec_mod
    if not hasattr(_dino_dec_mod, "dist_fn"):
        _dino_dec_mod.dist_fn = type("DistFn", (), {"all_reduce": lambda x: None})()

    cameras = [c.strip() for c in args.cameras.split(",")]
    for c in cameras:
        if c not in CAMERA_CONFIG:
            raise ValueError(f"Unknown camera: {c}. Choose from {list(CAMERA_CONFIG.keys())}")

    emb_dim = args.emb_dim or infer_emb_dim_from_checkpoint(args.checkpoint)
    print(f"Loading decoder (emb_dim={emb_dim}) from {args.checkpoint}")

    device = torch.device(args.device)
    decoder = VQVAE(emb_dim=emb_dim).to(device)
    decoder.load_state_dict(torch.load(args.checkpoint, map_location=device))
    decoder.eval()

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
        run_name = args.wandb_run_name or f"eval_{'+'.join(cameras)}"
        wandb.init(project=args.wandb_project, name=run_name, config=get_wandb_config("eval_decoder", args))

    with torch.no_grad():
        for data in tqdm(eval_loader, desc="Evaluating"):
            batch_loss = 0.0
            preds_per_cam = {}
            targets_per_cam = {}
            for cam in cameras:
                embd = data[f"{cam}_embd"].to(device)
                target = data[f"{cam}_image"].to(device)
                B, T, N, D = embd.shape
                embd = ensure_patch_grid(embd, target_side=args.patch_side)
                pred, _ = decoder(embd)
                pred = rearrange(pred, "(b t) c h w -> b t c h w", b=B, t=T)
                target = target.permute(0, 1, 4, 2, 3)
                if target.shape[2] != 3:
                    target = target[:, :, :3]
                batch_loss += nn.MSELoss()(pred, target).item()
                preds_per_cam[cam] = pred
                targets_per_cam[cam] = target
            batch_loss /= len(cameras)
            total_loss += batch_loss
            n_batches += 1

            if (args.output_dir or wandb_images is not None) and samples_saved < args.num_samples:
                if args.output_dir:
                    os.makedirs(args.output_dir, exist_ok=True)
                for i in range(min(B, args.num_samples - samples_saved)):
                    for cam in cameras:
                        gt = targets_per_cam[cam][i, 0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                        pred_img = preds_per_cam[cam][i, 0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                        if args.output_dir:
                            gt_path = os.path.join(args.output_dir, f"sample_{samples_saved}_{cam}_gt.png")
                            pred_path = os.path.join(args.output_dir, f"sample_{samples_saved}_{cam}_pred.png")
                            Image.fromarray((gt * 255).astype(np.uint8)).save(gt_path)
                            Image.fromarray((pred_img * 255).astype(np.uint8)).save(pred_path)
                        if wandb_images is not None and samples_saved == 0:
                            wandb_images[f"eval_{cam}_gt"] = wandb.Image(gt, caption=f"{cam} ground truth")
                            wandb_images[f"eval_{cam}_pred"] = wandb.Image(pred_img, caption=f"{cam} decoded")
                    samples_saved += 1
                    if samples_saved >= args.num_samples:
                        break

    mean_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"\nTest MSE Loss: {mean_loss:.6f}")
    if args.output_dir:
        print(f"Saved {samples_saved} sample images to {args.output_dir}")

    # Log episode videos (gt / pred / diff rows) to wandb
    if args.wandb and HAS_WANDB and args.num_episode_videos > 0:
        episode_videos = _create_episode_videos(
            decoder=decoder,
            eval_ds=eval_ds,
            cameras=cameras,
            args=args,
            device=device,
            ensure_patch_grid=ensure_patch_grid,
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
