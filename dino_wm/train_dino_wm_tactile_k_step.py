"""
Train K-step DINO world model on tactile dataset (LPB-style).

Uses num_hist, num_pred, frameskip like LPB:
- num_hist history frames, num_pred target frames
- frameskip: collapse frameskip actions per frame
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
from einops import rearrange
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from tactile_dataset import CAMERA_CONFIG, load_full_episode
from tactile_k_step_dataset import TactileKStepDataset, load_full_episode_subsampled
from dino_wm.dino_models import NormStats
from dino_wm.dino_models_k_step import TactileKStepTransformer

# VQVAE Quantize needs dist_fn for training; we use a no-op for eval
import dino_wm.dino_decoder as _dino_dec_mod
if not hasattr(_dino_dec_mod, "dist_fn"):
    _dino_dec_mod.dist_fn = type("DistFn", (), {"all_reduce": lambda x: None})()
from dino_wm.dino_decoder import VQVAE

try:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
except ImportError:
    DDIMScheduler = None

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


def apply_ddim_action_noise(
    inp_acs: torch.Tensor,
    noise_scheduler,
    args,
    device: str,
    train_steps_per_epoch: int,
    total_epochs: int,
    iter_i: int,
) -> torch.Tensor:
    """
    Optionally apply DDIM noise + ramping mask to normalized history actions.
    Independent of loss_mode; no-ops when args.noised is False or scheduler is missing.
    For multi_step loss, call once per rollout step (each step gets fresh noise / mask draw).
    """
    if not getattr(args, "noised", False) or noise_scheduler is None:
        return inp_acs
    out = inp_acs.clone()
    B = out.shape[0]
    geom = torch.distributions.Geometric(
        probs=torch.full((B,), args.action_noise_geom_p, device=device, dtype=torch.float32)
    )
    timesteps = torch.clamp(
        geom.sample().to(device),
        0,
        args.action_noise_timesteps - 1,
    ).long()
    noise = torch.randn_like(out)
    noised_action = noise_scheduler.add_noise(out, noise, timesteps)
    epoch_idx = iter_i // train_steps_per_epoch
    chance_of_mask = min(0.5, epoch_idx / total_epochs)
    mask = torch.rand(B, device=device) < chance_of_mask
    if mask.any():
        expand = (B,) + (1,) * (out.ndim - 1)
        out = torch.where(mask.view(expand), noised_action, out)
    return out


def parse_args():
    import sys
    from config import load_config

    config_path = str(Path(__file__).parent / "configs" / "default.yaml")
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
    cfg_wm = load_config("train_wm", config_path)
    cfg_k = load_config("train_wm_k_step", config_path)
    cfg = {**cfg_wm, **cfg_k}  # k_step section overrides

    p = argparse.ArgumentParser(
        description="Train K-step DINO world model: (e_t, s_t, K actions) -> e_{t+K}"
    )
    p.add_argument("--config", type=str, default=config_path)
    p.add_argument("--hdf5_path", type=str, default=cfg.get("hdf5_path", None))
    p.add_argument("--norm_stats_json", type=str, default=cfg.get("norm_stats_json", None))
    p.add_argument("--cameras", type=str, default=cfg.get("cameras", "camera_0,camera_1,camera_2"))
    p.add_argument("--condition_cameras", type=str, default=None)
    p.add_argument("--predict_cameras", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=cfg.get("output_dir", "checkpoints_k_step"))
    p.add_argument("--batch_size", type=int, default=cfg.get("batch_size", 16))
    p.add_argument("--eval_batch_size", type=int, default=cfg.get("eval_batch_size", 8))
    p.add_argument("--num_hist", type=int, default=cfg.get("num_hist", 1),
                   help="Number of history (conditioning) frames (LPB-style).")
    p.add_argument("--num_pred", type=int, default=cfg.get("num_pred", 1),
                   help="Number of prediction (target) frames (LPB-style).")
    p.add_argument("--frameskip", type=int, default=cfg.get("frameskip", 8),
                   help="Collapse frameskip actions per frame (LPB-style).")
    p.add_argument(
        "--temporal_mode",
        type=str,
        choices=["lpb", "consecutive"],
        default=cfg.get("temporal_mode", "lpb"),
        help='Window layout: "lpb" = subsample every frameskip (default); '
        '"consecutive" = dense hist t-nh+1..t, targets t+fs..t+np*fs, aligned action blocks.',
    )
    p.add_argument("--num_test", type=int, default=cfg.get("num_test", 50))
    p.add_argument("--iters", type=int, default=cfg.get("iters", 100000))
    p.add_argument("--eval_every", type=int, default=cfg.get("eval_every", 1000))
    p.add_argument("--device", type=str, default=cfg.get("device", "cuda:0"))
    p.add_argument("--wandb", action="store_true", default=cfg.get("use_wandb", False))
    p.add_argument("--no_consolidated", action="store_true")
    p.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    p.add_argument("--decoder_dir", type=str, default=cfg.get("decoder_dir"))
    p.add_argument("--eval_rollout_horizon", type=int, default=cfg.get("eval_rollout_horizon", 16))
    p.add_argument("--num_eval_videos", type=int, default=cfg.get("num_eval_videos", 1))
    p.add_argument("--video_fps", type=int, default=cfg.get("video_fps", 10))
    p.add_argument("--video_every_eval", type=int, default=cfg.get("video_every_eval", 3))
    p.add_argument("--num_workers", type=int, default=cfg.get("num_workers", 4),
                   help="Set to 0 if DataLoader collate fails with 'storage not resizable' (h5py+multiprocessing)")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--segment_sampling", type=str, choices=["uniform", "weighted"], default=cfg.get("segment_sampling", "uniform"))
    p.add_argument("--gripper_change_weight", type=float, default=cfg.get("gripper_change_weight", 5.0))
    p.add_argument("--weight_source", type=str, choices=["gripper_state", "gripper_change_json"], default=cfg.get("weight_source", "gripper_state"))
    p.add_argument("--gripper_change_json_path", type=str, default=cfg.get("gripper_change_json_path", None))
    p.add_argument("--normalize_states", action="store_true", default=cfg.get("normalize_states", False))
    p.set_defaults(normalize_states=cfg.get("normalize_states", False))
    p.add_argument(
        "--noised",
        action="store_true",
        default=cfg.get("noised", False),
        help="DDIM noise on history actions (mse and multi_step); independent of --loss_mode.",
    )
    p.add_argument(
        "--action_noise_geom_p",
        type=float,
        default=cfg.get("action_noise_geom_p", 0.05),
        help="Geometric(probs=p) for sampling diffusion timestep per batch element (~mean 1/p-1 trials).",
    )
    p.add_argument(
        "--action_noise_timesteps",
        type=int,
        default=cfg.get("action_noise_timesteps", 100),
        help="DDIMScheduler num_train_timesteps (timesteps clipped to [0, this-1]).",
    )
    p.add_argument(
        "--loss_mode",
        type=str,
        choices=["mse", "multi_step"],
        default=cfg.get("loss_mode", "mse"),
        help='"mse" = one forward vs GT; "multi_step" = rollout k steps (see multi_step_rollout_*); '
        "train dataset horizon is max(num_pred, multi_step_rollout_max) so k is independent of model num_pred.",
    )
    p.add_argument(
        "--multi_step_rollout_min",
        type=int,
        default=cfg.get("multi_step_rollout_min", 3),
        help="Minimum rollout length k for multi_step loss (inclusive).",
    )
    p.add_argument(
        "--multi_step_rollout_max",
        type=int,
        default=cfg.get("multi_step_rollout_max", 5),
        help="Maximum rollout length k for multi_step loss (inclusive). Train data loads this many future frames.",
    )
    args = p.parse_args()
    args.norm_stats_path = args.norm_stats_json or os.path.join(
        os.path.dirname(os.path.abspath(args.hdf5_path)), "norm_stats.json"
    )
    return args


def load_decoders(decoder_dir: str, predict_cameras: list, device: str) -> dict:
    decoders = {}
    vision_decoder_path = os.path.join(decoder_dir, "decoder_camera_0_camera_1.pth")
    tactile_decoder_path = os.path.join(decoder_dir, "decoder_camera_2.pth")
    if os.path.isfile(vision_decoder_path):
        dec = VQVAE(emb_dim=768).to(device)
        dec.load_state_dict(torch.load(vision_decoder_path, map_location=device))
        dec.eval()
        for c in predict_cameras:
            if CAMERA_EMB_DIMS.get(c, 768) == 768:
                decoders[c] = dec
    if os.path.isfile(tactile_decoder_path) and "camera_2" in predict_cameras:
        dec = VQVAE(emb_dim=512).to(device)
        dec.load_state_dict(torch.load(tactile_decoder_path, map_location=device))
        dec.eval()
        decoders["camera_2"] = dec
    return decoders


def multi_step_rollout_loss(
    transition: nn.Module,
    *,
    camera_embds: dict,
    state_all: torch.Tensor,
    act_collapsed: torch.Tensor,
    cond_inputs: dict,
    inp_states: torch.Tensor,
    predict_cameras: list,
    condition_cameras: list,
    num_hist: int,
    args,
    noise_scheduler,
    device: str,
    train_steps_per_epoch: int,
    total_epochs: int,
    iter_i: int,
) -> torch.Tensor:
    """
    Sample k in [multi_step_rollout_min, multi_step_rollout_max] (independent of model num_pred).
    Batch must provide num_hist + k future-supervised frames (train dataset num_pred >= rollout_max).
    """
    lo = args.multi_step_rollout_min
    hi = args.multi_step_rollout_max
    if hi < lo:
        lo, hi = hi, lo
    k = random.randint(lo, hi)

    cur_inputs = {c: cond_inputs[c].clone() for c in condition_cameras}
    cur_states = inp_states.clone()
    mse = nn.MSELoss()
    loss_ms = torch.tensor(0.0, device=device)

    for step in range(1, k + 1):
        slot0 = step - 1
        inp_acs = act_collapsed[:, slot0 : slot0 + num_hist].clone()
        if args.noised:
            inp_acs = apply_ddim_action_noise(
                inp_acs,
                noise_scheduler,
                args,
                device,
                train_steps_per_epoch,
                total_epochs,
                iter_i,
            )

        preds, pred_state = transition(cur_inputs, cur_states, inp_acs)
        tgt_idx = num_hist + step - 1
        loss_ms = loss_ms + mse(pred_state[:, 0], state_all[:, tgt_idx])
        for cam in predict_cameras:
            loss_ms = loss_ms + mse(preds[cam][:, 0], camera_embds[cam][:, tgt_idx])

        if step < k:
            for cam in condition_cameras:
                if cam in predict_cameras:
                    next_f = preds[cam][:, 0:1].detach()
                else:
                    next_f = camera_embds[cam][:, tgt_idx : tgt_idx + 1]
                cur_inputs[cam] = torch.cat([cur_inputs[cam][:, 1:], next_f], dim=1)
            cur_states = torch.cat([cur_states[:, 1:], pred_state[:, 0:1].detach()], dim=1)

    return loss_ms


def create_eval_video(
    transition,
    decoders: dict,
    eval_ds: TactileKStepDataset,
    predict_cameras: list,
    condition_cameras: list,
    args,
    device: str,
    norm_stats: NormStats,
) -> np.ndarray:
    """
    LPB-style rollout: subsample by frameskip, at each step feed (hist frames, hist actions) -> pred next frame.
    """
    if not decoders:
        return None
    traj_ids = eval_ds.trajectory_ids[: args.num_eval_videos]
    if not traj_ids:
        return None

    traj_id = traj_ids[0]
    ep_data = load_full_episode_subsampled(
        args.hdf5_path,
        traj_id,
        list(set(predict_cameras) | set(condition_cameras)),
        args.frameskip,
        not args.no_consolidated,
    )
    if "states" not in ep_data or "actions_full" not in ep_data:
        return None

    with torch.inference_mode():
        T_sub = ep_data[f"{predict_cameras[0]}_embd"].shape[0]
        T = min(T_sub, args.eval_rollout_horizon)
        num_hist, num_pred = args.num_hist, args.num_pred
        if T <= num_hist + num_pred:
            return None

        all_gt_imgs = []
        for cam in predict_cameras:
            if f"{cam}_image" in ep_data:
                gt_img = np.asarray(ep_data[f"{cam}_image"][:T], dtype=np.float32)
            else:
                gt_img = np.zeros((T, 224, 224, 3), dtype=np.float32)
            if gt_img.ndim == 3:
                gt_img = np.stack([gt_img] * 3, axis=-1)
            if gt_img.max() > 1.0:
                gt_img = gt_img / 255.0
            all_gt_imgs.append(np.clip(gt_img, 0, 1))

        states = torch.tensor(ep_data["states"][:T], dtype=torch.float32, device=device).unsqueeze(0)
        if args.normalize_states:
            states = norm_stats.normalize_states(states)
        obs_idx = ep_data["obs_indices"]
        actions_full = torch.tensor(ep_data["actions_full"], dtype=torch.float32, device=device).unsqueeze(0)
        if actions_full.shape[-1] > 7:
            actions_full = actions_full[..., :7]
        actions_full = norm_stats.normalize_acs(actions_full)

        inputs = {}
        for c in condition_cameras:
            emb = torch.tensor(ep_data[f"{c}_embd"][:T], dtype=torch.float32, device=device).unsqueeze(0)
            inputs[c] = ensure_patch_grid(emb, PATCH_SIDE_DINOV3)

        pred_imgs = [[all_gt_imgs[i][t].copy() for t in range(T)] for i in range(len(predict_cameras))]
        pred_embds = {c: inputs[c].clone() for c in condition_cameras}

        fs = args.frameskip
        for t in range(num_hist, T - num_pred + 1):
            act_start = obs_idx[t - num_hist]
            act_end = obs_idx[t]
            act_slice = actions_full[:, act_start:act_end]
            act_collapsed = rearrange(act_slice, "b (n f) d -> b n (f d)", n=num_hist, f=fs, d=7)
            inp_act = act_collapsed

            cond_inputs = {c: pred_embds[c][:, t - num_hist : t] for c in condition_cameras}
            inp_states = states[:, t - num_hist : t]

            preds, _ = transition(cond_inputs, inp_states, inp_act)

            for i, cam in enumerate(predict_cameras):
                if cam not in decoders:
                    continue
                pred_emb = preds[cam]
                pred_img, _ = decoders[cam](pred_emb)
                pred_img = rearrange(pred_img, "(b t) c h w -> b t c h w", t=num_pred)
                pred_img = pred_img[0, 0].cpu().numpy().transpose(1, 2, 0)
                pred_imgs[i][t] = np.clip(pred_img, 0, 1)

            for c in condition_cameras:
                if c in predict_cameras:
                    pred_embds[c][:, t : t + 1] = preds[c]
                else:
                    pred_embds[c][:, t : t + 1] = inputs[c][:, t : t + 1]

        rows = []
        for t in range(T):
            gt_row = np.concatenate([all_gt_imgs[i][t] for i in range(len(predict_cameras))], axis=1)
            pred_row = np.concatenate([pred_imgs[i][t] for i in range(len(predict_cameras))], axis=1)
            diff_row = np.abs(gt_row.astype(np.float32) - pred_row.astype(np.float32))
            if diff_row.shape[-1] == 1:
                diff_row = np.repeat(diff_row, 3, axis=-1)
            frame = np.vstack([gt_row, pred_row, diff_row])
            rows.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

        video = np.stack(rows, axis=0)
        video = np.transpose(video, (0, 3, 1, 2))
        return video


def main():
    args = parse_args()

    cameras = [c.strip() for c in args.cameras.split(",")]
    condition_cameras = [c.strip() for c in args.condition_cameras.split(",")] if args.condition_cameras else cameras
    predict_cameras = [c.strip() for c in args.predict_cameras.split(",")] if args.predict_cameras else cameras

    for c in cameras + condition_cameras + predict_cameras:
        if c not in CAMERA_CONFIG:
            raise SystemExit(f"Unknown camera: {c}")

    if args.wandb and HAS_WANDB:
        from config import get_wandb_config
        wandb_cfg = get_wandb_config("train_wm", args)
        wandb_cfg.update(
            cameras=cameras,
            num_hist=args.num_hist,
            num_pred=args.num_pred,
            frameskip=args.frameskip,
            temporal_mode=args.temporal_mode,
            noised=args.noised,
            loss_mode=args.loss_mode,
            action_noise_timesteps=args.action_noise_timesteps,
            action_noise_geom_p=args.action_noise_geom_p,
        )
        wandb.init(project="dino-wm-tactile-k-step", name=f"h{args.num_hist}_p{args.num_pred}_fs{args.frameskip}", config=wandb_cfg)

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    num_hist = args.num_hist
    num_pred = args.num_pred
    frameskip = args.frameskip
    device = args.device
    is_consolidated = not args.no_consolidated
    norm_stats = NormStats(args.norm_stats_path, device)

    # Model num_pred is independent of multi-step rollout length; extend train segments when needed.
    num_pred_train = (
        max(num_pred, args.multi_step_rollout_max)
        if args.loss_mode == "multi_step"
        else num_pred
    )

    train_ds = TactileKStepDataset(
        args.hdf5_path, cameras=cameras,
        num_hist=num_hist, num_pred=num_pred_train, frameskip=frameskip,
        split="train", num_test=args.num_test, is_consolidated=is_consolidated, seed=args.seed,
        temporal_mode=args.temporal_mode,
    )
    eval_ds = TactileKStepDataset(
        args.hdf5_path, cameras=cameras,
        num_hist=num_hist, num_pred=num_pred, frameskip=frameskip,
        split="test", num_test=args.num_test, is_consolidated=is_consolidated, seed=args.seed,
        temporal_mode=args.temporal_mode,
    )

    def collate_fn(batch):
        """Custom collate to avoid 'storage that is not resizable' with multiprocessing + h5py."""
        if len(batch) == 0:
            return {}
        keys = batch[0].keys()
        return {k: torch.stack([b[k].contiguous().clone() for b in batch]) for k in keys}

    base_dl_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.num_workers > 0,
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_fn,
    }
    train_dl_kwargs = {**base_dl_kwargs, "shuffle": True}
    eval_dl_kwargs = {**base_dl_kwargs, "batch_size": args.eval_batch_size, "shuffle": True}

    train_loader = iter(DataLoader(train_ds, **train_dl_kwargs))
    eval_loader = iter(DataLoader(eval_ds, **eval_dl_kwargs))

    camera_dims = {c: CAMERA_EMB_DIMS.get(c, 768) for c in cameras}
    transition = TactileKStepTransformer(
        cameras=cameras,
        camera_dims=camera_dims,
        condition_cameras=condition_cameras,
        predict_cameras=predict_cameras,
        common_dim=384,
        ac_dim=64,
        state_dim=8,
        num_hist=num_hist,
        num_pred=num_pred,
        frameskip=frameskip,
        patches_per_frame=PATCH_SIDE_DINOV3 * PATCH_SIDE_DINOV3,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
    ).to(device)
    if not args.no_compile and hasattr(torch, "compile"):
        transition = torch.compile(transition, mode="reduce-overhead")
        print("Using torch.compile")
    transition.train()

    param_groups = [
        {"params": transition.transformer.parameters(), "lr": 5e-5},
        {"params": transition.state_head.parameters(), "lr": 5e-5},
        {"params": transition.action_encoder.parameters(), "lr": 5e-4},
        {"params": [transition.pos_embedding], "lr": 5e-4},
    ]
    for cam in predict_cameras:
        param_groups.append({"params": transition.heads[cam].parameters(), "lr": 5e-5})
    for cam in condition_cameras:
        param_groups.append({"params": transition.project_in[cam].parameters(), "lr": 5e-5})

    optimizer = AdamW(param_groups)
    best_eval = float("inf")
    decoders = {}
    if args.decoder_dir and os.path.isdir(args.decoder_dir):
        decoders = load_decoders(args.decoder_dir, predict_cameras, device)
        if decoders:
            print(f"Loaded decoders: {list(decoders.keys())}")

    train_steps_per_epoch = max(1, len(train_ds) // args.batch_size)
    total_epochs = max(1, (args.iters + train_steps_per_epoch - 1) // train_steps_per_epoch)

    print(
        f"K-step training: loss_mode={args.loss_mode}, temporal_mode={args.temporal_mode}, "
        f"num_hist={num_hist}, num_pred(model)={num_pred}, num_pred(train_data)={num_pred_train}, "
        f"frameskip={frameskip}"
    )
    if args.loss_mode == "multi_step":
        print(
            f"  multi_step rollout k in [{args.multi_step_rollout_min}, {args.multi_step_rollout_max}] "
            f"(independent of model num_pred)"
        )
    noise_scheduler = None
    if args.noised:
        if DDIMScheduler is None:
            raise ImportError(
                "--noised requires diffusers. Install with: pip install diffusers"
            )
        noise_scheduler = DDIMScheduler(num_train_timesteps=args.action_noise_timesteps)
        print(
            f"  action noise: DDIMScheduler.add_noise on inp_acs, "
            f"mask ramp to 0.5 over {total_epochs} epochs (~{train_steps_per_epoch} iters/epoch), "
            f"geom_p={args.action_noise_geom_p}, T={args.action_noise_timesteps}"
        )

    for i in tqdm(range(args.iters), desc="Training", unit="iter"):
        try:
            data = next(train_loader)
        except StopIteration:
            train_loader = iter(DataLoader(train_ds, **train_dl_kwargs))
            data = next(train_loader)

        non_blocking = base_dl_kwargs.get("pin_memory", False)
        # data: {cam}_embd (B, num_frames, N, D), state (B, num_frames, state_dim), action (B, num_frames*frameskip, 7)
        acs_raw = data["action"].to(device, non_blocking=non_blocking)
        if acs_raw.shape[-1] > 7:
            acs_raw = acs_raw[..., :7]
        acs_norm = norm_stats.normalize_acs(acs_raw)
        act_collapsed = rearrange(
            acs_norm, "b (n f) d -> b n (f d)", n=num_hist + num_pred_train, f=frameskip, d=7
        )

        state_all = data["state"].to(device, non_blocking=non_blocking)
        if args.normalize_states:
            state_all = norm_stats.normalize_states(state_all)

        cond_inputs = {}
        for cam in condition_cameras:
            embd = data[f"{cam}_embd"].to(device, non_blocking=non_blocking)
            embd = ensure_patch_grid(embd, PATCH_SIDE_DINOV3)
            cond_inputs[cam] = embd[:, :num_hist]

        inp_states = state_all[:, :num_hist]

        cams_embd = list(set(predict_cameras) | set(condition_cameras))
        camera_embds = {}
        for cam in cams_embd:
            embd = data[f"{cam}_embd"].to(device, non_blocking=non_blocking)
            embd = ensure_patch_grid(embd, PATCH_SIDE_DINOV3)
            camera_embds[cam] = embd

        optimizer.zero_grad()
        mse = nn.MSELoss()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            if args.loss_mode == "mse":
                target_embds = {
                    cam: camera_embds[cam][:, num_hist : num_hist + num_pred] for cam in predict_cameras
                }
                target_state = state_all[:, num_hist : num_hist + num_pred]
                inp_acs = apply_ddim_action_noise(
                    act_collapsed[:, :num_hist].clone(),
                    noise_scheduler,
                    args,
                    device,
                    train_steps_per_epoch,
                    total_epochs,
                    i,
                )
                preds, pred_state = transition(cond_inputs, inp_states, inp_acs)
                loss = mse(pred_state, target_state)
                for cam in predict_cameras:
                    loss = loss + mse(preds[cam], target_embds[cam])
            else:
                loss = multi_step_rollout_loss(
                    transition,
                    camera_embds=camera_embds,
                    state_all=state_all,
                    act_collapsed=act_collapsed,
                    cond_inputs=cond_inputs,
                    inp_states=inp_states,
                    predict_cameras=predict_cameras,
                    condition_cameras=condition_cameras,
                    num_hist=num_hist,
                    args=args,
                    noise_scheduler=noise_scheduler,
                    device=device,
                    train_steps_per_epoch=train_steps_per_epoch,
                    total_epochs=total_epochs,
                    iter_i=i,
                )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if args.wandb and HAS_WANDB:
            wandb.log({"train_loss": loss.item()})

        if i % 50 == 0:
            print(f"\rIter {i}, loss: {loss.item():.4f}", end="", flush=True)

        if (i + 1) % args.eval_every == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                eval_data = next(eval_loader)
            except StopIteration:
                eval_loader = iter(DataLoader(eval_ds, **eval_dl_kwargs))
                eval_data = next(eval_loader)

            transition.eval()
            with torch.inference_mode():
                acs_raw = eval_data["action"].to(device)
                if acs_raw.shape[-1] > 7:
                    acs_raw = acs_raw[..., :7]
                acs_norm = norm_stats.normalize_acs(acs_raw)
                act_collapsed = rearrange(acs_norm, "b (n f) d -> b n (f d)", n=num_hist + num_pred, f=frameskip, d=7)

                cond_inputs = {}
                for cam in condition_cameras:
                    embd = eval_data[f"{cam}_embd"].to(device)
                    embd = ensure_patch_grid(embd, PATCH_SIDE_DINOV3)
                    cond_inputs[cam] = embd[:, :num_hist]
                inp_states = eval_data["state"].to(device)[:, :num_hist]
                if args.normalize_states:
                    inp_states = norm_stats.normalize_states(inp_states)
                inp_acs = act_collapsed[:, :num_hist]

                target_embds = {cam: eval_data[f"{cam}_embd"].to(device)[:, num_hist : num_hist + num_pred] for cam in predict_cameras}
                for cam in predict_cameras:
                    embd = eval_data[f"{cam}_embd"].to(device)
                    embd = ensure_patch_grid(embd, PATCH_SIDE_DINOV3)
                    target_embds[cam] = embd[:, num_hist : num_hist + num_pred]
                target_state = eval_data["state"].to(device)[:, num_hist : num_hist + num_pred]
                if args.normalize_states:
                    target_state = norm_stats.normalize_states(target_state)

                preds_eval, pred_state_eval = transition(cond_inputs, inp_states, inp_acs)
                eval_loss = mse(pred_state_eval, target_state)
                for cam in predict_cameras:
                    eval_loss = eval_loss + mse(preds_eval[cam], target_embds[cam])
                eval_loss_val = eval_loss.item()

            print()
            print(f"Iter {i}, Eval Loss: {eval_loss_val:.4f}")

            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(transition.state_dict(), f"{args.output_dir}/wm_kstep_h{num_hist}_p{num_pred}_fs{frameskip}_iter{i+1}.pth")
            if eval_loss_val < best_eval:
                best_eval = eval_loss_val
                torch.save(transition.state_dict(), f"{args.output_dir}/best_wm_kstep_h{num_hist}_p{num_pred}_fs{frameskip}.pth")

            if args.wandb and HAS_WANDB:
                wandb.log({"eval_loss": eval_loss_val})

            eval_count = (i + 1) // args.eval_every
            if args.wandb and HAS_WANDB and decoders and eval_count % args.video_every_eval == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                video = create_eval_video(
                    transition, decoders, eval_ds, predict_cameras,
                    condition_cameras, args, device, norm_stats,
                )
                if video is not None:
                    wandb.log({"eval_video": wandb.Video(video, fps=args.video_fps, format="mp4")})

            transition.train()


if __name__ == "__main__":
    main()
