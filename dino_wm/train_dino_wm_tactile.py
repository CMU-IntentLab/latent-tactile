"""
Train DINO world model on tactile dataset.

Conditions on a user-selected combination of camera embeddings (camera_0, camera_1, camera_2)
and predicts future embeddings for those cameras. Uses DINOv3 (768-dim) for vision cameras
and supports AnyTouch (512-dim) for tactile. Similar to train_dino_wm.py but with:
- TactileTrajectoryDataset instead of SplitTrajectoryDataset
- DINOv3 embeddings (768-dim) instead of DINOv2 (384-dim)
- Configurable cameras via --cameras, --condition_cameras, --predict_cameras
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

from tactile_dataset import TactileTrajectoryDataset, CAMERA_CONFIG, load_full_episode
from dino_models import TactileVideoTransformer, normalize_acs, normalize_states

# VQVAE Quantize needs dist_fn for training; we use a no-op for eval
import dino_decoder as _dino_dec_mod
if not hasattr(_dino_dec_mod, "dist_fn"):
    _dino_dec_mod.dist_fn = type("DistFn", (), {"all_reduce": lambda x: None})()
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


def parse_args():
    import sys
    from config import load_config

    config_path = str(Path(__file__).parent / "configs" / "default.yaml")
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
    cfg = load_config("train_wm", config_path)

    p = argparse.ArgumentParser(
        description="Train DINO world model on tactile dataset (see configs/default.yaml)"
    )
    p.add_argument("--config", type=str, default=config_path, help="Path to config YAML")
    p.add_argument("--hdf5_path", type=str, required=True, help="Path to consolidated HDF5")
    p.add_argument("--norm_stats_json", type=str, default=None)
    p.add_argument("--cameras", type=str, default=cfg.get("cameras", "camera_0,camera_1,camera_2"))
    p.add_argument("--condition_cameras", type=str, default=None)
    p.add_argument("--predict_cameras", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=cfg.get("output_dir", "checkpoints"))
    p.add_argument("--batch_size", type=int, default=cfg.get("batch_size", 16))
    p.add_argument("--segment_length", type=int, default=cfg.get("segment_length", 4))
    p.add_argument("--num_test", type=int, default=cfg.get("num_test", 50))
    p.add_argument("--iters", type=int, default=cfg.get("iters", 100000))
    p.add_argument("--eval_every", type=int, default=cfg.get("eval_every", 1000))
    p.add_argument("--device", type=str, default=cfg.get("device", "cuda:0"))
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no_consolidated", action="store_true")
    p.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    p.add_argument("--ar_loss_weight", type=float, default=cfg.get("ar_loss_weight", 0.5))
    p.add_argument("--ar_loss_mode", type=str, choices=["single", "multi_step"], default=cfg.get("ar_loss_mode", "single"))
    p.add_argument("--decoder_dir", type=str, default=cfg.get("decoder_dir"))
    p.add_argument("--eval_rollout_horizon", type=int, default=cfg.get("eval_rollout_horizon", 16))
    p.add_argument("--num_eval_videos", type=int, default=cfg.get("num_eval_videos", 1))
    p.add_argument("--video_fps", type=int, default=cfg.get("video_fps", 10))
    p.add_argument("--video_every_eval", type=int, default=cfg.get("video_every_eval", 3))
    p.add_argument("--num_workers", type=int, default=cfg.get("num_workers", 4))
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--segment_sampling", type=str, choices=["uniform", "weighted"], default=cfg.get("segment_sampling", "uniform"))
    p.add_argument("--gripper_change_weight", type=float, default=cfg.get("gripper_change_weight", 5.0))
    p.add_argument("--weight_source", type=str, choices=["gripper_state", "gripper_change_json"], default=cfg.get("weight_source", "gripper_state"))
    p.add_argument("--gripper_change_json_path", type=str, default=None)
    p.add_argument("--normalize_states", action="store_true", help="Normalize states using norm_stats.json (min-max to [0,1])")
    p.set_defaults(normalize_states=cfg.get("normalize_states", False))
    args = p.parse_args()
    args.norm_stats_path = args.norm_stats_json or os.path.join(
        os.path.dirname(os.path.abspath(args.hdf5_path)), "norm_stats.json"
    )
    return args


def load_decoders(decoder_dir: str, predict_cameras: list, device: str) -> dict:
    """Load decoders for each camera. Vision cams share decoder_camera_0_camera_1.pth."""
    decoders = {}
    # Vision cameras share one decoder (768 dim)
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


def create_eval_video(
    transition,
    decoders: dict,
    eval_ds: TactileTrajectoryDataset,
    predict_cameras: list,
    condition_cameras: list,
    args,
    device: str,
) -> np.ndarray:
    """
    Rollout on one episode, decode to images, create video: top=GT, middle=pred, bottom=diff.
    Cameras concatenated horizontally. Returns (T, C, H, W) uint8 for wandb.Video.
    """
    if not decoders:
        return None
    traj_ids = eval_ds.trajectory_ids[: args.num_eval_videos]
    if not traj_ids:
        return None

    traj_id = traj_ids[0]
    ep_data = load_full_episode(
        args.hdf5_path,
        traj_id,
        predict_cameras + [c for c in condition_cameras if c not in predict_cameras],
        is_consolidated=not args.no_consolidated,
    )
    if "states" not in ep_data or "actions" not in ep_data:
        return None

    T_total = ep_data[f"{predict_cameras[0]}_embd"].shape[0]
    T = min(T_total, args.eval_rollout_horizon)
    H = args.segment_length - 1

    if T <= H:
        return None

    # Load GT images and embeddings for all predict cameras
    all_gt_imgs = []
    for cam in predict_cameras:
        gt_img = np.asarray(ep_data[f"{cam}_image"][:T], dtype=np.float32)
        if gt_img.ndim == 3:
            gt_img = np.stack([gt_img] * 3, axis=-1)
        if gt_img.max() > 1.0:
            gt_img = gt_img / 255.0
        all_gt_imgs.append(np.clip(gt_img, 0, 1))

    # Build initial inputs from episode
    states = torch.tensor(ep_data["states"][:T], dtype=torch.float32, device=device).unsqueeze(0)
    if args.normalize_states:
        states = normalize_states(states, device, args.norm_stats_path)
    actions = torch.tensor(ep_data["actions"][:T], dtype=torch.float32, device=device).unsqueeze(0)
    if actions.shape[-1] > 7:
        actions = actions[..., :7]
    actions = normalize_acs(actions, device, norm_stats_path=args.norm_stats_path)

    inputs = {}
    for c in condition_cameras:
        emb = torch.tensor(ep_data[f"{c}_embd"][:T], dtype=torch.float32, device=device).unsqueeze(0)
        inputs[c] = ensure_patch_grid(emb, PATCH_SIDE_DINOV3)[:, :H]
    inp_states = states[:, :H]
    inp_acs = actions[:, :H]

    # pred_imgs[cam_idx][t] = image for camera at timestep t
    pred_imgs = [[all_gt_imgs[i][t] for t in range(T)] for i in range(len(predict_cameras))]

    # Rollout: predict frames H, H+1, ..., T-1
    for k in range(T - H):
        preds, pred_state = transition(inputs, inp_states, inp_acs)
        next_embds = {c: preds[c][:, -1:] for c in predict_cameras}
        next_state = pred_state[:, -1:]

        for i, cam in enumerate(predict_cameras):
            if cam in decoders:
                emb = next_embds[cam]
                with torch.no_grad():
                    pred_img, _ = decoders[cam](emb)
                pred_img = rearrange(pred_img, "(b t) c h w -> b t c h w", t=1)
                pred_img = pred_img[0, 0].cpu().numpy().transpose(1, 2, 0)
                pred_imgs[i][H + k] = np.clip(pred_img, 0, 1)

        # Shift window for next step
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

    # Build video: each frame = [gt_row | pred_row | diff_row], cameras concatenated horizontally
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
            raise SystemExit(f"Unknown camera: {c}. Choose from {list(CAMERA_CONFIG.keys())}")

    if args.wandb and HAS_WANDB:
        from config import get_wandb_config
        wandb_cfg = get_wandb_config("train_wm", args)
        wandb_cfg.update(cameras=cameras, condition_cameras=condition_cameras, predict_cameras=predict_cameras)
        wandb.init(project="dino-wm-tactile", name=f"wm_{'+'.join(cameras)}", config=wandb_cfg)

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # cuDNN benchmark: find optimal conv algorithms (faster when input sizes are fixed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    BL = args.segment_length
    print('segment length: ', BL)
    H = BL - 1
    EVAL_H = 16
    device = args.device
    is_consolidated = not args.no_consolidated

    if args.segment_sampling == "weighted" and args.weight_source == "gripper_change_json":
        if not args.gripper_change_json_path or not os.path.isfile(args.gripper_change_json_path):
            raise FileNotFoundError(
                "gripper_change_json_path must point to existing file when weight_source=gripper_change_json"
            )

    train_ds = TactileTrajectoryDataset(
        args.hdf5_path,
        cameras=cameras,
        segment_length=BL,
        split="train",
        num_test=args.num_test,
        is_consolidated=is_consolidated,
        seed=args.seed,
        segment_sampling=args.segment_sampling,
        gripper_state_idx=-1,
        gripper_change_weight=args.gripper_change_weight,
        weight_source=args.weight_source,
        gripper_change_json_path=args.gripper_change_json_path,
    )
    eval_ds = TactileTrajectoryDataset(
        args.hdf5_path,
        cameras=cameras,
        segment_length=BL,
        split="test",
        num_test=args.num_test,
        is_consolidated=is_consolidated,
        seed=args.seed,
    )
    imagine_ds = TactileTrajectoryDataset(
        args.hdf5_path,
        cameras=cameras,
        segment_length=32,
        split="test",
        num_test=args.num_test,
        is_consolidated=is_consolidated,
        seed=args.seed,
    )

    train_sampler = train_ds.get_sampler()
    base_dl_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.num_workers > 0,
        "persistent_workers": args.num_workers > 0,
    }
    train_dl_kwargs = {**base_dl_kwargs}
    if train_sampler is not None:
        train_dl_kwargs["sampler"] = train_sampler
        train_dl_kwargs["shuffle"] = False
    else:
        train_dl_kwargs["shuffle"] = True
    eval_dl_kwargs = {**base_dl_kwargs, "shuffle": True}
    train_loader = iter(DataLoader(train_ds, **train_dl_kwargs))
    eval_loader = iter(DataLoader(eval_ds, **eval_dl_kwargs))
    imagine_dl_kwargs = {**base_dl_kwargs, "batch_size": 1, "shuffle": True}
    imagine_loader = iter(DataLoader(imagine_ds, **imagine_dl_kwargs))

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
        num_frames=BL - 1,
        dropout=0.1,
    ).to(device)
    if not args.no_compile and hasattr(torch, "compile"):
        transition = torch.compile(transition, mode="reduce-overhead")
        print("Using torch.compile (mode=reduce-overhead)")
    transition.train()

    param_groups = [
        {"params": transition.transformer.parameters(), "lr": 5e-5},
        {"params": transition.state_head.parameters(), "lr": 5e-5},
        {"params": transition.action_encoder.parameters(), "lr": 5e-4},
        {"params": [transition.pos_embedding], "lr": 5e-4},
        {"params": [transition.temp_embedding], "lr": 5e-4},
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
            print(f"Loaded decoders for: {list(decoders.keys())}")

    for i in tqdm(range(args.iters), desc="Training", unit="iter"):
        try:
            data = next(train_loader)
        except StopIteration:
            train_loader = iter(DataLoader(train_ds, **train_dl_kwargs))
            data = next(train_loader)

        # Build camera inputs: (B, T-1, N, D)
        non_blocking = base_dl_kwargs.get("pin_memory", False)
        camera_inputs = {}
        camera_outputs = {}
        for cam in cameras:
            embd = data[f"{cam}_embd"].to(device, non_blocking=non_blocking)
            embd = ensure_patch_grid(embd, PATCH_SIDE_DINOV3)
            inputs_t = embd[:, :-1]
            output_t = embd[:, 1:]
            camera_inputs[cam] = inputs_t
            camera_outputs[cam] = output_t

        cond_inputs = {c: camera_inputs[c] for c in condition_cameras}

        data_state = data["state"].to(device, non_blocking=non_blocking)
        if args.normalize_states:
            data_state = normalize_states(data_state, device, args.norm_stats_path)
        inputs_states = data_state[:, :-1]
        output_state = data_state[:, 1:]

        data_acs = data["action"].to(device, non_blocking=non_blocking)
        if data_acs.shape[-1] > 7:
            data_acs = data_acs[..., :7]
        norm_acs = normalize_acs(data_acs, device, norm_stats_path=args.norm_stats_path)
        acs = norm_acs[:, :-1]

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            preds, pred_state = transition(cond_inputs, inputs_states, acs)
            loss_tf = nn.MSELoss()(pred_state, output_state)
            for cam in predict_cameras:
                loss_tf = loss_tf + nn.MSELoss()(preds[cam], camera_outputs[cam])

        # Autoregressive loss
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            detach_preds = {c: preds[c].detach() for c in predict_cameras}
            detach_pred_state = pred_state.detach()

            if args.ar_loss_mode == "single":
                # Default: 1 step ahead - predict step 2 from step 0->1
                ar_inputs = {}
                for cam in condition_cameras:
                    first = camera_inputs[cam][:, [0]]
                    if cam in predict_cameras:
                        next_frame = detach_preds[cam][:, [0]]
                    else:
                        next_frame = camera_outputs[cam][:, [0]]
                    ar_inputs[cam] = torch.cat([first, next_frame], dim=1)
                ar_states = torch.cat([data_state[:, [0]], detach_pred_state[:, [0]]], dim=1)
                ar_acs = norm_acs[:, [0, 1]]

                preds_ar, pred_state_ar = transition(ar_inputs, ar_states, ar_acs)
                loss_ar = nn.MSELoss()(pred_state_ar[:, 1], data_state[:, 2])
                for cam in predict_cameras:
                    loss_ar = loss_ar + nn.MSELoss()(preds_ar[cam][:, 1], camera_outputs[cam][:, 1])
            else:
                # multi_step: sample k from 3-5, rollout k steps, sum loss over steps 1..k
                k = random.randint(3, 5)
                # segment has frames 0..segment_length-1; we need GT for frames 2..k+1
                max_k = data_state.shape[1] - 2  # need data_state[:, step+1] and camera_outputs[:, step]
                k = min(k, max(1, max_k))
                if k < 1:
                    loss_ar = torch.tensor(0.0, device=device)
                else:
                    # Start with frames [0, 1] (use pred for frame 1 if we predict it, else GT)
                    cur_inputs = {}
                    for cam in condition_cameras:
                        first = camera_inputs[cam][:, [0]]
                        if cam in predict_cameras:
                            next_frame = detach_preds[cam][:, [0]]
                        else:
                            next_frame = camera_outputs[cam][:, [0]]
                        cur_inputs[cam] = torch.cat([first, next_frame], dim=1)
                    cur_states = torch.cat([data_state[:, [0]], detach_pred_state[:, [0]]], dim=1)
                    loss_ar = torch.tensor(0.0, device=device)
                    mse = nn.MSELoss(reduction="mean")

                    for step in range(1, k + 1):
                        # Actions: from step-1 to step
                        ar_acs = norm_acs[:, [step - 1, step]]
                        preds_ar, pred_state_ar = transition(cur_inputs, cur_states, ar_acs)
                        # Compare predicted (step+1) with GT: state at step+1, camera at step
                        loss_ar = loss_ar + mse(pred_state_ar[:, 1], data_state[:, step + 1])
                        for cam in predict_cameras:
                            loss_ar = loss_ar + mse(preds_ar[cam][:, 1], camera_outputs[cam][:, step])
                        # Shift window for next step: use pred as new last frame
                        if step < k:
                            for cam in condition_cameras:
                                if cam in predict_cameras:
                                    next_frame = preds_ar[cam][:, [1]].detach()
                                else:
                                    next_frame = camera_outputs[cam][:, [step]]
                                cur_inputs[cam] = torch.cat([cur_inputs[cam][:, 1:], next_frame], dim=1)
                            cur_states = torch.cat([cur_states[:, 1:], pred_state_ar[:, [1]].detach()], dim=1)

        loss = loss_tf + args.ar_loss_weight * loss_ar

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if args.wandb and HAS_WANDB:
            log_dict = {"train_loss": loss_tf.item(), "train_loss_ar": loss_ar.item()}
            for cam in predict_cameras:
                log_dict[f"loss_{cam}"] = nn.MSELoss()(preds[cam], camera_outputs[cam]).item()
            wandb.log(log_dict)

        if i % 50 == 0:
            parts = [f"TF: {loss_tf.item():.4f}", f"AR: {loss_ar.item():.4f}"]
            for cam in predict_cameras:
                parts.append(f"{cam}: {nn.MSELoss()(preds[cam], camera_outputs[cam]).item():.4f}")
            print(f"\rIter {i}, " + ", ".join(parts), end="", flush=True)

        if (i + 1) % args.eval_every == 0:
            try:
                eval_data = next(eval_loader)
            except StopIteration:
                eval_loader = iter(DataLoader(eval_ds, **eval_dl_kwargs))
                eval_data = next(eval_loader)

            transition.eval()
            with torch.no_grad():
                non_blocking = base_dl_kwargs.get("pin_memory", False)
                eval_inputs = {}
                eval_outputs = {}
                for cam in cameras:
                    embd = eval_data[f"{cam}_embd"].to(device, non_blocking=non_blocking)
                    embd = ensure_patch_grid(embd, PATCH_SIDE_DINOV3)
                    eval_inputs[cam] = embd[:, :-1]
                    eval_outputs[cam] = embd[:, 1:]
                cond_inputs = {c: eval_inputs[c] for c in condition_cameras}
                eval_states = eval_data["state"].to(device, non_blocking=non_blocking)[:, :-1]
                eval_output_state = eval_data["state"].to(device, non_blocking=non_blocking)[:, 1:]
                if args.normalize_states:
                    eval_states = normalize_states(eval_states, device, args.norm_stats_path)
                    eval_output_state = normalize_states(eval_output_state, device, args.norm_stats_path)
                eval_acs_raw = eval_data["action"].to(device, non_blocking=non_blocking)
                if eval_acs_raw.shape[-1] > 7:
                    eval_acs_raw = eval_acs_raw[..., :7]
                eval_acs = normalize_acs(eval_acs_raw, device, norm_stats_path=args.norm_stats_path)[:, :-1]

                preds_eval, pred_state_eval = transition(cond_inputs, eval_states, eval_acs)
                eval_loss = nn.MSELoss()(pred_state_eval, eval_output_state)
                for cam in predict_cameras:
                    eval_loss = eval_loss + nn.MSELoss()(preds_eval[cam], eval_outputs[cam])

            print()
            print(f"Iter {i}, Eval Loss: {eval_loss.item():.4f}")

            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(transition.state_dict(), f"{args.output_dir}/wm_tactile_iter{i+1}.pth")
            if eval_loss.item() < best_eval:
                best_eval = eval_loss.item()
                torch.save(transition.state_dict(), f"{args.output_dir}/best_wm_tactile.pth")

            if args.wandb and HAS_WANDB:
                wandb.log({"eval_loss": eval_loss.item()})

            # Log eval video every N evals (top=GT, middle=pred, bottom=diff; cameras concatenated)
            eval_count = (i + 1) // args.eval_every
            should_log_video = (
                args.wandb and HAS_WANDB and decoders
                and eval_count % args.video_every_eval == 0
            )
            if should_log_video:
                video = create_eval_video(
                    transition,
                    decoders,
                    eval_ds,
                    predict_cameras,
                    condition_cameras,
                    args,
                    device,
                )
                if video is not None:
                    wandb.log({"eval_video": wandb.Video(video, fps=args.video_fps, format="mp4")})

            transition.train()


if __name__ == "__main__":
    main()
