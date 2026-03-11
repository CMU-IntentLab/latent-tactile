"""
Train DINO decoder on tactile dataset from hdf5_to_dataset_tactile.py.

Supports training modes:
- tactile (default): Vision (camera_0, camera_1) share one decoder (emb_dim=384);
  tactile (camera_2) has its own decoder with native embedding dim. No projectors.
- per_camera: One decoder per camera
- all: One shared decoder for all cameras
- custom: Subset via --cameras

Dataset format (from hdf5_to_dataset_tactile.py):
- camera_0, camera_1: DINO patch embeddings (14×14)
- camera_2: AnyTouch patch embeddings

RAE loss (--loss rae): L1 + LPIPS + adversarial, following
  "Diffusion Transformers with Representation Autoencoders" (https://arxiv.org/abs/2510.11690)
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from einops import rearrange
from torchvision import models
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

from tactile_dataset import TactileTrajectoryDataset, CAMERA_CONFIG

# Provide no-op dist_fn for VQVAE Quantize (dino_decoder uses it when training)
import dino_decoder as _dino_dec_mod
if not hasattr(_dino_dec_mod, "dist_fn"):
    _dino_dec_mod.dist_fn = type("DistFn", (), {"all_reduce": lambda x: None})()

from dino_decoder import VQVAE


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_patch_grid(embd: torch.Tensor, target_side: int) -> torch.Tensor:
    """
    Reshape or interpolate patch embeddings to target_side x target_side grid.
    Input: (B, T, num_patches, emb_dim)
    Output: (B, T, target_side*target_side, emb_dim)
    """
    B, T, N, D = embd.shape
    side = int(N ** 0.5)
    if side * side != N:
        # Non-square: flatten and pad/truncate to target_side^2
        target_n = target_side * target_side
        if N >= target_n:
            embd = embd[:, :, :target_n, :]
        else:
            pad = torch.zeros(B, T, target_n - N, D, device=embd.device, dtype=embd.dtype)
            embd = torch.cat([embd, pad], dim=2)
        N = target_n
        side = target_side

    if side != target_side:
        # Interpolate spatial dims: (B*T, D, side, side) -> (B*T, D, target_side, target_side)
        embd = rearrange(embd, "b t (h w) d -> (b t) d h w", h=side, w=side)
        embd = nn.functional.interpolate(
            embd, size=(target_side, target_side), mode="bilinear", align_corners=False
        )
        embd = rearrange(embd, "(b t) d h w -> b t (h w) d", h=target_side, w=target_side)
    return embd


def parse_args():
    p = argparse.ArgumentParser(
        description="Train DINO decoder on tactile dataset (camera_0, camera_1, camera_2)"
    )
    p.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="Path to consolidated HDF5 or directory with .hdf5 files",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["tactile", "per_camera", "all", "custom"],
        default="tactile",
        help="tactile: vision (cam0+1) share decoder 384, tactile has own decoder; per_camera/all/custom",
    )
    p.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="For custom mode: comma-separated list, e.g. 'camera_0,camera_1' or 'camera_0,camera_2'",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    p.add_argument(
        "--segment_length",
        type=int,
        default=1,
        help="Timesteps per segment (1 = single frame)",
    )
    p.add_argument(
        "--num_test",
        type=int,
        default=20,
        help="Number of trajectories for validation split",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=5000,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    p.add_argument(
        "--patch_side",
        type=int,
        default=14,
        help="Expected patch grid side (14 for DINO 14x14)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Log to Weights & Biases",
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
        help="Random seed for train/test split",
    )
    p.add_argument(
        "--loss",
        type=str,
        choices=["mse", "l1", "l1_edge", "perceptual", "l1_perceptual", "rae"],
        default="mse",
        help="Loss: mse, l1, l1_edge, perceptual, l1_perceptual, or rae (L1+LPIPS+GAN per paper)",
    )
    p.add_argument(
        "--l1_weight",
        type=float,
        default=1.0,
        help="Weight for L1 term when using l1_edge or l1_perceptual loss",
    )
    p.add_argument(
        "--edge_weight",
        type=float,
        default=0.5,
        help="Weight for edge/gradient term when using l1_edge loss",
    )
    p.add_argument(
        "--perceptual_weight",
        type=float,
        default=0.1,
        help="Weight for perceptual term when using l1_perceptual loss",
    )
    p.add_argument(
        "--resize_to_224",
        action="store_true",
        help="Resize images to 224x224 (default: keep original size)",
    )
    # RAE loss args (from https://arxiv.org/abs/2510.11690)
    p.add_argument(
        "--lpips_weight",
        type=float,
        default=1.0,
        help="Weight for LPIPS in RAE loss (paper: 1.0)",
    )
    p.add_argument(
        "--gan_weight",
        type=float,
        default=0.75,
        help="Base weight for GAN in RAE loss before adaptive lambda (paper: 0.75)",
    )
    p.add_argument(
        "--disc_start_iter",
        type=int,
        default=600,
        help="Start training discriminator at this iter (paper: epoch 6)",
    )
    p.add_argument(
        "--adv_start_iter",
        type=int,
        default=800,
        help="Start adding adversarial loss to decoder at this iter (paper: epoch 8)",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers for parallel data loading (0 = main thread only)",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision for faster training",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster training (PyTorch 2.0+)",
    )
    return p.parse_args()


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gradient (edge) loss: L1 between spatial gradients of pred and target."""
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return nn.functional.l1_loss(pred_dx, target_dx) + nn.functional.l1_loss(pred_dy, target_dy)


# ImageNet normalization for VGG perceptual loss
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
VGG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss. Compares features at relu1_2, relu2_2, relu3_3, relu4_3."""

    def __init__(self, device: torch.device):
        super().__init__()
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        except AttributeError:
            vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]) # relu3_3
        self.slice4 = nn.Sequential(*list(vgg.children())[16:23]) # relu4_3
        for p in self.parameters():
            p.requires_grad = False
        self.to(device)
        self.mean = VGG_MEAN.to(device)
        self.std = VGG_STD.to(device)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [0, 1] to ImageNet stats."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Inputs: (N, C, H, W) in [0, 1]. Resizes to 224 if needed."""
        if pred.shape[-2:] != (224, 224):
            pred = nn.functional.interpolate(pred, size=(224, 224), mode="bilinear", align_corners=False)
        if target.shape[-2:] != (224, 224):
            target = nn.functional.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)
        pred = self._normalize(pred)
        target = self._normalize(target)
        p1, p2, p3, p4 = self._features(pred)
        t1, t2, t3, t4 = self._features(target)
        return (
            nn.functional.l1_loss(p1, t1)
            + nn.functional.l1_loss(p2, t2)
            + nn.functional.l1_loss(p3, t3)
            + nn.functional.l1_loss(p4, t4)
        )

    def _features(self, x: torch.Tensor):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4


_perceptual_loss_cache = {}
_lpips_loss_cache = {}


def _get_lpips_loss(device: torch.device):
    """LPIPS for RAE. Expects [0,1] or [-1,1]; we use [0,1] and normalize to [-1,1] internally."""
    if device not in _lpips_loss_cache:
        if HAS_LPIPS:
            _lpips_loss_cache[device] = lpips.LPIPS(net="vgg").to(device)
        else:
            _lpips_loss_cache[device] = _get_perceptual_loss(device)
    return _lpips_loss_cache[device]


def _ensure_rgb(x: torch.Tensor) -> torch.Tensor:
    """Ensure (N,C,H,W) has C=3 for LPIPS/discriminator."""
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x[:, :3]


def lpips_loss_fn(pred: torch.Tensor, target: torch.Tensor, device: torch.device) -> torch.Tensor:
    """LPIPS loss. Inputs (N,C,H,W) in [0,1]. Normalize to [-1,1] for lpips."""
    pred = _ensure_rgb(pred)
    target = _ensure_rgb(target)
    if HAS_LPIPS:
        # LPIPS expects [-1, 1]
        pred_n = pred * 2 - 1
        target_n = target * 2 - 1
        if pred_n.shape[-2:] != (224, 224):
            pred_n = nn.functional.interpolate(pred_n, size=(224, 224), mode="bilinear", align_corners=False)
        if target_n.shape[-2:] != (224, 224):
            target_n = nn.functional.interpolate(target_n, size=(224, 224), mode="bilinear", align_corners=False)
        return _get_lpips_loss(device)(pred_n, target_n).mean()
    else:
        ploss = _get_perceptual_loss(device)
        return ploss(pred, target)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for 224x224 images. Outputs per-patch real/fake logits."""

    def __init__(self, ndf=64, n_layers=3):
        super().__init__()
        layers = []
        in_c = 3
        for n in range(n_layers):
            out_c = min(ndf * (2 ** n), 512)
            layers += [
                nn.Conv2d(in_c, out_c, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.InstanceNorm2d(out_c),
            ]
            in_c = out_c
        self.model = nn.Sequential(*layers)
        self.final = nn.Conv2d(in_c, 1, 4, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.model(x)
        return self.final(h)


def _get_perceptual_loss(device: torch.device) -> PerceptualLoss:
    if device not in _perceptual_loss_cache:
        _perceptual_loss_cache[device] = PerceptualLoss(device)
    return _perceptual_loss_cache[device]


def compute_loss(pred: torch.Tensor, target: torch.Tensor, args, return_rae_components: bool = False):
    """Compute loss based on args.loss. Resizes target to pred size if needed.
    When return_rae_components=True and loss=rae, returns (loss, (l1, lpips)).
    """
    if target.shape[-2:] != pred.shape[-2:]:
        target = nn.functional.interpolate(
            target.reshape(-1, *target.shape[2:]),
            size=pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        target = target.reshape(pred.shape)
    if args.loss == "mse":
        return nn.MSELoss()(pred, target)
    if args.loss == "l1":
        return nn.L1Loss()(pred, target)
    if args.loss == "l1_edge":
        l1 = nn.L1Loss()(pred, target)
        edge = edge_loss(pred, target)
        return args.l1_weight * l1 + args.edge_weight * edge
    if args.loss == "perceptual":
        ploss = _get_perceptual_loss(pred.device)
        pred_flat = pred.reshape(-1, *pred.shape[2:])
        target_flat = target.reshape(-1, *target.shape[2:])
        return ploss(pred_flat, target_flat)
    if args.loss == "l1_perceptual":
        l1 = nn.L1Loss()(pred, target)
        ploss = _get_perceptual_loss(pred.device)
        pred_flat = pred.reshape(-1, *pred.shape[2:])
        target_flat = target.reshape(-1, *target.shape[2:])
        perc = ploss(pred_flat, target_flat)
        return args.l1_weight * l1 + args.perceptual_weight * perc
    if args.loss == "rae":
        # RAE uses L1 + LPIPS; GAN is handled separately in training loop
        l1 = nn.L1Loss()(pred, target)
        pred_flat = pred.reshape(-1, *pred.shape[2:])
        target_flat = target.reshape(-1, *target.shape[2:])
        if target_flat.shape[-2:] != pred_flat.shape[-2:]:
            target_flat = nn.functional.interpolate(
                target_flat, size=pred_flat.shape[-2:], mode="bilinear", align_corners=False
            )
        lpips_val = lpips_loss_fn(pred_flat, target_flat, pred.device)
        loss = l1 + args.lpips_weight * lpips_val
        if return_rae_components:
            return loss, (l1.item(), lpips_val.item())
        return loss
    raise ValueError(f"Unknown loss: {args.loss}")


def compute_rae_loss_components(pred: torch.Tensor, target: torch.Tensor, args) -> tuple:
    """Return (l1, lpips) for RAE loss logging."""
    if target.shape[-2:] != pred.shape[-2:]:
        target = nn.functional.interpolate(
            target.reshape(-1, *target.shape[2:]),
            size=pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        target = target.reshape(pred.shape)
    l1 = nn.L1Loss()(pred, target)
    pred_flat = pred.reshape(-1, *pred.shape[2:])
    target_flat = target.reshape(-1, *target.shape[2:])
    if target_flat.shape[-2:] != pred_flat.shape[-2:]:
        target_flat = nn.functional.interpolate(
            target_flat, size=pred_flat.shape[-2:], mode="bilinear", align_corners=False
        )
    lpips_val = lpips_loss_fn(pred_flat, target_flat, pred.device)
    return l1.item(), lpips_val.item()


def get_cameras_to_train(args) -> list[list[str]]:
    """Return list of camera groups. Each group is trained together (one decoder per group)."""
    if args.mode == "tactile":
        return [["camera_0", "camera_1"], ["camera_2"]]
    if args.mode == "per_camera":
        return [[c] for c in CAMERA_CONFIG.keys()]
    if args.mode == "all":
        return [list(CAMERA_CONFIG.keys())]
    # custom
    if not args.cameras:
        raise ValueError("--cameras required when --mode custom")
    cams = [c.strip() for c in args.cameras.split(",")]
    for c in cams:
        if c not in CAMERA_CONFIG:
            raise ValueError(f"Unknown camera: {c}. Choose from {list(CAMERA_CONFIG.keys())}")
    return [cams]


def train_one_decoder(
    camera_group: list[str],
    train_ds: TactileTrajectoryDataset,
    eval_ds: TactileTrajectoryDataset,
    args,
) -> VQVAE:
    """Train a single decoder for the given camera group. No projectors."""
    device = torch.device(args.device)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Infer emb_dim from first sample
    sample = train_ds[0]
    dims = [int(sample[f"{cam}_embd"].shape[-1]) for cam in camera_group]
    if len(set(dims)) > 1:
        raise ValueError(f"Cameras in group have different emb_dims: {dict(zip(camera_group, dims))}")
    emb_dim = dims[0]
    print(f"  Inferred emb_dim={emb_dim}")

    decoder = VQVAE(emb_dim=emb_dim).to(device)
    if args.compile and hasattr(torch, "compile"):
        decoder = torch.compile(decoder, mode="reduce-overhead")
        print("  Using torch.compile (mode=reduce-overhead)")
    decoder_for_save = getattr(decoder, "_orig_mod", decoder)
    params = list(decoder.parameters())
    optimizer = AdamW(params, lr=args.lr)
    best_eval = float("inf")
    best_eval_adv = float("inf")  # best in adversarial phase (i >= adv_start_iter)
    train_iter = iter(train_loader)
    eval_iter = iter(eval_loader)

    discriminators = None
    disc_optimizer = None
    if args.loss == "rae":
        discriminators = nn.ModuleDict(
            {cam: PatchDiscriminator(ndf=64, n_layers=3).to(device) for cam in camera_group}
        )
        disc_optimizer = AdamW(
            [p for d in discriminators.values() for p in d.parameters()],
            lr=args.lr,
            betas=(0.5, 0.9),
        )
        print(f"  Using per-camera discriminators: {list(discriminators.keys())}")
        if not HAS_LPIPS:
            print("  Warning: lpips not installed. Using VGG perceptual as LPIPS fallback. pip install lpips for RAE.")

    group_name = "+".join(camera_group)
    if args.wandb and HAS_WANDB:
        wandb.init(project="dino-decoder-tactile", name=f"decoder_{group_name}")

    use_amp = args.amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    non_blocking = torch.cuda.is_available()
    if use_amp:
        print("  Using AMP (automatic mixed precision)")

    for i in range(args.iters):
        # Refresh loaders when exhausted
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data = next(train_iter)

        decoder.train()
        optimizer.zero_grad()
        total_loss = 0.0
        preds_dict = {}
        targets_dict = {}
        loss_l1_sum, loss_lpips_sum = 0.0, 0.0
        loss_gan_val, loss_disc_val = None, None
        loss_d_per_cam = {}

        with torch.amp.autocast("cuda", enabled=use_amp):
            for cam in camera_group:
                embd = data[f"{cam}_embd"].to(device, non_blocking=non_blocking)
                target = data[f"{cam}_image"].to(device, non_blocking=non_blocking)

                B, T, N, D = embd.shape
                embd = ensure_patch_grid(embd, target_side=args.patch_side)

                pred, _ = decoder(embd)
                pred = rearrange(pred, "(b t) c h w -> b t c h w", b=B, t=T)
                target = target.permute(0, 1, 4, 2, 3)  # B T H W C -> B T C H W
                if target.shape[2] != 3:
                    target = target[:, :, :3]
                preds_dict[cam] = pred
                targets_dict[cam] = target

                if args.loss == "rae":
                    loss, (l1_val, lpips_val) = compute_loss(pred, target, args, return_rae_components=True)
                    loss_l1_sum += l1_val
                    loss_lpips_sum += lpips_val
                else:
                    loss = compute_loss(pred, target, args)
                total_loss = total_loss + loss

            total_loss = total_loss / len(camera_group)
            if args.loss == "rae":
                loss_l1_sum /= len(camera_group)
                loss_lpips_sum /= len(camera_group)
            loss_rec = total_loss.item() if args.loss == "rae" else None

            # RAE: add adversarial loss with adaptive lambda (paper Sec 3, Table 12)
            # Per-camera discriminators to prevent mode collapse across different camera modalities
            if args.loss == "rae" and i >= args.adv_start_iter and discriminators is not None:
            total_gan = 0.0
            for cam in camera_group:
                pred = preds_dict[cam]
                target = targets_dict[cam]
                disc = discriminators[cam]
                pred_flat = _ensure_rgb(pred.reshape(-1, *pred.shape[2:]))
                target_flat = _ensure_rgb(target.reshape(-1, *target.shape[2:]))
                if pred_flat.shape[-2:] != (224, 224):
                    pred_224 = nn.functional.interpolate(
                        pred_flat, size=(224, 224), mode="bilinear", align_corners=False
                    )
                else:
                    pred_224 = pred_flat
                fake_score = disc(pred_224)
                gan_loss_gen = -fake_score.mean()  # hinge: generator maximizes D(fake)

                L_rec_cam = compute_loss(pred, target, args)
                grad_rec = torch.autograd.grad(
                    L_rec_cam, pred, retain_graph=True, create_graph=False
                )[0]
                grad_gan = torch.autograd.grad(
                    gan_loss_gen, pred, retain_graph=True, create_graph=False
                )[0]
                lam = (grad_rec.norm() / (grad_gan.norm() + 1e-8)).detach().clamp(0.01, 100)
                total_gan = total_gan + lam * gan_loss_gen
            total_gan = total_gan / len(camera_group)
            loss_gan_val = total_gan.item()
            total_loss = total_loss + args.gan_weight * total_gan

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # RAE: train discriminators (one per camera to prevent mode collapse)
        if args.loss == "rae" and i >= args.disc_start_iter and discriminators is not None:
            disc_optimizer.zero_grad()
            loss_d_total = 0.0
            loss_d_per_cam = {}
            for cam in camera_group:
                pred = preds_dict[cam].detach()
                target = targets_dict[cam]
                disc = discriminators[cam]
                pred_flat = _ensure_rgb(pred.reshape(-1, *pred.shape[2:]))
                target_flat = _ensure_rgb(target.reshape(-1, *target.shape[2:]))
                if pred_flat.shape[-2:] != (224, 224):
                    pred_224 = nn.functional.interpolate(
                        pred_flat, size=(224, 224), mode="bilinear", align_corners=False
                    )
                    target_224 = nn.functional.interpolate(
                        target_flat, size=(224, 224), mode="bilinear", align_corners=False
                    )
                else:
                    pred_224 = pred_flat
                    target_224 = target_flat
                real_score = disc(target_224)
                fake_score = disc(pred_224)
                loss_d_real = nn.functional.relu(1.0 - real_score).mean()
                loss_d_fake = nn.functional.relu(1.0 + fake_score).mean()
                loss_d_cam = loss_d_real + loss_d_fake
                loss_d_total = loss_d_total + loss_d_cam
                loss_d_per_cam[cam] = loss_d_cam.item()
            loss_disc_val = (loss_d_total / len(camera_group)).item()
            loss_d_scaled = loss_d_total / len(camera_group)
            if scaler is not None:
                scaler.scale(loss_d_scaled).backward()
                scaler.step(disc_optimizer)
                scaler.update()
            else:
                loss_d_scaled.backward()
                disc_optimizer.step()

        if args.wandb and HAS_WANDB:
            log_dict = {"train_loss": total_loss.item()}
            if args.loss == "rae":
                log_dict["loss_l1"] = loss_l1_sum
                log_dict["loss_lpips"] = loss_lpips_sum
                if loss_rec is not None:
                    log_dict["loss_rec"] = loss_rec
                if loss_gan_val is not None:
                    log_dict["loss_gan"] = loss_gan_val
                if loss_disc_val is not None:
                    log_dict["loss_disc"] = loss_disc_val
                for cam, val in loss_d_per_cam.items():
                    log_dict[f"loss_disc_{cam}"] = val
            wandb.log(log_dict)
        if i % 50 == 0:
            parts = [f"Loss: {total_loss.item():.4f}"]
            if args.loss == "rae":
                parts.append(f"L1: {loss_l1_sum:.4f}")
                parts.append(f"LPIPS: {loss_lpips_sum:.4f}")
                if loss_gan_val is not None:
                    parts.append(f"GAN: {loss_gan_val:.4f}")
                if loss_disc_val is not None:
                    parts.append(f"D: {loss_disc_val:.4f}")
            print(f"\r[{group_name}] Iter {i}, " + ", ".join(parts), end="", flush=True)

        if i % 100 == 0:
            try:
                eval_data = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_loader)
                eval_data = next(eval_iter)

            decoder.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                eval_loss = 0.0
                log_images = {}
                for cam in camera_group:
                    embd = eval_data[f"{cam}_embd"].to(device, non_blocking=non_blocking)
                    target = eval_data[f"{cam}_image"].to(device, non_blocking=non_blocking)
                    B, T, N, D = embd.shape
                    embd = ensure_patch_grid(embd, target_side=args.patch_side)
                    pred, _ = decoder(embd)
                    pred = rearrange(pred, "(b t) c h w -> b t c h w", b=B, t=T)
                    target = target.permute(0, 1, 4, 2, 3)
                    if target.shape[2] != 3:
                        target = target[:, :, :3]
                    eval_loss += compute_loss(pred, target, args).item()

                    if args.wandb and HAS_WANDB:
                        gt_img = target[0, 0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                        pred_img = pred[0, 0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                        log_images[f"eval_{cam}_gt"] = wandb.Image(gt_img, caption=f"{cam} ground truth")
                        log_images[f"eval_{cam}_pred"] = wandb.Image(pred_img, caption=f"{cam} decoded")
                eval_loss /= len(camera_group)

            in_adv_phase = (
                args.loss == "rae"
                and i >= args.adv_start_iter
                and discriminators is not None
            )
            if in_adv_phase:
                if eval_loss < best_eval_adv:
                    best_eval_adv = eval_loss
                    os.makedirs(args.output_dir, exist_ok=True)
                    base = f"decoder_{group_name.replace('+', '_')}_adv"
                    torch.save(decoder_for_save.state_dict(), os.path.join(args.output_dir, f"{base}.pth"))
                    if discriminators is not None:
                        torch.save(
                            discriminators.state_dict(),
                            os.path.join(args.output_dir, f"discriminators_{group_name.replace('+', '_')}_adv.pth"),
                        )
            elif eval_loss < best_eval:
                best_eval = eval_loss
                os.makedirs(args.output_dir, exist_ok=True)
                base = f"decoder_{group_name.replace('+', '_')}"
                torch.save(decoder_for_save.state_dict(), os.path.join(args.output_dir, f"{base}.pth"))
                if discriminators is not None:
                    torch.save(
                        discriminators.state_dict(),
                        os.path.join(args.output_dir, f"discriminators_{group_name.replace('+', '_')}.pth"),
                    )
            ## also save every 5000 iters
            if i % 5000 == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                ckpt_path = os.path.join(args.output_dir, f"decoder_{group_name.replace('+', '_')}_{i}.pth")
                torch.save(decoder_for_save.state_dict(), ckpt_path)
                if discriminators is not None:
                    disc_path = os.path.join(args.output_dir, f"discriminators_{group_name.replace('+', '_')}_{i}.pth")
                    torch.save(discriminators.state_dict(), disc_path)

            if args.wandb and HAS_WANDB:
                wandb.log({
                    "eval_loss": eval_loss,
                    "best_eval": best_eval,
                    "best_eval_adv": best_eval_adv,
                    **log_images,
                })
            print()
            best_str = f"best: {best_eval:.4f}"
            if args.loss == "rae" and discriminators is not None:
                best_str += f", best_adv: {best_eval_adv:.4f}"
            print(f"  Eval Loss: {eval_loss:.4f} ({best_str})")
            decoder.train()

    # Save last checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    base = f"decoder_{group_name.replace('+', '_')}_last"
    torch.save(decoder_for_save.state_dict(), os.path.join(args.output_dir, f"{base}.pth"))
    if discriminators is not None:
        torch.save(
            discriminators.state_dict(),
            os.path.join(args.output_dir, f"discriminators_{group_name.replace('+', '_')}_last.pth"),
        )
    print(f"  Saved last checkpoint to {args.output_dir}/")

    return decoder


def main():
    args = parse_args()
    camera_groups = get_cameras_to_train(args)

    is_consolidated = not args.no_consolidated
    for group in camera_groups:
        train_ds = TactileTrajectoryDataset(
            args.hdf5_path,
            cameras=group,
            segment_length=args.segment_length,
            split="train",
            num_test=args.num_test,
            is_consolidated=is_consolidated,
            seed=args.seed,
            resize_to_224=args.resize_to_224,
        )
        eval_ds = TactileTrajectoryDataset(
            args.hdf5_path,
            cameras=group,
            segment_length=args.segment_length,
            split="test",
            num_test=args.num_test,
            is_consolidated=is_consolidated,
            seed=args.seed,
            resize_to_224=args.resize_to_224,
        )
        print(f"Training decoder for {group}: {len(train_ds)} train, {len(eval_ds)} eval samples")
        train_one_decoder(group, train_ds, eval_ds, args)


if __name__ == "__main__":
    main()
