"""
FVD, FID, LPIPS metrics adopted from SAILOR-FM.
Works directly with tensors, no checkpoint path or disk I/O.
I3D/Inception weights auto-download on first use.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg

# Lazy init
_I3D_FEATURES: Optional["I3DFeatures"] = None
_INCEPTION_FEATURES: Optional["InceptionV3Features"] = None

try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    HAS_INCEPTION = True
except ImportError:
    HAS_INCEPTION = False

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

_LPIPS_MODEL = None


def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of features. features: (N, D)."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Fréchet distance between two Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m} too large")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)


class I3DFeatures(nn.Module):
    """
    I3D (Inflated Inception V3) for FVD. Standard from Unterthiner et al. 2019.
    Weights auto-download from PyTorch-I3D.
    """

    _WEIGHT_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"

    def __init__(self):
        super().__init__()
        self.model = self._load_i3d_model()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _load_i3d_model():
        cache_dir = os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "i3d",
        )
        os.makedirs(cache_dir, exist_ok=True)
        filepath = os.path.join(cache_dir, "i3d_torchscript.pt")
        if not os.path.isfile(filepath):
            print(f"Downloading I3D weights to {filepath} ...")
            torch.hub.download_url_to_file(I3DFeatures._WEIGHT_URL, filepath, progress=True)
        return torch.jit.load(filepath, map_location="cpu")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, T, H, W) in [0, 1]. Returns (B, 400)."""
        x = x * 2 - 1
        if x.shape[-2:] != (224, 224):
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = x.reshape(B, T, C, 224, 224).permute(0, 2, 1, 3, 4).contiguous()
        return self.model(x, return_features=True)


@torch.no_grad()
def compute_fvd(
    real_videos: torch.Tensor,
    fake_videos: torch.Tensor,
    batch_size: int = 16,
    device: str = "cuda",
    verbose: bool = True,
) -> float:
    """
    Compute FVD between real and generated videos (I3D-based, SAILOR-FM style).

    Args:
        real_videos: (N, 3, T, H, W) or (N, T, 3, H, W) in [0, 1]
        fake_videos: (M, 3, T, H, W) or (M, T, 3, H, W) in [0, 1]
        batch_size: Batch size for feature extraction
        device: Device to run on
        verbose: Whether to print progress

    Returns:
        fvd: FVD score (lower is better). nan if N or M < 2 (singular covariance).
    """
    if real_videos.shape[0] < 2 or fake_videos.shape[0] < 2:
        return float("nan")
    if real_videos.ndim == 5 and real_videos.shape[2] == 3:
        real_videos = real_videos.permute(0, 2, 1, 3, 4)
    if fake_videos.ndim == 5 and fake_videos.shape[2] == 3:
        fake_videos = fake_videos.permute(0, 2, 1, 3, 4)

    global _I3D_FEATURES
    if _I3D_FEATURES is None:
        _I3D_FEATURES = I3DFeatures()
    _I3D_FEATURES.to(device)
    _I3D_FEATURES.eval()

    def extract(videos):
        feats = []
        for i in range(0, len(videos), batch_size):
            batch = videos[i : i + batch_size].to(device)
            feats.append(_I3D_FEATURES(batch).cpu().numpy())
        return np.concatenate(feats, axis=0)

    if verbose:
        print("Extracting I3D features from real videos...")
    real_features = extract(real_videos)
    if verbose:
        print("Extracting I3D features from fake videos...")
    fake_features = extract(fake_videos)

    mu_real, sigma_real = compute_statistics(real_features)
    mu_fake, sigma_fake = compute_statistics(fake_features)
    fvd = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    _I3D_FEATURES.cpu()
    torch.cuda.empty_cache()
    return fvd


# =============================================================================
# FID (Inception V3) - For Images
# =============================================================================


class InceptionV3Features(nn.Module):
    """Inception V3 for FID. Returns 2048-dim features."""

    def __init__(self):
        super().__init__()
        if not HAS_INCEPTION:
            raise ImportError("torchvision required for FID")
        self.inception = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,
        )
        self.inception.eval()
        self.inception.fc = nn.Identity()
        for p in self.inception.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [0, 1]. Returns (B, 2048)."""
        x = x * 2 - 1
        if x.shape[-2:] != (299, 299):
            x = nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return self.inception(x)


@torch.no_grad()
def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    """
    FID between real and fake images (Inception V3).
    real/fake: (N, 3, H, W) or (N, H, W, 3) in [0, 1].
    """
    if not HAS_INCEPTION or real_images.shape[0] < 2 or fake_images.shape[0] < 2:
        return float("nan")
    if real_images.ndim == 4 and real_images.shape[-1] == 3:
        real_images = real_images.permute(0, 3, 1, 2)
    if fake_images.ndim == 4 and fake_images.shape[-1] == 3:
        fake_images = fake_images.permute(0, 3, 1, 2)
    global _INCEPTION_FEATURES
    if _INCEPTION_FEATURES is None:
        _INCEPTION_FEATURES = InceptionV3Features()
    _INCEPTION_FEATURES.to(device)
    _INCEPTION_FEATURES.eval()

    def extract(imgs):
        feats = []
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i : i + batch_size].to(device)
            feats.append(_INCEPTION_FEATURES(batch).cpu().numpy())
        return np.concatenate(feats, axis=0)

    mu_r, sig_r = compute_statistics(extract(real_images))
    mu_f, sig_f = compute_statistics(extract(fake_images))
    fid_val = calculate_frechet_distance(mu_r, sig_r, mu_f, sig_f)
    _INCEPTION_FEATURES.cpu()
    torch.cuda.empty_cache()
    return fid_val


@torch.no_grad()
def compute_lpips_batch(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    """
    LPIPS between real and fake images. Batched for speed.
    real/fake: (N, 3, H, W) in [0, 1]. Resized to 224 internally.
    """
    if not HAS_LPIPS or len(real_images) == 0:
        return float("nan")
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="vgg")
    _LPIPS_MODEL.to(device)
    _LPIPS_MODEL.eval()
    if real_images.ndim == 4 and real_images.shape[-1] == 3:
        real_images = real_images.permute(0, 3, 1, 2)
    if fake_images.ndim == 4 and fake_images.shape[-1] == 3:
        fake_images = fake_images.permute(0, 3, 1, 2)
    x = real_images * 2 - 1
    y = fake_images * 2 - 1
    if x.shape[-2:] != (224, 224):
        x = nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        y = nn.functional.interpolate(y, size=(224, 224), mode="bilinear", align_corners=False)
    scores = []
    for i in range(0, len(x), batch_size):
        bx = x[i : i + batch_size].to(device)
        by = y[i : i + batch_size].to(device)
        scores.append(_LPIPS_MODEL(bx, by).mean().item())
    _LPIPS_MODEL.cpu()
    torch.cuda.empty_cache()
    return float(np.mean(scores))


def frames_to_image_tensor(
    frames: list[np.ndarray],
    resolution: int = 224,
    max_frames: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert frames (H,W,3) to (N, 3, H, W) tensor in [0, 1].
    If max_frames set, subsample uniformly.
    """
    import cv2

    if not frames:
        return torch.zeros(0, 3, resolution, resolution)
    if max_frames and len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]
    arrs = []
    for f in frames:
        if f.shape[-1] == 3:
            a = np.asarray(f, dtype=np.float32) / 255.0
        else:
            a = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            a = np.asarray(a, dtype=np.float32) / 255.0
        if a.shape[:2] != (resolution, resolution):
            a = cv2.resize(a, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        arrs.append(a)
    arr = np.stack(arrs, axis=0)
    arr = arr.transpose(0, 3, 1, 2)
    return torch.from_numpy(arr).float()


def frames_to_fvd_tensor(
    frames_list: list[np.ndarray],
    num_frames: int = 16,
    resolution: int = 224,
) -> torch.Tensor:
    """
    Convert list of frames (H, W, 3) to (1, 3, T, H, W) tensor in [0, 1].
    For multiple videos, pass list of frame lists and stack to (N, 3, T, H, W).
    """
    import cv2

    if not frames_list:
        return torch.zeros(1, 3, 0, resolution, resolution)
    n = min(num_frames, len(frames_list))
    indices = np.linspace(0, len(frames_list) - 1, n, dtype=int)
    selected = [frames_list[i] for i in indices]
    arrs = []
    for f in selected:
        if f.shape[-1] == 3:
            f = np.asarray(f, dtype=np.float32) / 255.0
        else:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f = np.asarray(f, dtype=np.float32) / 255.0
        if f.shape[:2] != (resolution, resolution):
            f = cv2.resize(f, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        arrs.append(f)
    arr = np.stack(arrs, axis=0)  # (T, H, W, 3)
    arr = arr.transpose(0, 3, 1, 2)  # (T, 3, H, W)
    t = torch.from_numpy(arr).float().unsqueeze(0)  # (1, T, 3, H, W)
    return t.permute(0, 2, 1, 3, 4)  # (1, 3, T, H, W)
