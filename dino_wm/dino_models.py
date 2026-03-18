# Based on DINO-WM https://arxiv.org/abs/2411.04983



import json
import os

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Tuple, Optional, List, Dict
from torchvision import transforms
from scipy.spatial.transform import Rotation

class NormStats:
    """
    Load norm stats once, reuse for normalize/unnormalize.
    """

    _DEFAULT_MIN_AC = [-0.78933347, -1.0, -0.95038878, -0.3243517, -0.30636792, -0.30071826, -1.0]
    _DEFAULT_MAX_AC = [0.89928758, 0.71893158, 0.69869383, 0.32456627, 0.51343921, 0.28401476, 1.0]
    _cache: Dict[str, Dict[str, torch.Tensor]] = {}

    @staticmethod
    def _load(norm_stats_path: Optional[str]) -> Dict[str, torch.Tensor]:
        cache_key = norm_stats_path if (norm_stats_path and os.path.isfile(norm_stats_path)) else "_default"
        if cache_key in NormStats._cache:
            return NormStats._cache[cache_key]
        if cache_key == "_default":
            stats = {
                "min_ac": torch.tensor(NormStats._DEFAULT_MIN_AC, dtype=torch.float32),
                "max_ac": torch.tensor(NormStats._DEFAULT_MAX_AC, dtype=torch.float32),
                "min_state": None,
                "max_state": None,
            }
        else:
            with open(norm_stats_path) as f:
                info = json.load(f)
            print(f"Loaded norm stats: {info}")
            stats = {
                "min_ac": torch.tensor(info["min_acs"], dtype=torch.float32),
                "max_ac": torch.tensor(info["max_acs"], dtype=torch.float32),
                "min_state": torch.tensor(info["min_states"], dtype=torch.float32),
                "max_state": torch.tensor(info["max_states"], dtype=torch.float32),
            }
        NormStats._cache[cache_key] = stats
        return stats

    def __init__(self, norm_stats_path: Optional[str] = None, device: str = "cuda:0"):
        self._path = norm_stats_path
        self._device = device
        self._stats = self._load(norm_stats_path)
        self._min_ac = self._stats["min_ac"].to(device)
        self._max_ac = self._stats["max_ac"].to(device)
        min_s, max_s = self._stats["min_state"], self._stats["max_state"]
        self._min_state = min_s.to(device) if min_s is not None else None
        self._max_state = max_s.to(device) if max_s is not None else None

    def normalize_acs(self, acs: torch.Tensor) -> torch.Tensor:
        ac_dim = acs.shape[-1]
        min_ac = self._min_ac[:ac_dim]
        max_ac = self._max_ac[:ac_dim]
        return (acs - min_ac) / (max_ac - min_ac).clamp(min=1e-6)

    def unnormalize_acs(self, acs: torch.Tensor) -> torch.Tensor:
        ac_dim = acs.shape[-1]
        min_ac = self._min_ac[:ac_dim]
        max_ac = self._max_ac[:ac_dim]
        return acs * (max_ac - min_ac) + min_ac

    def normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        if self._min_state is None:
            return states
        state_dim = states.shape[-1]
        min_s = self._min_state[:state_dim]
        max_s = self._max_state[:state_dim]
        return (states - min_s) / (max_s - min_s).clamp(min=1e-6)

    def unnormalize_states(self, states: torch.Tensor) -> torch.Tensor:
        if self._min_state is None:
            return states
        state_dim = states.shape[-1]
        min_s = self._min_state[:state_dim]
        max_s = self._max_state[:state_dim]
        return states * (max_s - min_s) + min_s


def batch_quat_to_rotvec(quaternions):
    """
    Convert a batch of quaternions to axis-angle using PyTorch and scipy.

    Args:
        quaternions (torch.Tensor): A tensor of shape (N, 4), where each quaternion is (w, x, y, z).

    Returns:
        axes (torch.Tensor): A tensor of shape (N, 3), representing the rotation axes.
        angles (torch.Tensor): A tensor of shape (N,), representing the rotation angles in radians.
    """
    # Convert PyTorch tensor to NumPy array
    quaternions_np = quaternions.cpu().numpy()

    # Use scipy for the quaternion-to-axis-angle conversion
    r = Rotation.from_quat(quaternions_np)
    rotvecs = r.as_rotvec()
    return rotvecs

def batch_rotvec_to_quat(rotvecs):
    """
    Convert a batch of quaternions to axis-angle using PyTorch and scipy.

    Args:
        quaternions (torch.Tensor): A tensor of shape (N, 4), where each quaternion is (w, x, y, z).

    Returns:
        axes (torch.Tensor): A tensor of shape (N, 3), representing the rotation axes.
        angles (torch.Tensor): A tensor of shape (N,), representing the rotation angles in radians.
    """
    # Convert PyTorch tensor to NumPy array
    rotvecs_np = rotvecs.cpu().numpy()

    # Use scipy for the quaternion-to-axis-angle conversion
    r = Rotation.from_rotvec(rotvecs_np)
    quaternions = r.as_quat()
    return quaternions

class ResidualBlock2(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock2, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Decoder(nn.Module):
    def __init__(self, in_channels=384, out_channels=3):
        super(Decoder, self).__init__()
        
        # Two residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock2(in_channels),
            ResidualBlock2(in_channels),            
            ResidualBlock2(in_channels),
            ResidualBlock2(in_channels)

        )
        
        # Three transposed convolutions to go from 16x16 to 224x224
        self.transposed_convs = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # New intermediate layer
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # New additional layer with no change in resolution
            nn.ConvTranspose2d(in_channels // 8, in_channels // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels // 8, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.resize_transform = transforms.Resize((224, 224))

        
    
    def forward(self, x):

        x = x.view(-1, 16, 16, 384)  # Reshape to (16, 16, 384) where 16x16 is the spatial grid
        x = x.permute(0, 3, 1, 2)        # Pass through residual blocks
        x = self.residual_blocks(x)
        # Pass through transposed convolutions
        x = self.transposed_convs(x)
        x = self.resize_transform(x)
        
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0., 
                 num_frames: int = 2, patches_per_frame: int = 256):
        super().__init__()
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Register buffer instead of creating mask every forward pass
        mask = self._create_causal_mask(num_frames, patches_per_frame)
        self.register_buffer("mask", mask)
        
    def _create_causal_mask(self, num_frames: int, patches_per_frame: int) -> torch.Tensor:
        total_patches = num_frames * patches_per_frame
        mask = torch.zeros(total_patches, total_patches)
        
        for i in range(num_frames):
            start_idx = i * patches_per_frame
            end_idx = (i + 1) * patches_per_frame
            
            # Allow attention within current frame
            mask[start_idx:end_idx, start_idx:end_idx] = 1
            
            # Allow attention to previous frames
            if i > 0:
                mask[start_idx:end_idx, :start_idx] = 1
                
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Use registered mask buffer
        mask = self.mask[:seq_len, :seq_len]
        dots = dots.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0., 
                 num_frames: int = 2, patches_per_frame: int = 256):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads, dim_head, dropout, num_frames, patches_per_frame)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class VideoTransformer(nn.Module):
    def __init__(
        self,
        *,
        image_size: Tuple[int, int],
        dim: int,
        ac_dim: int,
        state_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        num_frames: int = 2,
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.device = device
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(device)
        
        # Improved action embedding
        self.action_encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, ac_dim),
            nn.LayerNorm(ac_dim)
        ).to(device)
        
        total_dim = 2*dim + ac_dim + state_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, total_dim) * 0.02)
        self.temp_embedding = nn.Parameter(torch.randn(1, num_frames, total_dim) * 0.02)
        
        self.dropout = nn.Dropout(emb_dropout)
        
        # Use TransformerBlock instead of separate components
        self.transformer = nn.ModuleList([
            TransformerBlock(total_dim, heads, dim_head, mlp_dim, dropout, num_frames)
            for _ in range(depth)
        ])
        
        # Separate prediction heads
        self.wrist_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, dim)
        )
                
        
        self.front_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, dim)
        )
        
        self.state_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, state_dim)
        )

        self.failure_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, 1)
        )

        

    
    def forward(
        self,
        video1: torch.Tensor,
        video2: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x = self.forward_features(video1, video2, states, actions)

        # Generate predictions
        pred1 = self.front_head(x)
        pred2 = self.wrist_head(x)
        state_preds = self.state_pred(x)
        failure_preds = self.failure_pred(x)
        
        return pred1, pred2, state_preds, failure_preds

    def forward_features(self,
        video1: torch.Tensor,
        video2: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode actions
        action_embeddings = self.action_encoder(actions).unsqueeze(2).expand(-1, -1, 256, -1)
        state_embeddings = states.unsqueeze(2).expand(-1, -1, 256, -1)
        
        # Combine features
        batch_size, num_frames, _, _ = video1.shape
    
        x = torch.cat((video1, video2, action_embeddings, state_embeddings), dim=3)
        # Add positional embeddings
        x = x + self.pos_embedding
        x = x + self.temp_embedding[:, :num_frames].unsqueeze(2)

        # Reshape for transformer
        x = rearrange(x, 'b s n d -> b (s n) d')
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer:
            x = block(x)
            
        # Reshape back
        x = rearrange(x, 'b (s n) d -> b s n d', s=num_frames)
        return x

    def failure_pred(self, features):
        failure_preds = self.failure_head(features)
        failure_preds = torch.mean(failure_preds, dim=2)  # Average over patches
        return failure_preds
    
    def state_pred(self, features):
        state_preds = self.state_head(features)
        state_preds = torch.mean(state_preds, dim=2)  # Average over patches
        return state_preds


    @torch.no_grad()
    def get_dino_features(self, video: torch.Tensor) -> torch.Tensor:
        """Extract DINO features from video frames."""
        b, f, c, h, w = video.shape
        video = video.view(b * f, c, h, w)
        features = self.dino.forward_features(video)['x_norm_patchtokens']
        return features.view(b, f, -1, features.shape[-1])


class TactileVideoTransformer(nn.Module):
    """
    Flexible world model for tactile dataset with configurable cameras.
    Conditions on a subset of camera embeddings and predicts future embeddings
    for selected cameras. Uses DINOv3 (768-dim) for vision, supports AnyTouch (512-dim) for tactile.
    """

    # Default embedding dims: DINOv3 ViT-B/16 = 768, AnyTouch = 512
    DEFAULT_CAMERA_DIMS = {"camera_0": 768, "camera_1": 768, "camera_2": 512}

    def __init__(
        self,
        *,
        cameras: List[str],
        camera_dims: Optional[Dict[str, int]] = None,
        condition_cameras: Optional[List[str]] = None,
        predict_cameras: Optional[List[str]] = None,
        common_dim: int = 384,
        ac_dim: int = 10,
        state_dim: int = 8,
        patches_per_frame: int = 196,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 2048,
        num_frames: int = 3,
        dim_head: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        camera_dims = camera_dims or self.DEFAULT_CAMERA_DIMS
        condition_cameras = condition_cameras or cameras
        predict_cameras = predict_cameras or cameras

        for c in condition_cameras + predict_cameras:
            if c not in camera_dims:
                raise ValueError(f"Unknown camera '{c}'. Choose from {list(camera_dims.keys())}")

        self.cameras = cameras
        self.condition_cameras = condition_cameras
        self.predict_cameras = predict_cameras
        self.camera_dims = camera_dims
        self.common_dim = common_dim
        self.patches_per_frame = patches_per_frame

        # Project each camera to common_dim
        self.project_in = nn.ModuleDict()
        for cam in condition_cameras:
            dim_in = camera_dims[cam]
            self.project_in[cam] = nn.Sequential(
                nn.Linear(dim_in, common_dim),
                nn.LayerNorm(common_dim),
                nn.ReLU(),
            )

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, ac_dim),
            nn.LayerNorm(ac_dim),
        )

        total_dim = len(condition_cameras) * common_dim + ac_dim + state_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, patches_per_frame, total_dim) * 0.02)
        self.temp_embedding = nn.Parameter(torch.randn(1, num_frames, total_dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.ModuleList([
            TransformerBlock(total_dim, heads, dim_head, mlp_dim, dropout, num_frames, patches_per_frame)
            for _ in range(depth)
        ])

        # Prediction heads: one per predict_camera, output to original dim
        self.heads = nn.ModuleDict()
        for cam in predict_cameras:
            dim_out = camera_dims[cam]
            self.heads[cam] = nn.Sequential(
                LayerNorm(total_dim),
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.Linear(total_dim, dim_out),
            )

        self.state_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, state_dim),
        )

    def forward(
        self,
        camera_inputs: Dict[str, torch.Tensor],
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            camera_inputs: Dict[cam_name, (B, T, num_patches, dim)]
            states: (B, T, state_dim)
            actions: (B, T, 7)

        Returns:
            preds: Dict[cam_name, (B, T, num_patches, dim)]
            pred_state: (B, T, state_dim)
        """
        x = self.forward_features(camera_inputs, states, actions)
        preds = {}
        for cam in self.predict_cameras:
            preds[cam] = self.heads[cam](x)
        pred_state = self.state_head(x)
        pred_state = torch.mean(pred_state, dim=2)
        return preds, pred_state

    def forward_features(
        self,
        camera_inputs: Dict[str, torch.Tensor],
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate and process camera embeddings."""
        batch_size, num_frames, num_patches, _ = next(iter(camera_inputs.values())).shape
        if num_patches != self.patches_per_frame:
            raise ValueError(
                f"Expected {self.patches_per_frame} patches per frame, got {num_patches}. "
                "Use ensure_patch_grid to resize embeddings."
            )

        # Project each camera
        projected = []
        for cam in self.condition_cameras:
            if cam not in camera_inputs:
                raise ValueError(f"Missing camera '{cam}' in inputs")
            proj = self.project_in[cam](camera_inputs[cam])
            projected.append(proj)
        x = torch.cat(projected, dim=-1)

        # Action and state embeddings
        action_emb = self.action_encoder(actions).unsqueeze(2).expand(-1, -1, num_patches, -1)
        state_emb = states.unsqueeze(2).expand(-1, -1, num_patches, -1)
        x = torch.cat([x, action_emb, state_emb], dim=-1)

        x = x + self.pos_embedding
        x = x + self.temp_embedding[:, :num_frames].unsqueeze(2)

        x = rearrange(x, "b s n d -> b (s n) d")
        x = self.dropout(x)

        for block in self.transformer:
            x = block(x)

        x = rearrange(x, "b (s n) d -> b s n d", s=num_frames)
        return x


import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def horizontal_forward(network, x, input_shape=(-1,), output_shape=(-1,)):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    x = x.reshape(-1, *input_shape)
    x = network(x)
    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x

def create_normal_dist(
    x,
    std=None,
    mean_scale=1,
    init_std=0,
    min_std=0.1,
    activation=None,
    event_shape=None,
):
    if std == None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = F.softplus(std + init_std) + min_std
    else:
        mean = x
    dist = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist
    

class TransposedConvDecoder(nn.Module):
    def __init__(self, observation_shape=(3, 224, 224), emb_dim=512, activation=nn.ReLU, depth=64, kernel_size=5, stride=3):
        super().__init__()

        activation = activation()
        self.observation_shape = observation_shape
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.emb_dim = emb_dim

        self.network = nn.Sequential(
            nn.Linear(
                emb_dim, self.depth * 32
            ),
            nn.Unflatten(1, (self.depth * 32, 1)),
            nn.Unflatten(2, (1,1)),
            nn.ConvTranspose2d(
                self.depth * 32,
                self.depth * 8,
                self.kernel_size,
                self.stride,
                padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 8,
                self.depth * 4,
                self.kernel_size,
                self.stride,
                padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 4,
                self.depth * 2,
                self.kernel_size,
                self.stride,
                padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 2,
                self.depth * 1,
                self.kernel_size,
                self.stride,
                padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 1,
                self.observation_shape[0],
                self.kernel_size,
                self.stride,
                padding=1
            ),
            nn.Upsample(size=(observation_shape[1], observation_shape[2]), mode='bilinear', align_corners=False)
        )
        self.network.apply(initialize_weights)

    def forward(self, posterior):
        x = horizontal_forward(
            self.network, posterior, input_shape=[self.emb_dim],output_shape=self.observation_shape
        )
        dist = create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        img = dist.mean.squeeze(2)
        img = einops.rearrange(img, "b t c h w -> (b t) c h w")
        return img, torch.zeros(1).to(posterior.device) # dummy placeholder