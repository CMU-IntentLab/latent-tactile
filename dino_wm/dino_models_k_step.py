"""
K-step dynamics model for tactile dataset (LPB-style).

Uses num_hist, num_pred, frameskip like LPB VisualDynamicsModel:
- num_hist: history (conditioning) frames
- num_pred: prediction (target) frames
- frameskip: collapse frameskip actions per frame into one vector
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from einops import rearrange

from dino_wm.dino_models import LayerNorm, TransformerBlock


class TactileKStepTransformer(nn.Module):
    """
    LPB-style world model: (e_0..e_{H-1}, s_0..s_{H-1}, act_0..act_{H-1}) -> predict e_H..e_{H+P-1}.

    Input:
        camera_inputs: Dict[cam, (B, num_hist, N, D)]
        states: (B, num_hist, state_dim)
        actions: (B, num_hist, frameskip*7) - per-frame collapsed actions

    Output:
        preds: Dict[cam, (B, num_pred, N, D)]
        pred_state: (B, num_pred, state_dim)
    """

    DEFAULT_CAMERA_DIMS = {"camera_0": 768, "camera_1": 768, "camera_2": 512}

    def __init__(
        self,
        *,
        cameras: List[str],
        camera_dims: Optional[Dict[str, int]] = None,
        condition_cameras: Optional[List[str]] = None,
        predict_cameras: Optional[List[str]] = None,
        common_dim: int = 384,
        ac_dim: int = 64,
        state_dim: int = 8,
        num_hist: int = 1,
        num_pred: int = 1,
        frameskip: int = 8,
        patches_per_frame: int = 196,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 2048,
        dim_head: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        camera_dims = camera_dims or self.DEFAULT_CAMERA_DIMS
        condition_cameras = condition_cameras or cameras
        predict_cameras = predict_cameras or cameras

        self.cameras = cameras
        self.condition_cameras = condition_cameras
        self.predict_cameras = predict_cameras
        self.camera_dims = camera_dims
        self.common_dim = common_dim
        self.patches_per_frame = patches_per_frame
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.frameskip = frameskip

        self.project_in = nn.ModuleDict()
        for cam in condition_cameras:
            dim_in = camera_dims[cam]
            self.project_in[cam] = nn.Sequential(
                nn.Linear(dim_in, common_dim),
                nn.LayerNorm(common_dim),
                nn.ReLU(),
            )

        self.action_per_frame_dim = frameskip * 7
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_per_frame_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, ac_dim),
            nn.LayerNorm(ac_dim),
        )

        total_dim = len(condition_cameras) * common_dim + ac_dim + state_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, patches_per_frame, total_dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)

        if num_hist > 1:
            self.temp_embedding = nn.Parameter(torch.randn(1, num_hist, total_dim) * 0.02)
        else:
            self.temp_embedding = None

        self.transformer = nn.ModuleList([
            TransformerBlock(
                total_dim, heads, dim_head, mlp_dim, dropout,
                num_frames=num_hist,
                patches_per_frame=patches_per_frame,
            )
            for _ in range(depth)
        ])

        # Prediction heads: output num_pred future frames per camera
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
            camera_inputs: Dict[cam_name, (B, num_hist, num_patches, dim)]
            states: (B, num_hist, state_dim)
            actions: (B, num_hist, frameskip*7) - per-frame collapsed actions

        Returns:
            preds: Dict[cam_name, (B, num_pred, num_patches, dim)]
            pred_state: (B, num_pred, state_dim)
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
        """
        Encode (e_0..e_{H-1}, s_0..s_{H-1}, act_0..act_{H-1}) and run transformer.

        Args:
            camera_inputs: Dict[cam_name, (B, num_hist, num_patches, dim)]
            states: (B, num_hist, state_dim)
            actions: (B, num_hist, frameskip*7)

        Returns:
            (B, num_pred, num_patches, total_dim) - representation for prediction
        """
        batch_size, num_frames, num_patches, _ = next(iter(camera_inputs.values())).shape

        projected = []
        for cam in self.condition_cameras:
            proj = self.project_in[cam](camera_inputs[cam])
            projected.append(proj)
        x = torch.cat(projected, dim=-1)

        # actions: (B, num_hist, frameskip*7) -> encode per frame
        action_emb = self.action_encoder(actions)  # (B, num_hist, ac_dim)
        action_emb = action_emb.unsqueeze(2).expand(-1, -1, num_patches, -1)
        state_emb = states.unsqueeze(2).expand(-1, -1, num_patches, -1)

        x = torch.cat([x, action_emb, state_emb], dim=-1)
        x = x + self.pos_embedding
        if self.temp_embedding is not None:
            x = x + self.temp_embedding[:, :num_frames].unsqueeze(2)  # (1, num_hist, 1, dim)
        x = self.dropout(x)

        x = rearrange(x, "b t n d -> b (t n) d")
        for block in self.transformer:
            x = block(x)
        x = rearrange(x, "b (t n) d -> b t n d", t=num_frames)
        x = x[:, -1:]  # Use last context frame for prediction
        return x
