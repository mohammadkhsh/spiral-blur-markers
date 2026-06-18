from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from spiral_markers.ml.p4_grouped_blur_estimator import (
    _PatchEncoder,
    angle_to_vec_display,
    build_group_blur_features,
    display_angle_from_center_to_point,
    vec_to_angle_deg_display,
)


@dataclass
class P4GroupedBlurV2Output:
    blur_presence_logit: torch.Tensor
    blur_length: torch.Tensor
    blur_axis_vec: torch.Tensor
    blur_sign_logit: torch.Tensor


class P4GroupedBlurEstimatorV2(nn.Module):
    def __init__(
        self,
        slot_patch_size: int = 40,
        robot_crop_size: int = 128,
        token_dim: int = 160,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
    ) -> None:
        super().__init__()
        self.slot_patch_size = int(slot_patch_size)
        self.robot_crop_size = int(robot_crop_size)
        self.token_dim = int(token_dim)

        self.slot_patch_encoder = _PatchEncoder(out_dim=96)
        self.robot_crop_encoder = _PatchEncoder(out_dim=128)
        self.slot_class_encoder = nn.Sequential(
            nn.Linear(3, 24),
            nn.SiLU(inplace=True),
            nn.Linear(24, 32),
            nn.LayerNorm(32),
        )
        self.slot_fusion = nn.Sequential(
            nn.Linear(96 + 32, token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.robot_proj = nn.Sequential(
            nn.Linear(128, token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.slot_embed = nn.Parameter(torch.randn(4, token_dim) * 0.02)
        self.robot_token = nn.Parameter(torch.randn(1, token_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.blur_presence_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, 1))
        self.blur_length_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, 1))
        self.blur_axis_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, 2))

        # Sign branch uses both global pooled context and the predicted axis token.
        self.sign_head = nn.Sequential(
            nn.Linear(token_dim + 2, token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim, token_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim // 2, 1),
        )

    def forward(
        self,
        slot_patches: torch.Tensor,
        slot_classes: torch.Tensor,
        robot_crop: torch.Tensor,
    ) -> P4GroupedBlurV2Output:
        batch_size = slot_patches.shape[0]
        slot_feat = self.slot_patch_encoder(slot_patches.reshape(batch_size * 4, *slot_patches.shape[2:])).reshape(batch_size, 4, -1)
        class_feat = self.slot_class_encoder(slot_classes.reshape(batch_size * 4, -1)).reshape(batch_size, 4, -1)
        slot_tokens = self.slot_fusion(torch.cat([slot_feat, class_feat], dim=-1))
        slot_tokens = slot_tokens + self.slot_embed.unsqueeze(0)

        robot_feat = self.robot_crop_encoder(robot_crop)
        robot_token = self.robot_proj(robot_feat).unsqueeze(1) + self.robot_token.unsqueeze(0)
        tokens = torch.cat([robot_token, slot_tokens], dim=1)
        encoded = self.encoder(tokens)
        pooled = encoded[:, 0] + torch.mean(encoded[:, 1:], dim=1)

        blur_axis_vec = self.blur_axis_head(pooled)
        blur_axis_vec = blur_axis_vec / blur_axis_vec.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        sign_input = torch.cat([pooled, blur_axis_vec], dim=-1)

        return P4GroupedBlurV2Output(
            blur_presence_logit=self.blur_presence_head(pooled).squeeze(-1),
            blur_length=self.blur_length_head(pooled).squeeze(-1).relu(),
            blur_axis_vec=blur_axis_vec,
            blur_sign_logit=self.sign_head(sign_input).squeeze(-1),
        )


def axis_vec_to_angle_deg(axis_x: float, axis_y: float) -> float:
    # blur_axis_vec models (cos 2θ, sin 2θ); recover θ modulo 180 in [0, 180)
    return float((0.5 * vec_to_angle_deg_display(axis_x, axis_y)) % 180.0)


def compose_relative_angle_deg(
    axis_x: float,
    axis_y: float,
    sign_positive: bool,
) -> float:
    base_angle = axis_vec_to_angle_deg(axis_x, axis_y)
    return float((base_angle + (180.0 if sign_positive else 0.0)) % 360.0)


__all__ = [
    "P4GroupedBlurEstimatorV2",
    "P4GroupedBlurV2Output",
    "axis_vec_to_angle_deg",
    "compose_relative_angle_deg",
    "angle_to_vec_display",
    "build_group_blur_features",
    "display_angle_from_center_to_point",
]
