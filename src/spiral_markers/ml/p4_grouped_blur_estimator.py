from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from spiral_markers.ml.runtime_env import patch_windows_platform_for_torch

patch_windows_platform_for_torch()

import torch
from torch import nn


def angle_to_vec_display(angle_deg: float) -> tuple[float, float]:
    angle_rad = np.deg2rad(float(angle_deg))
    return float(np.cos(angle_rad)), float(np.sin(angle_rad))


def vec_to_angle_deg_display(x: float, y: float) -> float:
    return float(np.rad2deg(np.arctan2(float(y), float(x))) % 360.0)


def display_angle_from_center_to_point(
    center_xy: tuple[float, float],
    point_xy: tuple[float, float],
) -> float:
    dx = float(point_xy[0]) - float(center_xy[0])
    dy = float(point_xy[1]) - float(center_xy[1])
    return float(np.rad2deg(np.arctan2(dy, dx)) % 360.0)


def _crop_patch(
    image_rgb: np.ndarray,
    center_xy: tuple[float, float],
    patch_size_px: float,
    out_size: int,
) -> np.ndarray:
    size_px = max(8, int(round(float(patch_size_px))))
    patch = cv2.getRectSubPix(
        image_rgb.astype(np.float32),
        (size_px, size_px),
        (float(center_xy[0]), float(center_xy[1])),
    )
    patch = cv2.resize(patch, (int(out_size), int(out_size)), interpolation=cv2.INTER_LINEAR)
    return np.clip(patch, 0.0, 1.0).astype(np.float32)


_CANONICAL_SLOT_CENTERS = {
    0: (96.0, 64.0),
    1: (64.0, 96.0),
    2: (32.0, 64.0),
    3: (64.0, 32.0),
}


def build_group_blur_features(
    image_rgb: np.ndarray,
    slot_centers_xy: dict[int, tuple[float, float]],
    slot_class_ids: dict[int, int],
    robot_crop_size: int = 128,
    slot_patch_size: int = 40,
    slot_patch_extent_px: float = 42.0,
) -> dict[str, np.ndarray]:
    required_slots = (0, 1, 2, 3)
    if any(slot_index not in slot_centers_xy for slot_index in required_slots):
        raise ValueError("All four slots are required to build grouped blur features.")

    src = np.asarray([slot_centers_xy[idx] for idx in required_slots], dtype=np.float32)
    dst = np.asarray([_CANONICAL_SLOT_CENTERS[idx] for idx in required_slots], dtype=np.float32)
    warp = cv2.getPerspectiveTransform(src, dst)
    canonical_rgb = cv2.warpPerspective(
        image_rgb.astype(np.float32),
        warp,
        (int(robot_crop_size), int(robot_crop_size)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    canonical_rgb = np.clip(canonical_rgb, 0.0, 1.0).astype(np.float32)

    slot_patches: list[np.ndarray] = []
    slot_classes: list[list[float]] = []
    for slot_index in required_slots:
        patch = _crop_patch(
            canonical_rgb,
            _CANONICAL_SLOT_CENTERS[slot_index],
            patch_size_px=float(slot_patch_extent_px),
            out_size=int(slot_patch_size),
        )
        slot_patches.append(np.transpose(patch, (2, 0, 1)))
        class_one_hot = [0.0, 0.0, 0.0]
        class_id = int(slot_class_ids[slot_index])
        if 0 <= class_id < 3:
            class_one_hot[class_id] = 1.0
        slot_classes.append(class_one_hot)

    return {
        "robot_crop": np.transpose(canonical_rgb, (2, 0, 1)).astype(np.float32),
        "slot_patches": np.stack(slot_patches, axis=0).astype(np.float32),
        "slot_classes": np.asarray(slot_classes, dtype=np.float32),
    }


@dataclass
class P4GroupedBlurOutput:
    blur_presence_logit: torch.Tensor
    blur_length: torch.Tensor
    blur_dir_vec: torch.Tensor
    blur_axis_vec: torch.Tensor


class _PatchEncoder(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class P4GroupedBlurEstimator(nn.Module):
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
        self.blur_dir_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, 2))
        self.blur_axis_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, 2))

    def forward(
        self,
        slot_patches: torch.Tensor,
        slot_classes: torch.Tensor,
        robot_crop: torch.Tensor,
    ) -> P4GroupedBlurOutput:
        # slot_patches: [B, 4, 3, H, W], slot_classes: [B, 4, 3], robot_crop: [B, 3, H, W]
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

        blur_dir_vec = self.blur_dir_head(pooled)
        blur_dir_vec = blur_dir_vec / blur_dir_vec.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        blur_axis_vec = self.blur_axis_head(pooled)
        blur_axis_vec = blur_axis_vec / blur_axis_vec.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)

        return P4GroupedBlurOutput(
            blur_presence_logit=self.blur_presence_head(pooled).squeeze(-1),
            blur_length=self.blur_length_head(pooled).squeeze(-1).relu(),
            blur_dir_vec=blur_dir_vec,
            blur_axis_vec=blur_axis_vec,
        )
