from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from spiral_markers.ml.runtime_env import patch_windows_platform_for_torch

patch_windows_platform_for_torch()

import torch
from torch import nn


def angle_to_vec(angle_deg: float) -> tuple[float, float]:
    angle_rad = np.deg2rad(float(angle_deg))
    return float(np.cos(angle_rad)), float(np.sin(angle_rad))


def vec_to_angle_deg(x: float, y: float) -> float:
    return float(np.rad2deg(np.arctan2(float(y), float(x))))


def crop_spiral_patch(
    image_rgb: np.ndarray,
    center_xy: tuple[float, float],
    box_size_px: float,
    out_size: int = 64,
) -> np.ndarray:
    patch_size = max(8, int(round(float(box_size_px))))
    patch = cv2.getRectSubPix(
        image_rgb.astype(np.float32),
        (patch_size, patch_size),
        (float(center_xy[0]), float(center_xy[1])),
    )
    patch = cv2.resize(patch, (int(out_size), int(out_size)), interpolation=cv2.INTER_LINEAR)
    patch = np.clip(patch, 0.0, 1.0).astype(np.float32)
    return patch


def build_spiral_node_features(
    image_rgb: np.ndarray,
    detections: list[dict[str, Any]],
    image_size: tuple[int, int],
    crop_size: int = 64,
    max_nodes: int = 24,
) -> dict[str, torch.Tensor]:
    height, width = int(image_size[0]), int(image_size[1])
    selected = sorted(detections, key=lambda item: float(item.get("confidence", 1.0)), reverse=True)[:max_nodes]
    patches: list[np.ndarray] = []
    geom_feats: list[list[float]] = []
    for det in selected:
        x0, y0, x1, y1 = [float(v) for v in det["bbox_xyxy"]]
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        w = max(x1 - x0, 2.0)
        h = max(y1 - y0, 2.0)
        cls = int(det["class_id"])
        conf = float(det.get("confidence", 1.0))
        patch = crop_spiral_patch(image_rgb, (cx, cy), max(w, h), out_size=crop_size)
        patches.append(np.transpose(patch, (2, 0, 1)))
        class_one_hot = [0.0, 0.0, 0.0]
        if 0 <= cls < 3:
            class_one_hot[cls] = 1.0
        geom_feats.append(
            [
                cx / max(width, 1),
                cy / max(height, 1),
                w / max(width, 1),
                h / max(height, 1),
                conf,
                *class_one_hot,
            ]
        )
    node_count = len(selected)
    if node_count == 0:
        patches = [np.zeros((3, crop_size, crop_size), dtype=np.float32)]
        geom_feats = [[0.0] * 8]
        node_count = 0
    return {
        "patches": torch.from_numpy(np.stack(patches, axis=0)).float(),
        "geom": torch.tensor(geom_feats, dtype=torch.float32),
        "node_count": torch.tensor([int(node_count)], dtype=torch.int64),
    }


@dataclass
class P4ReasonerOutput:
    query_presence_logits: torch.Tensor
    query_center_xy: torch.Tensor
    query_heading_vec: torch.Tensor
    query_blur_vec: torch.Tensor
    query_blur_len: torch.Tensor
    query_id_logits: torch.Tensor
    node_robot_logits: torch.Tensor
    node_slot_logits: torch.Tensor
    node_blur_vec: torch.Tensor
    node_blur_len: torch.Tensor


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


class P4BlurReasoner(nn.Module):
    def __init__(
        self,
        crop_size: int = 64,
        token_dim: int = 192,
        num_robot_queries: int = 4,
        num_robot_ids: int = 8,
        num_slots: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.crop_size = int(crop_size)
        self.token_dim = int(token_dim)
        self.num_robot_queries = int(num_robot_queries)
        self.num_robot_ids = int(num_robot_ids)
        self.num_slots = int(num_slots)

        self.patch_encoder = _PatchEncoder(out_dim=128)
        self.geom_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
        )
        self.node_fusion = nn.Sequential(
            nn.Linear(128 + 64, token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.node_pos = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.query_embed = nn.Parameter(torch.randn(num_robot_queries, token_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.query_presence_head = nn.Linear(token_dim, 1)
        self.query_center_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, 2))
        self.query_heading_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, 2))
        self.query_blur_vec_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, 2))
        self.query_blur_len_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, 1))
        self.query_id_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, num_robot_ids))

        self.node_robot_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, num_robot_queries + 1))
        self.node_slot_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, num_slots))
        self.node_blur_vec_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, 2))
        self.node_blur_len_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, 1))

    def forward(
        self,
        patches: torch.Tensor,
        geom: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> P4ReasonerOutput:
        # patches: [B, N, 3, H, W], geom: [B, N, 8]
        bsz, num_nodes = patches.shape[:2]
        patch_feat = self.patch_encoder(patches.reshape(bsz * num_nodes, *patches.shape[2:])).reshape(bsz, num_nodes, -1)
        geom_feat = self.geom_encoder(geom.reshape(bsz * num_nodes, -1)).reshape(bsz, num_nodes, -1)
        node_tokens = self.node_fusion(torch.cat([patch_feat, geom_feat], dim=-1))
        node_tokens = node_tokens + self.node_pos(geom[..., :2])
        if node_mask is None:
            encoded = self.encoder(node_tokens)
        else:
            encoded = self.encoder(node_tokens, src_key_padding_mask=node_mask)

        query = self.query_embed.unsqueeze(0).expand(bsz, -1, -1)
        decoded = self.decoder(
            tgt=query,
            memory=encoded,
            memory_key_padding_mask=node_mask,
        )
        heading_vec = self.query_heading_head(decoded)
        heading_vec = heading_vec / heading_vec.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        blur_vec = self.query_blur_vec_head(decoded)
        blur_vec = blur_vec / blur_vec.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)

        node_blur_vec = self.node_blur_vec_head(encoded)
        node_blur_vec = node_blur_vec / node_blur_vec.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)

        return P4ReasonerOutput(
            query_presence_logits=self.query_presence_head(decoded).squeeze(-1),
            query_center_xy=self.query_center_head(decoded).sigmoid(),
            query_heading_vec=heading_vec,
            query_blur_vec=blur_vec,
            query_blur_len=self.query_blur_len_head(decoded).squeeze(-1).relu(),
            query_id_logits=self.query_id_head(decoded),
            node_robot_logits=self.node_robot_head(encoded),
            node_slot_logits=self.node_slot_head(encoded),
            node_blur_vec=node_blur_vec,
            node_blur_len=self.node_blur_len_head(encoded).squeeze(-1).relu(),
        )


def pad_node_batch(items: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_nodes = max(int(item["patches"].shape[0]) for item in items)
    bsz = len(items)
    crop_h = int(items[0]["patches"].shape[-2])
    crop_w = int(items[0]["patches"].shape[-1])
    patch_batch = torch.zeros((bsz, max_nodes, 3, crop_h, crop_w), dtype=torch.float32)
    geom_batch = torch.zeros((bsz, max_nodes, 8), dtype=torch.float32)
    node_mask = torch.ones((bsz, max_nodes), dtype=torch.bool)
    for idx, item in enumerate(items):
        count = int(item["patches"].shape[0])
        patch_batch[idx, :count] = item["patches"]
        geom_batch[idx, :count] = item["geom"]
        node_mask[idx, :count] = False
    return patch_batch, geom_batch, node_mask
