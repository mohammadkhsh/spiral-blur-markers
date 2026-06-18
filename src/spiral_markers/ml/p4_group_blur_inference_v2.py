from __future__ import annotations

from typing import Any

import numpy as np
import torch

from spiral_markers.ml.p4_ai_inference import _build_classical_groups
from spiral_markers.ml.p4_grouped_blur_estimator_v2 import (
    P4GroupedBlurEstimatorV2,
    build_group_blur_features,
    compose_relative_angle_deg,
    display_angle_from_center_to_point,
)


def _prepare_group_batch(
    image_rgb: np.ndarray,
    groups: list[dict[str, Any]],
) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    group_indices: list[int] = []
    robot_crops: list[np.ndarray] = []
    slot_patches: list[np.ndarray] = []
    slot_classes: list[np.ndarray] = []

    for group_index, group in enumerate(groups):
        selected_slots = dict(group.get("selected_slots", {}))
        if any(slot_index not in selected_slots for slot_index in (0, 1, 2, 3)):
            continue
        slot_centers = {
            slot_index: tuple(float(v) for v in selected_slots[slot_index]["center_xy"])
            for slot_index in (0, 1, 2, 3)
        }
        slot_class_ids = {
            slot_index: int(selected_slots[slot_index]["class_id"])
            for slot_index in (0, 1, 2, 3)
        }
        features = build_group_blur_features(
            image_rgb=image_rgb,
            slot_centers_xy=slot_centers,
            slot_class_ids=slot_class_ids,
        )
        group_indices.append(int(group_index))
        robot_crops.append(features["robot_crop"])
        slot_patches.append(features["slot_patches"])
        slot_classes.append(features["slot_classes"])

    if not group_indices:
        return group_indices, torch.empty(0), torch.empty(0), torch.empty(0)
    return (
        group_indices,
        torch.from_numpy(np.stack(robot_crops, axis=0)).float(),
        torch.from_numpy(np.stack(slot_patches, axis=0)).float(),
        torch.from_numpy(np.stack(slot_classes, axis=0)).float(),
    )


@torch.no_grad()
def predict_group_blur_robots_v2(
    blur_model: P4GroupedBlurEstimatorV2,
    image_rgb: np.ndarray,
    det_rows: list[dict[str, Any]],
    device: torch.device,
    blur_presence_thresh: float = 0.5,
    blur_length_thresh_px: float = 0.35,
    num_clusters: int = 4,
    dedupe_radius_px: float = 28.0,
) -> list[dict[str, Any]]:
    groups = _build_classical_groups(
        det_rows,
        num_clusters=int(num_clusters),
        dedupe_radius_px=float(dedupe_radius_px),
    )
    if not groups:
        return []

    group_indices, robot_crops, slot_patches, slot_classes = _prepare_group_batch(image_rgb, groups)
    blur_predictions: dict[int, dict[str, float]] = {}
    if group_indices:
        robot_crops = robot_crops.to(device)
        slot_patches = slot_patches.to(device)
        slot_classes = slot_classes.to(device)
        output = blur_model(slot_patches, slot_classes, robot_crops)
        presence = torch.sigmoid(output.blur_presence_logit).detach().cpu().numpy()
        blur_len = output.blur_length.detach().cpu().numpy() * 30.0
        axis_vec = output.blur_axis_vec.detach().cpu().numpy()
        sign_prob = torch.sigmoid(output.blur_sign_logit).detach().cpu().numpy()
        for batch_index, group_index in enumerate(group_indices):
            relative_angle_deg = compose_relative_angle_deg(
                float(axis_vec[batch_index, 0]),
                float(axis_vec[batch_index, 1]),
                bool(float(sign_prob[batch_index]) >= 0.5),
            )
            blur_predictions[int(group_index)] = {
                "presence": float(presence[batch_index]),
                "length_px": float(max(0.0, blur_len[batch_index])),
                "relative_angle_deg": float(relative_angle_deg),
                "sign_prob": float(sign_prob[batch_index]),
            }

    rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        center_xy = tuple(float(v) for v in group["center_xy"])
        heading_deg = float(group["heading_deg"]) if group.get("heading_deg") is not None else 0.0
        zero_center_xy = group.get("zero_center_xy")
        blur_angle_deg = 0.0
        blur_length_px = 0.0
        blur_source = "default_zero"
        blur_sign_prob = 0.5
        pred = blur_predictions.get(int(group_index))
        if pred is not None and zero_center_xy is not None:
            heading_display_deg = display_angle_from_center_to_point(center_xy, tuple(float(v) for v in zero_center_xy))
            blur_length_px = float(pred["length_px"])
            blur_sign_prob = float(pred["sign_prob"])
            if float(pred["presence"]) >= float(blur_presence_thresh) and float(blur_length_px) >= float(blur_length_thresh_px):
                blur_angle_deg = float((heading_display_deg + float(pred["relative_angle_deg"])) % 360.0)
                blur_source = "grouped_blur_axis_sign_head"
            else:
                blur_angle_deg = 0.0
                blur_source = "grouped_blur_axis_sign_head_no_blur"
        rows.append(
            {
                "confidence": float(group["confidence"]),
                "center_xy": center_xy,
                "center_source": str(group["center_source"]),
                "heading_deg": heading_deg,
                "heading_source": str(group.get("heading_source", "unavailable")),
                "zero_center_xy": zero_center_xy,
                "selected_slots": dict(group.get("selected_slots", {})),
                "blur_angle_deg": float(blur_angle_deg),
                "blur_length_px": float(blur_length_px),
                "blur_sign_prob": float(blur_sign_prob),
                "blur_source": blur_source,
                "robot_id": int(group["robot_id"]) if group.get("robot_id") is not None else 0,
            }
        )
    rows.sort(key=lambda row: float(row.get("confidence", 0.0)), reverse=True)
    return rows
