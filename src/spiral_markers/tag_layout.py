from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spiral_markers.io_utils import MethodConfig, resolve_baseline_identity_theta_deg


@dataclass
class TagPrimitive:
    center_xy: tuple[float, float]
    twist_angle_deg: float
    radius: float
    role: str
    slot_index: int
    bit_value: int | None = None


@dataclass
class AnchorPrimitive:
    center_xy: tuple[float, float]
    radius: float
    intensity: float
    index: int
    angle_deg: float


@dataclass
class RobotTagLayout:
    robot_id: int
    patch_size: int
    center_xy: tuple[float, float]
    orientation_heading_deg: float
    primitives: list[TagPrimitive]
    anchors: list[AnchorPrimitive]


def encode_robot_id_bits(robot_id: int, num_slots: int) -> list[int]:
    if num_slots <= 0:
        return []
    clipped = int(robot_id) % (2 ** num_slots)
    return [int((clipped >> shift) & 1) for shift in range(num_slots - 1, -1, -1)]


def decode_robot_id_bits(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return int(value)


def slot_angles_deg(cfg: MethodConfig) -> list[float]:
    total_slots = max(1, cfg.layout.identity_slots + 1)
    step = 360.0 / total_slots
    # Paper baseline uses clockwise ordering after the theta=0 orientation spiral.
    return [float(cfg.layout.orientation_slot_angle_deg - slot_index * step) for slot_index in range(total_slots)]


def _uses_sign_coded_identity(cfg: MethodConfig) -> bool:
    return str(cfg.synthesis.method_key).lower() in {
        "baseline",
        "paper_v4",
        "paper_v4_sharp",
        "paper_v4_sharp_native",
        "ecc_sr",
        "mb_psf",
        "aso",
    }


def build_robot_tag_layout(robot_id: int, cfg: MethodConfig) -> RobotTagLayout:
    angles = slot_angles_deg(cfg)
    patch_radius = cfg.layout.group_radius + cfg.synthesis.radius + 16.0
    patch_size = int(np.ceil(2.0 * patch_radius))
    center = np.array([0.5 * (patch_size - 1), 0.5 * (patch_size - 1)], dtype=np.float32)
    bits = encode_robot_id_bits(robot_id, cfg.layout.identity_slots)
    baseline_theta_id = resolve_baseline_identity_theta_deg(cfg.synthesis)

    primitives: list[TagPrimitive] = []
    for slot_index, angle_deg in enumerate(angles):
        angle_rad = np.deg2rad(angle_deg)
        offset = np.array(
            [cfg.layout.group_radius * np.cos(angle_rad), -cfg.layout.group_radius * np.sin(angle_rad)],
            dtype=np.float32,
        )
        primitive_center = tuple((center + offset).tolist())
        if slot_index == 0:
            primitives.append(
                TagPrimitive(
                    center_xy=primitive_center,
                    twist_angle_deg=float(cfg.layout.orientation_theta_deg),
                    radius=float(cfg.synthesis.radius),
                    role="orientation",
                    slot_index=0,
                    bit_value=None,
                )
            )
            continue
        bit_value = int(bits[slot_index - 1])
        if _uses_sign_coded_identity(cfg):
            twist = float(baseline_theta_id if bit_value == 1 else -baseline_theta_id)
        else:
            twist = float(cfg.synthesis.high_bit_angle_deg if bit_value == 1 else cfg.synthesis.low_bit_angle_deg)
        primitives.append(
            TagPrimitive(
                center_xy=primitive_center,
                twist_angle_deg=twist,
                radius=float(cfg.synthesis.radius),
                role="identity",
                slot_index=slot_index - 1,
                bit_value=bit_value,
            )
        )

    anchors: list[AnchorPrimitive] = []
    if cfg.layout.anchor_count > 0:
        anchor_radius = float(cfg.layout.anchor_size_ratio * cfg.layout.group_radius)
        count = min(cfg.layout.anchor_count, len(cfg.layout.anchor_angles_deg))
        for anchor_index in range(count):
            angle_deg = float(cfg.layout.anchor_angles_deg[anchor_index])
            ratio = float(cfg.layout.anchor_radius_ratios[min(anchor_index, len(cfg.layout.anchor_radius_ratios) - 1)])
            intensity = float(cfg.layout.anchor_intensities[min(anchor_index, len(cfg.layout.anchor_intensities) - 1)])
            angle_rad = np.deg2rad(angle_deg)
            offset = np.array(
                [
                    cfg.layout.group_radius * ratio * np.cos(angle_rad),
                    -cfg.layout.group_radius * ratio * np.sin(angle_rad),
                ],
                dtype=np.float32,
            )
            anchors.append(
                AnchorPrimitive(
                    center_xy=tuple((center + offset).tolist()),
                    radius=anchor_radius,
                    intensity=intensity,
                    index=anchor_index,
                    angle_deg=angle_deg,
                )
            )

    return RobotTagLayout(
        robot_id=int(robot_id),
        patch_size=patch_size,
        center_xy=(float(center[0]), float(center[1])),
        orientation_heading_deg=0.0,
        primitives=primitives,
        anchors=anchors,
    )
