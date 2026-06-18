from __future__ import annotations

from typing import Iterable

import numpy as np


def polar_grid(shape: tuple[int, int], center_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(shape, dtype=np.float32)
    dx = xx - center_xy[0]
    dy = yy - center_xy[1]
    radius = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    return radius, angle


def normalize_angle_rad(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def normalize_angle_deg(angle_deg: np.ndarray | float) -> np.ndarray | float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def suggested_num_legs(base_legs: int, theta_deg: float, min_legs: int = 2) -> int:
    if abs(theta_deg) < 1.0e-6:
        return min_legs
    return max(min_legs, int(round(base_legs * abs(np.sin(np.deg2rad(theta_deg))))))


def annulus_bounds(
    radius: float,
    ring_count: int,
    inner_hole_ratio: float = 0.18,
    gap_ratio: float = 0.08,
) -> list[tuple[float, float]]:
    if ring_count <= 0:
        return []
    inner_radius = 0.0 if float(inner_hole_ratio) <= 0.0 else max(2.0, radius * inner_hole_ratio)
    raw_edges = np.linspace(inner_radius, radius, ring_count + 1)
    gap = gap_ratio * (radius - inner_radius) / max(ring_count, 1)
    bounds: list[tuple[float, float]] = []
    for idx, (left, right) in enumerate(zip(raw_edges[:-1], raw_edges[1:])):
        inner = left + gap * 0.5
        outer = max(left + gap * 0.75, right - gap * 0.5)
        if idx == 0 and inner_radius <= 0.0:
            inner = 0.0
        bounds.append((inner, outer))
    return bounds


def clockwise_sort_indices(points_xy: np.ndarray, center_xy: tuple[float, float], heading_deg: float = 0.0) -> np.ndarray:
    if len(points_xy) == 0:
        return np.array([], dtype=int)
    dx = points_xy[:, 0] - center_xy[0]
    dy = center_xy[1] - points_xy[:, 1]
    rel = normalize_angle_deg(np.degrees(np.arctan2(dy, dx)) - heading_deg)
    return np.argsort(-rel)


def circular_mean_deg(values_deg: Iterable[float]) -> float:
    values = np.asarray(list(values_deg), dtype=np.float32)
    if values.size == 0:
        return 0.0
    complex_mean = np.exp(1j * np.deg2rad(values)).mean()
    return float(np.degrees(np.angle(complex_mean)))


def project_points(homography: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xy.shape[0], 1), dtype=np.float32)
    homogeneous = np.concatenate([points_xy.astype(np.float32), ones], axis=1)
    warped = homogeneous @ homography.T
    return warped[:, :2] / np.clip(warped[:, 2:3], 1.0e-6, None)


def rotation_matrix(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)


def angle_from_center_deg(point_xy: tuple[float, float], center_xy: tuple[float, float]) -> float:
    dx = point_xy[0] - center_xy[0]
    dy = center_xy[1] - point_xy[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def circle_intersection_area(
    center_a_xy: tuple[float, float],
    radius_a: float,
    center_b_xy: tuple[float, float],
    radius_b: float,
) -> float:
    radius_a = max(float(radius_a), 0.0)
    radius_b = max(float(radius_b), 0.0)
    if radius_a <= 0.0 or radius_b <= 0.0:
        return 0.0
    distance = float(np.linalg.norm(np.asarray(center_a_xy, dtype=np.float32) - np.asarray(center_b_xy, dtype=np.float32)))
    if distance >= radius_a + radius_b:
        return 0.0
    if distance <= abs(radius_a - radius_b):
        return float(np.pi * min(radius_a, radius_b) ** 2)
    term_a = radius_a * radius_a * np.arccos(np.clip((distance * distance + radius_a * radius_a - radius_b * radius_b) / (2.0 * distance * radius_a), -1.0, 1.0))
    term_b = radius_b * radius_b * np.arccos(np.clip((distance * distance + radius_b * radius_b - radius_a * radius_a) / (2.0 * distance * radius_b), -1.0, 1.0))
    term_c = 0.5 * np.sqrt(
        max(
            0.0,
            (-distance + radius_a + radius_b)
            * (distance + radius_a - radius_b)
            * (distance - radius_a + radius_b)
            * (distance + radius_a + radius_b),
        )
    )
    return float(term_a + term_b - term_c)


def circle_iou(
    center_a_xy: tuple[float, float],
    radius_a: float,
    center_b_xy: tuple[float, float],
    radius_b: float,
) -> float:
    area_a = float(np.pi * max(float(radius_a), 0.0) ** 2)
    area_b = float(np.pi * max(float(radius_b), 0.0) ** 2)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    intersection = circle_intersection_area(center_a_xy, radius_a, center_b_xy, radius_b)
    union = area_a + area_b - intersection
    return float(intersection / max(union, 1.0e-6))
