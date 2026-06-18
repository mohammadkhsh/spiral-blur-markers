from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from spiral_markers.geometry import angle_from_center_deg, project_points, rotation_matrix
from spiral_markers.io_utils import MethodConfig, SheetConfig, ensure_dir
from spiral_markers.synthesis import generate_spiral_image
from spiral_markers.tag_layout import RobotTagLayout, build_robot_tag_layout


@dataclass
class RenderedTagGroundTruth:
    robot_id: int
    center_xy: tuple[float, float]
    heading_deg: float
    orientation_center_xy: tuple[float, float]
    spiral_centers_xy: list[tuple[float, float]]
    spiral_twists_deg: list[float]
    render_scale: float = 1.0
    effective_radius_px: float = 0.0
    fit_circle_radius_px: float = 0.0


@dataclass
class RenderedScene:
    image: np.ndarray
    tags: list[RenderedTagGroundTruth]
    color_image: np.ndarray | None = None


@dataclass
class RenderedRobotLayer:
    robot_id: int
    gray_layer: np.ndarray
    color_layer: np.ndarray | None
    alpha_layer: np.ndarray
    homography: np.ndarray
    placement: "ScenePlacement"
    gt: RenderedTagGroundTruth
    color_alpha_layer: np.ndarray | None = None


@dataclass
class RenderedLayeredScene:
    background_gray: np.ndarray
    background_rgb: np.ndarray | None
    clean_scene: RenderedScene
    layers: list[RenderedRobotLayer]


@dataclass
class ScenePlacement:
    robot_id: int
    center_xy: tuple[float, float]
    scale: float
    rotation_deg: float
    tilt_x_deg: float
    tilt_y_deg: float


def _alpha_blend(base: np.ndarray, patch: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if base.ndim == 3 and alpha.ndim == 2:
        alpha = alpha[..., None]
    return patch * alpha + base * (1.0 - alpha)


def render_robot_tag_patch(robot_id: int, cfg: MethodConfig) -> tuple[np.ndarray, np.ndarray, RobotTagLayout, float]:
    method_key = str(cfg.synthesis.method_key).lower()
    patch_scale_adjust = 1.0
    patch_cfg = cfg
    if method_key == "paper_v4_sharp":
        oversample = 4
        patch_scale_adjust = 1.0 / float(oversample)
        patch_cfg = replace(
            cfg,
            synthesis=replace(
                cfg.synthesis,
                method_key="paper_v4_sharp_native",
                image_size=int(cfg.synthesis.image_size * oversample),
                radius=float(cfg.synthesis.radius) * float(oversample),
                center_disk_radius=float(cfg.synthesis.center_disk_radius) * float(oversample),
                anti_alias_sigma=0.0,
            ),
            layout=replace(
                cfg.layout,
                group_radius=float(cfg.layout.group_radius) * float(oversample),
            ),
        )

    layout = build_robot_tag_layout(robot_id, patch_cfg)
    patch = np.ones((layout.patch_size, layout.patch_size), dtype=np.float32)
    alpha = np.zeros_like(patch)

    primitive_size = int(patch_cfg.synthesis.image_size)
    half = primitive_size // 2
    for primitive in layout.primitives:
        primitive_image, primitive_alpha = generate_spiral_image(
            patch_cfg.synthesis,
            twist_angle_deg=primitive.twist_angle_deg,
            role=primitive.role,
            bit_value=primitive.bit_value,
        )
        center_x = int(round(primitive.center_xy[0]))
        center_y = int(round(primitive.center_xy[1]))
        x_start = center_x - half
        y_start = center_y - half
        x0 = max(0, x_start)
        y0 = max(0, y_start)
        x1 = min(layout.patch_size, x_start + primitive_size)
        y1 = min(layout.patch_size, y_start + primitive_size)
        px0 = max(0, -x_start)
        py0 = max(0, -y_start)
        px1 = px0 + (x1 - x0)
        py1 = py0 + (y1 - y0)
        current_alpha = primitive_alpha[py0:py1, px0:px1]
        patch_region = patch[y0:y1, x0:x1]
        patch[y0:y1, x0:x1] = _alpha_blend(patch_region, primitive_image[py0:py1, px0:px1], current_alpha)
        alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], current_alpha)

    if layout.anchors:
        yy, xx = np.indices(patch.shape, dtype=np.float32)
        for anchor in layout.anchors:
            distance = np.hypot(xx - anchor.center_xy[0], yy - anchor.center_xy[1])
            anchor_alpha = np.exp(-0.5 * (distance / max(anchor.radius, 1.0)) ** 2).astype(np.float32)
            anchor_image = np.full_like(patch, fill_value=1.0 - 0.92 * anchor.intensity, dtype=np.float32)
            patch = _alpha_blend(patch, anchor_image, 0.85 * anchor_alpha)
            alpha = np.maximum(alpha, 0.65 * anchor_alpha)

    return patch, alpha, layout, float(patch_scale_adjust)


def _destination_corners(
    patch_size: int,
    center_xy: tuple[float, float],
    scale: float,
    rotation_deg: float,
    tilt_x_deg: float,
    tilt_y_deg: float,
) -> np.ndarray:
    half = 0.5 * patch_size * scale
    local = np.array(
        [
            [-half, -half],
            [half, -half],
            [half, half],
            [-half, half],
        ],
        dtype=np.float32,
    )
    tx = np.tan(np.deg2rad(tilt_x_deg)) * 0.15
    ty = np.tan(np.deg2rad(tilt_y_deg)) * 0.15
    warped_local = local.copy()
    warped_local[:, 0] *= 1.0 + ty * (local[:, 1] / max(half, 1.0))
    warped_local[:, 1] *= 1.0 + tx * (local[:, 0] / max(half, 1.0))
    rotated = warped_local @ rotation_matrix(rotation_deg).T
    rotated[:, 0] += center_xy[0]
    rotated[:, 1] += center_xy[1]
    return rotated.astype(np.float32)


def _place_patch(
    canvas: np.ndarray,
    patch: np.ndarray,
    alpha: np.ndarray,
    dst_corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    warped_patch, warped_alpha, homography = _warp_patch_and_alpha(
        canvas_shape=canvas.shape,
        patch=patch,
        alpha=alpha,
        dst_corners=dst_corners,
        border_value=1.0,
    )
    blended = _alpha_blend(canvas, warped_patch, warped_alpha)
    return blended, homography


def _warp_patch_and_alpha(
    canvas_shape: tuple[int, ...],
    patch: np.ndarray,
    alpha: np.ndarray,
    dst_corners: np.ndarray,
    border_value: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = canvas_shape[:2]
    src = np.array(
        [
            [0.0, 0.0],
            [patch.shape[1] - 1.0, 0.0],
            [patch.shape[1] - 1.0, patch.shape[0] - 1.0],
            [0.0, patch.shape[0] - 1.0],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src, dst_corners)
    warped_patch = cv2.warpPerspective(
        patch.astype(np.float32),
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )
    warped_alpha = cv2.warpPerspective(
        alpha.astype(np.float32),
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return warped_patch, warped_alpha, homography


def textured_green_background(image_size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    height, width = image_size
    yy, xx = np.indices((height, width), dtype=np.float32)
    xx_norm = xx / max(width - 1, 1)
    yy_norm = yy / max(height - 1, 1)

    coarse = cv2.GaussianBlur(rng.random((height, width)).astype(np.float32), (0, 0), 18.0)
    medium = cv2.GaussianBlur(rng.random((height, width)).astype(np.float32), (0, 0), 8.0)
    fine = cv2.GaussianBlur(rng.random((height, width)).astype(np.float32), (0, 0), 2.2)
    stripe_phase = rng.uniform(0.0, 2.0 * np.pi)
    stripe = 0.5 + 0.5 * np.sin(2.0 * np.pi * (1.7 * xx_norm + 0.8 * yy_norm) + stripe_phase)
    texture = 0.45 * coarse + 0.35 * medium + 0.2 * fine
    texture = texture / np.maximum(texture.max(), 1.0e-6)

    red = np.clip(0.08 + 0.10 * texture + 0.03 * stripe, 0.0, 1.0)
    green = np.clip(0.32 + 0.45 * texture + 0.08 * stripe, 0.0, 1.0)
    blue = np.clip(0.08 + 0.12 * texture + 0.02 * (1.0 - stripe), 0.0, 1.0)
    background = np.stack([red, green, blue], axis=-1)
    return np.clip(background, 0.0, 1.0).astype(np.float32)


def sample_scene_placements(
    cfg: MethodConfig,
    rng: np.random.Generator,
    robot_ids: list[int],
    image_size: tuple[int, int],
    min_robots: int,
    max_robots: int,
    scale_range: tuple[float, float],
    force_scale: float | None = None,
    force_tilt_deg: float | None = None,
) -> list[ScenePlacement]:
    count = int(rng.integers(min_robots, max_robots + 1))
    chosen_ids = rng.choice(robot_ids, size=min(count, len(robot_ids)), replace=False).tolist()
    height, width = image_size
    base_layout = build_robot_tag_layout(chosen_ids[0] if chosen_ids else 0, cfg)
    base_radius = 0.5 * base_layout.patch_size
    placements: list[ScenePlacement] = []

    for robot_id in chosen_ids:
        scale = float(force_scale) if force_scale is not None else float(rng.uniform(*scale_range))
        footprint_radius = base_radius * scale
        chosen_center: np.ndarray | None = None
        for _ in range(max(200, cfg.scene.max_attempts)):
            candidate = np.array(
                [
                    rng.uniform(footprint_radius + 8.0, width - footprint_radius - 8.0),
                    rng.uniform(footprint_radius + 8.0, height - footprint_radius - 8.0),
                ],
                dtype=np.float32,
            )
            if not placements:
                chosen_center = candidate
                break
            okay = True
            for prev in placements:
                prev_radius = base_radius * prev.scale
                min_distance = max(cfg.scene.min_center_distance, footprint_radius + prev_radius + 8.0)
                if np.linalg.norm(candidate - np.asarray(prev.center_xy, dtype=np.float32)) < min_distance:
                    okay = False
                    break
            if okay:
                chosen_center = candidate
                break
        if chosen_center is None:
            continue
        placements.append(
            ScenePlacement(
                robot_id=int(robot_id),
                center_xy=(float(chosen_center[0]), float(chosen_center[1])),
                scale=scale,
                rotation_deg=float(rng.uniform(*cfg.scene.rotation_range_deg)),
                tilt_x_deg=float(force_tilt_deg) if force_tilt_deg is not None else float(rng.uniform(*cfg.scene.tilt_range_deg)),
                tilt_y_deg=float(force_tilt_deg) if force_tilt_deg is not None else float(rng.uniform(*cfg.scene.tilt_range_deg)),
            )
        )

    return placements


def render_scene_from_placements(
    cfg: MethodConfig,
    placements: list[ScenePlacement],
    background_rgb: np.ndarray | None = None,
) -> RenderedScene:
    height, width = cfg.scene.image_size
    gray_canvas = np.full((height, width), cfg.scene.background, dtype=np.float32)
    color_canvas = None if background_rgb is None else background_rgb.copy().astype(np.float32)
    rendered_tags: list[RenderedTagGroundTruth] = []

    for placement in placements:
        patch, alpha, layout, patch_scale_adjust = render_robot_tag_patch(placement.robot_id, cfg)
        dst_corners = _destination_corners(
            patch_size=layout.patch_size,
            center_xy=placement.center_xy,
            scale=placement.scale * patch_scale_adjust,
            rotation_deg=placement.rotation_deg,
            tilt_x_deg=placement.tilt_x_deg,
            tilt_y_deg=placement.tilt_y_deg,
        )
        gray_canvas, homography = _place_patch(gray_canvas, patch, alpha, dst_corners)
        if color_canvas is not None:
            patch_rgb = np.repeat(patch[..., None], 3, axis=2)
            color_canvas, _ = _place_patch(color_canvas, patch_rgb, alpha, dst_corners)

        primitive_points = np.array([primitive.center_xy for primitive in layout.primitives], dtype=np.float32)
        projected_points = project_points(homography, primitive_points)
        center_point = np.mean(projected_points, axis=0)
        orientation_point = projected_points[0]
        heading_deg = angle_from_center_deg(tuple(orientation_point.tolist()), tuple(center_point.tolist()))
        effective_radius = 0.5 * layout.patch_size * placement.scale * patch_scale_adjust
        fit_circle_radius = float(
            max(
                np.linalg.norm(projected_points[index] - center_point)
                + layout.primitives[index].radius * placement.scale * patch_scale_adjust
                for index in range(len(layout.primitives))
            )
        )

        rendered_tags.append(
            RenderedTagGroundTruth(
                robot_id=placement.robot_id,
                center_xy=(float(center_point[0]), float(center_point[1])),
                heading_deg=heading_deg,
                orientation_center_xy=(float(orientation_point[0]), float(orientation_point[1])),
                spiral_centers_xy=[(float(x), float(y)) for x, y in projected_points],
                spiral_twists_deg=[primitive.twist_angle_deg for primitive in layout.primitives],
                render_scale=placement.scale,
                effective_radius_px=float(effective_radius),
                fit_circle_radius_px=fit_circle_radius,
            )
        )

    return RenderedScene(image=np.clip(gray_canvas, 0.0, 1.0), tags=rendered_tags, color_image=color_canvas)


def render_scene_layers_from_placements(
    cfg: MethodConfig,
    placements: list[ScenePlacement],
    background_rgb: np.ndarray | None = None,
) -> RenderedLayeredScene:
    height, width = cfg.scene.image_size
    background_gray = np.full((height, width), cfg.scene.background, dtype=np.float32)
    clean_gray = background_gray.copy()
    clean_color = None if background_rgb is None else background_rgb.copy().astype(np.float32)
    layers: list[RenderedRobotLayer] = []
    rendered_tags: list[RenderedTagGroundTruth] = []

    for placement in placements:
        patch, alpha, layout, patch_scale_adjust = render_robot_tag_patch(placement.robot_id, cfg)
        dst_corners = _destination_corners(
            patch_size=layout.patch_size,
            center_xy=placement.center_xy,
            scale=placement.scale * patch_scale_adjust,
            rotation_deg=placement.rotation_deg,
            tilt_x_deg=placement.tilt_x_deg,
            tilt_y_deg=placement.tilt_y_deg,
        )
        warped_gray, warped_alpha, homography = _warp_patch_and_alpha(
            canvas_shape=clean_gray.shape,
            patch=patch,
            alpha=alpha,
            dst_corners=dst_corners,
            border_value=1.0,
        )
        clean_gray = _alpha_blend(clean_gray, warped_gray, warped_alpha)

        warped_color = None
        if clean_color is not None:
            patch_rgb = np.repeat(patch[..., None], 3, axis=2)
            warped_color, _, _ = _warp_patch_and_alpha(
                canvas_shape=clean_color.shape,
                patch=patch_rgb,
                alpha=alpha,
                dst_corners=dst_corners,
                border_value=1.0,
            )
            clean_color = _alpha_blend(clean_color, warped_color, warped_alpha)

        primitive_points = np.array([primitive.center_xy for primitive in layout.primitives], dtype=np.float32)
        projected_points = project_points(homography, primitive_points)
        center_point = np.mean(projected_points, axis=0)
        orientation_point = projected_points[0]
        heading_deg = angle_from_center_deg(tuple(orientation_point.tolist()), tuple(center_point.tolist()))
        effective_radius = 0.5 * layout.patch_size * placement.scale * patch_scale_adjust
        fit_circle_radius = float(
            max(
                np.linalg.norm(projected_points[index] - center_point)
                + layout.primitives[index].radius * placement.scale * patch_scale_adjust
                for index in range(len(layout.primitives))
            )
        )
        gt = RenderedTagGroundTruth(
            robot_id=placement.robot_id,
            center_xy=(float(center_point[0]), float(center_point[1])),
            heading_deg=heading_deg,
            orientation_center_xy=(float(orientation_point[0]), float(orientation_point[1])),
            spiral_centers_xy=[(float(x), float(y)) for x, y in projected_points],
            spiral_twists_deg=[primitive.twist_angle_deg for primitive in layout.primitives],
            render_scale=placement.scale,
            effective_radius_px=float(effective_radius),
            fit_circle_radius_px=fit_circle_radius,
        )
        rendered_tags.append(gt)
        layers.append(
            RenderedRobotLayer(
                robot_id=int(placement.robot_id),
                gray_layer=warped_gray.astype(np.float32),
                color_layer=None if warped_color is None else warped_color.astype(np.float32),
                alpha_layer=warped_alpha.astype(np.float32),
                color_alpha_layer=None,
                homography=homography.astype(np.float32),
                placement=placement,
                gt=gt,
            )
        )

    return RenderedLayeredScene(
        background_gray=np.clip(background_gray, 0.0, 1.0),
        background_rgb=None if background_rgb is None else np.clip(background_rgb.astype(np.float32), 0.0, 1.0),
        clean_scene=RenderedScene(
            image=np.clip(clean_gray, 0.0, 1.0),
            tags=rendered_tags,
            color_image=None if clean_color is None else np.clip(clean_color, 0.0, 1.0),
        ),
        layers=layers,
    )


def render_scene(
    cfg: MethodConfig,
    robot_ids: list[int],
    rng: np.random.Generator,
) -> RenderedScene:
    height, width = cfg.scene.image_size
    placements: list[ScenePlacement] = []
    base_layout = build_robot_tag_layout(robot_ids[0] if robot_ids else 0, cfg)
    base_radius = 0.5 * base_layout.patch_size

    for robot_id in robot_ids[: cfg.scene.num_tags]:
        scale = float(rng.uniform(*cfg.scene.tag_scale_range))
        footprint_radius = base_radius * scale
        chosen_center: np.ndarray | None = None
        for _ in range(cfg.scene.max_attempts):
            candidate = np.array(
                [
                    rng.uniform(footprint_radius + 8.0, width - footprint_radius - 8.0),
                    rng.uniform(footprint_radius + 8.0, height - footprint_radius - 8.0),
                ],
                dtype=np.float32,
            )
            if not placements:
                chosen_center = candidate
                break
            distances = [
                float(np.linalg.norm(candidate - np.asarray(prev.center_xy)) - max(cfg.scene.min_center_distance, footprint_radius + base_radius * prev.scale))
                for prev in placements
            ]
            if min(distances) >= 8.0:
                chosen_center = candidate
                break
        if chosen_center is None:
            continue
        placements.append(
            ScenePlacement(
                robot_id=int(robot_id),
                center_xy=(float(chosen_center[0]), float(chosen_center[1])),
                scale=scale,
                rotation_deg=float(rng.uniform(*cfg.scene.rotation_range_deg)),
                tilt_x_deg=float(rng.uniform(*cfg.scene.tilt_range_deg)),
                tilt_y_deg=float(rng.uniform(*cfg.scene.tilt_range_deg)),
            )
        )

    return render_scene_from_placements(cfg, placements, background_rgb=None)


def export_marker_sheet(
    cfg: MethodConfig,
    out_path: str | Path,
    sheet_cfg: SheetConfig | None = None,
) -> Path:
    sheet = sheet_cfg or cfg.sheet
    ids = list(sheet.ids)
    columns = max(1, int(sheet.columns))
    rows = int(np.ceil(len(ids) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=sheet.page_size_inches, dpi=sheet.dpi)
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    for axis in axes_array.flat:
        axis.axis("off")
    for axis, robot_id in zip(axes_array.flat, ids):
        patch, _, _, _ = render_robot_tag_patch(robot_id, cfg)
        axis.imshow(patch, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"ID {robot_id}", fontsize=10)
        axis.axis("off")
    fig.tight_layout(pad=0.25)
    out_file = Path(out_path)
    ensure_dir(out_file.parent)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
