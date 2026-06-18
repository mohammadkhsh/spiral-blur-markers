from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from spiral_markers.io_utils import CorruptionConfig


def kernel_angle_to_display_deg(angle_deg: float) -> float:
    return float((180.0 - float(angle_deg)) % 360.0)


def motion_blur(image: np.ndarray, length: int, angle_deg: float, beta: float = 0.0) -> np.ndarray:
    if length <= 1:
        return image.copy()
    kernel = np.zeros((length, length), dtype=np.float32)
    samples = np.linspace(-1.0, 1.0, num=length, dtype=np.float32)
    weights = np.clip(1.0 + beta * samples, 0.05, None)
    kernel[length // 2, :] = weights
    center = (0.5 * (length - 1), 0.5 * (length - 1))
    rotation = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    kernel /= np.maximum(kernel.sum(), 1.0)
    return cv2.filter2D(image.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT)


def illumination(image: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return np.clip(image * scale + offset, 0.0, 1.0)


def gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0.0:
        return image.copy()
    noisy = image + rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def contrast_reduction(image: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1.0e-6:
        return image.copy()
    mean = float(np.mean(image))
    reduced = mean + scale * (image - mean)
    return np.clip(reduced, 0.0, 1.0)


def soft_shadow(
    image: np.ndarray,
    strength: float,
    coverage: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if strength <= 0.0 or coverage <= 0.0:
        return image.copy()
    height, width = image.shape[:2]
    lowres = np.zeros((max(16, height // 8), max(16, width // 8)), dtype=np.float32)
    center_count = max(1, int(round(2 + 6 * coverage)))
    yy, xx = np.indices(lowres.shape, dtype=np.float32)
    for _ in range(center_count):
        cx = rng.uniform(0, lowres.shape[1] - 1)
        cy = rng.uniform(0, lowres.shape[0] - 1)
        rx = rng.uniform(2, max(3, lowres.shape[1] * coverage))
        ry = rng.uniform(2, max(3, lowres.shape[0] * coverage))
        lowres += np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
    shadow = cv2.resize(lowres, (width, height), interpolation=cv2.INTER_CUBIC)
    shadow = shadow / np.maximum(shadow.max(), 1.0e-6)
    shadow = ndi.gaussian_filter(shadow, sigma=max(height, width) * 0.01, mode="reflect")
    attenuation = 1.0 - strength * shadow
    if image.ndim == 3:
        attenuation = attenuation[..., None]
    return np.clip(image * attenuation, 0.0, 1.0)


def partial_occlusion(image: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    if ratio <= 0.0:
        return image.copy()
    height, width = image.shape[:2]
    occlusion = np.zeros((height, width), dtype=np.float32)
    target = ratio * height * width
    filled = 0.0
    while filled < target:
        shape_type = rng.choice(["rect", "circle"])
        if shape_type == "rect":
            w = int(rng.uniform(0.08, 0.22) * width)
            h = int(rng.uniform(0.08, 0.22) * height)
            x0 = int(rng.uniform(0, max(1, width - w)))
            y0 = int(rng.uniform(0, max(1, height - h)))
            occlusion[y0 : y0 + h, x0 : x0 + w] = 1.0
            filled += w * h
        else:
            radius = int(rng.uniform(0.04, 0.12) * min(height, width))
            x0 = int(rng.uniform(radius, max(radius + 1, width - radius)))
            y0 = int(rng.uniform(radius, max(radius + 1, height - radius)))
            cv2.circle(occlusion, (x0, y0), radius, 1.0, thickness=-1)
            filled = float(occlusion.sum())
    if image.ndim == 2:
        fill_value = float(np.clip(np.median(image) * 0.4, 0.0, 1.0))
        return np.where(occlusion > 0.0, fill_value, image)
    fill_value = np.clip(np.median(image, axis=(0, 1)) * 0.4, 0.0, 1.0)
    return np.where(occlusion[..., None] > 0.0, fill_value[None, None, :], image)


def apply_corruptions(
    image: np.ndarray,
    cfg: CorruptionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    current = image.astype(np.float32)
    current = motion_blur(current, cfg.motion_blur_length, cfg.motion_blur_angle_deg, beta=cfg.motion_beta)
    current = illumination(current, cfg.illumination_scale, cfg.illumination_offset)
    current = soft_shadow(current, cfg.shadow_strength, cfg.shadow_coverage, rng)
    current = contrast_reduction(current, cfg.contrast_scale)
    current = gaussian_noise(current, cfg.gaussian_noise_sigma, rng)
    current = partial_occlusion(current, cfg.occlusion_ratio, rng)
    return np.clip(current, 0.0, 1.0)


def generate_shadow_mask(
    image_size: tuple[int, int],
    strength_range: tuple[float, float],
    coverage_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    height, width = image_size
    strength = float(rng.uniform(*strength_range))
    coverage = float(rng.uniform(*coverage_range))
    lowres = np.zeros((max(16, height // 8), max(16, width // 8)), dtype=np.float32)
    center_count = max(1, int(round(2 + 6 * coverage)))
    yy, xx = np.indices(lowres.shape, dtype=np.float32)
    blob_params: list[dict[str, float]] = []
    for _ in range(center_count):
        cx = float(rng.uniform(0, lowres.shape[1] - 1))
        cy = float(rng.uniform(0, lowres.shape[0] - 1))
        rx = float(rng.uniform(2, max(3, lowres.shape[1] * coverage)))
        ry = float(rng.uniform(2, max(3, lowres.shape[0] * coverage)))
        lowres += np.exp(-(((xx - cx) / max(rx, 1.0e-6)) ** 2 + ((yy - cy) / max(ry, 1.0e-6)) ** 2))
        blob_params.append({"cx": cx, "cy": cy, "rx": rx, "ry": ry})
    shadow = cv2.resize(lowres, (width, height), interpolation=cv2.INTER_CUBIC)
    shadow = shadow / np.maximum(shadow.max(), 1.0e-6)
    shadow = ndi.gaussian_filter(shadow, sigma=max(height, width) * 0.01, mode="reflect")
    attenuation = np.clip(1.0 - strength * shadow, 0.0, 1.0).astype(np.float32)
    metadata = {
        "shadow_strength": strength,
        "shadow_coverage": coverage,
        "shadow_blob_count": float(center_count),
    }
    metadata["shadow_mean_attenuation"] = float(np.mean(attenuation))
    metadata["shadow_min_attenuation"] = float(np.min(attenuation))
    metadata["shadow_max_attenuation"] = float(np.max(attenuation))
    metadata["shadow_blobs"] = blob_params  # type: ignore[assignment]
    return attenuation, metadata


def apply_shadow_mask(image: np.ndarray, attenuation: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return np.clip(image * attenuation[..., None], 0.0, 1.0)
    return np.clip(image * attenuation, 0.0, 1.0)


def generate_robot_occlusion_masks(
    image_size: tuple[int, int],
    tags,
    target_fraction_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    height, width = image_size
    full_mask = np.zeros((height, width), dtype=np.float32)
    robot_metadata: list[dict[str, object]] = []

    for robot_index, gt in enumerate(tags):
        target_fraction = float(rng.uniform(*target_fraction_range))
        robot_mask = np.zeros((height, width), dtype=np.float32)
        circle_mask = np.zeros((height, width), dtype=np.uint8)
        center = (int(round(gt.center_xy[0])), int(round(gt.center_xy[1])))
        radius = max(4, int(round(gt.fit_circle_radius_px)))
        cv2.circle(circle_mask, center, radius, 1, thickness=-1)
        circle_area = float(np.sum(circle_mask))
        shapes: list[dict[str, object]] = []
        attempts = 0
        current_fraction = 0.0
        while current_fraction < target_fraction and attempts < 32:
            attempts += 1
            shape_type = str(rng.choice(["rect", "circle", "ellipse"]))
            if shape_type == "rect":
                w = int(rng.uniform(0.30, 0.80) * radius)
                h = int(rng.uniform(0.30, 0.80) * radius)
                cx = int(rng.uniform(center[0] - 0.5 * radius, center[0] + 0.5 * radius))
                cy = int(rng.uniform(center[1] - 0.5 * radius, center[1] + 0.5 * radius))
                angle = float(rng.uniform(0.0, 180.0))
                rect = ((float(cx), float(cy)), (float(max(w, 3)), float(max(h, 3))), angle)
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.fillConvexPoly(robot_mask, box, 1.0)
                shapes.append({"type": "rect", "cx": cx, "cy": cy, "w": int(max(w, 3)), "h": int(max(h, 3)), "angle_deg": angle})
            elif shape_type == "circle":
                occ_radius = int(rng.uniform(0.18, 0.45) * radius)
                cx = int(rng.uniform(center[0] - 0.45 * radius, center[0] + 0.45 * radius))
                cy = int(rng.uniform(center[1] - 0.45 * radius, center[1] + 0.45 * radius))
                cv2.circle(robot_mask, (cx, cy), max(occ_radius, 2), 1.0, thickness=-1)
                shapes.append({"type": "circle", "cx": cx, "cy": cy, "radius": int(max(occ_radius, 2))})
            else:
                axes = (
                    int(rng.uniform(0.20, 0.45) * radius),
                    int(rng.uniform(0.12, 0.35) * radius),
                )
                cx = int(rng.uniform(center[0] - 0.45 * radius, center[0] + 0.45 * radius))
                cy = int(rng.uniform(center[1] - 0.45 * radius, center[1] + 0.45 * radius))
                angle = float(rng.uniform(0.0, 180.0))
                cv2.ellipse(robot_mask, (cx, cy), (max(axes[0], 2), max(axes[1], 2)), angle, 0.0, 360.0, 1.0, thickness=-1)
                shapes.append({"type": "ellipse", "cx": cx, "cy": cy, "axes": [int(max(axes[0], 2)), int(max(axes[1], 2))], "angle_deg": angle})
            current_fraction = float(np.sum((robot_mask > 0.0) & (circle_mask > 0)) / max(circle_area, 1.0))

        robot_mask = np.where(circle_mask > 0, robot_mask, 0.0).astype(np.float32)
        full_mask = np.maximum(full_mask, robot_mask)
        actual_fraction = float(np.sum((robot_mask > 0.0) & (circle_mask > 0)) / max(circle_area, 1.0))
        robot_metadata.append(
            {
                "robot_index": int(robot_index),
                "robot_id": int(gt.robot_id),
                "target_occlusion_fraction": target_fraction,
                "actual_occlusion_fraction": actual_fraction,
                "circle_radius_px": int(radius),
                "shapes": shapes,
            }
        )
    return np.clip(full_mask, 0.0, 1.0), robot_metadata


def composite_with_background(
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    if foreground.ndim == 3:
        return np.where(mask[..., None] > 0.0, background, foreground).astype(np.float32)
    return np.where(mask > 0.0, background, foreground).astype(np.float32)


def apply_local_motion_blur(
    background_gray: np.ndarray,
    background_rgb: np.ndarray | None,
    robot_layers,
    blur_params: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray | None]:
    scene_gray = background_gray.copy().astype(np.float32)
    scene_rgb = None if background_rgb is None else background_rgb.copy().astype(np.float32)
    for layer, params in zip(robot_layers, blur_params):
        blur_length = int(params["blur_length_px"])
        blur_angle = float(params.get("kernel_blur_angle_deg", params["blur_angle_deg"]))
        beta = float(params.get("motion_beta", 0.0))
        alpha = layer.alpha_layer.astype(np.float32)
        if np.max(alpha) <= 1.0e-6:
            continue
        ys, xs = np.where(alpha > 1.0e-3)
        if xs.size == 0 or ys.size == 0:
            continue
        pad = max(4, int(np.ceil(blur_length)) + 2)
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(alpha.shape[1], int(xs.max()) + pad + 1)
        y1 = min(alpha.shape[0], int(ys.max()) + pad + 1)

        gray_roi = layer.gray_layer[y0:y1, x0:x1].astype(np.float32)
        alpha_roi = alpha[y0:y1, x0:x1].astype(np.float32)
        blurred_gray = motion_blur(gray_roi, blur_length, blur_angle, beta=beta)
        blurred_alpha = np.clip(motion_blur(alpha_roi, blur_length, blur_angle, beta=beta), 0.0, 1.0)
        scene_gray[y0:y1, x0:x1] = blurred_gray * blurred_alpha + scene_gray[y0:y1, x0:x1] * (1.0 - blurred_alpha)

        if scene_rgb is not None and layer.color_layer is not None:
            color_roi = layer.color_layer[y0:y1, x0:x1].astype(np.float32)
            color_alpha_layer = getattr(layer, "color_alpha_layer", None)
            if color_alpha_layer is None:
                color_alpha_roi = alpha_roi
            else:
                color_alpha_roi = color_alpha_layer[y0:y1, x0:x1].astype(np.float32)
            blurred_channels = [
                motion_blur(color_roi[..., channel], blur_length, blur_angle, beta=beta) for channel in range(color_roi.shape[2])
            ]
            blurred_color = np.stack(blurred_channels, axis=-1).astype(np.float32)
            blurred_color_alpha = np.clip(motion_blur(color_alpha_roi, blur_length, blur_angle, beta=beta), 0.0, 1.0)
            scene_rgb[y0:y1, x0:x1] = (
                blurred_color * blurred_color_alpha[..., None]
                + scene_rgb[y0:y1, x0:x1] * (1.0 - blurred_color_alpha[..., None])
            )
    return np.clip(scene_gray, 0.0, 1.0), None if scene_rgb is None else np.clip(scene_rgb, 0.0, 1.0)
