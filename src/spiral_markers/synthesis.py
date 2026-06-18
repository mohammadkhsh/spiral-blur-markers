from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import cv2
import numpy as np
from scipy import ndimage as ndi

from spiral_markers.geometry import annulus_bounds, polar_grid, suggested_num_legs
from spiral_markers.io_utils import SpiralSynthesisConfig


def _marker_tones(background: float, contrast: float) -> tuple[float, float]:
    dark = np.clip(background * (1.0 - contrast), 0.0, 1.0)
    light = np.clip(background + contrast * (1.0 - background), 0.0, 1.0)
    return float(dark), float(light)


def _wrap_pi(angle_rad: np.ndarray) -> np.ndarray:
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def _smoothstep01(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0).astype(np.float32)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _sector_repeated_angle(
    angle_rad: np.ndarray,
    sector_count: int,
    phase_offset_deg: float,
) -> np.ndarray:
    if sector_count <= 1:
        return angle_rad
    span = 2.0 * np.pi / float(sector_count)
    shifted = angle_rad - np.deg2rad(phase_offset_deg)
    return _wrap_pi(((shifted + 0.5 * span) % span) - 0.5 * span)


def _ecc_sector_mask(
    angle_rad: np.ndarray,
    sector_count: int,
    gap_deg: float,
    phase_offset_deg: float,
    soft_edge_deg: float = 0.0,
) -> np.ndarray:
    if sector_count <= 1:
        return np.ones_like(angle_rad, dtype=np.float32)
    span = 2.0 * np.pi / float(sector_count)
    gap_rad = float(np.clip(np.deg2rad(gap_deg), 0.0, span - 1.0e-6))
    active_half_width = 0.5 * max(span - gap_rad, 1.0e-6)
    repeated = np.abs(_sector_repeated_angle(angle_rad, sector_count, phase_offset_deg))
    if soft_edge_deg <= 1.0e-6:
        return (repeated <= active_half_width).astype(np.float32)

    soft_edge_rad = float(np.clip(np.deg2rad(soft_edge_deg), 0.0, active_half_width))
    hard_half_width = max(active_half_width - soft_edge_rad, 0.0)
    mask = np.zeros_like(angle_rad, dtype=np.float32)
    hard_region = repeated <= hard_half_width
    mask[hard_region] = 1.0
    if soft_edge_rad > 1.0e-6:
        taper_region = (repeated > hard_half_width) & (repeated <= active_half_width)
        taper_coord = (repeated[taper_region] - hard_half_width) / soft_edge_rad
        mask[taper_region] = 0.5 + 0.5 * np.cos(np.pi * np.clip(taper_coord, 0.0, 1.0))
    return mask.astype(np.float32)


def _class_label_from_twist(theta_deg: float) -> str:
    if abs(float(theta_deg)) <= 1.0e-6:
        return "zero"
    return "pos" if float(theta_deg) > 0.0 else "neg"


def _aso_orientation_window(
    angle_rad: np.ndarray,
    notch_center_deg: float,
    notch_width_deg: float,
    notch_depth: float,
) -> np.ndarray:
    if notch_depth <= 1.0e-6 or notch_width_deg <= 1.0e-6:
        return np.ones_like(angle_rad, dtype=np.float32)
    delta = _wrap_pi(angle_rad - np.deg2rad(float(notch_center_deg)))
    half_width = np.deg2rad(float(notch_width_deg)) * 0.5
    normalized = np.clip(np.abs(delta) / max(half_width, 1.0e-6), 0.0, 1.0)
    notch = 0.5 + 0.5 * np.cos(np.pi * normalized)
    return (1.0 - float(np.clip(notch_depth, 0.0, 1.0)) * notch).astype(np.float32)


def _apply_self_normalized_reference_rings(
    image: np.ndarray,
    annulus_alpha: np.ndarray,
    rr: np.ndarray,
    cfg: SpiralSynthesisConfig,
    dark: float,
    light: float,
    inner_blend: float = 1.0,
    outer_blend: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not bool(cfg.self_norm_ring_pair):
        return image, annulus_alpha
    ref_width = max(1.0, float(cfg.self_norm_ring_width_ratio) * float(cfg.radius))
    inner_center = float(cfg.self_norm_inner_radius_ratio) * float(cfg.radius)
    outer_center = float(cfg.self_norm_outer_radius_ratio) * float(cfg.radius)
    inner_mask = np.abs(rr - inner_center) <= 0.5 * ref_width
    outer_mask = np.abs(rr - outer_center) <= 0.5 * ref_width
    inner_blend = float(np.clip(inner_blend, 0.0, 1.0))
    outer_blend = float(np.clip(outer_blend, 0.0, 1.0))
    image[inner_mask] = (1.0 - inner_blend) * image[inner_mask] + inner_blend * light
    image[outer_mask] = (1.0 - outer_blend) * image[outer_mask] + outer_blend * dark
    annulus_alpha[inner_mask | outer_mask] = 1.0
    return image, annulus_alpha


def _ring_masks(
    shape: tuple[int, int],
    center_xy: tuple[float, float],
    cfg: SpiralSynthesisConfig,
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[float, float]]]:
    rr, _ = polar_grid(shape, center_xy)
    bounds = annulus_bounds(
        radius=cfg.radius,
        ring_count=max(cfg.ring_count, 1),
        inner_hole_ratio=cfg.inner_hole_ratio,
        gap_ratio=cfg.ring_gap_ratio,
    )
    ring_masks = [(rr >= inner) & (rr <= outer) for inner, outer in bounds]
    annulus_alpha = np.zeros(shape, dtype=bool)
    for mask in ring_masks:
        annulus_alpha |= mask
    return rr, ring_masks, bounds


def _spiral_phase(
    radius: np.ndarray,
    angle: np.ndarray,
    ring_mid_radius: float,
    num_legs: int,
    theta_deg: float,
    phase_offset_deg: float,
) -> np.ndarray:
    # Practical baseline approximation:
    # phase = L * angle + k(theta) * log(r/r0), with k=0 at theta=0 -> radial rays.
    theta_rad = np.deg2rad(theta_deg)
    curvature = float(np.clip(2.0 * num_legs * np.tan(theta_rad), -60.0, 60.0))
    log_term = np.log(np.maximum(radius, 1.0) / max(ring_mid_radius, 1.0))
    return num_legs * angle + curvature * log_term + np.deg2rad(phase_offset_deg)


def generate_spiral_image(
    cfg: SpiralSynthesisConfig,
    twist_angle_deg: float | None = None,
    image_size: int | None = None,
    active_rings: Sequence[int] | None = None,
    render_mode: str = "primary",
    phase_offset_deg: float = 0.0,
    role: str = "identity",
    bit_value: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    del bit_value
    size = int(image_size or cfg.image_size)
    method_key = str(cfg.method_key).lower()
    if method_key == "paper_v4_sharp" and render_mode != "supersampled":
        supersample = 2
        scale = float(supersample)
        hi_cfg = replace(
            cfg,
            method_key="paper_v4_sharp_native",
            image_size=int(size * supersample),
            radius=float(cfg.radius) * scale,
            center_disk_radius=float(cfg.center_disk_radius) * scale,
            anti_alias_sigma=0.0,
        )
        hi_image, hi_alpha = generate_spiral_image(
            hi_cfg,
            twist_angle_deg=twist_angle_deg,
            image_size=int(size * supersample),
            active_rings=active_rings,
            render_mode="supersampled",
            phase_offset_deg=phase_offset_deg,
            role=role,
        )
        image = cv2.resize(hi_image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
        alpha = cv2.resize(hi_alpha, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
        return np.clip(image, 0.0, 1.0), np.clip(alpha, 0.0, 1.0)

    center = (0.5 * (size - 1), 0.5 * (size - 1))
    theta_deg = float(cfg.twist_angle_deg if twist_angle_deg is None else twist_angle_deg)
    rr, aa = polar_grid((size, size), center)
    _, ring_masks, bounds = _ring_masks((size, size), center, cfg)
    dark, light = _marker_tones(cfg.background, cfg.contrast)

    image = np.full((size, size), fill_value=cfg.background, dtype=np.float32)
    annulus_alpha = np.zeros((size, size), dtype=np.float32)
    ring_weights = np.asarray(cfg.ring_weights, dtype=np.float32)
    if ring_weights.size != max(cfg.ring_count, 1):
        ring_weights = np.ones(max(cfg.ring_count, 1), dtype=np.float32)
    active = set(range(max(cfg.ring_count, 1))) if active_rings is None else set(active_rings)

    for ring_index, (ring_mask, (inner, outer)) in enumerate(zip(ring_masks, bounds)):
        if ring_index not in active:
            continue
        ring_mid = 0.5 * (inner + outer)
        freq_mult = float(cfg.ring_frequency_multipliers[min(ring_index, len(cfg.ring_frequency_multipliers) - 1)])
        l0 = int(round(cfg.base_legs * freq_mult))
        if method_key in {"paper_v4", "paper_v4_sharp", "paper_v4_sharp_native"}:
            legs = 12 if (abs(theta_deg) <= 1.0e-6 or str(role) == "orientation") else 6
        elif abs(theta_deg) <= 1.0e-6:
            # Empirically stable odd spoke count keeps GST twist close to zero for the orientation marker.
            legs = max(max(2, cfg.min_legs), 5)
        else:
            legs = suggested_num_legs(l0, theta_deg, min_legs=max(2, cfg.min_legs))
        phase_angles = aa
        class_label = _class_label_from_twist(theta_deg)

        if method_key == "paper_v4":
            phase = _spiral_phase(rr, phase_angles, ring_mid, legs, theta_deg, phase_offset_deg=phase_offset_deg)
            ring_signal = 0.5 + 0.5 * np.cos(phase)
            band_width = max(float(outer - inner), 1.0)
            edge_width = max(1.4, 0.22 * band_width)
            inner_soft = _smoothstep01((rr - float(inner)) / edge_width)
            outer_soft = _smoothstep01((float(outer) - rr) / edge_width)
            ring_alpha = np.clip(inner_soft * outer_soft, 0.0, 1.0).astype(np.float32)
            ring_alpha *= ((rr >= float(inner) - edge_width) & (rr <= float(outer) + edge_width)).astype(np.float32)
            ring_image = dark + (light - dark) * ring_signal
        elif method_key in {"paper_v4_sharp", "paper_v4_sharp_native"}:
            phase = _spiral_phase(rr, phase_angles, ring_mid, legs, theta_deg, phase_offset_deg=phase_offset_deg)
            ring_signal = (np.cos(phase) >= 0.0).astype(np.float32)
            ring_alpha = ring_mask.astype(np.float32)
            ring_image = dark + (light - dark) * ring_signal
        elif method_key == "ecc_sr" and int(cfg.sector_count) > 1:
            sector_count = int(cfg.sector_count)
            phase = _spiral_phase(rr, phase_angles, ring_mid, legs, theta_deg, phase_offset_deg=phase_offset_deg)
            harmonic = 0.5 + 0.5 * np.cos(phase)
            binary = (harmonic >= 0.5).astype(np.float32)
            ring_signal = (1.0 - cfg.alias_safe_blend) * binary + cfg.alias_safe_blend * harmonic
            sector_mask = _ecc_sector_mask(
                angle_rad=aa,
                sector_count=sector_count,
                gap_deg=float(cfg.ecc_sector_gap_deg),
                phase_offset_deg=float(cfg.sector_phase_offset_deg),
                soft_edge_deg=float(cfg.ecc_sector_soft_edge_deg),
            )
            span = 2.0 * np.pi / float(sector_count)
            shifted_angle = (aa - np.deg2rad(float(cfg.sector_phase_offset_deg))) % (2.0 * np.pi)
            sector_index = np.floor(shifted_angle / span).astype(np.int32)
            parity_weight = float(np.clip(cfg.ecc_parity_sector_weight, 0.0, 1.0))
            primary_weight = float(np.clip(cfg.ecc_primary_sector_weight, 0.0, 1.0))
            sector_weight = np.where((sector_index % 2) == 0, primary_weight, parity_weight).astype(np.float32)
            if float(cfg.sector_code_strength) > 1.0e-6:
                class_gain = {
                    "neg": np.where((sector_index % 3) == 0, 1.0, 1.0 - 0.12 * float(cfg.sector_code_strength)),
                    "zero": np.where((sector_index % 3) == 1, 1.0, 1.0 - 0.12 * float(cfg.sector_code_strength)),
                    "pos": np.where((sector_index % 3) == 2, 1.0, 1.0 - 0.12 * float(cfg.sector_code_strength)),
                }[class_label].astype(np.float32)
                sector_weight *= class_gain
            ring_signal = np.clip(0.5 + sector_weight * (ring_signal - 0.5), 0.0, 1.0)
            ring_image = dark + (light - dark) * ring_signal
            ring_alpha = ring_mask.astype(np.float32) * sector_mask
        else:
            phase = _spiral_phase(rr, phase_angles, ring_mid, legs, theta_deg, phase_offset_deg=phase_offset_deg)
            harmonic = 0.5 + 0.5 * np.cos(phase)
            binary = (harmonic >= 0.5).astype(np.float32)
            ring_signal = (1.0 - cfg.alias_safe_blend) * binary + cfg.alias_safe_blend * harmonic

        if method_key == "mb_psf" and int(cfg.ring_count) > 1:
            blur_guard = 1.0 - 0.10 * float(ring_index)
            ring_signal = np.clip(0.5 + blur_guard * (ring_signal - 0.5), 0.0, 1.0)

        if method_key == "aso" and abs(theta_deg) <= 1.0e-6 and str(role) == "orientation":
            asym_strength = float(np.clip(cfg.direction_diverse_mix, 0.0, 1.0))
            asym_phase = np.deg2rad(float(cfg.direction_diverse_phase_deg))
            asymmetry = 0.5 + 0.5 * np.cos(aa - asym_phase)
            harmonic_asym = 0.5 + 0.5 * np.cos(2.0 * (aa - asym_phase) - np.deg2rad(35.0))
            ring_signal = np.clip(
                ring_signal * (1.0 - 0.16 * asym_strength + 0.32 * asym_strength * asymmetry)
                + 0.10 * asym_strength * (harmonic_asym - 0.5),
                0.0,
                1.0,
            )
        if not (method_key == "ecc_sr" and int(cfg.sector_count) > 1):
            ring_image = dark + (light - dark) * ring_signal
            ring_alpha = ring_mask.astype(np.float32)

        weight = float(ring_weights[min(ring_index, ring_weights.size - 1)])
        composite_alpha = np.clip(weight * ring_alpha, 0.0, 1.0).astype(np.float32)
        image = ring_image * composite_alpha + image * (1.0 - composite_alpha)
        annulus_alpha = np.maximum(annulus_alpha, ring_alpha)

    if cfg.center_disk_radius > 0.0:
        center_disk = rr <= cfg.center_disk_radius
        image[center_disk] = dark

    image, annulus_alpha = _apply_self_normalized_reference_rings(
        image=image,
        annulus_alpha=annulus_alpha,
        rr=rr,
        cfg=cfg,
        dark=dark,
        light=light,
    )

    if cfg.anti_alias_sigma > 0.0:
        image = ndi.gaussian_filter(image, sigma=float(cfg.anti_alias_sigma), mode="reflect")
        annulus_alpha = ndi.gaussian_filter(annulus_alpha, sigma=float(cfg.anti_alias_sigma), mode="reflect")

    image = np.clip(image.astype(np.float32), 0.0, 1.0)
    annulus_alpha = np.clip(annulus_alpha.astype(np.float32), 0.0, 1.0)
    return image, annulus_alpha


def zero_mean_template(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if float(np.sum(alpha)) <= 1.0e-6:
        return image.astype(np.float32)
    masked = image.astype(np.float32)
    mean = float(np.sum(masked * alpha) / max(float(np.sum(alpha)), 1.0e-6))
    centered = (masked - mean) * alpha
    norm = float(np.sqrt(np.sum(centered * centered))) + 1.0e-6
    return centered / norm
