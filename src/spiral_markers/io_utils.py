from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar, get_args, get_origin, get_type_hints

import cv2
import numpy as np
import yaml


@dataclass
class SpiralSynthesisConfig:
    method_key: str = "baseline"
    image_size: int = 160
    radius: float = 54.0
    base_legs: int = 12
    twist_angle_deg: float = 32.0
    baseline_identity_theta_deg: float = 45.0
    low_bit_angle_deg: float = 18.0
    high_bit_angle_deg: float = 44.0
    contrast: float = 1.0
    background: float = 1.0
    center_disk_radius: float = 7.0
    anti_alias_sigma: float = 0.9
    ring_count: int = 1
    ring_frequency_multipliers: tuple[float, ...] = (1.0,)
    ring_weights: tuple[float, ...] = (1.0,)
    ring_gap_ratio: float = 0.08
    inner_hole_ratio: float = 0.18
    direction_diverse_mix: float = 0.0
    direction_diverse_phase_deg: float = 45.0
    min_legs: int = 2
    alias_safe_blend: float = 0.0
    self_norm_ring_pair: bool = False
    self_norm_inner_radius_ratio: float = 0.76
    self_norm_outer_radius_ratio: float = 0.94
    self_norm_ring_width_ratio: float = 0.04
    sector_count: int = 0
    sector_code_strength: float = 0.0
    sector_phase_offset_deg: float = 18.0
    ecc_sector_gap_deg: float = 8.0
    ecc_sector_soft_edge_deg: float = 1.5
    ecc_primary_sector_weight: float = 1.0
    ecc_parity_sector_weight: float = 0.85


@dataclass
class TagLayoutConfig:
    identity_slots: int = 4
    group_radius: float = 82.0
    orientation_theta_deg: float = 0.0
    orientation_slot_angle_deg: float = 0.0
    anchor_count: int = 0
    anchor_angles_deg: tuple[float, ...] = (25.0, 135.0, 220.0, 325.0)
    anchor_radius_ratios: tuple[float, ...] = (0.34, 0.46, 0.59, 0.74)
    anchor_intensities: tuple[float, ...] = (1.0, 0.8, 0.65, 0.9)
    anchor_size_ratio: float = 0.085


@dataclass
class DetectionConfig:
    detector_mode: str = "baseline"
    baseline_decode_mode: str = "balanced"
    use_gpu: bool = False
    gpu_device_id: int = 0
    gpu_frontend: bool = True
    gpu_slot_scoring: bool = True
    fast_baseline_mode: bool = False
    enable_expensive_pose_refinement: bool = False
    fast_slot_search_radius_px: int = 1
    balanced_slot_search_radius_px: int = 2
    balanced_refine_step_px: int = 2
    balanced_refine_step_deg: float = 2.0
    balanced_branch_gate_enabled: bool = False
    balanced_branch_min_orientation_margin: float = 0.25
    balanced_branch_min_id_margin: float = 3.0
    balanced_gpu_pose_batch: bool = True
    decode_direct_min_orientation_margin: float = 0.08
    decode_direct_min_bit_margin: float = 0.08
    decode_direct_min_id_margin: float = 0.12
    gaussian_sigma: float = 1.1
    gst_sigma: float = 2.2
    eps: float = 1.0e-6
    use_normalized_score: bool = False
    local_contrast_normalization: bool = False
    template_angles_deg: tuple[float, ...] = (0.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0)
    template_size: int = 160
    template_radius: float = 54.0
    score_threshold: float = 0.33
    nms_radius: int = 18
    max_detections: int = 96
    peak_large_component_area_factor: float = 2.0
    peak_large_component_top_k: int = 3
    peak_suppression_radius_scale: float = 1.0
    cluster_radius: float = 135.0
    expected_group_size: int = 5
    cluster_pair_alpha: float = 1.15
    cluster_pair_beta: float = 8.0
    cluster_pair_max_radius: float = 170.0
    cluster_split_enabled: bool = True
    cluster_split_min_size: int = 6
    cluster_split_diameter_factor: float = 1.35
    cluster_split_improvement: float = 0.04
    orientation_candidate_twist_deg: float = 20.0
    orientation_hypothesis_max: int = 6
    decode_w_ang: float = 1.0
    decode_w_rad: float = 0.9
    decode_w_conf: float = 0.4
    decode_w_twist: float = 0.35
    decode_slot_gate_ratio: float = 0.55
    decode_slot_gate_angle_deg: float = 65.0
    decode_assignment_cost_threshold: float = 1.65
    decode_layout_w_conf: float = 0.55
    decode_layout_w_visible: float = 0.75
    decode_layout_w_residual: float = 1.15
    decode_layout_w_slot_error: float = 1.0
    decode_min_spirals: int = 4
    decode_relaxed_min_spirals: int = 3
    decode_min_layout_score: float = 0.35
    decode_relaxed_min_layout_score: float = 0.58
    decode_max_residual_ratio: float = 0.52
    decode_relaxed_max_residual_ratio: float = 0.24
    decode_orientation_margin: float = 0.10
    robot_verification_enabled: bool = True
    robot_min_visible_spirals: int = 3
    robot_support_threshold: float = 0.16
    robot_slot_match_radius_ratio: float = 0.42
    robot_twist_consistency_tau_deg: float = 14.0
    robot_min_verification_score: float = 0.34
    robot_nms_iou_threshold: float = 0.28
    robot_nms_center_radius_ratio: float = 0.48
    orientation_angle_tolerance_deg: float = 8.0
    decode_twist_deadzone_deg: float = 6.0
    fusion_mode: str = "max"
    multiscale_sigmas: tuple[float, ...] = (1.0,)
    pyramid_scales: tuple[float, ...] = (1.0,)
    s3_small_radius_px: float = 26.0
    s3_scheduler_width_px: float = 6.0
    s3_coarse_threshold_scale: float = 0.68
    s3_candidate_multiplier: int = 2
    s3_refine_roi_radius_factor: float = 2.1
    s3_refine_search_ratio: float = 0.45
    s3_candidate_nms_scale: float = 0.75
    s3_candidate_pair_radius_scale: float = 0.18
    self_norm_weight: float = 0.0
    psn_self_norm_weight_min: float = 0.05
    psn_self_norm_weight_max: float = 0.55
    psn_variance_floor: float = 0.25
    psn_fallback_strength_ratio: float = 0.72
    psn_fallback_blend: float = 0.75
    small_peak_threshold_drop: float = 0.28
    sector_decode: bool = False
    sector_sample_radius_ratio: float = 0.72
    sector_inner_radius_ratio: float = 0.46
    sector_visibility_floor: float = 0.12
    sector_erasure_visibility_threshold: float = 0.18
    sector_min_visible_ratio: float = 0.35
    sector_min_confidence: float = 0.10
    sector_override_margin: float = 0.08
    anchor_analysis: bool = False
    blur_length_grid: tuple[int, ...] = (0, 3, 5, 7, 9, 11, 13, 15)
    blur_angle_step_deg: float = 15.0
    beta_grid: tuple[float, ...] = (-0.6, -0.3, 0.0, 0.3, 0.6)


@dataclass
class SceneConfig:
    image_size: tuple[int, int] = (768, 768)
    background: float = 0.92
    num_tags: int = 4
    tag_scale_range: tuple[float, float] = (0.82, 1.15)
    rotation_range_deg: tuple[float, float] = (-180.0, 180.0)
    tilt_range_deg: tuple[float, float] = (-10.0, 10.0)
    min_center_distance: float = 190.0
    max_attempts: int = 200


@dataclass
class SheetConfig:
    ids: tuple[int, ...] = tuple(range(8))
    columns: int = 4
    page_size_inches: tuple[float, float] = (8.27, 11.69)
    dpi: int = 220
    margin_inches: float = 0.35


@dataclass
class CorruptionConfig:
    motion_blur_length: int = 0
    motion_blur_angle_deg: float = 0.0
    motion_beta: float = 0.0
    illumination_scale: float = 1.0
    illumination_offset: float = 0.0
    shadow_strength: float = 0.0
    shadow_coverage: float = 0.0
    gaussian_noise_sigma: float = 0.0
    contrast_scale: float = 1.0
    occlusion_ratio: float = 0.0


@dataclass
class MethodConfig:
    name: str = "Baseline-GST-Spiral"
    seed: int = 13
    output_root: str = "outputs"
    synthesis: SpiralSynthesisConfig = field(default_factory=SpiralSynthesisConfig)
    layout: TagLayoutConfig = field(default_factory=TagLayoutConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    sheet: SheetConfig = field(default_factory=SheetConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)


@dataclass
class BenchmarkConfig:
    output_dir: str = "outputs/results/default_benchmark"
    figure_dir: str = "outputs/figures"
    demo_dir: str = "outputs/figures"
    method_configs: tuple[str, ...] = (
        "configs/baseline.yaml",
        "configs/proposed_multiring.yaml",
        "configs/proposed_norm.yaml",
        "configs/proposed_blurdiverse.yaml",
    )
    ids: tuple[int, ...] = tuple(range(8))
    seeds: tuple[int, ...] = (0, 1)
    scenes_per_setting: int = 3
    blur_angles_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    sweeps: dict[str, list[float]] = field(
        default_factory=lambda: {
            "motion_blur_length": [0, 5, 9, 13, 17],
            "illumination_scale": [0.6, 0.8, 1.0, 1.2, 1.4],
            "shadow_coverage": [0.0, 0.15, 0.3, 0.45, 0.6],
            "occlusion_ratio": [0.0, 0.1, 0.2, 0.3, 0.4],
        }
    )


@dataclass
class SizeStudyConfig:
    output_dir: str = "outputs/size_study_4000"
    method_configs: tuple[str, ...] = (
        "configs/proposed_multiring.yaml",
        "configs/proposed_norm.yaml",
        "configs/proposed_blurdiverse.yaml",
    )
    ids: tuple[int, ...] = tuple(range(8))
    num_scenes: int = 4000
    image_size: tuple[int, int] = (448, 448)
    min_robots: int = 2
    max_robots: int = 5
    tag_scale_range: tuple[float, float] = (0.48, 0.92)
    size_bins: int = 4
    seed: int = 20260320
    image_ext: str = ".jpg"
    save_images: bool = True
    save_overlays: bool = True


@dataclass
class PaperStudyConfig:
    output_dir: str = "outputs/paper_study_1500"
    method_configs: tuple[str, ...] = (
        "configs/paper_baseline.yaml",
        "configs/paper_psn_spiral.yaml",
        "configs/paper_aar_spiral.yaml",
        "configs/paper_s3_spiral.yaml",
        "configs/paper_kse_spiral.yaml",
        "configs/paper_na_psf_spiral.yaml",
        "configs/paper_sec_spiral.yaml",
    )
    ids: tuple[int, ...] = tuple(range(8))
    num_scenes: int = 1500
    image_size: tuple[int, int] = (448, 448)
    min_robots: int = 2
    max_robots: int = 5
    tag_scale_range: tuple[float, float] = (0.30, 0.82)
    size_bins: int = 4
    seed: int = 20260321
    image_ext: str = ".jpg"
    save_images: bool = True
    save_overlays: bool = True
    families: tuple[str, ...] = ("size", "illumination", "blur", "occlusion", "combined")
    blur_fixed_scale: float = 0.56
    blur_fixed_tilt_deg: float = 0.0
    motion_exposure_ms: float = 8.0


T = TypeVar("T")


def _coerce_value(type_hint: Any, value: Any) -> Any:
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    if value is None:
        return None
    if origin in (tuple, list):
        item_type = args[0] if args else Any
        items = [_coerce_value(item_type, item) for item in value]
        return tuple(items) if origin is tuple else items
    if origin is dict:
        return dict(value)
    if origin is None and is_dataclass(type_hint):
        return dataclass_from_dict(type_hint, value)
    if origin is None and type_hint in (Path,):
        return Path(value)
    if origin is None and type_hint in (int, float, str, bool):
        return type_hint(value)
    return value


def dataclass_from_dict(cls: type[T], payload: Mapping[str, Any]) -> T:
    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in payload:
            continue
        value = payload[f.name]
        target_type = type_hints.get(f.name, f.type)
        if is_dataclass(target_type):
            kwargs[f.name] = dataclass_from_dict(target_type, value)
        else:
            kwargs[f.name] = _coerce_value(target_type, value)
    return cls(**kwargs)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_method_config(path: str | Path) -> MethodConfig:
    return dataclass_from_dict(MethodConfig, load_yaml(path))


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    return dataclass_from_dict(BenchmarkConfig, load_yaml(path))


def load_size_study_config(path: str | Path) -> SizeStudyConfig:
    return dataclass_from_dict(SizeStudyConfig, load_yaml(path))


def load_paper_study_config(path: str | Path) -> PaperStudyConfig:
    return dataclass_from_dict(PaperStudyConfig, load_yaml(path))


def save_json(payload: Any, path: str | Path, indent: int = 2) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    def _default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, default=_default)
    return output_path


def save_csv(rows: list[Mapping[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def write_image(path: str | Path, image: np.ndarray) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    if image.ndim == 2:
        to_write = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        rgb = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        to_write = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), to_write)
    return output_path


def read_image(path: str | Path, grayscale: bool = True) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    if grayscale:
        return image.astype(np.float32) / 255.0
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def as_serializable_dataclass(instance: Any) -> Any:
    if is_dataclass(instance):
        return asdict(instance)
    return instance


def resolve_baseline_identity_theta_deg(cfg: SpiralSynthesisConfig) -> float:
    configured = float(cfg.baseline_identity_theta_deg)
    if configured > 0.0:
        return configured
    return max(abs(float(cfg.low_bit_angle_deg)), abs(float(cfg.high_bit_angle_deg)), 1.0)
