from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spiral_markers.ml.runtime_env import patch_windows_platform_for_torch

patch_windows_platform_for_torch()

from ultralytics import RTDETR

from spiral_markers.ml.p4_group_blur_inference_v2 import predict_group_blur_robots_v2
from spiral_markers.ml.p4_grouped_blur_estimator_v2 import P4GroupedBlurEstimatorV2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _resize_rgb(image_rgb: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(
        image_rgb.astype(np.float32),
        (int(size), int(size)),
        interpolation=cv2.INTER_AREA if image_rgb.shape[0] > int(size) or image_rgb.shape[1] > int(size) else cv2.INTER_LINEAR,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def _class_color_bgr(class_id: int) -> tuple[int, int, int]:
    if int(class_id) == 0:
        return (0, 140, 255)
    if int(class_id) == 1:
        return (0, 255, 255)
    return (255, 100, 0)


def _draw_arrow(
    image: np.ndarray,
    start_xy: tuple[float, float],
    angle_deg: float,
    length_px: float,
    color_bgr: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    sx, sy = float(start_xy[0]), float(start_xy[1])
    angle_rad = math.radians(float(angle_deg))
    ex = sx + float(length_px) * math.cos(angle_rad)
    ey = sy + float(length_px) * math.sin(angle_rad)
    cv2.arrowedLine(
        image,
        (int(round(sx)), int(round(sy))),
        (int(round(ex)), int(round(ey))),
        color_bgr,
        thickness,
        cv2.LINE_AA,
        tipLength=0.22,
    )


def _draw_arrow_to_point(
    image: np.ndarray,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    color_bgr: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.arrowedLine(
        image,
        (int(round(float(start_xy[0]))), int(round(float(start_xy[1])))),
        (int(round(float(end_xy[0]))), int(round(float(end_xy[1])))),
        color_bgr,
        thickness,
        cv2.LINE_AA,
        tipLength=0.22,
    )


def _build_det_rows(det_result) -> list[dict]:
    rows: list[dict] = []
    if det_result.boxes is None or len(det_result.boxes) == 0:
        return rows
    xyxy = det_result.boxes.xyxy.cpu().numpy()
    confs = det_result.boxes.conf.cpu().numpy()
    classes = det_result.boxes.cls.cpu().numpy().astype(int)
    for box, conf, cls in zip(xyxy, confs, classes):
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        rows.append(
            {
                "bbox_xyxy": [x0, y0, x1, y1],
                "class_id": int(cls),
                "confidence": float(conf),
                "center_xy": (float(0.5 * (x0 + x1)), float(0.5 * (y0 + y1))),
            }
        )
    return rows


def _filter_rows_by_confidence(det_rows: list[dict], min_confidence: float) -> list[dict]:
    min_confidence = float(min_confidence)
    if min_confidence <= 0.0:
        return list(det_rows)
    return [
        dict(row)
        for row in det_rows
        if float(row.get("confidence", 0.0)) >= min_confidence
    ]


def _dedupe_rows(det_rows: list[dict], merge_radius_px: float) -> list[dict]:
    kept: list[dict] = []
    for row in sorted(det_rows, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        center_xy = tuple(float(v) for v in row["center_xy"])
        if any(
            math.hypot(center_xy[0] - float(prev["center_xy"][0]), center_xy[1] - float(prev["center_xy"][1]))
            < float(merge_radius_px)
            for prev in kept
        ):
            continue
        kept.append(dict(row))
    return kept


def _estimate_grouping_params(det_rows: list[dict]) -> tuple[int, float]:
    if not det_rows:
        return 0, 28.0
    max_sides = [
        max(float(row["bbox_xyxy"][2] - row["bbox_xyxy"][0]), float(row["bbox_xyxy"][3] - row["bbox_xyxy"][1]))
        for row in det_rows
    ]
    median_side = float(np.median(np.asarray(max_sides, dtype=np.float32))) if max_sides else 70.0
    dedupe_radius_px = float(np.clip(0.40 * median_side, 28.0, 160.0))
    deduped = _dedupe_rows(det_rows, merge_radius_px=dedupe_radius_px)
    max_complete_groups = len(deduped) // 4
    zero_count = sum(1 for row in deduped if int(row["class_id"]) == 1)
    if max_complete_groups <= 0:
        return 0, dedupe_radius_px
    if zero_count <= 0:
        return max_complete_groups, dedupe_radius_px
    return max(1, min(int(zero_count), int(max_complete_groups))), dedupe_radius_px


def _draw_paper_style_overlay(image_rgb: np.ndarray, det_rows: list[dict], robot_preds: list[dict]) -> np.ndarray:
    image_bgr = cv2.cvtColor((np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    for det in det_rows:
        x0, y0, x1, y1 = [int(round(float(v))) for v in det["bbox_xyxy"]]
        color = _class_color_bgr(int(det["class_id"]))
        cv2.rectangle(image_bgr, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        cv2.circle(
            image_bgr,
            (int(round(float(det["center_xy"][0]))), int(round(float(det["center_xy"][1])))),
            3,
            color,
            -1,
            cv2.LINE_AA,
        )

    for robot in robot_preds:
        center_xy = tuple(float(v) for v in robot["center_xy"])
        center_px = (int(round(center_xy[0])), int(round(center_xy[1])))
        cv2.circle(image_bgr, center_px, 10, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(image_bgr, center_px, (0, 0, 255), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

        zero_center_xy = robot.get("zero_center_xy")
        if zero_center_xy is not None:
            zero_center_xy = tuple(float(v) for v in zero_center_xy)
            _draw_arrow_to_point(image_bgr, center_xy, zero_center_xy, (0, 255, 0), thickness=2)
            cv2.circle(
                image_bgr,
                (int(round(zero_center_xy[0])), int(round(zero_center_xy[1]))),
                4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        else:
            _draw_arrow(image_bgr, center_xy, float(robot.get("heading_deg", 0.0)), 34.0, (0, 255, 0), thickness=2)

        pred_label = f"P id={int(robot['robot_id'])}"
        cv2.putText(
            image_bgr,
            pred_label,
            (center_px[0] + 8, center_px[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            pred_label,
            (center_px[0] + 8, center_px[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return image_bgr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final sharp two-stage pipeline on real images using the paper qualitative overlay style.")
    parser.add_argument("--image-dir", default="real-tests")
    parser.add_argument("--detector-weights", default="outputs/stage1_detector_ablation/train/rtdetr_l_spiral_p5_n/weights/best.pt")
    parser.add_argument("--blur-weights", default="outputs/p4_ai_group_blur_train_v2_stage1/p4_grouped_blur_v2_stage1_rtdetr_l_p5_best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--resize-to", type=int, default=640)
    parser.add_argument("--no-resize", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--min-stage1-conf", type=float, default=0.20)
    parser.add_argument("--max-det", type=int, default=128)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    image_dir = (ROOT / args.image_dir).resolve()
    detector_weights = (ROOT / args.detector_weights).resolve()
    blur_weights = (ROOT / args.blur_weights).resolve()
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)
    if not detector_weights.exists():
        raise FileNotFoundError(detector_weights)
    if not blur_weights.exists():
        raise FileNotFoundError(blur_weights)

    detector = RTDETR(str(detector_weights))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blur_model = P4GroupedBlurEstimatorV2().to(device)
    state = torch.load(blur_weights, map_location=device)
    blur_model.load_state_dict(state["model"])
    blur_model.eval()

    glob_fn = image_dir.rglob if args.recursive else image_dir.glob
    image_paths = sorted(
        path
        for path in glob_fn("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and "_overlay" not in path.stem
        and "_detections" not in path.stem
        and "_resized" not in path.stem
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    summary_rows: list[dict[str, object]] = []
    for image_path in image_paths:
        image_rgb = _load_rgb(image_path)
        if args.no_resize:
            resized_rgb = image_rgb
            resize_suffix = "native"
        else:
            resized_rgb = _resize_rgb(image_rgb, int(args.resize_to))
            resize_suffix = f"resized{int(args.resize_to)}"
        det_result = detector.predict(
            source=(resized_rgb * 255.0).astype(np.uint8),
            imgsz=int(args.imgsz),
            device=str(args.device),
            conf=float(args.conf),
            verbose=False,
            max_det=int(args.max_det),
        )[0]
        raw_det_rows = _build_det_rows(det_result)
        det_rows = _filter_rows_by_confidence(raw_det_rows, float(args.min_stage1_conf))
        num_clusters, dedupe_radius_px = _estimate_grouping_params(det_rows)
        robot_preds = predict_group_blur_robots_v2(
            blur_model,
            resized_rgb,
            det_rows,
            device,
            num_clusters=int(num_clusters),
            dedupe_radius_px=float(dedupe_radius_px),
        )
        overlay_bgr = _draw_paper_style_overlay(resized_rgb, det_rows, robot_preds)

        resized_path = image_path.with_name(f"{image_path.stem}_{resize_suffix}{image_path.suffix}")
        overlay_path = image_path.with_name(f"{image_path.stem}_{resize_suffix}_overlay{image_path.suffix}")
        json_path = image_path.with_name(f"{image_path.stem}_{resize_suffix}_detections.json")
        cv2.imwrite(
            str(resized_path),
            cv2.cvtColor((np.clip(resized_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(overlay_path), overlay_bgr)
        json_path.write_text(
            json.dumps(
                {
                    "image_path": str(image_path),
                    "resized_image_path": str(resized_path),
                    "overlay_path": str(overlay_path),
                    "detector_weights": str(detector_weights),
                    "blur_weights": str(blur_weights),
                    "imgsz": int(args.imgsz),
                    "resize_to": None if args.no_resize else int(args.resize_to),
                    "no_resize": bool(args.no_resize),
                    "conf": float(args.conf),
                    "min_stage1_conf": float(args.min_stage1_conf),
                    "raw_spiral_detection_count": len(raw_det_rows),
                    "filtered_spiral_detection_count": len(det_rows),
                    "estimated_num_clusters": int(num_clusters),
                    "group_dedupe_radius_px": float(dedupe_radius_px),
                    "spiral_detections": det_rows,
                    "robot_detections": robot_preds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary_rows.append(
            {
                "image": image_path.name,
                "resized": resized_path.name,
                "overlay": overlay_path.name,
                "json": json_path.name,
                "raw_spiral_count": len(raw_det_rows),
                "spiral_count": len(det_rows),
                "robot_count": len(robot_preds),
                "estimated_num_clusters": int(num_clusters),
                "group_dedupe_radius_px": float(dedupe_radius_px),
            }
        )
        print(
            f"[done] {image_path.name} resized={resized_path.name} robots={len(robot_preds)} "
            f"spirals={len(det_rows)}/{len(raw_det_rows)} "
            f"clusters={num_clusters} dedupe={dedupe_radius_px:.1f}px"
        )

    summary_path = image_dir / "two_stage_real_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(json.dumps({"image_dir": str(image_dir), "num_images": len(image_paths), "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
