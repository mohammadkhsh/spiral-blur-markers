from __future__ import annotations

import argparse
import csv
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


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _load_manifest(root: Path) -> dict[str, dict[str, str]]:
    manifest_path = root / "manifest.csv"
    rows: dict[str, dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[str(Path(row["image_path"]).resolve())] = row
    return rows


def _load_scene_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _class_name(class_id: int) -> str:
    return {0: "neg", 1: "zero", 2: "pos"}.get(int(class_id), f"cls{class_id}")


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


def _select_blur_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    blur_rows = [row for row in rows if row.get("domain") == "blur"]
    if not blur_rows:
        return []
    scored = sorted(
        blur_rows,
        key=lambda row: (
            -float(row["blur_angle_error_deg"]),
            float(row["id_accuracy"]),
            float(row["robot_recall"]),
        ),
    )
    if len(scored) <= count:
        return scored
    picks: list[dict[str, str]] = []
    used: set[str] = set()
    for idx in np.linspace(0, len(scored) - 1, num=count):
        row = scored[int(round(float(idx)))]
        scene_name = row["scene_name"]
        if scene_name in used:
            continue
        picks.append(row)
        used.add(scene_name)
    if len(picks) < count:
        for row in scored:
            if row["scene_name"] in used:
                continue
            picks.append(row)
            used.add(row["scene_name"])
            if len(picks) >= count:
                break
    return picks[:count]


def _panel_text(image: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image.copy()
    y = 24
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        y += 22
    return out


def _draw_scene_overlay(image_rgb: np.ndarray, payload: dict, det_rows: list[dict], robot_preds: list[dict], metrics_row: dict[str, str]) -> np.ndarray:
    image_bgr = cv2.cvtColor((np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    for det in det_rows:
        x0, y0, x1, y1 = [int(round(float(v))) for v in det["bbox_xyxy"]]
        color = _class_color_bgr(det["class_id"])
        cv2.rectangle(image_bgr, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        cv2.circle(image_bgr, (int(round(det["center_xy"][0])), int(round(det["center_xy"][1]))), 2, color, -1, cv2.LINE_AA)
        label = f"{_class_name(det['class_id'])} {float(det['confidence']):.2f}"
        cv2.putText(image_bgr, label, (x0, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image_bgr, label, (x0, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    zero_spiral_centers: dict[int, tuple[float, float]] = {}
    for spiral in payload.get("spirals", []):
        if int(spiral.get("slot_index", -1)) == 0:
            zero_spiral_centers[int(spiral["robot_index"])] = tuple(float(v) for v in spiral["center_xy"])

    for robot in payload.get("robots", []):
        center_xy = tuple(float(v) for v in robot["center_xy"])
        zero_center_xy = zero_spiral_centers.get(int(robot["robot_index"]))
        cv2.drawMarker(image_bgr, (int(round(center_xy[0])), int(round(center_xy[1]))), (255, 255, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        if zero_center_xy is not None:
            _draw_arrow_to_point(image_bgr, center_xy, zero_center_xy, (255, 255, 255), thickness=2)
            cv2.circle(image_bgr, (int(round(zero_center_xy[0])), int(round(zero_center_xy[1]))), 4, (255, 255, 255), 1, cv2.LINE_AA)
        _draw_arrow(image_bgr, center_xy, float(robot["blur_angle_deg"]), max(12.0, float(robot["blur_length_px"]) * 3.0), (0, 165, 255), thickness=2)

    for idx, robot in enumerate(robot_preds):
        center_xy = tuple(float(v) for v in robot["center_xy"])
        cv2.circle(image_bgr, (int(round(center_xy[0])), int(round(center_xy[1]))), 10, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(image_bgr, (int(round(center_xy[0])), int(round(center_xy[1]))), 3, (0, 0, 255), -1, cv2.LINE_AA)
        zero_center_xy = robot.get("zero_center_xy")
        if zero_center_xy is not None:
            _draw_arrow_to_point(image_bgr, center_xy, tuple(float(v) for v in zero_center_xy), (0, 255, 0), thickness=2)
            cv2.circle(image_bgr, (int(round(float(zero_center_xy[0]))), int(round(float(zero_center_xy[1])))), 4, (0, 255, 0), 1, cv2.LINE_AA)
        _draw_arrow(image_bgr, center_xy, float(robot["blur_angle_deg"]), max(12.0, float(robot["blur_length_px"]) * 3.0), (0, 0, 255), thickness=2)
        pred_label = (
            f"P{idx} id={int(robot['robot_id'])} "
            f"b={float(robot['blur_length_px']):.1f}px "
            f"ang={float(robot['blur_angle_deg']):.1f} "
            f"sp={float(robot.get('blur_sign_prob', 0.5)):.2f}"
        )
        cv2.putText(image_bgr, pred_label, (int(round(center_xy[0])) + 8, int(round(center_xy[1])) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    lines = [
        f"{metrics_row['scene_name']}  domain={metrics_row['domain']}",
        f"robot R={float(metrics_row['robot_recall']):.2f} P={float(metrics_row['robot_precision']):.2f} ID={float(metrics_row['id_accuracy']):.2f}",
        f"err center={float(metrics_row['center_error_px']):.2f}px heading={float(metrics_row['heading_error_deg']):.2f}deg",
        f"err blurA={float(metrics_row['blur_angle_error_deg']):.2f}deg blurL={float(metrics_row['blur_length_error_px']):.2f}px",
        f"time det={float(metrics_row['detector_time_ms']):.1f}ms blur={float(metrics_row['reasoner_time_ms']):.1f}ms total={float(metrics_row['total_time_ms']):.1f}ms",
        "white=GT heading, orange=GT blur, green=pred heading, red=pred blur",
    ]
    return _panel_text(image_bgr, lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export blurred-scene overlays for grouped blur v2.")
    parser.add_argument("--data-root", default="outputs/p4_ai_dataset_v1_sharp")
    parser.add_argument("--metrics-csv", default="outputs/p4_ai_eval_sharp_v8_grouped_blur_v2_val30/scene_metrics.csv")
    parser.add_argument("--detector-weights", default="outputs/p4_ai_rtdetr_train_sharp/yolov8n_rtdetr_p4_sharp_70_30_e50_v2/weights/best.pt")
    parser.add_argument("--blur-weights", default="outputs/p4_ai_group_blur_train_v2/p4_grouped_blur_v2_70_30_b128e12_best.pt")
    parser.add_argument("--out", default="outputs/p4_ai_eval_sharp_v8_grouped_blur_v2_val30/blur_overlay_samples")
    parser.add_argument("--num-scenes", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    data_root = (ROOT / args.data_root).resolve()
    out_root = (ROOT / args.out).resolve()
    raw_root = out_root / "raw_images"
    overlay_root = out_root / "overlays"
    meta_root = out_root / "metadata"
    for path in (out_root, raw_root, overlay_root, meta_root):
        path.mkdir(parents=True, exist_ok=True)

    metrics_rows = _load_scene_metrics((ROOT / args.metrics_csv).resolve())
    chosen_rows = _select_blur_rows(metrics_rows, int(args.num_scenes))
    manifest = _load_manifest(data_root)

    detector = RTDETR(str((ROOT / args.detector_weights).resolve()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blur_model = P4GroupedBlurEstimatorV2().to(device)
    state = torch.load((ROOT / args.blur_weights).resolve(), map_location=device)
    blur_model.load_state_dict(state["model"])
    blur_model.eval()

    exported = []
    for row in chosen_rows:
        image_path = None
        json_path = None
        for key, item in manifest.items():
            if Path(key).name == row["scene_name"]:
                image_path = Path(key)
                json_path = Path(item["json_path"]).resolve()
                break
        if image_path is None or json_path is None:
            continue

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        image_rgb = _load_rgb(image_path)
        det_result = detector.predict(
            source=(image_rgb * 255.0).astype(np.uint8),
            imgsz=int(args.imgsz),
            device=str(args.device),
            conf=float(args.conf),
            verbose=False,
            max_det=24,
        )[0]

        det_rows: list[dict] = []
        if det_result.boxes is not None and len(det_result.boxes) > 0:
            xyxy = det_result.boxes.xyxy.cpu().numpy()
            confs = det_result.boxes.conf.cpu().numpy()
            classes = det_result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls in zip(xyxy, confs, classes):
                det_rows.append(
                    {
                        "bbox_xyxy": [float(v) for v in box.tolist()],
                        "class_id": int(cls),
                        "confidence": float(conf),
                        "center_xy": (float(0.5 * (box[0] + box[2])), float(0.5 * (box[1] + box[3]))),
                    }
                )

        robot_preds = predict_group_blur_robots_v2(blur_model, image_rgb, det_rows, device)
        overlay = _draw_scene_overlay(image_rgb, payload, det_rows, robot_preds, row)

        raw_bgr = cv2.cvtColor((np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        raw_path = raw_root / row["scene_name"]
        overlay_path = overlay_root / row["scene_name"]
        meta_path = meta_root / f"{Path(row['scene_name']).stem}.json"
        cv2.imwrite(str(raw_path), raw_bgr)
        cv2.imwrite(str(overlay_path), overlay)
        meta_payload = {
            "scene_name": row["scene_name"],
            "metrics": row,
            "image_path": str(image_path),
            "json_path": str(json_path),
            "det_rows": det_rows,
            "robot_preds": robot_preds,
            "robot_gt": payload.get("robots", []),
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
        exported.append(
            {
                "scene_name": row["scene_name"],
                "raw_path": str(raw_path),
                "overlay_path": str(overlay_path),
                "meta_path": str(meta_path),
            }
        )

    (out_root / "index.json").write_text(json.dumps(exported, indent=2), encoding="utf-8")
    print(json.dumps({"out_root": str(out_root), "num_exported": len(exported)}, indent=2))


if __name__ == "__main__":
    main()
