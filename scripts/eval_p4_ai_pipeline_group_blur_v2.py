from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spiral_markers.ml.runtime_env import patch_windows_platform_for_torch

patch_windows_platform_for_torch()

import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import RTDETR

from spiral_markers.ml.p4_group_blur_inference_v2 import predict_group_blur_robots_v2
from spiral_markers.ml.p4_grouped_blur_estimator_v2 import P4GroupedBlurEstimatorV2


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _load_manifest(root: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with (root / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[str(Path(row["image_path"]).resolve())] = row
    return rows


def _angle_error_deg(pred: float, target: float) -> float:
    delta = (float(pred) - float(target) + 180.0) % 360.0 - 180.0
    return abs(delta)


def _spiral_match_metrics(preds: list[dict], gts: list[dict], max_dist: float = 24.0) -> tuple[float, float, float]:
    if not preds:
        return 0.0, 0.0, 0.0
    if not gts:
        return 0.0, 0.0, 0.0
    cost = np.full((len(preds), len(gts)), fill_value=1.0e6, dtype=np.float32)
    for i, pred in enumerate(preds):
        px, py = pred["center_xy"]
        for j, gt in enumerate(gts):
            gx, gy = gt["center_xy"]
            dist = math.hypot(px - gx, py - gy)
            if dist <= max_dist:
                cost[i, j] = dist
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        if cost[i, j] < 1.0e5:
            matches.append((i, j))
    recall = len(matches) / max(len(gts), 1)
    precision = len(matches) / max(len(preds), 1)
    class_acc = float(np.mean([preds[i]["class_id"] == gts[j]["class_id"] for i, j in matches])) if matches else 0.0
    return float(recall), float(precision), float(class_acc)


def _robot_match_metrics(preds: list[dict], gts: list[dict], max_dist: float = 48.0) -> dict[str, float]:
    if not preds and not gts:
        return {
            "robot_recall": 1.0,
            "robot_precision": 1.0,
            "id_accuracy": 1.0,
            "center_error_px": 0.0,
            "heading_error_deg": 0.0,
            "blur_angle_error_deg": 0.0,
            "blur_length_error_px": 0.0,
        }
    if not preds:
        return {
            "robot_recall": 0.0,
            "robot_precision": 0.0,
            "id_accuracy": 0.0,
            "center_error_px": float("nan"),
            "heading_error_deg": float("nan"),
            "blur_angle_error_deg": float("nan"),
            "blur_length_error_px": float("nan"),
        }
    cost = np.full((len(preds), len(gts)), fill_value=1.0e6, dtype=np.float32)
    for i, pred in enumerate(preds):
        px, py = pred["center_xy"]
        for j, gt in enumerate(gts):
            gx, gy = gt["center_xy"]
            dist = math.hypot(px - gx, py - gy)
            if dist <= max_dist:
                cost[i, j] = dist
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        if cost[i, j] < 1.0e5:
            matches.append((i, j))
    return {
        "robot_recall": len(matches) / max(len(gts), 1),
        "robot_precision": len(matches) / max(len(preds), 1),
        "id_accuracy": float(np.mean([preds[i]["robot_id"] == gts[j]["robot_id"] for i, j in matches])) if matches else 0.0,
        "center_error_px": float(np.mean([cost[i, j] for i, j in matches])) if matches else float("nan"),
        "heading_error_deg": float(np.mean([_angle_error_deg(preds[i]["heading_deg"], gts[j]["heading_deg"]) for i, j in matches])) if matches else float("nan"),
        "blur_angle_error_deg": float(np.mean([_angle_error_deg(preds[i]["blur_angle_deg"], gts[j]["blur_angle_deg"]) for i, j in matches])) if matches else float("nan"),
        "blur_length_error_px": float(np.mean([abs(preds[i]["blur_length_px"] - gts[j]["blur_length_px"]) for i, j in matches])) if matches else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR + grouped blur v2 head on the paper_v4 sharp dataset.")
    parser.add_argument("--data-root", default="outputs/p4_ai_dataset_v1_sharp")
    parser.add_argument("--detector-weights", default="outputs/p4_ai_rtdetr_train_sharp/yolov8n_rtdetr_p4_sharp_70_30_e50_v2/weights/best.pt")
    parser.add_argument("--blur-weights", default="outputs/p4_ai_group_blur_train_v2/p4_grouped_blur_v2_70_30_best.pt")
    parser.add_argument("--out", default="outputs/p4_ai_eval_sharp_v8_grouped_blur_v2_val30")
    parser.add_argument("--split-file", default="outputs/p4_ai_dataset_v1_sharp/val_30.txt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    out_root = (ROOT / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "eval.log"
    scene_rows = []

    detector = RTDETR(str((ROOT / args.detector_weights).resolve()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blur_model = P4GroupedBlurEstimatorV2().to(device)
    state = torch.load((ROOT / args.blur_weights).resolve(), map_location=device)
    blur_model.load_state_dict(state["model"])
    blur_model.eval()

    data_root = (ROOT / args.data_root).resolve()
    manifest = _load_manifest(data_root)
    split_path = Path(args.split_file)
    if not split_path.is_absolute():
        split_path = (ROOT / split_path).resolve()
    eval_items: list[tuple[Path, Path]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        image_path_str = line.strip()
        if not image_path_str:
            continue
        image_path = Path(image_path_str).resolve()
        row = manifest.get(str(image_path))
        if row is None:
            raise KeyError(f"Image path not found in manifest: {image_path}")
        eval_items.append((image_path, Path(row["json_path"]).resolve()))

    with log_path.open("w", encoding="utf-8") as log_handle:
        start = time.perf_counter()
        for idx, (image_path, json_path) in enumerate(eval_items, start=1):
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            image_rgb = _load_rgb(image_path)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            det_result = detector.predict(
                source=(image_rgb * 255.0).astype(np.uint8),
                imgsz=int(args.imgsz),
                device=str(args.device),
                conf=float(args.conf),
                verbose=False,
                max_det=24,
            )[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            det_rows = []
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
            spiral_gt = list(payload["spirals"])
            spiral_metrics = _spiral_match_metrics(det_rows, spiral_gt)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            robot_preds = predict_group_blur_robots_v2(blur_model, image_rgb, det_rows, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            robot_gt = list(payload["robots"])
            robot_metrics = _robot_match_metrics(robot_preds, robot_gt)
            row = {
                "scene_name": payload["scene_name"],
                "domain": payload["domain"],
                "spiral_recall": spiral_metrics[0],
                "spiral_precision": spiral_metrics[1],
                "spiral_class_accuracy": spiral_metrics[2],
                **robot_metrics,
                "detector_time_ms": float((t1 - t0) * 1000.0),
                "reasoner_time_ms": float((t3 - t2) * 1000.0),
                "total_time_ms": float((t1 - t0 + t3 - t2) * 1000.0),
            }
            scene_rows.append(row)
            line = json.dumps(row)
            print(line)
            log_handle.write(line + "\n")
            log_handle.flush()
            if idx % 50 == 0:
                elapsed = time.perf_counter() - start
                print(f"[progress] {idx}/{len(eval_items)} elapsed={elapsed/60.0:.1f}m")

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in scene_rows if not math.isnan(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "num_scenes": len(scene_rows),
        "spiral_recall": _mean("spiral_recall"),
        "spiral_precision": _mean("spiral_precision"),
        "spiral_class_accuracy": _mean("spiral_class_accuracy"),
        "robot_recall": _mean("robot_recall"),
        "robot_precision": _mean("robot_precision"),
        "id_accuracy": _mean("id_accuracy"),
        "center_error_px": _mean("center_error_px"),
        "heading_error_deg": _mean("heading_error_deg"),
        "blur_angle_error_deg": _mean("blur_angle_error_deg"),
        "blur_length_error_px": _mean("blur_length_error_px"),
        "detector_time_ms": _mean("detector_time_ms"),
        "reasoner_time_ms": _mean("reasoner_time_ms"),
        "total_time_ms": _mean("total_time_ms"),
        "log_path": str(log_path),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_root / "scene_metrics.csv").open("w", encoding="utf-8") as handle:
        keys = list(scene_rows[0].keys()) if scene_rows else []
        handle.write(",".join(keys) + "\n")
        for row in scene_rows:
            handle.write(",".join(str(row[k]) for k in keys) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
