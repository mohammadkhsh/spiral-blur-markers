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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ultralytics import RTDETR

from spiral_markers.ml.p4_ai_inference import _build_classical_groups
from spiral_markers.ml.p4_grouped_blur_estimator import display_angle_from_center_to_point
from spiral_markers.ml.p4_grouped_blur_estimator_v2 import (
    P4GroupedBlurEstimatorV2,
    angle_to_vec_display,
    build_group_blur_features,
)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _load_manifest(root: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with (root / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[str(Path(row["image_path"]).resolve())] = row
    return rows


def _angle_error_deg(pred: float, target: float) -> float:
    delta = (float(pred) - float(target) + 180.0) % 360.0 - 180.0
    return abs(delta)


def _build_det_rows(det_result) -> list[dict]:
    det_rows = []
    if det_result.boxes is None or len(det_result.boxes) == 0:
        return det_rows
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
    return det_rows


def _match_groups_to_gt(groups: list[dict], robots: list[dict], max_dist_px: float = 48.0) -> list[tuple[dict, dict]]:
    if not groups or not robots:
        return []
    pairs: list[tuple[float, int, int]] = []
    for gi, group in enumerate(groups):
        gcx, gcy = group["center_xy"]
        for ri, robot in enumerate(robots):
            rcx, rcy = robot["center_xy"]
            dist = math.hypot(float(gcx) - float(rcx), float(gcy) - float(rcy))
            if dist <= float(max_dist_px):
                pairs.append((dist, gi, ri))
    pairs.sort(key=lambda item: item[0])
    used_groups: set[int] = set()
    used_robots: set[int] = set()
    matches: list[tuple[dict, dict]] = []
    for _, gi, ri in pairs:
        if gi in used_groups or ri in used_robots:
            continue
        group = groups[gi]
        selected_slots = group.get("selected_slots", {})
        if any(slot_index not in selected_slots for slot_index in (0, 1, 2, 3)):
            continue
        used_groups.add(gi)
        used_robots.add(ri)
        matches.append((group, robots[ri]))
    return matches


def _precompute_samples(
    data_root: Path,
    split_file: Path,
    detector_weights: Path,
    imgsz: int,
    device_arg: str,
    conf: float,
) -> tuple[list[dict], dict[str, float]]:
    manifest = _load_manifest(data_root)
    eval_items: list[tuple[Path, Path]] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        image_path_str = line.strip()
        if not image_path_str:
            continue
        image_path = Path(image_path_str).resolve()
        row = manifest.get(str(image_path))
        if row is None:
            raise KeyError(f"Image path not found in manifest: {image_path}")
        eval_items.append((image_path, Path(row["json_path"]).resolve()))

    detector = RTDETR(str(detector_weights))
    use_cuda = torch.cuda.is_available() and str(device_arg).lower() != "cpu"

    samples: list[dict] = []
    matched_groups = 0
    total_robots = 0
    scene_times_ms: list[float] = []
    start = time.perf_counter()
    for idx, (image_path, json_path) in enumerate(eval_items, start=1):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        image_rgb = _load_rgb(image_path)
        total_robots += len(payload.get("robots", []))
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        det_result = detector.predict(
            source=(image_rgb * 255.0).astype(np.uint8),
            imgsz=int(imgsz),
            device=str(device_arg),
            conf=float(conf),
            verbose=False,
            max_det=24,
        )[0]
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        scene_times_ms.append(float((t1 - t0) * 1000.0))
        det_rows = _build_det_rows(det_result)
        groups = _build_classical_groups(det_rows)
        matches = _match_groups_to_gt(groups, list(payload.get("robots", [])))
        matched_groups += len(matches)
        for group, robot in matches:
            selected_slots = group["selected_slots"]
            slot_centers = {
                int(slot_index): tuple(float(v) for v in slot["center_xy"])
                for slot_index, slot in selected_slots.items()
            }
            slot_classes = {
                int(slot_index): int(slot["class_id"])
                for slot_index, slot in selected_slots.items()
            }
            pred_center = tuple(float(v) for v in group["center_xy"])
            pred_zero = tuple(float(v) for v in group["zero_center_xy"])
            pred_heading_display_deg = display_angle_from_center_to_point(pred_center, pred_zero)
            blur_length_px = float(robot["blur_length_px"])
            blur_present = 1.0 if blur_length_px > 1.0e-6 else 0.0
            relative_angle_deg = 0.0
            if blur_present > 0.5:
                relative_angle_deg = float((float(robot["blur_angle_deg"]) - pred_heading_display_deg) % 360.0)
            samples.append(
                {
                    "image_path": str(image_path),
                    "slot_centers_xy": {str(k): [float(v[0]), float(v[1])] for k, v in slot_centers.items()},
                    "slot_class_ids": {str(k): int(v) for k, v in slot_classes.items()},
                    "blur_present": float(blur_present),
                    "blur_length_px": float(blur_length_px),
                    "relative_angle_deg": float(relative_angle_deg),
                }
            )
        if idx % 100 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[precompute] {idx}/{len(eval_items)} scenes "
                f"matched_groups={matched_groups}/{total_robots} elapsed={elapsed/60.0:.1f}m"
            )
    stats = {
        "num_scenes": float(len(eval_items)),
        "num_robot_samples": float(len(samples)),
        "num_gt_robots": float(total_robots),
        "match_rate": float(matched_groups / max(total_robots, 1)),
        "detector_runtime_ms": float(np.mean(scene_times_ms)) if scene_times_ms else float("nan"),
    }
    return samples, stats


class PredictedGroupBlurDatasetV2(Dataset):
    def __init__(
        self,
        samples: list[dict],
        robot_crop_size: int = 128,
        slot_patch_size: int = 40,
    ) -> None:
        self.samples = list(samples)
        self.robot_crop_size = int(robot_crop_size)
        self.slot_patch_size = int(slot_patch_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image_rgb = _load_rgb(Path(sample["image_path"]))
        slot_centers_xy = {
            int(slot_index): (float(center_xy[0]), float(center_xy[1]))
            for slot_index, center_xy in sample["slot_centers_xy"].items()
        }
        slot_class_ids = {
            int(slot_index): int(class_id)
            for slot_index, class_id in sample["slot_class_ids"].items()
        }
        features = build_group_blur_features(
            image_rgb=image_rgb,
            slot_centers_xy=slot_centers_xy,
            slot_class_ids=slot_class_ids,
            robot_crop_size=self.robot_crop_size,
            slot_patch_size=self.slot_patch_size,
        )
        blur_length_px = float(sample["blur_length_px"])
        blur_present = float(sample["blur_present"])
        relative_angle_deg = float(sample["relative_angle_deg"])
        axis_angle_deg = float(relative_angle_deg % 180.0)
        axis_x, axis_y = angle_to_vec_display(2.0 * axis_angle_deg)
        sign_target = 1.0 if relative_angle_deg >= 180.0 else 0.0
        return {
            "robot_crop": torch.from_numpy(features["robot_crop"]).float(),
            "slot_patches": torch.from_numpy(features["slot_patches"]).float(),
            "slot_classes": torch.from_numpy(features["slot_classes"]).float(),
            "blur_present": torch.tensor(float(blur_present), dtype=torch.float32),
            "blur_length": torch.tensor(float(blur_length_px) / 30.0, dtype=torch.float32),
            "blur_axis_vec": torch.tensor([axis_x, axis_y], dtype=torch.float32),
            "blur_sign": torch.tensor(float(sign_target), dtype=torch.float32),
            "relative_angle_deg": torch.tensor(float(relative_angle_deg), dtype=torch.float32),
        }


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "robot_crop": torch.stack([item["robot_crop"] for item in batch], dim=0),
        "slot_patches": torch.stack([item["slot_patches"] for item in batch], dim=0),
        "slot_classes": torch.stack([item["slot_classes"] for item in batch], dim=0),
        "blur_present": torch.stack([item["blur_present"] for item in batch], dim=0),
        "blur_length": torch.stack([item["blur_length"] for item in batch], dim=0),
        "blur_axis_vec": torch.stack([item["blur_axis_vec"] for item in batch], dim=0),
        "blur_sign": torch.stack([item["blur_sign"] for item in batch], dim=0),
        "relative_angle_deg": torch.stack([item["relative_angle_deg"] for item in batch], dim=0),
    }


def _axis_error_deg_from_vec(pred_vec: torch.Tensor, tgt_vec: torch.Tensor) -> torch.Tensor:
    pred_angle = 0.5 * torch.atan2(pred_vec[:, 1], pred_vec[:, 0])
    tgt_angle = 0.5 * torch.atan2(tgt_vec[:, 1], tgt_vec[:, 0])
    delta = torch.remainder(pred_angle - tgt_angle + 0.5 * math.pi, math.pi) - 0.5 * math.pi
    return torch.abs(torch.rad2deg(delta))


def _compose_pred_relative_angles(
    pred_axis_vec: torch.Tensor,
    pred_sign_logit: torch.Tensor,
) -> torch.Tensor:
    axis_deg = torch.rad2deg(0.5 * torch.atan2(pred_axis_vec[:, 1], pred_axis_vec[:, 0]))
    axis_deg = torch.remainder(axis_deg, 180.0)
    sign_positive = (torch.sigmoid(pred_sign_logit) >= 0.5).float()
    return torch.remainder(axis_deg + 180.0 * sign_positive, 360.0)


def _batch_loss(
    output,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    sign_length_thresh_norm: float = 0.12,
) -> tuple[torch.Tensor, dict[str, float]]:
    presence_targets = batch["blur_present"]
    blur_present_mask = presence_targets > 0.5
    sign_mask = blur_present_mask & (batch["blur_length"] >= float(sign_length_thresh_norm))

    presence_loss = F.binary_cross_entropy_with_logits(output.blur_presence_logit, presence_targets, reduction="mean")
    length_loss = F.smooth_l1_loss(output.blur_length, batch["blur_length"], reduction="mean")
    if torch.any(blur_present_mask):
        axis_loss = torch.mean(
            1.0 - torch.sum(output.blur_axis_vec[blur_present_mask] * batch["blur_axis_vec"][blur_present_mask], dim=-1)
        )
    else:
        axis_loss = torch.zeros((), device=device)
    if torch.any(sign_mask):
        sign_loss = F.binary_cross_entropy_with_logits(output.blur_sign_logit[sign_mask], batch["blur_sign"][sign_mask], reduction="mean")
    else:
        sign_loss = torch.zeros((), device=device)

    total_loss = 0.4 * presence_loss + 0.9 * length_loss + 1.0 * axis_loss + 0.8 * sign_loss
    metrics = {
        "total_loss": float(total_loss.detach().cpu()),
        "presence_loss": float(presence_loss.detach().cpu()),
        "length_loss": float(length_loss.detach().cpu()),
        "axis_loss": float(axis_loss.detach().cpu()),
        "sign_loss": float(sign_loss.detach().cpu()),
        "sign_mask_count": int(sign_mask.detach().cpu().sum().item()),
    }
    return total_loss, metrics


@torch.no_grad()
def _evaluate(model: P4GroupedBlurEstimatorV2, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    dir_errs: list[float] = []
    axis_errs: list[float] = []
    len_errs: list[float] = []
    sign_accs: list[float] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(batch["slot_patches"], batch["slot_classes"], batch["robot_crop"])
        loss, _ = _batch_loss(output, batch, device)
        losses.append(float(loss.detach().cpu()))
        blur_mask = batch["blur_present"] > 0.5
        if torch.any(blur_mask):
            axis_errs.extend(_axis_error_deg_from_vec(output.blur_axis_vec[blur_mask], batch["blur_axis_vec"][blur_mask]).detach().cpu().tolist())
            pred_relative = _compose_pred_relative_angles(output.blur_axis_vec[blur_mask], output.blur_sign_logit[blur_mask])
            tgt_relative = batch["relative_angle_deg"][blur_mask]
            dir_errs.extend(
                [_angle_error_deg(float(pred_val), float(tgt_val)) for pred_val, tgt_val in zip(pred_relative.detach().cpu().tolist(), tgt_relative.detach().cpu().tolist())]
            )
            len_errs.extend((torch.abs(output.blur_length[blur_mask] - batch["blur_length"][blur_mask]) * 30.0).detach().cpu().tolist())
            informative_mask = blur_mask & (batch["blur_length"] >= 0.12)
            if torch.any(informative_mask):
                sign_pred = (torch.sigmoid(output.blur_sign_logit[informative_mask]) >= 0.5).float()
                sign_accs.append(float((sign_pred == batch["blur_sign"][informative_mask]).float().mean().detach().cpu()))
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_dir_err_deg": float(np.mean(dir_errs)) if dir_errs else float("nan"),
        "val_axis_err_deg": float(np.mean(axis_errs)) if axis_errs else float("nan"),
        "val_len_err_px": float(np.mean(len_errs)) if len_errs else float("nan"),
        "val_sign_acc": float(np.mean(sign_accs)) if sign_accs else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train grouped blur estimator v2 from Stage-1 detector predictions.")
    parser.add_argument("--data-root", default="outputs/p4_ai_dataset_v1_sharp")
    parser.add_argument("--project", default="outputs/p4_ai_group_blur_train_v2_stage1")
    parser.add_argument("--name", default="p4_grouped_blur_v2_stage1_rtdetr_l_p5")
    parser.add_argument("--train-split-file", default="outputs/p4_ai_dataset_v1_sharp/train_70.txt")
    parser.add_argument("--val-split-file", default="outputs/p4_ai_dataset_v1_sharp/val_30.txt")
    parser.add_argument("--detector-weights", default="outputs/stage1_detector_ablation/train/rtdetr_l_spiral_p5_n/weights/best.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.15)
    args = parser.parse_args()

    data_root = _resolve_path(args.data_root)
    train_split = _resolve_path(args.train_split_file)
    val_split = _resolve_path(args.val_split_file)
    detector_weights = _resolve_path(args.detector_weights)
    project_dir = _resolve_path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)
    log_path = project_dir / f"{args.name}_train.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        tee = _Tee(sys.__stdout__, log_handle)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = tee
        sys.stderr = tee
        try:
            device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
            print(f"log_path: {log_path}")
            print(f"device: {device}")
            print(f"detector_weights: {detector_weights}")

            train_samples, train_stats = _precompute_samples(data_root, train_split, detector_weights, args.imgsz, args.device, args.det_conf)
            val_samples, val_stats = _precompute_samples(data_root, val_split, detector_weights, args.imgsz, args.device, args.det_conf)
            print(json.dumps({"train_precompute": train_stats, "val_precompute": val_stats}, indent=2))

            (project_dir / f"{args.name}_train_samples_stats.json").write_text(json.dumps(train_stats, indent=2), encoding="utf-8")
            (project_dir / f"{args.name}_val_samples_stats.json").write_text(json.dumps(val_stats, indent=2), encoding="utf-8")

            train_ds = PredictedGroupBlurDatasetV2(train_samples)
            val_ds = PredictedGroupBlurDatasetV2(val_samples)
            print(f"train_items: {len(train_ds)}")
            print(f"val_items: {len(val_ds)}")

            train_loader = DataLoader(
                train_ds,
                batch_size=int(args.batch),
                shuffle=True,
                num_workers=int(args.num_workers),
                pin_memory=device.type == "cuda",
                collate_fn=_collate,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=int(args.batch),
                shuffle=False,
                num_workers=max(0, int(args.num_workers) // 2),
                pin_memory=device.type == "cuda",
                collate_fn=_collate,
            )

            model = P4GroupedBlurEstimatorV2().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1.0e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

            best_val = float("inf")
            history = []
            for epoch in range(1, int(args.epochs) + 1):
                model.train()
                epoch_start = time.perf_counter()
                train_losses: list[float] = []
                for step, batch in enumerate(train_loader, start=1):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type if device.type != "cpu" else "cpu", enabled=device.type == "cuda"):
                        output = model(batch["slot_patches"], batch["slot_classes"], batch["robot_crop"])
                        loss, metrics = _batch_loss(output, batch, device)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_losses.append(float(loss.detach().cpu()))
                    if step % 100 == 0:
                        print(
                            f"epoch={epoch} step={step}/{len(train_loader)} "
                            f"loss={metrics['total_loss']:.4f} axis={metrics['axis_loss']:.4f} "
                            f"sign={metrics['sign_loss']:.4f} signN={metrics['sign_mask_count']}"
                        )

                val_metrics = _evaluate(model, val_loader, device)
                epoch_time = time.perf_counter() - epoch_start
                epoch_row = {
                    "epoch": int(epoch),
                    "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                    "val_loss": float(val_metrics["val_loss"]),
                    "val_dir_err_deg": float(val_metrics["val_dir_err_deg"]),
                    "val_axis_err_deg": float(val_metrics["val_axis_err_deg"]),
                    "val_len_err_px": float(val_metrics["val_len_err_px"]),
                    "val_sign_acc": float(val_metrics["val_sign_acc"]),
                    "epoch_sec": float(epoch_time),
                }
                history.append(epoch_row)
                print(json.dumps(epoch_row))
                torch.save({"model": model.state_dict(), "epoch": epoch, "history": history}, project_dir / f"{args.name}_last.pt")
                if epoch_row["val_loss"] < best_val:
                    best_val = epoch_row["val_loss"]
                    torch.save({"model": model.state_dict(), "epoch": epoch, "history": history}, project_dir / f"{args.name}_best.pt")

            (project_dir / f"{args.name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


if __name__ == "__main__":
    main()
