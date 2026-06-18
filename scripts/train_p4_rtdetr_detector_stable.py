from __future__ import annotations

import argparse
import random
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))

from spiral_markers.ml.runtime_env import patch_windows_platform_for_torch

patch_windows_platform_for_torch()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _domain_key(image_path: str) -> str:
    stem = Path(image_path).stem.lower()
    if stem.startswith("clean_"):
        return "clean"
    if stem.startswith("blur_"):
        return "blur"
    return "other"


def _ensure_split(dataset_root: Path, train_ratio: float, seed: int) -> Path:
    all_paths: list[str] = []
    seen: set[str] = set()
    for name in ("train.txt", "val.txt", "test.txt", "train_70.txt", "val_30.txt"):
        for item in _read_lines(dataset_root / name):
            norm = str(Path(item).resolve())
            if norm not in seen:
                seen.add(norm)
                all_paths.append(norm)

    if not all_paths:
        raise RuntimeError(f"no image paths found under {dataset_root}")

    groups: dict[str, list[str]] = {"clean": [], "blur": [], "other": []}
    for item in all_paths:
        groups[_domain_key(item)].append(item)

    rng = random.Random(seed)
    train_paths: list[str] = []
    val_paths: list[str] = []
    for items in groups.values():
        if not items:
            continue
        items = list(items)
        rng.shuffle(items)
        split_idx = max(1, min(len(items) - 1, int(round(len(items) * train_ratio))))
        train_paths.extend(items[:split_idx])
        val_paths.extend(items[split_idx:])

    rng.shuffle(train_paths)
    rng.shuffle(val_paths)

    train_txt = dataset_root / "train_70.txt"
    val_txt = dataset_root / "val_30.txt"
    data_yaml = dataset_root / "data_spiral_70_30.yaml"
    train_txt.write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val_paths) + "\n", encoding="utf-8")
    data_yaml.write_text(
        f"path: {dataset_root}\n"
        f"train: {train_txt.resolve()}\n"
        f"val: {val_txt.resolve()}\n"
        "names:\n"
        "  0: neg\n"
        "  1: zero\n"
        "  2: pos\n",
        encoding="utf-8",
    )
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable RT-DETR training launcher for the p4 spiral dataset.")
    parser.add_argument("--data", default="")
    parser.add_argument("--dataset-root", default="spiral_markers/outputs/p4_ai_dataset_v1")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--model", default="spiral_markers/configs/yolov8n_rtdetr_p4_spiral.yaml")
    parser.add_argument("--project", default="spiral_markers/outputs/p4_ai_rtdetr_train")
    parser.add_argument("--name", default="yolov8n_rtdetr_p4_70_30")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--resume-weights", default="")
    args = parser.parse_args()

    project_dir = (REPO_ROOT / args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    log_path = project_dir / f"{args.name}_train.log"

    with log_path.open("w", encoding="utf-8") as log_handle:
        tee = Tee(sys.stdout, log_handle)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = tee
        sys.stderr = tee
        try:
            print(f"log_path: {log_path}")
            dataset_root = (REPO_ROOT / args.dataset_root).resolve()
            data_yaml = (REPO_ROOT / args.data).resolve() if args.data else _ensure_split(dataset_root, float(args.train_ratio), int(args.seed))
            print(f"data_yaml: {data_yaml}")
            print("importing ultralytics RTDETR...")
            from ultralytics import RTDETR
            print("ultralytics import complete")
            model_spec = args.resume_weights if args.resume_weights else str((REPO_ROOT / args.model).resolve())
            model = RTDETR(str((REPO_ROOT / model_spec).resolve()) if not Path(model_spec).is_absolute() else model_spec)
            model.train(
                data=str(data_yaml),
                epochs=int(args.epochs),
                imgsz=int(args.imgsz),
                batch=int(args.batch),
                device=str(args.device),
                workers=int(args.workers),
                project=str(project_dir),
                name=str(args.name),
                exist_ok=True,
                pretrained=bool(args.resume_weights),
                amp=True,
                mosaic=float(args.mosaic),
                close_mosaic=int(args.close_mosaic),
                fliplr=float(args.fliplr),
                seed=int(args.seed),
                cache=False,
                deterministic=True,
                patience=20,
                save_period=1,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    main()
