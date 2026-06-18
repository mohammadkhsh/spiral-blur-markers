from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spiral_markers.corruptions import apply_local_motion_blur
from spiral_markers.io_utils import ensure_dir, load_method_config, seeded_rng
from spiral_markers.rendering import ScenePlacement, render_scene_layers_from_placements, textured_green_background


def _log(log_path: Path, message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _write_image_jpg(path: Path, image_rgb) -> None:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def _load_manifest(root: Path) -> list[dict[str, str]]:
    with (root / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _make_placements(payload: dict[str, object]) -> list[ScenePlacement]:
    placements: list[ScenePlacement] = []
    for row in payload["placements"]:
        placements.append(
            ScenePlacement(
                robot_id=int(row["robot_id"]),
                center_xy=(float(row["center_xy"][0]), float(row["center_xy"][1])),
                scale=float(row["scale"]),
                rotation_deg=float(row["rotation_deg"]),
                tilt_x_deg=float(row["tilt_x_deg"]),
                tilt_y_deg=float(row["tilt_y_deg"]),
            )
        )
    return placements


def _robot_blur_params(payload: dict[str, object]) -> list[dict[str, float]]:
    robots = sorted(payload["robots"], key=lambda row: int(row["robot_index"]))
    return [
        {
            "blur_length_px": float(row["blur_length_px"]),
            "blur_angle_deg": float(row["blur_angle_deg"]),
            "kernel_blur_angle_deg": float(row.get("kernel_blur_angle_deg", row["blur_angle_deg"])),
            "motion_beta": float(row["motion_beta"]),
        }
        for row in robots
    ]


def _render_scene_from_payload(cfg, payload: dict[str, object]) -> object:
    domain = str(payload["domain"])
    local_index = int(payload["local_index"])
    target_h, target_w = [int(v) for v in payload["image_size"]]

    seed_offset = 0 if domain == "clean" else 1_000_000
    rng = seeded_rng(int(cfg.seed) + seed_offset + local_index)
    background_rgb = textured_green_background(tuple(int(v) for v in cfg.scene.image_size), rng)
    placements = _make_placements(payload)
    layered = render_scene_layers_from_placements(cfg, placements=placements, background_rgb=background_rgb.copy())
    blur_params = _robot_blur_params(payload)

    if domain == "blur":
        gray, color = apply_local_motion_blur(
            background_gray=layered.background_gray,
            background_rgb=layered.background_rgb,
            robot_layers=layered.layers,
            blur_params=blur_params,
        )
        rgb = color if color is not None else gray[..., None].repeat(3, axis=2)
    else:
        rgb = (
            layered.clean_scene.color_image
            if layered.clean_scene.color_image is not None
            else layered.clean_scene.image[..., None].repeat(3, axis=2)
        )
    resized = cv2.resize(
        (rgb.clip(0.0, 1.0) * 255.0).astype("uint8"),
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def _rewrite_split_file(src_file: Path, dst_file: Path, old_root: Path, new_root: Path) -> None:
    lines: list[str] = []
    for line in src_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        lines.append(str(Path(line.replace(str(old_root), str(new_root))).resolve()))
    dst_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the paper_v4 AI dataset with sharp spiral legs.")
    parser.add_argument("--source-root", default="outputs/p4_ai_dataset_v1")
    parser.add_argument("--config", default="configs/eval_paper_v4_sharp_fast.yaml")
    parser.add_argument("--out", default="outputs/p4_ai_dataset_v1_sharp")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    source_root = (ROOT / args.source_root).resolve()
    out_root = ensure_dir(ROOT / args.out)
    cfg = load_method_config(ROOT / args.config)

    images_root = ensure_dir(out_root / "images")
    labels_root = out_root / "labels"
    ann_root = out_root / "annotations"
    logs_root = ensure_dir(out_root / "logs")
    log_path = logs_root / "generation.log"
    log_path.write_text("", encoding="utf-8")

    _log(log_path, f"source_root={source_root}")
    _log(log_path, f"out_root={out_root}")
    _log(log_path, f"config={(ROOT / args.config).resolve()}")

    if labels_root.exists():
        shutil.rmtree(labels_root)
    if ann_root.exists():
        shutil.rmtree(ann_root)
    _copy_tree(source_root / "labels", labels_root)
    _copy_tree(source_root / "annotations", ann_root)

    for split in ("train", "val", "test"):
        ensure_dir(images_root / split)

    manifest_rows = _load_manifest(source_root)
    start_time = time.perf_counter()
    new_manifest: list[dict[str, object]] = []
    split_lists: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for idx, row in enumerate(manifest_rows, start=1):
        json_path = Path(row["json_path"]).resolve()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        split = str(payload["split"])
        image_name = str(payload["scene_name"])
        image_rgb = _render_scene_from_payload(cfg, payload)
        image_path = images_root / split / image_name
        _write_image_jpg(image_path, image_rgb)

        new_row = {
            "scene_name": image_name,
            "split": row["split"],
            "domain": row["domain"],
            "image_path": str(image_path.resolve()),
            "label_path": str((labels_root / split / f"{Path(image_name).stem}.txt").resolve()),
            "json_path": str((ann_root / "scenes" / split / f"{Path(image_name).stem}.json").resolve()),
            "robot_count": row["robot_count"],
            "spiral_count": row["spiral_count"],
        }
        new_manifest.append(new_row)
        split_lists[split].append(str(image_path.resolve()))

        if idx % int(args.log_every) == 0:
            elapsed = time.perf_counter() - start_time
            _log(log_path, f"rendered {idx}/{len(manifest_rows)} images elapsed={elapsed/60.0:.1f}m")

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["scene_name", "split", "domain", "image_path", "label_path", "json_path", "robot_count", "spiral_count"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_manifest)

    for split_name in ("train", "val", "test"):
        (out_root / f"{split_name}.txt").write_text("\n".join(split_lists[split_name]) + "\n", encoding="utf-8")

    _rewrite_split_file(source_root / "train_70.txt", out_root / "train_70.txt", source_root, out_root)
    _rewrite_split_file(source_root / "val_30.txt", out_root / "val_30.txt", source_root, out_root)

    data_yaml = out_root / "data_spiral.yaml"
    data_yaml.write_text(
        f"path: {out_root}\n"
        f"train: {(out_root / 'train.txt').resolve()}\n"
        f"val: {(out_root / 'val.txt').resolve()}\n"
        "names:\n"
        "  0: neg\n"
        "  1: zero\n"
        "  2: pos\n",
        encoding="utf-8",
    )
    data_7030_yaml = out_root / "data_spiral_70_30.yaml"
    data_7030_yaml.write_text(
        f"path: {out_root}\n"
        f"train: {(out_root / 'train_70.txt').resolve()}\n"
        f"val: {(out_root / 'val_30.txt').resolve()}\n"
        "names:\n"
        "  0: neg\n"
        "  1: zero\n"
        "  2: pos\n",
        encoding="utf-8",
    )

    source_stats = json.loads((source_root / "stats.json").read_text(encoding="utf-8"))
    elapsed = time.perf_counter() - start_time
    stats = {
        **source_stats,
        "config": str((ROOT / args.config).resolve()),
        "output_root": str(out_root),
        "elapsed_sec": float(elapsed),
        "log_path": str(log_path),
        "manifest_csv": str(manifest_path),
        "data_yaml": str(data_yaml),
        "data_70_30_yaml": str(data_7030_yaml),
        "source_root": str(source_root),
    }
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _log(log_path, f"done elapsed={elapsed/60.0:.1f}m images={len(new_manifest)}")


if __name__ == "__main__":
    main()
