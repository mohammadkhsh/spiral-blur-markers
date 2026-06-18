from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spiral_markers.io_utils import ensure_dir, load_method_config, seeded_rng, write_image
from spiral_markers.rendering import render_robot_tag_patch, textured_green_background
from spiral_markers.tag_layout import encode_robot_id_bits


def _mm_to_px(mm: float, dpi: int) -> int:
    return int(round(float(mm) / 25.4 * float(dpi)))


def _hex_to_rgb01(hex_color: str) -> np.ndarray:
    text = hex_color.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color}")
    return np.array([int(text[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def _crop_to_alpha(patch: np.ndarray, alpha: np.ndarray, pad: int = 4) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(alpha > 1.0e-4)
    if xs.size == 0 or ys.size == 0:
        return patch, alpha
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(patch.shape[1], int(xs.max()) + pad + 1)
    y1 = min(patch.shape[0], int(ys.max()) + pad + 1)
    return patch[y0:y1, x0:x1], alpha[y0:y1, x0:x1]


def _resize_image(image: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    width, height = size
    return cv2.resize(image.astype(np.float32), (int(width), int(height)), interpolation=interpolation)


def _dataset_green_rgb01(seed: int) -> np.ndarray:
    sample = textured_green_background((1024, 1024), seeded_rng(int(seed)))
    return np.mean(sample.reshape(-1, 3), axis=0).astype(np.float32)


def _draw_robot_top(
    canvas: np.ndarray,
    center_xy: tuple[int, int],
    radius_px: int,
    patch: np.ndarray,
    alpha: np.ndarray,
    robot_id: int,
    circle_fill_rgb: np.ndarray,
    inner_green_rgb: np.ndarray,
) -> dict[str, object]:
    cx, cy = int(center_xy[0]), int(center_xy[1])
    cv2.circle(canvas, (cx, cy), int(radius_px), tuple(float(v) for v in circle_fill_rgb.tolist()), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), int(radius_px), (0.52, 0.67, 0.76), thickness=3, lineType=cv2.LINE_AA)
    green_field_radius = int(round(0.90 * radius_px))
    cv2.circle(canvas, (cx, cy), green_field_radius, tuple(float(v) for v in inner_green_rgb.tolist()), thickness=-1, lineType=cv2.LINE_AA)

    cropped_patch, cropped_alpha = _crop_to_alpha(patch, alpha)
    target_diameter = max(8, int(round(1.44 * radius_px)))
    resized_patch = _resize_image(cropped_patch, (target_diameter, target_diameter), interpolation=cv2.INTER_CUBIC)
    resized_alpha = _resize_image(cropped_alpha, (target_diameter, target_diameter), interpolation=cv2.INTER_CUBIC)
    resized_alpha = np.clip(resized_alpha, 0.0, 1.0)

    x0 = cx - target_diameter // 2
    y0 = cy - target_diameter // 2
    x1 = x0 + target_diameter
    y1 = y0 + target_diameter
    canvas_h, canvas_w = canvas.shape[:2]
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(canvas_w, x1)
    dst_y1 = min(canvas_h, y1)
    if dst_x0 < dst_x1 and dst_y0 < dst_y1:
        src_x0 = dst_x0 - x0
        src_y0 = dst_y0 - y0
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        patch_alpha = resized_alpha[src_y0:src_y1, src_x0:src_x1][..., None]
        patch_rgb = np.repeat(resized_patch[src_y0:src_y1, src_x0:src_x1][..., None], 3, axis=2)
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = (
            patch_rgb * patch_alpha
            + canvas[dst_y0:dst_y1, dst_x0:dst_x1] * (1.0 - patch_alpha)
        )

    return {
        "robot_id": int(robot_id),
        "center_xy": [int(cx), int(cy)],
        "radius_px": int(radius_px),
    }


def _save_pdf_from_rgb(rgb: np.ndarray, out_path: Path, dpi: int) -> None:
    height, width = rgb.shape[:2]
    fig_w = width / float(dpi)
    fig_h = height / float(dpi)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(np.clip(rgb, 0.0, 1.0))
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a simple A3 printable sheet with light-blue robot tops and four spiral markers.")
    parser.add_argument("--config", default="configs/eval_paper_v4_sharp_fast.yaml")
    parser.add_argument("--out-dir", default="outputs/paper_figures/real_robot_a3_sheet")
    parser.add_argument("--ids", nargs="*", type=int, default=list(range(8)))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--page-width-mm", type=float, default=420.0)
    parser.add_argument("--page-height-mm", type=float, default=297.0)
    parser.add_argument("--margin-mm", type=float, default=12.0)
    parser.add_argument("--gutter-mm", type=float, default=5.0)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--circle-fill", default="#b8def5")
    args = parser.parse_args()

    cfg = load_method_config(ROOT / args.config)
    out_dir = ensure_dir(ROOT / args.out_dir)
    dpi = int(args.dpi)
    page_w_px = _mm_to_px(args.page_width_mm, dpi)
    page_h_px = _mm_to_px(args.page_height_mm, dpi)
    margin_px = _mm_to_px(args.margin_mm, dpi)
    gutter_px = _mm_to_px(args.gutter_mm, dpi)

    ids = [int(v) for v in args.ids]
    max_unique_ids = 2 ** int(cfg.layout.identity_slots)
    encoded_map: dict[tuple[int, ...], list[int]] = {}
    for robot_id in ids:
        bits = tuple(encode_robot_id_bits(robot_id, int(cfg.layout.identity_slots)))
        encoded_map.setdefault(bits, []).append(robot_id)
    collisions = [group for group in encoded_map.values() if len(group) > 1]
    if collisions:
        raise ValueError(
            "Requested IDs collide under the current four-spiral encoding. "
            f"With identity_slots={cfg.layout.identity_slots}, the pipeline supports {max_unique_ids} unique IDs. "
            f"Collisions: {collisions}"
        )
    columns = max(1, int(args.columns))
    rows = int(math.ceil(len(ids) / float(columns)))
    usable_w = page_w_px - 2 * margin_px
    usable_h = page_h_px - 2 * margin_px
    cell_w = int((usable_w - gutter_px * (columns - 1)) / columns)
    cell_h = int((usable_h - gutter_px * (rows - 1)) / rows)
    circle_radius = int(0.47 * min(cell_w, cell_h))

    page = np.ones((page_h_px, page_w_px, 3), dtype=np.float32)
    circle_fill_rgb = _hex_to_rgb01(args.circle_fill)
    inner_green_rgb = _dataset_green_rgb01(int(cfg.seed))
    placement_rows: list[dict[str, object]] = []

    for index, robot_id in enumerate(ids):
        row = index // columns
        col = index % columns
        x0 = margin_px + col * (cell_w + gutter_px)
        y0 = margin_px + row * (cell_h + gutter_px)
        cx = x0 + cell_w // 2
        cy = y0 + cell_h // 2

        patch, alpha, layout, _ = render_robot_tag_patch(robot_id, cfg)
        placement = _draw_robot_top(
            canvas=page,
            center_xy=(cx, cy),
            radius_px=circle_radius,
            patch=patch,
            alpha=alpha,
            robot_id=robot_id,
            circle_fill_rgb=circle_fill_rgb,
            inner_green_rgb=inner_green_rgb,
        )
        placement_rows.append(
            {
                **placement,
                "grid_row": int(row),
                "grid_col": int(col),
                "cell_origin_xy": [int(x0), int(y0)],
                "spiral_centers_xy": [[float(x), float(y)] for x, y in [primitive.center_xy for primitive in layout.primitives]],
                "spiral_twists_deg": [float(primitive.twist_angle_deg) for primitive in layout.primitives],
            }
        )

    png_path = out_dir / "real_robot_a3_sheet.png"
    pdf_path = out_dir / "real_robot_a3_sheet.pdf"
    metadata_path = out_dir / "real_robot_a3_sheet_metadata.json"
    write_image(png_path, page)
    _save_pdf_from_rgb(page, pdf_path, dpi=dpi)

    metadata = {
        "config": str((ROOT / args.config).resolve()),
        "dpi": dpi,
        "page_size_mm": [float(args.page_width_mm), float(args.page_height_mm)],
        "page_size_px": [int(page_w_px), int(page_h_px)],
        "margin_mm": float(args.margin_mm),
        "gutter_mm": float(args.gutter_mm),
        "grid": {"rows": int(rows), "columns": int(columns)},
        "robot_ids": ids,
        "dataset_green_rgb01": [float(v) for v in inner_green_rgb.tolist()],
        "placements": placement_rows,
        "png_path": str(png_path.resolve()),
        "pdf_path": str(pdf_path.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"png_path": str(png_path.resolve()), "pdf_path": str(pdf_path.resolve()), "metadata_path": str(metadata_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
