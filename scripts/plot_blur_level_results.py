from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _save_png_as_single_page_pdf(png_path: Path, pdf_path: Path, resolution: int = 350) -> None:
    from PIL import Image

    image = Image.open(png_path).convert("RGB")
    image.save(pdf_path, "PDF", resolution=resolution)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | float | int | None) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def _fmt_pct_trunc_1(value: float) -> str:
    if math.isnan(float(value)):
        return "nan"
    truncated = math.floor(float(value) * 10.0) / 10.0
    return f"{truncated:.1f}"


def _load_scene_blur_lengths(data_root: Path) -> dict[str, dict[str, float]]:
    manifest = _read_csv(data_root / "manifest.csv")
    blur_by_scene: dict[str, dict[str, float]] = {}
    for row in manifest:
        if row.get("domain") != "blur":
            continue
        json_path = Path(row["json_path"])
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        lengths = [_float(robot.get("blur_length_px")) for robot in payload.get("robots", [])]
        lengths = [v for v in lengths if not math.isnan(v)]
        if not lengths:
            continue
        blur_by_scene[row["scene_name"]] = {
            "mean_blur_px": float(np.mean(lengths)),
            "min_blur_px": float(np.min(lengths)),
            "max_blur_px": float(np.max(lengths)),
        }
    return blur_by_scene


def _bin_label(index: int, edges: np.ndarray) -> str:
    names = ["Low", "Medium", "High", "Very high"]
    lo = edges[index]
    hi = edges[index + 1]
    return f"{names[index]}\n{lo:.1f}-{hi:.1f}px"


def _assign_bin(mean_blur_px: float, edges: np.ndarray) -> int:
    index = int(np.searchsorted(edges, mean_blur_px, side="right") - 1)
    return int(np.clip(index, 0, len(edges) - 2))


def _summarize(
    metrics_path: Path,
    data_root: Path,
    method_name: str,
    edges: np.ndarray,
) -> list[dict[str, float | str | int]]:
    rows = [row for row in _read_csv(metrics_path) if row.get("domain") == "blur"]
    blur_meta = _load_scene_blur_lengths(data_root)
    bins: list[list[dict[str, str | float]]] = [[] for _ in range(len(edges) - 1)]
    for row in rows:
        scene_name = row["scene_name"]
        meta = blur_meta.get(scene_name)
        if meta is None:
            continue
        record: dict[str, str | float] = dict(row)
        record.update(meta)
        bins[_assign_bin(float(meta["mean_blur_px"]), edges)].append(record)

    summary: list[dict[str, float | str | int]] = []
    for index, items in enumerate(bins):
        summary.append(
            {
                "method": method_name,
                "level": ["low", "medium", "high", "very_high"][index],
                "level_label": _bin_label(index, edges),
                "range_low_px": float(edges[index]),
                "range_high_px": float(edges[index + 1]),
                "num_scenes": int(len(items)),
                "mean_gt_blur_px": _mean([_float(item.get("mean_blur_px")) for item in items]),
                "spiral_recall_pct": 100.0 * _mean([_float(item.get("spiral_recall")) for item in items]),
                "spiral_precision_pct": 100.0 * _mean([_float(item.get("spiral_precision")) for item in items]),
                "spiral_class_accuracy_pct": 100.0 * _mean([_float(item.get("spiral_class_accuracy")) for item in items]),
                "robot_recall_pct": 100.0 * _mean([_float(item.get("robot_recall")) for item in items]),
                "robot_precision_pct": 100.0 * _mean([_float(item.get("robot_precision")) for item in items]),
                "id_accuracy_pct": 100.0 * _mean([_float(item.get("id_accuracy")) for item in items]),
                "center_error_px": _mean([_float(item.get("center_error_px")) for item in items]),
                "heading_error_deg": _mean([_float(item.get("heading_error_deg")) for item in items]),
                "blur_angle_error_deg": _mean([_float(item.get("blur_angle_error_deg")) for item in items]),
                "blur_length_error_px": _mean([_float(item.get("blur_length_error_px")) for item in items]),
            }
        )
    return summary


def _write_summary_csv(rows: list[dict[str, float | str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_full_pipeline(rows: list[dict[str, float | str | int]], out_root: Path) -> None:
    title_fs = 15
    label_fs = 12
    tick_fs = 10
    value_fs = 8
    methods = ["Proposed sharp", "Faded variant"]
    px_metric_specs = [
        ("center_error_px", "Center err. [px]", "#5470c6"),
        ("blur_length_error_px", "Motion mag. err. [px]", "#fac858"),
    ]
    heading_spec = ("heading_error_deg", "Heading err. [deg]", "#91cc75")
    labels = [str(row["level_label"]) for row in rows if row["method"] == methods[0]]
    x = np.arange(len(labels), dtype=np.float32) * 0.86
    bar_width = 0.16

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), sharey=True)
    for panel_index, (ax, method) in enumerate(zip(axes, methods)):
        method_rows = [row for row in rows if row["method"] == method]
        for offset_index, (key, label, color) in enumerate(px_metric_specs):
            vals = np.asarray([float(row[key]) for row in method_rows], dtype=np.float32)
            positions = x + (-0.20 if offset_index == 0 else 0.20)
            bars = ax.bar(positions, vals, width=bar_width, color=color, alpha=0.88, label=label)
            for rect, value in zip(bars, vals):
                clipped = float(value) >= 3.6
                label_y = 3.68 if clipped else float(value) + 0.04
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    label_y,
                    f"{float(value):.2f}",
                    ha="center",
                    va="top" if clipped else "bottom",
                    fontsize=value_fs,
                    rotation=90,
                    clip_on=True,
                )

        ax_head = ax.twinx()
        heading_key, heading_label, heading_color = heading_spec
        heading_vals = np.asarray([float(row[heading_key]) for row in method_rows], dtype=np.float32)
        heading_bars = ax_head.bar(
            x,
            heading_vals,
            width=bar_width,
            color=heading_color,
            alpha=0.80,
            label=heading_label,
            zorder=3,
        )
        for rect, value in zip(heading_bars, heading_vals):
            ax_head.text(
                rect.get_x() + rect.get_width() / 2.0,
                min(float(value) + 0.005, 0.245),
                f"{float(value):.2f}",
                ha="center",
                va="bottom",
                fontsize=value_fs,
                rotation=90,
                color=heading_color,
                clip_on=True,
            )
        ax_head.set_ylim(0, 0.25)
        if panel_index == 1:
            ax_head.set_ylabel("Heading error [deg]", fontsize=label_fs, color=heading_color)
            ax_head.tick_params(axis="y", labelsize=tick_fs, colors=heading_color)
            ax_head.spines["right"].set_color(heading_color)
        else:
            ax_head.set_ylabel("")
            ax_head.tick_params(axis="y", right=False, labelright=False)
            ax_head.spines["right"].set_visible(False)

        ax.set_title(method, fontsize=title_fs, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=tick_fs)
        ax.set_xlabel("GT motion blur level from mean robot blur length", fontsize=label_fs)
        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 3.85)
        ax.set_yticks(np.arange(0.0, 3.61, 0.6))

        ax_acc = ax.twinx()
        ax_acc.spines["right"].set_position(("outward", 54))
        id_vals = np.asarray([float(row["id_accuracy_pct"]) for row in method_rows], dtype=np.float32)
        ax_acc.plot(
            x,
            id_vals,
            color="#111111",
            marker="o",
            linewidth=3.0,
            markersize=7,
            label="Robot ID acc. [%]",
        )
        ax_acc.set_ylim(0, 105)
        if panel_index == 1:
            ax_acc.set_ylabel("ID accuracy [%]", fontsize=label_fs, color="#111111", labelpad=12)
            ax_acc.tick_params(axis="y", labelsize=tick_fs, colors="#111111")
            ax_acc.spines["right"].set_color("#111111")
        else:
            ax_acc.set_ylabel("")
            ax_acc.tick_params(axis="y", right=False, labelright=False)
            ax_acc.spines["right"].set_visible(False)
        for xx, yy in zip(x, id_vals):
            ax_acc.annotate(
                f"{yy:.1f}",
                xy=(float(xx), float(yy)),
                xytext=(0, -9),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=value_fs,
                color="#111111",
                annotation_clip=False,
            )

    axes[0].set_ylabel("Pixel error [px]", fontsize=label_fs)
    center_proxy = plt.Rectangle((0, 0), 1, 1, color="#5470c6", alpha=0.88)
    length_proxy = plt.Rectangle((0, 0), 1, 1, color="#fac858", alpha=0.88)
    heading_proxy = plt.Rectangle((0, 0), 1, 1, color="#91cc75", alpha=0.80)
    id_proxy = plt.Line2D([0], [0], color="#111111", marker="o", linewidth=3.4, markersize=8)
    fig.legend(
        [center_proxy, length_proxy, heading_proxy, id_proxy],
        ["Center err. [px]", "Motion mag. err. [px]", "Heading err. [deg]", "Robot ID acc. [%]"],
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
        fontsize=12,
        handlelength=2.2,
        handleheight=1.0,
        columnspacing=1.3,
    )
    fig.suptitle("Effect of Motion Blur Level on Full Pipeline Performance", y=1.035, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    png_path = out_root / "blur_level_full_pipeline_sharp_vs_faded.png"
    pdf_path = out_root / "blur_level_full_pipeline_sharp_vs_faded.pdf"
    fig.savefig(png_path, dpi=350, bbox_inches="tight")
    plt.close(fig)
    _save_png_as_single_page_pdf(png_path, pdf_path)


def _plot_motion_direction(rows: list[dict[str, float | str | int]], out_root: Path) -> None:
    title_fs = 15
    label_fs = 12
    tick_fs = 10
    value_fs = 9
    methods = ["Proposed sharp", "Faded variant"]
    colors = ["#5470c6", "#ee6666"]
    labels = [str(row["level_label"]) for row in rows if row["method"] == methods[0]]
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for method_index, (method, color) in enumerate(zip(methods, colors)):
        method_rows = [row for row in rows if row["method"] == method]
        vals = np.asarray([float(row["blur_angle_error_deg"]) for row in method_rows], dtype=np.float32)
        positions = x + (method_index - 0.5) * width
        bars = ax.bar(positions, vals, width=width, color=color, alpha=0.9, label=method)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in vals], fontsize=value_fs, padding=2)

    ax.set_title("Motion Direction Error by Blur Level", fontsize=title_fs, fontweight="bold")
    ax.set_ylabel("Motion direction error [deg]", fontsize=label_fs)
    ax.set_xlabel("GT motion blur level from mean robot blur length", fontsize=label_fs)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=12, handlelength=2.2)
    fig.tight_layout()
    png_path = out_root / "blur_level_motion_direction_error_sharp_vs_faded.png"
    pdf_path = out_root / "blur_level_motion_direction_error_sharp_vs_faded.pdf"
    fig.savefig(png_path, dpi=350, bbox_inches="tight")
    plt.close(fig)
    _save_png_as_single_page_pdf(png_path, pdf_path)


def _plot_stage1(rows: list[dict[str, float | str | int]], out_root: Path) -> None:
    title_fs = 15
    label_fs = 12
    tick_fs = 10
    value_fs = 9
    methods = ["Proposed sharp", "Faded variant"]
    y_min = 97.0
    y_max = 100.0
    metric_specs = [
        ("spiral_recall_pct", "Spiral recall", "#5470c6", "o", -7),
        ("spiral_precision_pct", "Spiral precision", "#91cc75", "s", -7),
        ("spiral_class_accuracy_pct", "Spiral class acc.", "#ee6666", "^", -7),
    ]
    labels = [str(row["level_label"]) for row in rows if row["method"] == methods[0]]
    x = np.arange(len(labels), dtype=np.float32)

    method_rows_map = {method: [row for row in rows if row["method"] == method] for method in methods}
    faded_lower_vals = [
        float(row[key])
        for row in method_rows_map["Faded variant"]
        for key, *_ in metric_specs
        if float(row[key]) < y_min
    ]
    if faded_lower_vals:
        lower_y_min = max(0.0, math.floor(min(faded_lower_vals)) - 1.0)
        lower_y_max = min(y_min - 1.0, math.ceil(max(faded_lower_vals)) + 1.0)
        if lower_y_max - lower_y_min < 4.0:
            lower_y_max = min(y_min - 1.0, lower_y_min + 4.0)
    else:
        lower_y_min = y_min - 7.0
        lower_y_max = y_min - 1.0

    fig = plt.figure(figsize=(15.5, 5.1))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[3.3, 1.35],
        wspace=0.18,
        hspace=0.06,
    )
    ax_left = fig.add_subplot(gs[:, 0])
    ax_right_top = fig.add_subplot(gs[0, 1])
    ax_right_bottom = fig.add_subplot(gs[1, 1], sharex=ax_right_top)

    def _style_axis(ax: plt.Axes) -> None:
        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    def _plot_metric(ax: plt.Axes, vals: np.ndarray, color: str, marker: str, label: str | None) -> None:
        ax.plot(
            x,
            vals,
            marker=marker,
            linestyle="-",
            linewidth=3.0,
            markersize=7,
            color=color,
            label=label,
        )

    def _annotate_metric(
        ax: plt.Axes,
        x_vals: np.ndarray,
        vals: np.ndarray,
        color: str,
        default_offset: int,
        positive_offset: int | None = None,
    ) -> None:
        effective_offset = positive_offset if positive_offset is not None else default_offset
        for xx, yy in zip(x_vals, vals):
            ax.annotate(
                _fmt_pct_trunc_1(float(yy)),
                xy=(float(xx), float(yy)),
                xytext=(0, effective_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if effective_offset >= 0 else "top",
                fontsize=value_fs,
                color=color,
                annotation_clip=False,
            )

    def _draw_break_marks(ax_top: plt.Axes, ax_bottom: plt.Axes) -> None:
        d = 0.012
        kwargs = dict(color="k", clip_on=False, linewidth=1.1)
        ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)

    sharp_rows = method_rows_map["Proposed sharp"]
    for key, label, color, marker, label_offset in metric_specs:
        vals = np.asarray([float(row[key]) for row in sharp_rows], dtype=np.float32)
        _plot_metric(ax_left, vals, color, marker, label)
        _annotate_metric(ax_left, x, vals, color, label_offset)

    ax_left.set_title("Proposed sharp", fontsize=title_fs, fontweight="bold")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, fontsize=tick_fs)
    ax_left.set_xlabel("GT motion blur level from mean robot blur length", fontsize=label_fs)
    ax_left.set_ylabel("Stage-1 metric [%]", fontsize=label_fs)
    ax_left.set_ylim(y_min, y_max)
    ax_left.set_yticks([97.0, 98.0, 99.0, 100.0])
    _style_axis(ax_left)

    faded_rows = method_rows_map["Faded variant"]
    for key, label, color, marker, label_offset in metric_specs:
        vals = np.asarray([float(row[key]) for row in faded_rows], dtype=np.float32)
        _plot_metric(ax_right_top, vals, color, marker, label)
        _plot_metric(ax_right_bottom, vals, color, marker, None)
        top_mask = vals >= y_min
        if np.any(top_mask):
            _annotate_metric(ax_right_top, x[top_mask], vals[top_mask], color, label_offset)
        low_mask = vals < y_min
        if np.any(low_mask):
            _annotate_metric(
                ax_right_bottom,
                x[low_mask],
                vals[low_mask],
                color,
                5 if key == "spiral_class_accuracy_pct" else 4,
            )

    ax_right_top.set_title("Faded variant", fontsize=title_fs, fontweight="bold")
    ax_right_top.set_ylim(y_min, y_max)
    ax_right_top.set_yticks([97.0, 98.0, 99.0, 100.0])
    _style_axis(ax_right_top)
    ax_right_top.tick_params(axis="x", bottom=False, labelbottom=False)

    ax_right_bottom.set_ylim(lower_y_min, lower_y_max)
    bottom_tick_step = 1.0 if (lower_y_max - lower_y_min) <= 6.0 else 2.0
    ax_right_bottom.set_yticks(np.arange(math.ceil(lower_y_min), math.floor(lower_y_max) + 0.1, bottom_tick_step))
    ax_right_bottom.set_xticks(x)
    ax_right_bottom.set_xticklabels(labels, fontsize=tick_fs)
    ax_right_bottom.set_xlabel("GT motion blur level from mean robot blur length", fontsize=label_fs)
    _style_axis(ax_right_bottom)

    ax_right_top.spines["bottom"].set_visible(False)
    ax_right_bottom.spines["top"].set_visible(False)
    _draw_break_marks(ax_right_top, ax_right_bottom)

    handles, legend_labels = ax_left.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
        fontsize=12,
        handlelength=2.2,
        columnspacing=1.5,
        markerscale=1.25,
    )
    fig.suptitle("Effect of Motion Blur Level on Stage-1 Spiral Detection", y=1.035, fontsize=16, fontweight="bold")
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.065, right=0.985, wspace=0.18, hspace=0.06)
    png_path = out_root / "blur_level_stage1_sharp_vs_faded.png"
    pdf_path = out_root / "blur_level_stage1_sharp_vs_faded.pdf"
    fig.savefig(png_path, dpi=350, bbox_inches="tight")
    plt.close(fig)
    _save_png_as_single_page_pdf(png_path, pdf_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sharp/faded performance by motion blur level.")
    parser.add_argument("--sharp-metrics", default="outputs/p4_ai_eval_sharp_rtdetr_l_p5_stage2_aligned_val30/scene_metrics.csv")
    parser.add_argument("--faded-metrics", default="outputs/p4_ai_eval_faded_v8_grouped_blur_v2_val30/scene_metrics.csv")
    parser.add_argument("--sharp-data-root", default="outputs/p4_ai_dataset_v1_sharp")
    parser.add_argument("--faded-data-root", default="outputs/p4_ai_dataset_v1")
    parser.add_argument("--out", default="outputs/paper_figures/blur_level_analysis")
    parser.add_argument("--blur-min", type=float, default=1.0)
    parser.add_argument("--blur-max", type=float, default=30.0)
    args = parser.parse_args()

    out_root = (ROOT / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(float(args.blur_min), float(args.blur_max), 5)
    rows = []
    rows.extend(
        _summarize(
            metrics_path=(ROOT / args.sharp_metrics).resolve(),
            data_root=(ROOT / args.sharp_data_root).resolve(),
            method_name="Proposed sharp",
            edges=edges,
        )
    )
    rows.extend(
        _summarize(
            metrics_path=(ROOT / args.faded_metrics).resolve(),
            data_root=(ROOT / args.faded_data_root).resolve(),
            method_name="Faded variant",
            edges=edges,
        )
    )
    _write_summary_csv(rows, out_root / "blur_level_metrics_summary.csv")
    (out_root / "blur_level_metrics_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _plot_full_pipeline(rows, out_root)
    _plot_motion_direction(rows, out_root)
    _plot_stage1(rows, out_root)
    print(json.dumps(rows, indent=2))
    print(f"Saved figures to {out_root}")


if __name__ == "__main__":
    main()
