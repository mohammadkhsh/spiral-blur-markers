from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spiral_markers.io_utils import ensure_dir, load_method_config, write_image
from spiral_markers.synthesis import generate_spiral_image


def main() -> None:
    cfg = load_method_config(ROOT / "configs" / "eval_paper_v4_fast.yaml")
    out_dir = ensure_dir(ROOT / "outputs" / "paper_v4_marker_preview")
    twists = [(-45.0, "neg"), (0.0, "zero"), (45.0, "pos")]
    images = []
    for twist, name in twists:
        role = "orientation" if abs(twist) <= 1.0e-6 else "identity"
        image, _ = generate_spiral_image(cfg.synthesis, twist_angle_deg=twist, role=role)
        images.append((name, image))
        write_image(out_dir / f"{name}_spiral.png", image)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, (name, image) in zip(axes, images):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "tag_sheet.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(out_dir)


if __name__ == "__main__":
    main()
