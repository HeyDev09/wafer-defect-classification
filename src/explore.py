"""Exploratory data analysis: class counts and example wafer maps for each pattern."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from data import CLASSES, load_wafers

# 0 = no die (white), 1 = good die (light grey-green), 2 = failed die (red)
WAFER_CMAP = ListedColormap(["#ffffff", "#9fd3a8", "#d62728"])


def main():
    os.makedirs("results", exist_ok=True)
    maps, y, _ = load_wafers()
    counts = np.bincount(y, minlength=len(CLASSES))

    print("Labeled wafers per class:")
    for c, n in zip(CLASSES, counts):
        print(f"  {c:10s} {n:7,d}  ({n / len(y):.1%})")
    sizes = np.array([m.shape for m in maps])
    print(f"Map sizes range from {sizes.min(0)} to {sizes.max(0)}")

    fig, ax = plt.subplots(figsize=(8, 4))
    order = np.argsort(counts)
    ax.barh(np.array(CLASSES)[order], counts[order], color="#4c72b0")
    ax.set_xscale("log")
    ax.set_xlabel("Number of wafers (log scale)")
    ax.set_title("WM-811K labeled wafers per class: heavy class imbalance")
    for i, n in enumerate(counts[order]):
        ax.text(n, i, f" {n:,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig("results/class_distribution.png", dpi=130)
    plt.close(fig)

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(CLASSES), 5, figsize=(9, 16))
    for r, c in enumerate(CLASSES):
        idx = rng.choice(np.where(y == r)[0], 5, replace=False)
        for k, i in enumerate(idx):
            ax = axes[r, k]
            ax.imshow(maps[i], cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(c, fontsize=11)
    fig.suptitle("Real WM-811K wafer maps (red = failed die, green = good die)")
    fig.tight_layout()
    fig.savefig("results/sample_wafer_maps.png", dpi=110)
    plt.close(fig)
    print("Saved results/class_distribution.png and results/sample_wafer_maps.png")


if __name__ == "__main__":
    main()
