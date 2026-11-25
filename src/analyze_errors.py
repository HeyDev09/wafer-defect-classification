"""Compare both models and look at the wafers the Random Forest got wrong."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import CLASSES, load_wafers, train_test_split_by_lot
from explore import WAFER_CMAP


def compare_models():
    rf = json.load(open("results/random_forest_metrics.json"))
    cnn = json.load(open("results/cnn_metrics.json"))
    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, [rf["per_class_f1"][c] for c in CLASSES], 0.4,
           label=f"Random Forest (macro F1 {rf['macro_f1']:.2f})", color="#4c72b0")
    ax.bar(x + 0.2, [cnn["per_class_f1"][c] for c in CLASSES], 0.4,
           label=f"CNN (macro F1 {cnn['macro_f1']:.2f})", color="#dd8452")
    ax.set_xticks(x, CLASSES, rotation=30, ha="right")
    ax.set_ylabel("F1 score on test set")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1: Random Forest vs CNN")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("results/model_comparison.png", dpi=130)
    plt.close(fig)


def show_mistakes(n_per_class=5):
    maps, y, lots = load_wafers()
    _, test_idx = train_test_split_by_lot(y, lots)
    X = np.load("data/features.npy")
    rf = joblib.load("models/random_forest.joblib")
    pred = rf.predict(X[test_idx])
    y_test = y[test_idx]

    hard = ["Scratch", "Loc", "Edge-Loc"]  # the three weakest classes
    fig, axes = plt.subplots(len(hard), n_per_class, figsize=(10, 6.5))
    for r, c in enumerate(hard):
        k = CLASSES.index(c)
        wrong = np.where((y_test == k) & (pred != k))[0]
        print(f"{c}: {len(wrong)} of {(y_test == k).sum()} test wafers misclassified; "
              f"most often predicted as {CLASSES[np.bincount(pred[wrong]).argmax()]}")
        for j, i in enumerate(wrong[:n_per_class]):
            ax = axes[r, j]
            ax.imshow(maps[test_idx[i]], cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
            ax.set_title(f"true {c}\npred {CLASSES[pred[i]]}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Random Forest mistakes on the three hardest classes")
    fig.tight_layout()
    fig.savefig("results/misclassified_examples.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    compare_models()
    show_mistakes()
