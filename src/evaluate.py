"""Shared evaluation: per-class report, macro F1 and confusion matrix plot."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from data import CLASSES


def evaluate(y_true, y_pred, name, out_dir="results"):
    report = classification_report(y_true, y_pred, labels=range(len(CLASSES)),
                                   target_names=CLASSES, digits=3, zero_division=0)
    summary = {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "macro_f1_defect_classes_only": round(
            f1_score(y_true, y_pred, labels=range(8), average="macro"), 4),
        "per_class_f1": {c: round(v, 3) for c, v in zip(
            CLASSES, f1_score(y_true, y_pred, labels=range(len(CLASSES)), average=None))},
    }
    print(f"\n=== {name} ===\n{report}")
    print(json.dumps(summary, indent=2))
    with open(f"{out_dir}/{name}_report.txt", "w") as f:
        f.write(report)
    with open(f"{out_dir}/{name}_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Row-normalised confusion matrix: each row shows where the true class went.
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name}: confusion matrix (row-normalised)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{name}_confusion_matrix.png", dpi=130)
    plt.close(fig)
    return summary
