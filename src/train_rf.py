"""Model 1: Random Forest on hand-crafted wafer features."""
import os
import time
from multiprocessing import Pool

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from data import SEED, load_wafers, train_test_split_by_lot
from evaluate import evaluate
from features import FEATURE_NAMES, extract_features


def build_features(maps):
    with Pool() as pool:
        return np.array(pool.map(extract_features, maps, chunksize=500), dtype=np.float32)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    maps, y, lots = load_wafers()
    train_idx, test_idx = train_test_split_by_lot(y, lots)

    t = time.time()
    if os.path.exists("data/features.npy"):
        X = np.load("data/features.npy")
    else:
        X = build_features(maps)
        np.save("data/features.npy", X)
    print(f"Feature matrix {X.shape} ready in {time.time() - t:.0f}s")

    # class_weight="balanced_subsample" gives rare classes (Near-full, Donut)
    # more weight, so the forest does not just predict "none" everywhere.
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED)
    t = time.time()
    rf.fit(X[train_idx], y[train_idx])
    print(f"Trained in {time.time() - t:.0f}s")

    evaluate(y[test_idx], rf.predict(X[test_idx]), "random_forest")
    joblib.dump(rf, "models/random_forest.joblib", compress=3)

    imp = rf.feature_importances_
    order = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(FEATURE_NAMES)[order], imp[order], color="#4c72b0")
    ax.set_title("Random Forest: top 15 feature importances")
    fig.tight_layout()
    fig.savefig("results/rf_feature_importance.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
