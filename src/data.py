"""Load the labeled wafer maps and split them into train/test sets."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]
SEED = 42


def load_wafers(path="data/wm811k_labeled.npz"):
    """Return (list of 2D wafer maps, integer labels, lot names).

    Each wafer map is a 2D uint8 array: 0 = no die, 1 = good die, 2 = failed die.
    """
    d = np.load(path)
    sizes = d["shapes"].prod(axis=1)
    parts = np.split(d["flat"], np.cumsum(sizes)[:-1])
    maps = [p.reshape(s) for p, s in zip(parts, d["shapes"])]
    y = np.array([CLASSES.index(c) for c in d["labels"]])
    return maps, y, d["lots"]


def train_test_split_by_lot(y, lots):
    """80/20 split where all wafers from one lot stay on the same side.

    Wafers from the same lot look very similar. If a lot is split between
    train and test, the test score becomes too optimistic (data leakage).
    """
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(skf.split(np.zeros(len(y)), y, groups=lots))
    return train_idx, test_idx


def resize_map(m, size=32):
    """Resize a wafer map to size x size with nearest-neighbour sampling,
    so the values stay 0/1/2 (no blurring between classes)."""
    h, w = m.shape
    rows = (np.arange(size) * h / size).astype(int)
    cols = (np.arange(size) * w / size).astype(int)
    return m[rows][:, cols]
