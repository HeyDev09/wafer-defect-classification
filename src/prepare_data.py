"""Load the raw WM-811K pickle safely and save the labeled wafers in a compact format.

LSWMD.pkl was saved with a very old pandas, and pickle files can run code when
loaded. So we use a restricted unpickler that only allows the pandas/numpy
classes a DataFrame needs, and maps old module names to new ones.
"""
import argparse
import pickle

import numpy as np
from pandas.compat.pickle_compat import Unpickler as PandasUnpickler

ALLOWED = {
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.internals", "BlockManager"),
    ("pandas.indexes.base", "_new_Index"),
    ("pandas.indexes.base", "Index"),
    ("pandas.indexes.range", "RangeIndex"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("__builtin__", "slice"),
}
RENAMES = {
    ("pandas.indexes.base", "_new_Index"): ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.indexes.base", "Index"): ("pandas.core.indexes.base", "Index"),
    ("pandas.indexes.range", "RangeIndex"): ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.internals", "BlockManager"): ("pandas.core.internals.managers", "BlockManager"),
    ("numpy.core.multiarray", "_reconstruct"): ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"): ("numpy._core.multiarray", "scalar"),
    ("__builtin__", "slice"): ("builtins", "slice"),
}


class SafeUnpickler(PandasUnpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"Blocked class: {module}.{name}")
        module, name = RENAMES.get((module, name), (module, name))
        return super().find_class(module, name)


def label_of(x):
    """failureType is stored as a nested array like [['Center']] or [] if unlabeled."""
    a = np.asarray(x)
    return str(a.ravel()[0]) if a.size else None


def main(src, dst):
    with open(src, "rb") as f:
        df = SafeUnpickler(f, encoding="latin1").load()
    print("Raw dataset:", df.shape, list(df.columns))

    labels = df["failureType"].map(label_of)
    keep = labels.notna()
    print(f"Labeled wafers: {keep.sum():,} of {len(df):,}")

    maps = [np.asarray(m, dtype=np.uint8) for m in df.loc[keep, "waferMap"]]
    shapes = np.array([m.shape for m in maps], dtype=np.int32)
    flat = np.concatenate([m.ravel() for m in maps])
    np.savez_compressed(
        dst,
        flat=flat,
        shapes=shapes,
        labels=labels[keep].to_numpy().astype(str),
        lots=df.loc[keep, "lotName"].to_numpy().astype(str),
    )
    print("Saved", dst)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/LSWMD.pkl")
    p.add_argument("--dst", default="data/wm811k_labeled.npz")
    a = p.parse_args()
    main(a.src, a.dst)
