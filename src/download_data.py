"""Download the WM-811K dataset (LSWMD.pkl, ~2 GB) from Kaggle into data/."""
import os
import shutil

import kagglehub


def main():
    os.makedirs("data", exist_ok=True)
    folder = kagglehub.dataset_download("qingyi/wm811k-wafer-map")
    src = os.path.join(folder, "LSWMD.pkl")
    dst = os.path.join("data", "LSWMD.pkl")
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    print("Dataset ready at", dst)


if __name__ == "__main__":
    main()
