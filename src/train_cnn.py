"""Model 2: a small CNN that learns defect patterns directly from the wafer image."""
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from data import CLASSES, SEED, load_wafers, resize_map, train_test_split_by_lot
from evaluate import evaluate

SIZE = 32
EPOCHS = 15
BATCH = 128
N_NONE_TRAIN = 20000  # undersample the huge "none" class for training

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def to_tensor(maps):
    """2 input channels: channel 0 = where dies exist, channel 1 = where dies failed."""
    arr = np.stack([resize_map(m, SIZE) for m in maps])
    x = np.stack([(arr > 0), (arr == 2)], axis=1).astype(np.float32)
    return torch.from_numpy(x)


def augment(x):
    """Defect patterns don't depend on orientation, so random flips and
    90-degree rotations create extra realistic training examples."""
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(2, 3))


class WaferCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2))

        self.features = nn.Sequential(block(2, 32), block(32, 64), block(64, 128))  # 32 -> 4
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + 1024]).argmax(1) for i in range(0, len(X), 1024)]).numpy()


def main():
    torch.set_num_threads(os.cpu_count())
    os.makedirs("models", exist_ok=True)
    maps, y, lots = load_wafers()
    train_idx, test_idx = train_test_split_by_lot(y, lots)

    # Keep 10% of the training lots aside as a validation set to pick the best epoch.
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    fit_pos, val_pos = next(skf.split(train_idx, y[train_idx], groups=lots[train_idx]))
    fit_idx, val_idx = train_idx[fit_pos], train_idx[val_pos]

    none_id = CLASSES.index("none")
    none_idx = fit_idx[y[fit_idx] == none_id]
    fit_idx = np.concatenate([fit_idx[y[fit_idx] != none_id],
                              rng.choice(none_idx, N_NONE_TRAIN, replace=False)])

    X = to_tensor(maps)
    Y = torch.from_numpy(y)
    print(f"Train {len(fit_idx):,} | val {len(val_idx):,} | test {len(test_idx):,}")

    # Class weights (inverse square-root of frequency) so rare classes count more in the loss.
    counts = np.bincount(y[fit_idx], minlength=len(CLASSES))
    w = 1 / np.sqrt(counts)
    w = torch.tensor(w / w.mean(), dtype=torch.float32)

    model = WaferCNN()
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_f1 = -1
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t = time.time()
        perm = torch.from_numpy(rng.permutation(fit_idx))
        total = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            out = model(augment(X[b]))
            loss = loss_fn(out, Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        val_f1 = f1_score(y[val_idx], predict(model, X[val_idx]), average="macro")
        print(f"epoch {epoch:2d}  loss {total / len(perm):.4f}  val macro-F1 {val_f1:.4f}  "
              f"({time.time() - t:.0f}s)", flush=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "models/wafer_cnn.pt")

    model.load_state_dict(torch.load("models/wafer_cnn.pt"))
    print(f"Best validation macro-F1: {best_f1:.4f}")
    evaluate(y[test_idx], predict(model, X[test_idx]), "cnn")


if __name__ == "__main__":
    main()
