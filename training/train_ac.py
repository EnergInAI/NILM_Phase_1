import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

from preprocessing.load_ukdale import read_meter
from preprocessing.normalize import preprocess_df
from preprocessing.windowing import create_seq2point
from models.seq2point_cnn import Seq2PointCNN

# ================= CONFIG =================
H5_PATH = "data/ukdale.h5"

TRAIN_BUILDINGS = [1, 2, 3, 4]
MAINS_METER = 1
AC_METER = 10          # ⚠️ verify once

BATCH_SIZE = 256
EPOCHS = 40
LR = 3e-5

ON_THRESHOLD = 0.5     # kW (AC ON ≈ 500W+)
ON_WEIGHT = 4.0        # stronger than fridge
PATIENCE = 6

WINDOW_LENGTH = 599
MODEL_PATH = "models/ac_best.pth"
X_MEAN_PATH = "models/ac_X_mean.npy"
X_STD_PATH = "models/ac_X_std.npy"
# =========================================

# ================= REPRODUCIBILITY =================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ================= LOAD MULTI-BUILDING DATA =================
all_X, all_y = [], []

for b in TRAIN_BUILDINGS:
    try:
        print(f"Loading Building {b}...")

        mains_df = read_meter(H5_PATH, b, MAINS_METER)
        ac_df = read_meter(H5_PATH, b, AC_METER)

        mains = preprocess_df(mains_df, normalize=True)
        ac = preprocess_df(ac_df, normalize=False) / 1000.0  # W → kW

        min_len = min(len(mains), len(ac))
        mains = mains[:min_len]
        ac = ac[:min_len]

        X, y = create_seq2point(mains, ac)

        all_X.append(X)
        all_y.append(y)

    except Exception as e:
        print(f"⚠️ Skipping building {b}: {e}")

X = np.concatenate(all_X)
y = np.concatenate(all_y)

print("Total samples:", len(X))

# ================= NORMALIZATION =================
X_mean = X.mean()
X_std = X.std() + 1e-6
X = (X - X_mean) / X_std

np.save(X_MEAN_PATH, X_mean)
np.save(X_STD_PATH, X_std)

print("Aggregate standardized:",
      "mean =", X.mean(),
      "std =", X.std())

# ================= TRAIN / VAL SPLIT =================
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ================= TORCH DATA =================
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)

# ================= MODEL =================
model = Seq2PointCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

mse = torch.nn.MSELoss(reduction="none")

# ================= TRAINING =================
best_val = float("inf")
counter = 0

for epoch in range(EPOCHS):
    # -------- TRAIN --------
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        preds = model(xb)

        weights = torch.ones_like(yb)
        weights[yb > ON_THRESHOLD] = ON_WEIGHT

        loss = (mse(preds, yb) * weights).mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            preds = model(xb)

            weights = torch.ones_like(yb)
            weights[yb > ON_THRESHOLD] = ON_WEIGHT
            val_loss += (mse(preds, yb) * weights).mean().item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )

    # -------- EARLY STOP --------
    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print("✅ Best AC model saved")
    else:
        counter += 1
        if counter >= PATIENCE:
            print("⏹ Early stopping triggered")
            break

print("🏁 AC training finished")
