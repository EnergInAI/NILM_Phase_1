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
TV_METER = 6

BATCH_SIZE = 256
EPOCHS = 40
LR = 3e-5

ON_THRESHOLD = 0.06         # 20W
ON_RATIO = 0.6               # 60% ON, 40% OFF
ON_WEIGHT = 3.0
PATIENCE = 6

MODEL_PATH = "models/tv_best.pth"
X_MEAN_PATH = "models/tv_X_mean.npy"
X_STD_PATH = "models/tv_X_std.npy"
# =========================================

# ================= REPRODUCIBILITY =================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ================= LOAD DATA =================
all_X, all_y = [], []

for b in TRAIN_BUILDINGS:
    try:
        print(f"Loading Building {b}...")

        mains_df = read_meter(H5_PATH, b, MAINS_METER)
        tv_df = read_meter(H5_PATH, b, TV_METER)

        mains = preprocess_df(mains_df, normalize=True)
        tv = preprocess_df(tv_df, normalize=False) / 1000.0  # W → kW

        L = min(len(mains), len(tv))
        mains = mains[:L]
        tv = tv[:L]

        X, y = create_seq2point(mains, tv)

        all_X.append(X)
        all_y.append(y)

    except Exception as e:
        print(f"⚠️ Skipping building {b}: {e}")

X = np.concatenate(all_X)
y = np.concatenate(all_y)

print("Total windows before filtering:", len(X))

# ================= ON / OFF BALANCING =================
on_mask = y > ON_THRESHOLD
off_mask = ~on_mask

X_on, y_on = X[on_mask], y[on_mask]
X_off, y_off = X[off_mask], y[off_mask]

n_on = len(X_on)
n_off = int(n_on * (1 - ON_RATIO) / ON_RATIO)

idx = np.random.choice(len(X_off), size=min(n_off, len(X_off)), replace=False)

X_bal = np.concatenate([X_on, X_off[idx]])
y_bal = np.concatenate([y_on, y_off[idx]])

print(f"ON samples : {len(X_on)}")
print(f"OFF samples: {len(X_off[idx])}")
print(f"Final used : {len(X_bal)}")

# ================= NORMALIZE AGGREGATE =================
X_mean = X_bal.mean()
X_std = X_bal.std() + 1e-6
X_bal = (X_bal - X_mean) / X_std

np.save(X_MEAN_PATH, X_mean)
np.save(X_STD_PATH, X_std)

print("Aggregate standardized:",
      "mean =", X_bal.mean(),
      "std =", X_bal.std())

# ================= SPLIT =================
split = int(0.8 * len(X_bal))
X_train, X_val = X_bal[:split], X_bal[split:]
y_train, y_val = y_bal[:split], y_bal[split:]

# ================= TORCH DATA =================
train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)

val_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    ),
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
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

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

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

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

    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print("✅ Best TV model saved")
    else:
        counter += 1
        if counter >= PATIENCE:
            print("⏹ Early stopping triggered")
            break

print("🏁 TV training finished")
