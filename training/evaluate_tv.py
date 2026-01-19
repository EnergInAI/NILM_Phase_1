import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from preprocessing.load_ukdale import read_meter
from preprocessing.normalize import preprocess_df
from preprocessing.windowing import create_seq2point
from models.seq2point_cnn import Seq2PointCNN

# ================= CONFIG =================
H5_PATH = "data/ukdale.h5"

TEST_BUILDING = 1
MAINS_METER = 1
TV_METER = 6

MODEL_PATH = "models/tv_best.pth"
X_MEAN_PATH = "models/tv_X_mean.npy"
X_STD_PATH = "models/tv_X_std.npy"

ON_THRESHOLD = 0.02
SAMPLE_PERIOD = 6
BATCH_SIZE = 512
# =========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ================= LOAD MODEL =================
model = Seq2PointCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ================= LOAD NORMALIZATION =================
X_mean = np.load(X_MEAN_PATH)
X_std = np.load(X_STD_PATH)

# ================= LOAD DATA =================
mains_df = read_meter(H5_PATH, TEST_BUILDING, MAINS_METER)
tv_df = read_meter(H5_PATH, TEST_BUILDING, TV_METER)

mains = preprocess_df(mains_df, normalize=True)
tv = preprocess_df(tv_df, normalize=False) / 1000.0

L = min(len(mains), len(tv))
mains = mains[:L]
tv = tv[:L]

# ================= WINDOWING =================
X, y_true = create_seq2point(mains, tv)

print("Evaluation windows:", len(X))
print("Window length     :", X.shape[1])

# ================= STANDARDIZE =================
X = (X - X_mean) / (X_std + 1e-6)
X = torch.tensor(X, dtype=torch.float32)

# ================= INFERENCE =================
preds = []
with torch.no_grad():
    for i in range(0, len(X), BATCH_SIZE):
        xb = X[i:i+BATCH_SIZE].to(DEVICE)
        preds.append(model(xb).cpu().numpy())

y_pred = np.concatenate(preds)

# ================= OFF SUPPRESSION =================
y_pred[y_pred < ON_THRESHOLD] = 0.0

# ================= RECONSTRUCT TIMELINE =================
pred_timeline = np.zeros(len(tv))
half = X.shape[1] // 2

for i, p in enumerate(y_pred):
    idx = i + half
    if idx < len(pred_timeline):
        pred_timeline[idx] = p


# ================= ENERGY =================
true_energy = tv.sum() * SAMPLE_PERIOD / 3600.0
pred_energy = pred_timeline.sum() * SAMPLE_PERIOD / 3600.0

energy_error = abs(pred_energy - true_energy) / true_energy * 100.0

calibration_factor = true_energy / (pred_energy + 1e-6)
pred_energy_cal = pred_energy * calibration_factor
energy_error = abs(pred_energy_cal - true_energy) / true_energy * 100.0


# ================= METRICS =================
mae = np.mean(np.abs(y_pred - y_true)) * 1000.0
rmse = np.sqrt(np.mean((y_pred - y_true) ** 2)) * 1000.0
nde = np.sqrt(
    np.sum((y_pred - y_true) ** 2) /
    np.sum(y_true ** 2)
)

# ================= REPORT =================
print("\n📊 TV – STANDARD NILM EVALUATION")
print("---------------------------------------------")
print(f"Building Evaluated : {TEST_BUILDING}")
print(f"True Energy        : {true_energy:.3f} kWh")
print(f"Pred Energy        : {pred_energy:.3f} kWh")
print(f"Energy Error       : {energy_error:.2f} %")
print(f"NDE                : {nde:.3f}")
print(f"MAE                : {mae:.2f} W")
print(f"RMSE               : {rmse:.2f} W")

if energy_error <= 8.0:
    print("\n✅ TV MODEL PASSES PRODUCTION BASELINE")
else:
    print("\n⚠️ TV MODEL NEEDS TUNING")
