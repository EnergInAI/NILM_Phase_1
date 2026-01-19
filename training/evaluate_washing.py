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

TEST_BUILDING = 1          # WM exists here
MAINS_METER = 1
WM_METER = 7

MODEL_PATH = "models/wm_best.pth"
X_MEAN_PATH = "models/wm_X_mean.npy"
X_STD_PATH = "models/wm_X_std.npy"

ON_THRESHOLD = 0.6        # kW
SAMPLE_PERIOD = 6          # seconds
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

# ================= LOAD TEST DATA =================
mains_df = read_meter(H5_PATH, TEST_BUILDING, MAINS_METER)
wm_df = read_meter(H5_PATH, TEST_BUILDING, WM_METER)

mains = preprocess_df(mains_df, normalize=True)
wm = preprocess_df(wm_df, normalize=False) / 1000.0  # W → kW

L = min(len(mains), len(wm))
mains = mains[:L]
wm = wm[:L]

# ================= WINDOWING =================
X, y_true = create_seq2point(mains, wm)

print("Evaluation windows:", len(X))
print("Window length     :", X.shape[1])

# ================= STANDARDIZE =================
X = (X - X_mean) / (X_std + 1e-6)
X = torch.tensor(X, dtype=torch.float32)

# ================= INFERENCE (BATCHED) =================
preds = []
with torch.no_grad():
    for i in range(0, len(X), BATCH_SIZE):
        xb = X[i:i+BATCH_SIZE].to(DEVICE)
        preds.append(model(xb).cpu().numpy())

y_pred = np.concatenate(preds)

# ================= OFF SUPPRESSION =================
y_pred[y_pred < ON_THRESHOLD] = 0.0

# ================= RECONSTRUCT TIMELINE =================
pred_timeline = np.zeros(len(wm))
half = X.shape[1] // 2

for i, p in enumerate(y_pred):
    idx = i + half
    if idx < len(pred_timeline):
        pred_timeline[idx] = p

# ================= ENERGY (RAW, HONEST) =================
true_energy = wm.sum() * SAMPLE_PERIOD / 3600.0
pred_energy_raw = pred_timeline.sum() * SAMPLE_PERIOD / 3600.0

energy_error = abs(pred_energy_raw - true_energy) / true_energy * 100.0

# ================= CALIBRATION (REPORTING ONLY) =================
if pred_energy_raw < 0.05:
    calibration_factor = 1.0
else:
    calibration_factor = true_energy / (pred_energy_raw + 1e-6)

# ================= METRICS =================
mae = np.mean(np.abs(y_pred - y_true)) * 1000.0
rmse = np.sqrt(np.mean((y_pred - y_true) ** 2)) * 1000.0
nde = np.sqrt(
    np.sum((y_pred - y_true) ** 2) /
    np.sum(y_true ** 2)
)

# ================= REPORT =================
print("\n📊 WASHING MACHINE – STANDARD NILM EVALUATION")
print("------------------------------------------------")
print(f"Building Evaluated : {TEST_BUILDING}")
print(f"True Energy        : {true_energy:.3f} kWh")
print(f"Pred Energy        : {pred_energy_raw:.3f} kWh")
print(f"Energy Error       : {energy_error:.2f} %  (PRIMARY)")
print(f"Calibration Factor : {calibration_factor:.3f}")
print(f"NDE                : {nde:.3f}")
print(f"MAE                : {mae:.2f} W")
print(f"RMSE               : {rmse:.2f} W")

# ================= DECISION =================
if energy_error <= 10.0 and calibration_factor <= 1.6 and nde <= 2.0:
    print("\n✅ WASHING MACHINE MODEL PASSES PRODUCTION BASELINE")
else:
    print("\n⚠️ WASHING MACHINE MODEL NEEDS TUNING")
