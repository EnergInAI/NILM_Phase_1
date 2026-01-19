import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from preprocessing.load_ukdale import read_meter
from preprocessing.normalize import preprocess_df

# ================= CONFIG =================
H5_PATH = "data/ukdale.h5"
BUILDING = 1
MAINS_METER = 1

SAMPLE_PERIOD = 6
SAMPLES_PER_DAY = int(24 * 3600 / SAMPLE_PERIOD)

START_INDEX = 0   # change this to test different days
# =========================================

print("Loading aggregate mains...")

mains_df = read_meter(H5_PATH, BUILDING, MAINS_METER)
mains = preprocess_df(mains_df, normalize=True)  # kW

print("Total samples available:", len(mains))

# --- 1 DAY SLICE ---
end = START_INDEX + SAMPLES_PER_DAY
day_mains = mains[START_INDEX:end]

print("1-day samples:", len(day_mains))
print("Mean power (kW):", day_mains.mean())

os.makedirs("data", exist_ok=True)
np.save("data/mains_1day.npy", day_mains)

print("✅ mains_1day.npy saved")
