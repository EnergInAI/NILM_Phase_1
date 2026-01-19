import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from preprocessing.load_ukdale import read_meter
from preprocessing.normalize import preprocess_df

# ================= CONFIG =================
H5_PATH = "data/ukdale.h5"
BUILDING = 1        # choose any building for inference
MAINS_METER = 1
# =========================================

print("Loading aggregate mains...")

mains_df = read_meter(H5_PATH, BUILDING, MAINS_METER)

# preprocess_df converts to kW and resamples to 6s
mains = preprocess_df(mains_df, normalize=True)

print("Mains samples:", len(mains))
print("Mean power (kW):", mains.mean())

os.makedirs("data", exist_ok=True)
np.save("data/mains_input.npy", mains)

print("✅ mains_input.npy saved")
