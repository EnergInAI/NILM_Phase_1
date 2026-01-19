import numpy as np
from inference.load_models import load_all_models
from inference.predict_appliance import predict_series
from inference.postprocess import resolve_conflicts, compute_residual
from inference.energy_report import energy_kwh

SAMPLE_PERIOD = 6  # seconds

# ================= LOAD DATA =================
mains = np.load("data/mains_1day.npy")

# ================= LOAD MODELS =================
models = load_all_models()

stats = {
    "fridge": ("models/fridge_X_mean.npy", "models/fridge_X_std.npy"),
    "ac":     ("models/ac_X_mean.npy", "models/ac_X_std.npy"),
    "tv":     ("models/tv_X_mean.npy", "models/tv_X_std.npy"),
    "wm":     ("models/wm_X_mean.npy", "models/wm_X_std.npy"),
}

preds = {}

for app in stats:
    mean = np.load(stats[app][0])
    std  = np.load(stats[app][1])

    preds[app] = predict_series(
        models[app],
        mains,
        mean,
        std
    )

# ================= POST-PROCESS =================
preds = resolve_conflicts(mains, preds)
residual = compute_residual(mains, preds)

# ================= ENERGY =================
energy = {}
total_energy = energy_kwh(mains, SAMPLE_PERIOD)

for app in preds:
    energy[app] = energy_kwh(preds[app], SAMPLE_PERIOD)

energy["others"] = energy_kwh(residual, SAMPLE_PERIOD)

# ================= PERCENTAGE =================
percent = {
    k: (v / total_energy) * 100
    for k, v in energy.items()
}

# ================= REPORT =================
print("\n🔌 NILM – 1 DAY ENERGY BREAKDOWN")
print("--------------------------------")
for k in ["fridge", "ac", "tv", "wm", "others"]:
    print(f"{k:10s}: {energy[k]:6.2f} kWh | {percent[k]:5.1f} %")

print("--------------------------------")
print(f"TOTAL      : {total_energy:.2f} kWh | 100.0 %")
