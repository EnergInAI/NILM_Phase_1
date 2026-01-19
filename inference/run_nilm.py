import numpy as np
from inference.load_models import load_all_models
from inference.predict_appliance import predict_series
from inference.postprocess import calibrate, resolve_conflicts, compute_residual
from inference.energy_report import generate_report

SAMPLE_PERIOD = 6  # seconds

# Load mains data (already preprocessed to kW)
mains = np.load("data/mains_input.npy")

models = load_all_models()

# Load normalization stats
stats = {
    "fridge": ("models/fridge_X_mean.npy", "models/fridge_X_std.npy"),
    "ac":     ("models/ac_X_mean.npy", "models/ac_X_std.npy"),
    "tv":     ("models/tv_X_mean.npy", "models/tv_X_std.npy"),
    "wm":     ("models/wm_X_mean.npy", "models/wm_X_std.npy"),
}

preds = {}

for app in ["fridge", "ac", "tv", "wm"]:
    mean = np.load(stats[app][0])
    std  = np.load(stats[app][1])

    preds[app] = predict_series(
        models[app],
        mains,
        mean,
        std
    )

# Conflict resolution
preds = resolve_conflicts(mains, preds)

# Residual
residual = compute_residual(mains, preds)

# Final report
report = generate_report(preds, residual, SAMPLE_PERIOD)

print("\n🔌 NILM ENERGY BREAKDOWN (kWh)")
for k, v in report.items():
    print(f"{k:10s}: {v:.2f}")
