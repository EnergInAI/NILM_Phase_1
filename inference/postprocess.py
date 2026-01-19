import numpy as np

def calibrate(pred, true_energy, sample_period):
    pred_energy = pred.sum() * sample_period / 3600
    factor = true_energy / (pred_energy + 1e-6)
    return pred * factor, factor

def resolve_conflicts(mains, preds):
    """
    Priority:
    WM > AC > Fridge > TV
    """
    total = (
        preds["wm"] +
        preds["ac"] +
        preds["fridge"] +
        preds["tv"]
    )

    scale = np.minimum(1.0, mains / (total + 1e-6))

    preds["tv"]      *= scale
    preds["fridge"] *= scale
    preds["ac"]     *= scale
    preds["wm"]     *= scale

    return preds

def compute_residual(mains, preds):
    used = sum(preds.values())
    return np.clip(mains - used, 0, None)
