import numpy as np
import torch
from preprocessing.windowing import create_seq2point

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512

def predict_series(model, mains, X_mean, X_std):
    X, _ = create_seq2point(mains, mains)  # dummy target
    X = (X - X_mean) / (X_std + 1e-6)

    X = torch.tensor(X, dtype=torch.float32)

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = X[i:i+BATCH_SIZE].to(DEVICE)
            preds.append(model(xb).cpu().numpy())

    y_pred = np.concatenate(preds)

    timeline = np.zeros(len(mains))
    half = X.shape[1] // 2

    for i, p in enumerate(y_pred):
        idx = i + half
        if idx < len(timeline):
            timeline[idx] = p

    return timeline
