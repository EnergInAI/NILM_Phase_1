import numpy as np

WINDOW_SIZE = 599
HALF = WINDOW_SIZE // 2

def create_seq2point(aggregate, target, max_samples=60000):
    X, y = [], []

    for i in range(HALF, len(aggregate) - HALF):
        X.append(aggregate[i - HALF : i + HALF + 1])
        y.append(target[i])

        if len(X) >= max_samples:
            break

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
