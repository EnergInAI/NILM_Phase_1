import numpy as np

def preprocess_df(df, sample_period=6, normalize=True):
    power = df["power"].values

    usable_len = (len(power) // sample_period) * sample_period
    power = power[:usable_len]

    power = power.reshape(-1, sample_period).mean(axis=1)
    power = np.clip(power, 0, None)

    if normalize:
        power = power / 1000.0  # convert W → kW

    return power.astype(np.float32)
