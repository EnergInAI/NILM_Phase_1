import tables
import numpy as np
import pandas as pd

def read_meter(h5_path, building, meter):
    """
    Correct UK-DALE meter reader for tables with:
    [('index', '<i8'), ('values_block_0', '<f4', (1,))]
    """
    path = f"/building{building}/elec/meter{meter}/table"

    with tables.open_file(h5_path, mode="r") as f:
        table = f.get_node(path)
        data = table.read()

    # ✅ Extract real power column and FLATTEN
    if "values_block_0" in data.dtype.names:
        power = data["values_block_0"].reshape(-1).astype(np.float32)
    else:
        raise RuntimeError(
            f"values_block_0 not found in {path}. "
            f"Found columns: {data.dtype.names}"
        )

    return pd.DataFrame({"power": power})
def load_multi_building_seq2point(
    h5_path,
    buildings,
    mains_meter,
    appliance_meter,
    on_threshold
):
    import numpy as np
    from preprocessing.windowing import create_seq2point
    from preprocessing.normalize import preprocess_df
    from preprocessing.load_ukdale import read_meter

    X_all, y_all = [], []

    for b in buildings:
        mains = preprocess_df(
            read_meter(h5_path, b, mains_meter),
            normalize=True
        )

        app = preprocess_df(
            read_meter(h5_path, b, appliance_meter),
            normalize=False
        ) / 1000.0  # W → kW

        L = min(len(mains), len(app))
        X, y = create_seq2point(mains[:L], app[:L])

        mask = y > on_threshold
        X_all.append(X[mask])
        y_all.append(y[mask])

    return np.concatenate(X_all), np.concatenate(y_all)
