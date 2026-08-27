"""
Loads the Spark job execution dataset used to train the prediction model.
"""

import pandas as pd

from src.config import DATA_PATH


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    print("Retrieving Spark execution dataset …")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find dataset at '{path}'. "
            "Place Spark_realtime_metrices.csv in the data/ folder "
            "(see data/README.md)."
        ) from exc

    print(f"Dataset shape : {df.shape}")
    print(df.describe().round(2))
    return df
