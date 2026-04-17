"""
process_cfe.py - Load cached CFE pickle files and the original dataset,
                 compute normalised deltas, and save the output as parquet.

Run from the project root:
    python scripts/process_cfe.py
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_PATH = DATA_DIR / "a.pkl.gz"
SETUP_JSON_PATH = MODELS_DIR / "a-lr.json"

CACHE_LR = DATA_DIR / "cfe_batch_lr.pkl"
CACHE_EN = DATA_DIR / "cfe_batch_en.pkl"

OUTPUT_PATH = DATA_DIR / "all_deltas.parquet"
X_OUTPUT_PATH = DATA_DIR / "X.parquet"


def load_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Load the patient dataset and derive the feature column list.

    Returns
    -------
    X           : pd.DataFrame  (feature columns only, no outcome or _id cols)
    feature_cols: list[str]
    """
    data = pd.read_pickle(DATA_PATH)

    with open(SETUP_JSON_PATH) as f:
        setup = json.load(f)
    y_values = setup["experiment_setup"]["y"]

    df = data.loc[:, [col for col in data.columns if "_id" not in col]]
    df["outcome"] = y_values

    feature_cols = [col for col in df.columns if col != "outcome"]
    X = df[feature_cols]
    return X, feature_cols


def load_cache(cache_path: Path) -> list:
    """
    Load a pickled CFE results list from disk.

    Returns the list on success, raises on failure.
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"CFE cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    n_success = sum(r is not None for r in data)
    print(f"  Loaded {cache_path.name} ({n_success}/{len(data)} succeeded)")
    return data


def compute_normalised_deltas(
    cfe_results: list,
    X: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
) -> pd.DataFrame:
    """
    Transform raw counterfactual results into normalised proportional deltas.

    Parameters
    ----------
    cfe_results  : list where each entry is either None or a list of numpy arrays
                   (one array per counterfactual, each containing feature values)
    X            : DataFrame of original patient features
    feature_cols : list of feature column names
    model_name   : string label for this model

    Returns
    -------
    DataFrame where each row is one counterfactual with normalised deltas
    """
    feature_ranges = X[feature_cols].max().values - X[feature_cols].min().values

    # Avoid division by zero for any constant features
    feature_ranges[feature_ranges == 0] = np.nan

    rows = []
    for patient_idx, cfe_list in enumerate(cfe_results):
        if cfe_list is None:
            continue

        original = X[feature_cols].iloc[patient_idx].values

        for cf_idx, cf_values in enumerate(cfe_list):
            raw_delta = cf_values - original
            normalised_delta = raw_delta / feature_ranges

            row = {
                "patient_idx": patient_idx,
                "model": model_name,
                "cf_number": cf_idx,
            }
            for i, feature in enumerate(feature_cols):
                row[f"{feature}_raw"] = raw_delta[i]
                row[f"{feature}_norm"] = normalised_delta[i]

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    print("Loading dataset...")
    X, feature_cols = load_data()
    print(f"  {len(X)} patients, {len(feature_cols)} features.")

    print("Loading CFE caches...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfe_lr = load_cache(CACHE_LR)
        cfe_en = load_cache(CACHE_EN)

    print("Computing normalised deltas...")
    deltas_lr = compute_normalised_deltas(cfe_lr, X, feature_cols, "logistic_regression")
    deltas_en = compute_normalised_deltas(cfe_en, X, feature_cols, "elastic_net")

    all_deltas = pd.concat([deltas_lr, deltas_en], ignore_index=True)
    print(f"  {len(all_deltas)} rows ({len(deltas_lr)} LR + {len(deltas_en)} EN)")

    # pandas 3.x uses Arrow-backed strings by default; fastparquet requires
    # numpy object dtype strings — cast all non-numeric columns before writing.
    str_cols = all_deltas.select_dtypes(exclude="number").columns
    all_deltas[str_cols] = all_deltas[str_cols].astype(object)

    all_deltas.to_parquet(OUTPUT_PATH, engine="fastparquet", index=False)
    print(f"  Saved to {OUTPUT_PATH}")

    X.to_parquet(X_OUTPUT_PATH, engine="fastparquet", index=False)
    print(f"  Saved to {X_OUTPUT_PATH}")


if __name__ == "__main__":
    main()