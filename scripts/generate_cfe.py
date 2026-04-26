"""
generate_cfe.py - Generate and cache DiCE counterfactual explanations for both
              the Logistic Regression and Elastic Net models.

Run from the project root:
    python scripts/generate_cfe.py
"""

import json
import pickle
import warnings
from pathlib import Path

import joblib
import pandas as pd
import dice_ml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_LR_PATH = MODELS_DIR / "a-lr.pkl"
MODEL_EN_PATH = MODELS_DIR / "a-en.pkl"
SETUP_JSON_PATH = MODELS_DIR / "a-lr.json"
DATA_PATH = DATA_DIR / "a.pkl.gz"

CACHE_LR = DATA_DIR / "cfe_batch_lr.pkl"
CACHE_EN = DATA_DIR / "cfe_batch_en.pkl"

CACHE_LR_20 = DATA_DIR / "cfe_batch_lr_20.pkl"
CACHE_EN_20 = DATA_DIR / "cfe_batch_en_20.pkl"


def load_data():
    """
    Load both sklearn pipeline models, the patient data, and the setup JSON.

    Returns
    -------
    pipeline_lr  : sklearn Pipeline
    pipeline_en  : sklearn Pipeline
    df           : pd.DataFrame  (features + 'outcome' column, no _id columns)
    feature_cols : list[str]
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline_lr = joblib.load(MODEL_LR_PATH)
        pipeline_en = joblib.load(MODEL_EN_PATH)

    data = pd.read_pickle(DATA_PATH)

    with open(SETUP_JSON_PATH) as f:
        setup = json.load(f)
    y_values = setup["experiment_setup"]["y"]

    df = data.loc[:, [col for col in data.columns if "_id" not in col]]
    df["outcome"] = y_values

    feature_cols = [col for col in df.columns if col != "outcome"]
    return pipeline_lr, pipeline_en, df, feature_cols


def build_dice_objects(df, pipeline_lr, pipeline_en):
    """
    Construct DiCE Data, Model, and Dice explainer objects for both models.

    Parameters
    ----------
    df           : pd.DataFrame with features + 'outcome'
    pipeline_lr  : sklearn Pipeline (Logistic Regression)
    pipeline_en  : sklearn Pipeline (Elastic Net)

    Returns
    -------
    exp_lr : dice_ml.Dice  (genetic method)
    exp_en : dice_ml.Dice  (genetic method)
    """
    dice_data = dice_ml.Data(
        dataframe=df,
        continuous_features=[col for col in df.columns if col != "outcome"],
        outcome_name="outcome",
    )

    dice_model_lr = dice_ml.Model(model=pipeline_lr, backend="sklearn")
    dice_model_en = dice_ml.Model(model=pipeline_en, backend="sklearn")

    exp_lr = dice_ml.Dice(dice_data, dice_model_lr, method="genetic")
    exp_en = dice_ml.Dice(dice_data, dice_model_en, method="genetic")

    return exp_lr, exp_en


def load_cache(cache_path: Path) -> list | None:
    """
    Attempt to load a pickled CFE results list from disk.

    Returns the list if successful, None if the file does not exist or fails
    to load.
    """
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            n_success = sum(r is not None for r in data)
            print(f"  Loaded cache from {cache_path.name} ({n_success}/{len(data)} succeeded)")
            return data
        except Exception as e:
            print(f"  Cache {cache_path.name} exists but failed to load: {e}")
    return None


def generate_batch_cfe(
    explainer: dice_ml.Dice,
    X: pd.DataFrame,
    feature_cols: list,
    cache_path: Path,
    total_cfs: int = 3,
) -> list:
    """
    Generate counterfactuals for every patient in X and cache to disk.

    Each entry in the returned list is either:
      - A list of np.ndarray, one per counterfactual (length == total_cfs on success)
      - None if generation failed for that patient

    The result is pickled to cache_path before returning.
    """
    results = []
    for i in range(len(X)):
        try:
            cfe = explainer.generate_counterfactuals(
                query_instances=X.iloc[[i]],
                total_CFs=total_cfs,
                desired_class="opposite",
            )
            cf_df = cfe.cf_examples_list[0].final_cfs_df

            if cf_df is not None and len(cf_df) > 0:
                all_cfs = [row.values for _, row in cf_df[feature_cols].iterrows()]
                results.append(all_cfs)
            else:
                results.append(None)
        except Exception as e:
            print(f"  Patient {i}: failed ({e})")
            results.append(None)

    with open(cache_path, "wb") as f:
        pickle.dump(results, f)

    return results


def print_summary(cfe_lr: list, cfe_en: list) -> None:
    """Print summary statistics for both CFE result sets."""
    n = len(cfe_lr)
    lr_success = sum(r is not None for r in cfe_lr)
    en_success = sum(r is not None for r in cfe_en)
    both_success = sum(
        r_lr is not None and r_en is not None
        for r_lr, r_en in zip(cfe_lr, cfe_en)
    )
    print("\n--- Summary ---")
    print(f"  Total patients : {n}")
    print(f"  LR succeeded   : {lr_success}/{n} ({lr_success/n:.1%})")
    print(f"  EN succeeded   : {en_success}/{n} ({en_success/n:.1%})")
    print(f"  Both succeeded : {both_success}/{n} ({both_success/n:.1%})")


def generate_20_cfe_batch() -> None:
    """Generate and cache 20-CFE batches for both models."""
    print("Loading data and models...")
    pipeline_lr, pipeline_en, df, feature_cols = load_data()
    X = df[feature_cols]

    print("Building DiCE explainers...")
    exp_lr, exp_en = build_dice_objects(df, pipeline_lr, pipeline_en)

    print("Logistic Regression (20 CFEs):")
    cfe_lr = load_cache(CACHE_LR_20)
    if cfe_lr is None:
        print("  Generating...")
        cfe_lr = generate_batch_cfe(exp_lr, X, feature_cols, CACHE_LR_20, total_cfs=20)
        print(f"  Done. {sum(r is not None for r in cfe_lr)}/{len(cfe_lr)} succeeded.")

    print("Elastic Net (20 CFEs):")
    cfe_en = load_cache(CACHE_EN_20)
    if cfe_en is None:
        print("  Generating...")
        cfe_en = generate_batch_cfe(exp_en, X, feature_cols, CACHE_EN_20, total_cfs=20)
        print(f"  Done. {sum(r is not None for r in cfe_en)}/{len(cfe_en)} succeeded.")

    print_summary(cfe_lr, cfe_en)


def main() -> None:
    print("Loading data and models...")
    pipeline_lr, pipeline_en, df, feature_cols = load_data()
    print(f"  Loaded {len(df)} patients, {len(feature_cols)} features.")

    print("Building DiCE explainers...")
    exp_lr, exp_en = build_dice_objects(df, pipeline_lr, pipeline_en)

    X = df[feature_cols]

    print("Logistic Regression CFEs:")
    cfe_lr = load_cache(CACHE_LR)
    if cfe_lr is None:
        print("  Generating (this may take several minutes)...")
        cfe_lr = generate_batch_cfe(exp_lr, X, feature_cols, CACHE_LR)
        print(f"  Done. {sum(r is not None for r in cfe_lr)}/{len(cfe_lr)} succeeded.")

    print("Elastic Net CFEs:")
    cfe_en = load_cache(CACHE_EN)
    if cfe_en is None:
        print("  Generating (this may take several minutes)...")
        cfe_en = generate_batch_cfe(exp_en, X, feature_cols, CACHE_EN)
        print(f"  Done. {sum(r is not None for r in cfe_en)}/{len(cfe_en)} succeeded.")

    print_summary(cfe_lr, cfe_en)

    print("\n20-CFE batch:")
    generate_20_cfe_batch()


if __name__ == "__main__":
    main()