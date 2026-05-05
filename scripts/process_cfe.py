"""
process_cfe.py - Load cached CFE pickle files and the original datasets,
                 project all models to the 11 shared base features,
                 compute normalised deltas / PCA / individual batches,
                 and save a single nested cfe_data.pkl for the dashapp.

Run from the project root:
    python scripts/process_cfe.py

Output: data/cfe_data.pkl
    {
        "base_feature_cols": [...],   # 11 common features
        "pairs": {
            "LR vs EN":       { all_deltas, X, pca_results, indiv_cfe_batch,
                                model_a_name, model_b_name, spec_a, spec_b },
            "LR vs AC-SVM":   { ... },
            "EN vs AC-SVM":   { ... },
        }
    }
"""

import json
import pickle
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_PATH = DATA_DIR / "a.pkl.gz"
DATA_AC_PATH = DATA_DIR / "ac.pkl.gz"
SETUP_JSON_PATH = MODELS_DIR / "a-lr.json"
SETUP_AC_SVM_JSON = MODELS_DIR / "ac-svm.json"

CACHE_LR_20 = DATA_DIR / "cfe_batch_lr_20.pkl"
CACHE_EN_20 = DATA_DIR / "cfe_batch_en_20.pkl"
CACHE_AC_SVM_20 = DATA_DIR / "cfe_batch_ac_svm_20.pkl"

CFE_DATA_PATH = DATA_DIR / "cfe_data.pkl"

# 11 features present in both the a and ac datasets.
# IgG_arg_ctl exists in a but not ac — excluded so all pairs share identical schema.
BASE_FEATURE_COLS = [
    "rhf", "acpa", "IgG_cit", "IgG_car", "IgG_ace", "IgG_lys_ctl",
    "IgA_cit", "IgA_arg_ctl", "IgA_car", "IgA_ace", "IgA_lys_ctl",
]

# Full 12-feature order for the a dataset (LR / EN models)
A_FEAT_12 = [
    "rhf", "acpa", "IgG_cit", "IgG_arg_ctl", "IgG_car", "IgG_ace",
    "IgG_lys_ctl", "IgA_cit", "IgA_arg_ctl", "IgA_car", "IgA_ace", "IgA_lys_ctl",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_base_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset A and return:
      X_12  : DataFrame with all 12 a-features, reset_index (raw scale)
      X_base: DataFrame with the 11 common BASE_FEATURE_COLS (raw scale)
    """
    data = pd.read_pickle(DATA_PATH)
    with open(SETUP_JSON_PATH) as f:
        setup = json.load(f)
    y_values = setup["experiment_setup"]["y"]

    df = data.loc[:, [c for c in data.columns if "_id" not in c]]
    df["outcome"] = y_values
    X_12 = df[A_FEAT_12].reset_index(drop=True)
    X_base = X_12[BASE_FEATURE_COLS]
    return X_12, X_base


def load_ac_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Load dataset AC and return:
      X_ac       : DataFrame with all 25 AC features ([0,1] pre-scaled)
      ac_feat_25 : list of AC feature column names in dataset order
    """
    data = pd.read_pickle(DATA_AC_PATH)
    with open(SETUP_AC_SVM_JSON) as f:
        setup = json.load(f)
    y_values = setup["experiment_setup"]["y"]

    df = data.loc[:, [c for c in data.columns if "_id" not in c]]
    df["outcome"] = y_values
    ac_feat_25 = [c for c in df.columns if c != "outcome"]
    X_ac = df[ac_feat_25].reset_index(drop=True)
    return X_ac, ac_feat_25


def load_cache(cache_path: Path) -> list:
    if not cache_path.exists():
        raise FileNotFoundError(f"CFE cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    n_ok = sum(r is not None for r in data)
    print(f"  Loaded {cache_path.name} ({n_ok}/{len(data)} succeeded)")
    return data


# ---------------------------------------------------------------------------
# CFE projection helpers
# ---------------------------------------------------------------------------

def project_lr_en_to_base(cfe_results: list) -> list:
    """
    Drop IgG_arg_ctl (index 3 in A_FEAT_12) from each LR/EN CFE array,
    leaving the 11 BASE_FEATURE_COLS in the same relative order.
    """
    drop_idx = A_FEAT_12.index("IgG_arg_ctl")
    keep = [i for i in range(len(A_FEAT_12)) if i != drop_idx]
    out = []
    for cfe_list in cfe_results:
        if cfe_list is None:
            out.append(None)
        else:
            out.append([arr[keep] for arr in cfe_list])
    return out


def project_ac_svm_to_base(cfe_results: list, ac_feat_25: list, scaler) -> list:
    """
    Extract the 11 BASE_FEATURE_COLS from each AC-SVM CFE array (25 features,
    [0,1] scale) and inverse-scale them to the raw a-dataset scale using the
    fitted MinMaxScaler from the a-lr pipeline.

    The resulting arrays are in the same raw scale as the LR/EN CFE arrays.
    """
    # Indices of BASE_FEATURE_COLS in the ac 25-feature list
    base_idx_in_ac = [ac_feat_25.index(f) for f in BASE_FEATURE_COLS]

    # Indices of BASE_FEATURE_COLS in the a-lr 12-feature scaler (A_FEAT_12)
    scaler_idx = [A_FEAT_12.index(f) for f in BASE_FEATURE_COLS]
    data_min = scaler.data_min_[scaler_idx]
    data_range = scaler.data_range_[scaler_idx]

    out = []
    for cfe_list in cfe_results:
        if cfe_list is None:
            out.append(None)
        else:
            projected = []
            for arr in cfe_list:
                scaled_11 = arr[base_idx_in_ac]
                raw_11 = scaled_11 * data_range + data_min
                projected.append(raw_11)
            out.append(projected)
    return out


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------

def compute_normalised_deltas(
    cfe_results: list,
    X_base: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Compute range-normalised deltas for one model's CFE batch.
    All deltas are normalised by the base-dataset (a) feature ranges so that
    comparisons across model pairs are on a consistent scale.
    """
    feature_ranges = X_base.max().values - X_base.min().values
    feature_ranges[feature_ranges == 0] = np.nan

    rows = []
    for patient_idx, cfe_list in enumerate(cfe_results):
        if cfe_list is None:
            continue
        original = X_base.iloc[patient_idx].values
        for cf_idx, cf_values in enumerate(cfe_list):
            raw_delta = cf_values - original
            norm_delta = raw_delta / feature_ranges
            row = {"patient_idx": patient_idx, "model": model_name, "cf_number": cf_idx}
            for i, feat in enumerate(BASE_FEATURE_COLS):
                row[f"{feat}_raw"] = float(raw_delta[i])
                row[f"{feat}_norm"] = float(norm_delta[i])
            rows.append(row)
    return pd.DataFrame(rows)


def compute_pca_results(
    cfe_a: list,
    cfe_b: list,
    predict_proba_fn_a,
    predict_class_fn_a,
    predict_proba_fn_b,
    predict_class_fn_b,
    X_base: pd.DataFrame,
    model_name_a: str,
    model_name_b: str,
) -> pd.DataFrame:
    """
    Robust z-score all CFE data (pooled across both models in the pair),
    then run PCA per (patient, model) on the n_cfe × 11 standardised matrix.

    predict_*_fn callables accept a DataFrame with BASE_FEATURE_COLS columns
    in raw scale and return numpy arrays of predictions / probabilities.
    """
    from sklearn.decomposition import PCA
    from scipy.stats import iqr

    # Pool all CFE arrays for global z-score normalisation
    pool_rows = []
    for cfe_list in cfe_a + cfe_b:
        if cfe_list is None:
            continue
        pool_rows.extend(cfe_list)
    pool = np.array(pool_rows)
    global_median = np.median(pool, axis=0)
    global_iqr = iqr(pool, axis=0)
    global_iqr[global_iqr == 0] = 1.0

    rows = []
    for model_name, cfe_results, predict_proba_fn, predict_class_fn in [
        (model_name_a, cfe_a, predict_proba_fn_a, predict_class_fn_a),
        (model_name_b, cfe_b, predict_proba_fn_b, predict_class_fn_b),
    ]:
        for patient_idx, cfe_list in enumerate(cfe_results):
            if cfe_list is None:
                continue
            mat = np.array(cfe_list)
            z = (mat - global_median) / global_iqr

            pca = PCA(n_components=1)
            pca.fit(z)
            pc1 = pca.components_[0]
            pc1_ratio = float(pca.explained_variance_ratio_[0])
            mean_vec = z.mean(axis=0)

            orig_df = pd.DataFrame([X_base.iloc[patient_idx].values], columns=BASE_FEATURE_COLS)
            orig_class = int(predict_class_fn(orig_df)[0])
            target_class = 1 - orig_class

            cfe_df = pd.DataFrame(mat, columns=BASE_FEATURE_COLS)
            proba = predict_proba_fn(cfe_df)
            mean_confidence = float(proba[:, target_class].mean())

            row = {
                "patient_idx": patient_idx,
                "model": model_name,
                "pc1_ratio": pc1_ratio,
                "mean_confidence": mean_confidence,
            }
            for i in range(len(BASE_FEATURE_COLS)):
                row[f"pc1_v{i}"] = float(pc1[i])
                row[f"mean_v{i}"] = float(mean_vec[i])
            rows.append(row)

    return pd.DataFrame(rows)


def compute_indiv_cfe_batch(
    cfe_a: list,
    cfe_b: list,
    X_base: pd.DataFrame,
    model_name_a: str,
    model_name_b: str,
) -> pd.DataFrame:
    """
    Build a long-format DataFrame of every individual CFE for both models.
    Stores absolute feature values (raw scale) for orig and cfe per feature.
    """
    rows = []
    for model_name, cfe_results in [(model_name_a, cfe_a), (model_name_b, cfe_b)]:
        for patient_idx, cfe_list in enumerate(cfe_results):
            if cfe_list is None:
                continue
            orig_values = X_base.iloc[patient_idx].values
            for cf_idx, cf_arr in enumerate(cfe_list):
                row: dict = {
                    "patient_idx": patient_idx,
                    "model": model_name,
                    "cf_number": cf_idx,
                }
                for i, feat in enumerate(BASE_FEATURE_COLS):
                    row[f"orig_{feat}"] = float(orig_values[i])
                    row[f"cfe_{feat}"] = float(cf_arr[i])
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model spec construction
# ---------------------------------------------------------------------------

def _make_spec_base_model(
    pipeline_path: str,
    full_feature_cols: list,
    fill_raw: dict,
) -> dict:
    """
    Spec for a base-dataset model (LR or EN) whose pipeline accepts raw features
    and handles scaling internally.  fill_raw contains any features not in the
    11-base set that must be added before calling the pipeline.
    """
    return {
        "pipeline_path": pipeline_path,
        "full_feature_cols": full_feature_cols,
        "needs_scale_base": False,
        "feature_fill_raw": fill_raw,
        "scaler_data_min": None,
        "scaler_data_range": None,
        "extra_feature_fill": None,
    }


def _make_spec_ac_svm(
    pipeline_path: str,
    ac_feat_25: list,
    X_ac: pd.DataFrame,
    scaler,
) -> dict:
    """
    Spec for the AC-SVM model.  The SVM expects 25 pre-scaled [0,1] features.
    To call it from the 11-feature raw base space we:
      1. Scale the 11 base features using the a-lr scaler params
      2. Add the 14 clinical features at their ac-dataset medians (already [0,1])
    """
    scaler_idx = [A_FEAT_12.index(f) for f in BASE_FEATURE_COLS]
    extra_feats = [f for f in ac_feat_25 if f not in BASE_FEATURE_COLS]
    extra_medians = {f: float(X_ac[f].median()) for f in extra_feats}
    return {
        "pipeline_path": pipeline_path,
        "full_feature_cols": ac_feat_25,
        "needs_scale_base": True,
        "feature_fill_raw": {},
        "scaler_data_min": scaler.data_min_[scaler_idx].tolist(),
        "scaler_data_range": scaler.data_range_[scaler_idx].tolist(),
        "extra_feature_fill": extra_medians,
    }


# ---------------------------------------------------------------------------
# Predict function builders (used in process_cfe for PCA; shared with figures)
# ---------------------------------------------------------------------------

def make_predict_fns(spec: dict, pipeline):
    """
    Return (predict_proba_fn, predict_class_fn) from a model spec.

    Both functions accept a DataFrame with BASE_FEATURE_COLS columns in raw
    scale and return numpy arrays (float proba / int class).
    """
    full_cols = spec["full_feature_cols"]

    if not spec["needs_scale_base"]:
        fill = spec["feature_fill_raw"]

        def _full(X_base_df):
            X = X_base_df[BASE_FEATURE_COLS].copy()
            for feat, val in fill.items():
                X[feat] = val
            return X[full_cols]

        def predict_proba(X_base_df):
            return pipeline.predict_proba(_full(X_base_df))

        def predict_class(X_base_df):
            return pipeline.predict(_full(X_base_df)).astype(int)
    else:
        data_min = np.array(spec["scaler_data_min"])
        data_range = np.array(spec["scaler_data_range"])
        extra_fill = spec["extra_feature_fill"]

        def _full(X_base_df):
            raw = X_base_df[BASE_FEATURE_COLS].values.astype(float)
            scaled = (raw - data_min) / data_range
            X = pd.DataFrame(scaled, columns=BASE_FEATURE_COLS)
            for feat, val in extra_fill.items():
                X[feat] = val
            return X[full_cols]

        def predict_proba(X_base_df):
            return pipeline.predict_proba(_full(X_base_df))

        def predict_class(X_base_df):
            return pipeline.predict(_full(X_base_df)).astype(int)

    return predict_proba, predict_class


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_pair_schemas(pairs: dict) -> None:
    """Assert that every pair's DataFrames have the same columns."""
    ref_key = next(iter(pairs))
    ref = pairs[ref_key]
    ref_ad_cols = sorted(ref["all_deltas"].columns)
    ref_pca_cols = sorted(ref["pca_results"].columns)
    ref_ib_cols = sorted(ref["indiv_cfe_batch"].columns)

    for name, pair in pairs.items():
        assert sorted(pair["all_deltas"].columns) == ref_ad_cols, \
            f"all_deltas columns mismatch for pair '{name}'"
        assert sorted(pair["pca_results"].columns) == ref_pca_cols, \
            f"pca_results columns mismatch for pair '{name}'"
        assert sorted(pair["indiv_cfe_batch"].columns) == ref_ib_cols, \
            f"indiv_cfe_batch columns mismatch for pair '{name}'"

    print(f"  Schema validation passed for {len(pairs)} pairs.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading base dataset (A)...")
    X_12, X_base = load_base_data()
    print(f"  {len(X_base)} patients, base features: {BASE_FEATURE_COLS}")

    print("Loading AC dataset...")
    X_ac, ac_feat_25 = load_ac_data()
    print(f"  {len(X_ac)} patients, {len(ac_feat_25)} AC features.")

    print("Loading model pipelines...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr_pipeline = joblib.load(MODELS_DIR / "a-lr.pkl")
        en_pipeline = joblib.load(MODELS_DIR / "a-en.pkl")
        ac_svm = joblib.load(MODELS_DIR / "ac-svm.pkl")

    scaler = lr_pipeline.named_steps["columntransformer"].named_transformers_["num"]

    print("Loading 20-CFE caches...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfe_lr_20 = load_cache(CACHE_LR_20)
        cfe_en_20 = load_cache(CACHE_EN_20)

    has_ac_svm = CACHE_AC_SVM_20.exists()
    if has_ac_svm:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfe_ac_svm_20 = load_cache(CACHE_AC_SVM_20)
    else:
        print(f"  {CACHE_AC_SVM_20.name} not found — AC-SVM pairs will be skipped.")
        cfe_ac_svm_20 = None

    # Project all batches to 11 common base features in raw scale
    print("Projecting CFE batches to 11 base features...")
    cfe_lr_base = project_lr_en_to_base(cfe_lr_20)
    cfe_en_base = project_lr_en_to_base(cfe_en_20)
    if cfe_ac_svm_20 is not None:
        cfe_ac_base = project_ac_svm_to_base(cfe_ac_svm_20, ac_feat_25, scaler)
    else:
        cfe_ac_base = None

    # Build model specs (used for boundary predict functions in the dashapp)
    igG_arg_ctl_median = float(X_12["IgG_arg_ctl"].median())
    spec_lr = _make_spec_base_model(
        "models/a-lr.pkl", A_FEAT_12, {"IgG_arg_ctl": igG_arg_ctl_median}
    )
    spec_en = _make_spec_base_model(
        "models/a-en.pkl", A_FEAT_12, {"IgG_arg_ctl": igG_arg_ctl_median}
    )
    spec_ac = _make_spec_ac_svm("models/ac-svm.pkl", ac_feat_25, X_ac, scaler)

    # Build predict functions for PCA computation
    proba_lr, class_lr = make_predict_fns(spec_lr, lr_pipeline)
    proba_en, class_en = make_predict_fns(spec_en, en_pipeline)
    if cfe_ac_base is not None:
        proba_ac, class_ac = make_predict_fns(spec_ac, ac_svm)

    pairs: dict = {}

    # ---- Pair 1: LR vs EN ------------------------------------------------
    print("\nBuilding pair: LR vs EN")
    pairs["LR vs EN"] = {
        "all_deltas": _fix_str(pd.concat([
            compute_normalised_deltas(cfe_lr_base, X_base, "logistic_regression"),
            compute_normalised_deltas(cfe_en_base, X_base, "elastic_net"),
        ], ignore_index=True)),
        "X": X_base.copy(),
        "pca_results": _fix_str(compute_pca_results(
            cfe_lr_base, cfe_en_base,
            proba_lr, class_lr, proba_en, class_en,
            X_base, "logistic_regression", "elastic_net",
        )),
        "indiv_cfe_batch": _fix_str(compute_indiv_cfe_batch(
            cfe_lr_base, cfe_en_base, X_base, "logistic_regression", "elastic_net",
        )),
        "model_a_name": "logistic_regression",
        "model_b_name": "elastic_net",
        "spec_a": spec_lr,
        "spec_b": spec_en,
    }

    # ---- Pairs involving AC-SVM (only if batch exists) -------------------
    if cfe_ac_base is not None:
        print("\nBuilding pair: LR vs AC-SVM")
        pairs["LR vs AC-SVM"] = {
            "all_deltas": _fix_str(pd.concat([
                compute_normalised_deltas(cfe_lr_base, X_base, "logistic_regression"),
                compute_normalised_deltas(cfe_ac_base, X_base, "ac_svm"),
            ], ignore_index=True)),
            "X": X_base.copy(),
            "pca_results": _fix_str(compute_pca_results(
                cfe_lr_base, cfe_ac_base,
                proba_lr, class_lr, proba_ac, class_ac,
                X_base, "logistic_regression", "ac_svm",
            )),
            "indiv_cfe_batch": _fix_str(compute_indiv_cfe_batch(
                cfe_lr_base, cfe_ac_base, X_base, "logistic_regression", "ac_svm",
            )),
            "model_a_name": "logistic_regression",
            "model_b_name": "ac_svm",
            "spec_a": spec_lr,
            "spec_b": spec_ac,
        }

        print("\nBuilding pair: EN vs AC-SVM")
        pairs["EN vs AC-SVM"] = {
            "all_deltas": _fix_str(pd.concat([
                compute_normalised_deltas(cfe_en_base, X_base, "elastic_net"),
                compute_normalised_deltas(cfe_ac_base, X_base, "ac_svm"),
            ], ignore_index=True)),
            "X": X_base.copy(),
            "pca_results": _fix_str(compute_pca_results(
                cfe_en_base, cfe_ac_base,
                proba_en, class_en, proba_ac, class_ac,
                X_base, "elastic_net", "ac_svm",
            )),
            "indiv_cfe_batch": _fix_str(compute_indiv_cfe_batch(
                cfe_en_base, cfe_ac_base, X_base, "elastic_net", "ac_svm",
            )),
            "model_a_name": "elastic_net",
            "model_b_name": "ac_svm",
            "spec_a": spec_en,
            "spec_b": spec_ac,
        }

    print("\nValidating schemas across pairs...")
    validate_pair_schemas(pairs)

    print("Saving cfe_data.pkl...")
    cfe_data = {"base_feature_cols": BASE_FEATURE_COLS, "pairs": pairs}
    with open(CFE_DATA_PATH, "wb") as f:
        pickle.dump(cfe_data, f)
    print(f"  Saved {len(pairs)} pairs to {CFE_DATA_PATH}")
    for name, pair in pairs.items():
        n = len(pair["all_deltas"])
        print(f"    {name}: {n} all_deltas rows")


def _fix_str(df: pd.DataFrame) -> pd.DataFrame:
    """Cast Arrow-backed string columns to object dtype for pickle compatibility."""
    str_cols = df.select_dtypes(exclude="number").columns
    if len(str_cols):
        df = df.copy()
        df[str_cols] = df[str_cols].astype(object)
    return df


if __name__ == "__main__":
    main()
