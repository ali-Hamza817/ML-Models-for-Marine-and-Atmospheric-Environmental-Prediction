#!/usr/bin/env python3
"""
Label permutation sanity check with configurable K and p-value.
Outputs results to outputs/tables by default.
"""

import argparse
from pathlib import Path
import os
import yaml

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
RANDOM_SPLIT_DATASETS = {"biotoxin", "cast", "phyto_wide", "phyto_long"}

def _load_config():
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

CONFIG = _load_config()

def _get_date_col(dataset_name, df):
    cfg = CONFIG.get("datasets", {}).get(dataset_name, {})
    date_col = cfg.get("date_col")
    if date_col and date_col in df.columns:
        return date_col
    for col in df.columns:
        col_lower = col.lower()
        if "date" in col_lower or "time" in col_lower:
            return col
    return None

def _is_time_ordered(dataset_name, df):
    return _get_date_col(dataset_name, df) is not None

def _split_indices(df, y, dataset_name, seed, train_ratio=0.70, val_ratio=0.15):
    n = len(df)
    if n < 10:
        idx = np.arange(n)
        return idx, idx, idx, "insufficient"

    test_ratio = 1.0 - train_ratio - val_ratio
    date_col = _get_date_col(dataset_name, df)
    if dataset_name in RANDOM_SPLIT_DATASETS or date_col is None:
        # stratified random split on target quantiles
        q = min(10, max(3, n // 200))
        try:
            bins = pd.qcut(y, q=q, duplicates="drop")
        except Exception:
            bins = pd.qcut(y.rank(method="first"), q=q, duplicates="drop")

        sss1 = StratifiedShuffleSplit(
            n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed
        )
        train_idx, temp_idx = next(sss1.split(np.zeros(n), bins))

        bins_temp = bins.iloc[temp_idx]
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=seed
        )
        val_rel, test_rel = next(sss2.split(np.zeros(len(temp_idx)), bins_temp))
        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]
        return train_idx, val_idx, test_idx, "stratified-random-70/15/15"

    # chronological split
    try:
        date_values = pd.to_datetime(df[date_col], errors="coerce").values
        order_idx = np.argsort(date_values)
    except Exception:
        order_idx = np.arange(n)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    train_idx = order_idx[:train_end]
    val_idx = order_idx[train_end:val_end]
    test_idx = order_idx[val_end:]
    return train_idx, val_idx, test_idx, "chronological-70/15/15"


def get_target_column(dataset_name, df):
    target_mapping = {
        "biotoxin": "VALUE",
        "cast": "Bottom_D",
        "era5_daily": "wind10",
        "cleaned_data": "G2chla",
        "rolling_mean": "G2chla",
        "processed_seq": "G2chla",
        "hydrographic": "CHLOROPHYLL-a (µg l-1)",
        "phyto_wide": "Pseudo-nitzschia americana/brasiliana (cells l-1)",
        "phyto_long": "GYMNODINIALES Karlodinium-like",
    }

    target_col = target_mapping.get(dataset_name)

    # Hydrographic must use one consistent chlorophyll target across all models/tables.
    # Prefer explicit chlorophyll columns and avoid being overwritten by generic fallbacks.
    if dataset_name == "hydrographic":
        if target_col in df.columns:
            return target_col
        hydro_preferred = [
            col for col in df.columns
            if ("chlorophyll" in col.lower()) and ("g2chla" not in col.lower())
        ]
        if hydro_preferred:
            return hydro_preferred[0]
        if "G2chla" in df.columns:
            return "G2chla"

    if target_col not in df.columns:
        possible_targets = ["G2chla", "chla", "target", "y", "VALUE"]
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break

        if target_col not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            target_col = numeric_cols[-1] if len(numeric_cols) > 0 else None

    return target_col


def prepare_features(dataset_name, df, target_col):
    exclude_cols = ["Date", "date", "time", "Time", target_col]

    if dataset_name == "cast":
        geo_features = [
            "Lat_Dec", "Lat_Deg", "Lat_Min", "Lat_Hem",
            "Lon_Dec", "Lon_Deg", "Lon_Min", "Lon_Hem",
            "Rpt_Line", "St_Line", "Ac_Line",
            "Rpt_Sta", "St_Station", "Ac_Sta",
            "Sta_ID", "Sta_Code", "Orig_Sta_ID",
            "Cruise_ID", "Cast_ID", "DbSta_ID",
        ]
        exclude_cols.extend(geo_features)
    elif dataset_name in ["cleaned_data", "rolling_mean"]:
        suspicious_cols = [col for col in df.columns if "chla" in col.lower() and col != target_col]
        exclude_cols.extend(suspicious_cols)
    elif dataset_name == "era5_daily":
        wind_cols = [col for col in df.columns if "wind" in col.lower() and col != target_col]
        exclude_cols.extend(wind_cols)

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    return X, y


def _xgb_params():
    params = {}
    use_gpu = os.getenv("XGB_USE_GPU", "0") == "1"
    if use_gpu:
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
        params["gpu_id"] = 0
    return params, use_gpu


def _fit_model(model_type, random_seed, xgb_params=None):
    if model_type == "xgb":
        params = xgb_params or {}
        return xgb.XGBRegressor(n_estimators=50, random_state=random_seed, verbosity=0, **params)
    if model_type == "ridge":
        return Ridge(alpha=1.0, random_state=random_seed)
    raise ValueError(f"Unknown model_type: {model_type}")


def permutation_sanity_check(X, y, dataset_name, n_permutations, random_seed, model_type, split_rule):
    print(f"?? {dataset_name}: {X['train'].shape[0] + X['test'].shape[0]} samples, {X['train'].shape[1]} features")

    X_train, X_test = X["train"], X["test"]
    y_train, y_test = y["train"], y["test"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    xgb_params, use_gpu = _xgb_params()
    if model_type == "xgb" and use_gpu:
        print("   XGB GPU enabled via XGB_USE_GPU=1")

    try:
        model = _fit_model(model_type, random_seed, xgb_params=xgb_params)
        model.fit(X_train_scaled, y_train)
    except Exception as exc:
        if model_type == "xgb" and use_gpu:
            print(f"   XGB GPU failed, fallback to CPU: {exc}")
            xgb_params = {}
            model = _fit_model(model_type, random_seed, xgb_params=xgb_params)
            model.fit(X_train_scaled, y_train)
        else:
            raise

    y_pred = model.predict(X_test_scaled)
    original_r2 = r2_score(y_test, y_pred)

    permuted_r2_scores = []
    y_values = np.concatenate([y_train.values, y_test.values])
    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y_values)
        y_train_perm = y_permuted[:len(y_train)]
        y_test_perm = y_permuted[len(y_train):]

        model_perm = _fit_model(model_type, random_seed, xgb_params=xgb_params)
        model_perm.fit(X_train_scaled, y_train_perm)
        y_pred_perm = model_perm.predict(X_test_scaled)
        permuted_r2_scores.append(r2_score(y_test_perm, y_pred_perm))

    permuted_r2_scores = np.array(permuted_r2_scores, dtype=float)
    permuted_r2_mean = float(np.mean(permuted_r2_scores))

    b = int(np.sum(np.abs(permuted_r2_scores) >= abs(original_r2)))
    p_value = (b + 1) / (n_permutations + 1)

    pass_check = abs(permuted_r2_mean) < 0.15

    print(f"   original R2: {original_r2:.4f}")
    print(f"   permuted R2: {permuted_r2_mean:.4f}")
    print(f"   p-value: {p_value:.6f}")
    print(f"   status: {'PASS' if pass_check else 'FAIL'}")

    return {
        "dataset": dataset_name,
        "original_r2": original_r2,
        "permuted_r2": permuted_r2_mean,
        "p_value": p_value,
        "pass_sanity_check": pass_check,
        "n_features": X_train.shape[1],
        "n_samples": X_train.shape[0] + X_test.shape[0],
        "model_type": model_type,
        "n_permutations": n_permutations,
        "split_rule": split_rule,
    }

def run_all_datasets(n_permutations, random_seed, model_type, max_samples, datasets=None):
    all_datasets = [
        "biotoxin", "cast", "era5_daily", "cleaned_data",
        "rolling_mean", "processed_seq", "hydrographic",
        "phyto_wide", "phyto_long",
    ]

    if datasets:
        requested = [d.strip() for d in datasets if d.strip()]
        dataset_list = [d for d in all_datasets if d in requested]
        if not dataset_list:
            dataset_list = requested
    else:
        dataset_list = all_datasets

    results = []
    for dataset in dataset_list:
        print(f"\nChecking {dataset}")
        data_path = REPO_ROOT / "data" / "processed" / dataset / "clean.csv"
        try:
            df = pd.read_csv(data_path)
            if max_samples and len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=random_seed)
            target_col = get_target_column(dataset, df)
            if target_col is None or target_col not in df.columns:
                print("   target column not found")
                continue

            X_all, y_all = prepare_features(dataset, df, target_col)
            train_idx, val_idx, test_idx, split_rule = _split_indices(
                df, y_all, dataset, random_seed
            )
            X = {
                "train": X_all.iloc[train_idx],
                "test": X_all.iloc[test_idx],
            }
            y = {
                "train": y_all.iloc[train_idx],
                "test": y_all.iloc[test_idx],
            }

            if X["train"].shape[0] < 50 or X["test"].shape[0] < 10:
                print(f"   insufficient samples: train={X['train'].shape[0]}, test={X['test'].shape[0]}")
                results.append({
                    "dataset": dataset,
                    "original_r2": np.nan,
                    "permuted_r2": np.nan,
                    "p_value": np.nan,
                    "pass_sanity_check": False,
                    "n_features": X['train'].shape[1] if X['train'].shape[0] > 0 else 0,
                    "n_samples": X['train'].shape[0] + X['test'].shape[0],
                    "model_type": model_type,
                    "n_permutations": n_permutations,
                    "split_rule": split_rule,
                    "error": "Insufficient samples",
                })
                continue

            result = permutation_sanity_check(
                X, y, dataset,
                n_permutations=n_permutations,
                random_seed=random_seed,
                model_type=model_type,
                split_rule=split_rule,
            )
            results.append(result)
        except Exception as exc:
            print(f"   error: {exc}")
            results.append({
                "dataset": dataset,
                "original_r2": np.nan,
                "permuted_r2": np.nan,
                "p_value": np.nan,
                "pass_sanity_check": False,
                "n_features": 0,
                "n_samples": 0,
                "model_type": model_type,
                "n_permutations": n_permutations,
                "error": str(exc),
            })

    return results

def generate_report(results, output_path):
    results_df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved: {output_path}")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Permutation sanity check with configurable K")
    parser.add_argument("--n-permutations", type=int, default=10000, help="Permutation count (K)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", choices=["xgb", "ridge"], default="xgb", help="Model type")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional subsample size for speed")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset list to run (e.g., rolling_mean,cleaned_data,hydrographic)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "outputs" / "tables" / "complete_sanity_check_results.csv"),
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    datasets = [d for d in (args.datasets.split(",") if args.datasets else [])]
    results = run_all_datasets(
        n_permutations=args.n_permutations,
        random_seed=args.seed,
        model_type=args.model,
        max_samples=args.max_samples,
        datasets=datasets,
    )
    generate_report(results, Path(args.output))


if __name__ == "__main__":
    main()
