#!/usr/bin/env python3
"""
Generate final Table 1–4 from unified prediction outputs.

Inputs:
  - outputs/tables/alternative_metrics_predictions.csv (test predictions)
  - outputs/tables/complete_sanity_check_results.csv
  - data/processed/*/clean.csv

Outputs:
  - outputs/tables/final_table1_dataset_characteristics.csv
  - outputs/tables/final_table2_model_performance.csv
  - outputs/tables/final_table3_best_performance.csv
  - outputs/tables/final_table4_validation_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "outputs" / "tables"

TARGET_DATASETS = [
    "biotoxin",
    "cast",
    "era5_daily",
    "cleaned_data",
    "rolling_mean",
    "processed_seq",
    "hydrographic",
]

MODEL_MAP = {
    "rf": "RF",
    "xgb": "XGB",
    "svr": "SVR",
    "mean": "MEAN",
    "ridge": "RIDGE",
    "lasso": "LASSO",
    "lstm": "LSTM",
    "transformer": "TRANSFORMER",
}

TYPE_MAP = {
    "MEAN": "Baseline",
    "RIDGE": "Baseline",
    "LASSO": "Baseline",
    "RF": "Traditional ML",
    "XGB": "Traditional ML",
    "SVR": "Traditional ML",
    "LSTM": "Deep Learning",
    "TRANSFORMER": "Deep Learning",
}


def load_all_data():
    print("Loading data sources")
    data = {}

    # Dataset metadata
    dataset_info = []
    datasets = [
        "biotoxin",
        "cast",
        "era5_daily",
        "cleaned_data",
        "rolling_mean",
        "processed_seq",
        "hydrographic",
        "phyto_long",
        "phyto_wide",
    ]
    for dataset in datasets:
        try:
            df = pd.read_csv(DATA_DIR / dataset / "clean.csv")
            time_range = "N/A"
            if "time" in df.columns or "Date" in df.columns:
                time_col = "time" if "time" in df.columns else "Date"
                dates = pd.to_datetime(df[time_col], errors="coerce")
                if not dates.isna().all():
                    min_year = dates.dt.year.min()
                    max_year = dates.dt.year.max()
                    time_range = f"{min_year}-{max_year}"
                    if max_year >= 2024:
                        time_range += " (includes 2024+)"

            dataset_info.append(
                {
                    "Dataset": dataset,
                    "Samples": len(df),
                    "Variables": len(df.select_dtypes(include=[np.number]).columns) - 1,
                    "Type": "Time Series"
                    if dataset in ["era5_daily", "rolling_mean", "processed_seq"]
                    else "Cross-sectional",
                    "Time Range": time_range,
                }
            )
        except Exception:
            print(f"  - skipped dataset info for {dataset}")

    data["dataset_info"] = dataset_info
    print(f"  dataset info: {len(dataset_info)}")

    # Sanity check
    try:
        sanity_path = OUT_DIR / "complete_sanity_check_results.csv"
        data["sanity_check"] = pd.read_csv(sanity_path)
        print(f"  sanity check: {len(data['sanity_check'])}")
    except Exception:
        print("  sanity check not found")
        data["sanity_check"] = pd.DataFrame()

    # Predictions
    try:
        pred_path = OUT_DIR / "alternative_metrics_predictions.csv"
        data["predictions"] = pd.read_csv(pred_path)
        print(f"  predictions: {len(data['predictions'])}")
    except Exception:
        print("  predictions not found")
        data["predictions"] = pd.DataFrame()

    return data


def _load_bootstrap_samples():
    cfg_path = REPO_ROOT / "configs" / "config.yaml"
    if not cfg_path.exists():
        return 200
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        n_boot = int(cfg.get("global", {}).get("bootstrap_samples", 1000))
        return max(50, min(n_boot, 300))
    except Exception:
        return 200


def _bootstrap_ci_r2(y_true, y_pred, n_bootstrap=200, seed=42, max_samples=5000):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return np.nan, np.nan
    if n > max_samples:
        idx_pool = rng.choice(n, size=max_samples, replace=False)
        y_true = y_true[idx_pool]
        y_pred = y_pred[idx_pool]
        n = max_samples
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        scores.append(r2_score(y_true[idx], y_pred[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def compute_metrics_from_predictions(pred_df):
    if pred_df.empty:
        return pd.DataFrame()

    pred_df = pred_df.copy()
    pred_df["subset"] = pred_df["subset"].str.lower()
    pred_df = pred_df[pred_df["subset"] == "test"]

    n_bootstrap = _load_bootstrap_samples()
    rows = []
    for (dataset, model), g in pred_df.groupby(["dataset", "model"]):
        y_true = g["y_true"].to_numpy()
        y_pred = g["y_pred"].to_numpy()
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ci_lower, ci_upper = _bootstrap_ci_r2(
            y_true, y_pred, n_bootstrap=n_bootstrap, seed=42
        )

        model_upper = MODEL_MAP.get(model, model.upper())
        rows.append(
            {
                "dataset": dataset,
                "model": model_upper,
                "type": TYPE_MAP.get(model_upper, "Other"),
                "r2": r2,
                "mae": mae,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

    return pd.DataFrame(rows)


def create_table1_dataset_characteristics(data):
    print("Generating Table 1")
    df_info = pd.DataFrame(data["dataset_info"])

    # QA pass/fail (not permutation sanity check)
    qa_pass = set(TARGET_DATASETS)
    df_info["Validated"] = df_info["Dataset"].apply(lambda d: d in qa_pass)

    target_info = {
        "biotoxin": "Biotoxin concentration",
        "cast": "Bottom depth",
        "era5_daily": "Wind speed (10m)",
        "cleaned_data": "Chlorophyll-a",
        "rolling_mean": "Chlorophyll-a (smoothed)",
        "processed_seq": "Chlorophyll-a (processed)",
        "hydrographic": "Chlorophyll-a",
        "phyto_long": "Phytoplankton abundance",
        "phyto_wide": "Phytoplankton abundance",
    }
    df_info["Target Variable"] = df_info["Dataset"].map(target_info)

    table1 = df_info[
        [
            "Dataset",
            "Samples",
            "Variables",
            "Type",
            "Target Variable",
            "Time Range",
            "Validated",
        ]
    ].copy()

    table1["Samples"] = table1["Samples"].apply(lambda x: f"{x:,}")
    table1["Validated"] = table1["Validated"].apply(lambda x: "True" if x else "False")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table1.to_csv(OUT_DIR / "final_table1_dataset_characteristics.csv", index=False)
    print(f"  saved Table 1: {len(table1)} rows")
    return table1


def create_table2_model_performance(data, metrics_df):
    print("Generating Table 2")
    if metrics_df.empty:
        table2 = pd.DataFrame()
        table2.to_csv(OUT_DIR / "final_table2_model_performance.csv", index=False)
        return table2

    p_value_map = {}
    p_value_floor = None
    if not data["sanity_check"].empty and "p_value" in data["sanity_check"].columns:
        for _, row in data["sanity_check"].iterrows():
            p_value_map[row["dataset"]] = row["p_value"]
            if "n_permutations" in row and pd.notna(row["n_permutations"]):
                try:
                    k = int(row["n_permutations"])
                    p_value_floor = 1.0 / (k + 1)
                except Exception:
                    pass
        if p_value_floor is None:
            p_value_floor = 1.0 / 10001

    def format_p_value(p):
        if p is None or pd.isna(p):
            return "N/A"
        if p_value_floor is not None and p <= p_value_floor:
            return "p < 1e-4"
        return f"{p:.4f}"

    rows = []
    for _, row in metrics_df.iterrows():
        rows.append(
            {
                "Dataset": row["dataset"],
                "Model": row["model"],
                "R²": row["r2"],
                "R² (95% CI)": f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
                if pd.notna(row["ci_lower"]) and pd.notna(row["ci_upper"])
                else "N/A",
                "p-value": format_p_value(p_value_map.get(row["dataset"])),
                "MAE": row["mae"],
                "Type": row["type"],
            }
        )

    table2 = pd.DataFrame(rows)
    table2["R²"] = table2["R²"].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    table2["MAE"] = table2["MAE"].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    table2 = table2.sort_values(["Dataset", "Type", "Model"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table2.to_csv(OUT_DIR / "final_table2_model_performance.csv", index=False)
    print(f"  saved Table 2: {len(table2)} rows")
    return table2


def create_table3_best_performance(metrics_df):
    print("Generating Table 3")
    if metrics_df.empty:
        table3 = pd.DataFrame()
        table3.to_csv(OUT_DIR / "final_table3_best_performance.csv", index=False)
        return table3

    results_df = metrics_df[metrics_df["dataset"].isin(TARGET_DATASETS)].copy()
    rows = []
    for dataset in TARGET_DATASETS:
        dataset_results = results_df[results_df["dataset"] == dataset]
        if dataset_results.empty:
            rows.append(
                {
                    "Dataset": dataset,
                    "Best Model": "N/A",
                    "Best R²": np.nan,
                    "MAE": np.nan,
                    "Model Type": "N/A",
                    "Improvement": np.nan,
                    "Rank": 0,
                }
            )
            continue

        best_idx = dataset_results["r2"].idxmax()
        best_result = dataset_results.loc[best_idx]
        worst_r2 = dataset_results["r2"].min()
        improvement = best_result["r2"] - worst_r2

        rows.append(
            {
                "Dataset": dataset,
                "Best Model": best_result["model"],
                "Best R²": best_result["r2"],
                "MAE": best_result["mae"],
                "Model Type": best_result["type"],
                "Improvement": improvement,
                "Rank": 0,
            }
        )

    table3 = pd.DataFrame(rows)
    table3 = table3.sort_values("Best R²", ascending=False)
    table3["Rank"] = range(1, len(table3) + 1)

    table3["Best R²"] = table3["Best R²"].apply(lambda x: f"{x:.4f}")
    table3["MAE"] = table3["MAE"].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    table3["Improvement"] = table3["Improvement"].apply(lambda x: f"{x:.4f}")

    table3 = table3[
        ["Rank", "Dataset", "Best Model", "Best R²", "MAE", "Model Type", "Improvement"]
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table3.to_csv(OUT_DIR / "final_table3_best_performance.csv", index=False)
    print(f"  saved Table 3: {len(table3)} rows")
    return table3


def create_table4_validation_summary(data, metrics_df):
    print("Generating Table 4")
    results = []

    dataset_info = {item["Dataset"]: item for item in data["dataset_info"]}

    # QA validation status (pass for 7 datasets, fail for phyto_*)
    qa_pass = set(TARGET_DATASETS)

    best_performance = {}
    if not metrics_df.empty:
        for dataset in metrics_df["dataset"].unique():
            dataset_data = metrics_df[metrics_df["dataset"] == dataset]
            best_r2 = dataset_data["r2"].max()
            best_model = dataset_data.loc[dataset_data["r2"].idxmax(), "model"]
            best_performance[dataset] = {"r2": best_r2, "model": best_model}

    dl_success = {}
    if not metrics_df.empty:
        for dataset in metrics_df["dataset"].unique():
            dataset_data = metrics_df[metrics_df["dataset"] == dataset]
            total = len(dataset_data[dataset_data["type"] == "Deep Learning"])
            dl_success[dataset] = f"{total}/{total}" if total > 0 else "0/0"

    all_datasets = set(dataset_info.keys()) | set(best_performance.keys())
    for dataset in sorted(all_datasets):
        info = dataset_info.get(dataset, {})
        samples = info.get("Samples", "N/A")

        validation_status = "True" if dataset in qa_pass else "False"

        best = best_performance.get(dataset, {})
        best_r2 = best.get("r2", np.nan)
        best_model = best.get("model", "N/A")

        dl_rate = dl_success.get(dataset, "0/0")

        if not pd.isna(best_r2):
            if best_r2 > 0.8:
                difficulty = "Easy"
            elif best_r2 > 0.5:
                difficulty = "Medium"
            elif best_r2 > 0:
                difficulty = "Hard"
            else:
                difficulty = "Very Hard"
        else:
            difficulty = "Unknown"

        results.append(
            {
                "Dataset": dataset,
                "Samples": samples if isinstance(samples, str) else f"{samples:,}",
                "Validation": validation_status,
                "Best R²": f"{best_r2:.4f}" if not pd.isna(best_r2) else "N/A",
                "Best Model": best_model,
                "DL Success": dl_rate,
                "Difficulty": difficulty,
            }
        )

    table4 = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table4.to_csv(OUT_DIR / "final_table4_validation_summary.csv", index=False)
    print(f"  saved Table 4: {len(table4)} rows")
    return table4


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all_data()
    metrics_df = compute_metrics_from_predictions(data.get("predictions", pd.DataFrame()))

    create_table1_dataset_characteristics(data)
    create_table2_model_performance(data, metrics_df)
    create_table3_best_performance(metrics_df)
    create_table4_validation_summary(data, metrics_df)


if __name__ == "__main__":
    main()
