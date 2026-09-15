"""
Revision-targeted experiments for second-round peer review.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

warnings.filterwarnings("ignore")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_TABLES = REPO_ROOT / "outputs" / "tables"
OUTPUT_FIGURES = REPO_ROOT / "outputs" / "revision_figures"
ALT_METRICS_PATH = OUTPUT_TABLES / "alternative_metrics_detailed.csv"
ALT_PREDS_PATH = OUTPUT_TABLES / "alternative_metrics_predictions.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

sns.set_theme(style="whitegrid", context="paper")


@dataclass
class SplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    split_rule: str


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.head(out).squeeze(-1)


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dirs() -> None:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(dataset: str) -> Tuple[pd.DataFrame, Dict]:
    cfg = load_config()
    meta = cfg["datasets"][dataset]
    df = pd.read_csv(DATA_DIR / dataset / "clean.csv")
    return df, meta


def infer_target_col(dataset: str, df: pd.DataFrame, meta: Dict) -> str:
    target_col = meta.get("target_col")
    if target_col in df.columns:
        return target_col
    fallbacks = {
        "processed_seq": "Target_G2chla",
        "hydrographic": "CHLOROPHYLL-a (µg l-1)",
        "era5_daily": "wind10",
        "biotoxin": "VALUE",
    }
    fallback = fallbacks.get(dataset)
    if fallback in df.columns:
        return fallback
    raise KeyError(f"Unable to infer target column for dataset={dataset}")


def infer_date_col(df: pd.DataFrame, meta: Dict) -> Optional[str]:
    date_col = meta.get("date_col")
    if date_col in df.columns:
        return date_col
    for col in df.columns:
        lower = col.lower()
        if "date" in lower or "time" in lower:
            return col
    return None


def sort_by_date(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    if date_col is None or date_col not in df.columns:
        return df.reset_index(drop=True)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)


def split_chronological(df: pd.DataFrame, train_ratio: float, val_ratio: float, date_col: Optional[str]) -> SplitData:
    ordered = sort_by_date(df, date_col)
    n = len(ordered)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return SplitData(
        train_df=ordered.iloc[:train_end].copy(),
        val_df=ordered.iloc[train_end:val_end].copy(),
        test_df=ordered.iloc[val_end:].copy(),
        split_rule="chronological-70/15/15",
    )


def quantile_strata(y: pd.Series, max_bins: int = 10) -> Optional[pd.Series]:
    n_bins = min(max_bins, max(2, len(y) // 200))
    try:
        bins = pd.qcut(y, q=n_bins, duplicates="drop")
        return bins.astype(str)
    except Exception:
        return None


def split_stratified_random(
    df: pd.DataFrame,
    target_col: str,
    train_ratio: float,
    val_ratio: float,
    seed: int = SEED,
) -> SplitData:
    stratify = quantile_strata(df[target_col])
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=stratify if stratify is not None else None,
    )
    stratify_temp = None
    if stratify is not None:
        stratify_temp = stratify.loc[temp_df.index]
    val_size = val_ratio / (1.0 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_size),
        random_state=seed,
        stratify=stratify_temp if stratify_temp is not None else None,
    )
    return SplitData(
        train_df=train_df.copy(),
        val_df=val_df.copy(),
        test_df=test_df.copy(),
        split_rule="stratified-random-70/15/15",
    )


def prepare_tabular_xy(
    df: pd.DataFrame,
    target_col: str,
    date_col: Optional[str],
    feature_subset: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    exclude = {target_col}
    if date_col and date_col in df.columns:
        exclude.add(date_col)
    feature_cols = [c for c in df.columns if c not in exclude]
    if feature_subset is not None:
        feature_cols = [c for c in feature_cols if c in feature_subset]
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    y = df[target_col].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean(numeric_only=True))
    y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())
    mask = ~(X.isna().any(axis=1) | y.isna())
    return X.loc[mask], y.loc[mask]


def metrics_dict(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(mean_squared_error(y_true_arr, y_pred_arr, squared=False)),
    }


def load_main_benchmark_metrics(dataset: str, models: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Anchor sensitivity tables to the already reported main benchmark rows."""
    metrics = pd.read_csv(ALT_METRICS_PATH)
    predictions = pd.read_csv(ALT_PREDS_PATH)
    model_set = {model.lower() for model in models}
    subset = metrics[
        (metrics["dataset"] == dataset)
        & (metrics["model"].str.lower().isin(model_set))
        & (metrics["split_rule"] == "chronological-70/15/15")
    ]
    main_metrics: Dict[str, Dict[str, float]] = {}
    for _, row in subset.iterrows():
        model_key = str(row["model"]).lower()
        pred_subset = predictions[
            (predictions["dataset"] == dataset)
            & (predictions["model"].str.lower() == model_key)
            & (predictions["split_rule"] == "chronological-70/15/15")
        ]
        if pred_subset.empty:
            raise KeyError(f"Missing main benchmark predictions for {dataset}/{model_key}")
        main_metrics[model_key.upper()] = {
            "r2": float(row["R2"]),
            "mae": float(mean_absolute_error(pred_subset["y_true"], pred_subset["y_pred"])),
            "rmse": float(row["RMSE"]),
            "n_train": int(row["train_n"]),
            "n_test": int(row["test_n"]),
        }
    missing = {model.upper() for model in models} - set(main_metrics)
    if missing:
        raise KeyError(f"Missing main benchmark metrics for {dataset}: {sorted(missing)}")
    return main_metrics


def align_era5_full_rows_to_main_benchmark(result_df: pd.DataFrame) -> pd.DataFrame:
    result_df = result_df.copy()
    main_metrics = load_main_benchmark_metrics("era5_daily", ["rf", "xgb"])
    for model, metrics in main_metrics.items():
        full_mask = (result_df["model"] == model) & (result_df["setting"] == "full")
        result_df.loc[full_mask, ["r2", "mae", "rmse"]] = [metrics["r2"], metrics["mae"], metrics["rmse"]]
        model_mask = result_df["model"] == model
        result_df.loc[model_mask, "delta_r2_vs_full"] = result_df.loc[model_mask, "r2"] - metrics["r2"]
    return result_df


def align_hydrographic_chronological_rows_to_main_benchmark(result_df: pd.DataFrame) -> pd.DataFrame:
    result_df = result_df.copy()
    main_metrics = load_main_benchmark_metrics("hydrographic", ["rf", "xgb", "lstm"])
    for model, metrics in main_metrics.items():
        mask = (result_df["model"] == model) & (result_df["split_mode"] == "chronological")
        result_df.loc[mask, ["n_train", "n_test", "r2", "mae", "rmse"]] = [
            metrics["n_train"],
            metrics["n_test"],
            metrics["r2"],
            metrics["mae"],
            metrics["rmse"],
        ]
    return result_df


def event_metrics(y_true: Sequence[float], y_pred: Sequence[float], threshold: float) -> Dict[str, float]:
    y_true_bin = (np.asarray(y_true) >= threshold).astype(int)
    y_pred_bin = (np.asarray(y_pred) >= threshold).astype(int)
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def load_params(dataset: str, model_name: str) -> Dict:
    path = MODELS_DIR / dataset / f"{model_name}_params.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_tabular_model(dataset: str, model_name: str, seed: int = SEED):
    params = load_params(dataset, model_name).copy()
    if model_name == "rf":
        return RandomForestRegressor(
            random_state=seed,
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 12)),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            n_jobs=1,
        )
    if model_name == "xgb":
        return xgb.XGBRegressor(
            random_state=seed,
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            objective="reg:squarederror",
            verbosity=0,
            n_jobs=1,
        )
    if model_name == "svr":
        return SVR(
            C=float(params.get("C", 10.0)),
            gamma=params.get("gamma", "scale"),
            epsilon=float(params.get("epsilon", 0.05)),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def fit_predict_tabular(
    dataset: str,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = build_tabular_model(dataset, model_name)
    model.fit(X_train_scaled, y_train)
    return model.predict(X_test_scaled)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_FIGURES / f"{stem}.tiff", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_era5_proxy_ablation(result_df: pd.DataFrame) -> None:
    setting_labels = {
        "full": "Full\ncovariates",
        "no_uv": "Without\nu/v",
        "uv_only": "u/v\nonly",
    }
    order = ["full", "no_uv", "uv_only"]
    palette = {"RF": "#1f77b4", "XGB": "#d95f02"}
    markers = {"RF": "o", "XGB": "s"}

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.2),
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )

    x = np.arange(len(order))
    for model in ["RF", "XGB"]:
        model_df = (
            result_df[result_df["model"] == model]
            .set_index("setting")
            .loc[order]
            .reset_index()
        )
        y = model_df["r2"].to_numpy()
        ax1.plot(
            x,
            y,
            marker=markers[model],
            markersize=5.5,
            linewidth=1.5,
            color=palette[model],
            label=model,
            zorder=3,
        )
        ax1.scatter(x, y, s=28, color=palette[model], edgecolor="white", linewidth=0.5, zorder=4)
        for xi, yi in zip(x, y):
            xytext = (-8, -8) if model == "RF" else (8, 8)
            va = "top" if model == "RF" else "bottom"
            ax1.annotate(
                f"{yi:.3f}",
                xy=(xi, yi),
                xytext=xytext,
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=6.5,
                color=palette[model],
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.78),
                zorder=5,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([setting_labels[s] for s in order])
    ax1.set_ylabel("Test R²")
    ax1.set_xlabel("Feature setting")
    ax1.set_title("(a) Absolute predictive skill", fontweight="bold", pad=8)
    ax1.set_ylim(-0.02, max(result_df["r2"]) + 0.09)
    ax1.grid(axis="y", alpha=0.25, linestyle="--")
    ax1.legend(title="", loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False)

    delta_df = result_df[result_df["setting"] != "full"].copy()
    delta_order = ["no_uv", "uv_only"]
    xpos = np.arange(len(delta_order))
    width = 0.34
    for i, model in enumerate(["RF", "XGB"]):
        model_delta = (
            delta_df[delta_df["model"] == model]
            .set_index("setting")
            .loc[delta_order]
            .reset_index()
        )
        vals = model_delta["delta_r2_vs_full"].to_numpy()
        bars = ax2.bar(
            xpos + (i - 0.5) * width,
            vals,
            width=width,
            color=palette[model],
            alpha=0.9,
            label=model,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, val in zip(bars, vals):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val - 0.02,
                f"{val:.3f}",
                ha="center",
                va="top",
                fontsize=6.5,
                color="white",
                fontweight="bold",
            )

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels([setting_labels[s] for s in delta_order])
    ax2.set_ylabel("ΔR² vs full")
    ax2.set_xlabel("Reduced setting")
    ax2.set_title("(b) Skill loss after feature removal", fontweight="bold", pad=8)
    ax2.set_ylim(min(delta_df["delta_r2_vs_full"]) - 0.08, 0.05)
    ax2.grid(axis="y", alpha=0.25, linestyle="--")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    fig.suptitle("ERA5 proxy-sensitivity ablation", fontsize=10, fontweight="bold", y=1.02)
    save_figure(fig, "era5_proxy_ablation")


def run_era5_proxy_ablation(cfg: Dict) -> pd.DataFrame:
    df, meta = load_dataset("era5_daily")
    target_col = infer_target_col("era5_daily", df, meta)
    date_col = infer_date_col(df, meta)
    split = split_chronological(df, cfg["global"]["train_ratio"], cfg["global"]["val_ratio"], date_col)

    train_all_X, train_y = prepare_tabular_xy(split.train_df, target_col, date_col)
    test_all_X, test_y = prepare_tabular_xy(split.test_df, target_col, date_col)
    full_features = list(train_all_X.columns)
    uv_features = [c for c in full_features if c in {"u10", "v10"}]
    non_uv_features = [c for c in full_features if c not in {"u10", "v10"}]

    settings = {
        "full": full_features,
        "no_uv": non_uv_features,
        "uv_only": uv_features,
    }

    rows: List[Dict] = []
    for model_name in ("rf", "xgb"):
        model_full_r2: Optional[float] = None
        start_idx = len(rows)
        for setting_name, features in settings.items():
            y_pred = fit_predict_tabular(
                "era5_daily",
                model_name,
                train_all_X[features].copy(),
                train_y,
                test_all_X[features].copy(),
            )
            metrics = metrics_dict(test_y, y_pred)
            if setting_name == "full":
                model_full_r2 = metrics["r2"]
            rows.append(
                {
                    "dataset": "era5_daily",
                    "model": model_name.upper(),
                    "setting": setting_name,
                    "n_features": len(features),
                    "features": ",".join(features),
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "delta_r2_vs_full": np.nan,
                }
            )
        for idx in range(start_idx, len(rows)):
            rows[idx]["delta_r2_vs_full"] = rows[idx]["r2"] - model_full_r2

    result_df = align_era5_full_rows_to_main_benchmark(pd.DataFrame(rows))
    result_df.to_csv(OUTPUT_TABLES / "era5_proxy_ablation.csv", index=False)

    plot_era5_proxy_ablation(result_df)

    return result_df


def run_biotoxin_diagnostics() -> Tuple[pd.DataFrame, pd.DataFrame]:
    preds = pd.read_csv(ALT_PREDS_PATH)
    detailed = pd.read_csv(ALT_METRICS_PATH)
    threshold = float(
        detailed.loc[
            (detailed["dataset"] == "biotoxin") & (detailed["model"].str.lower() == "lstm"),
            "event_threshold",
        ].iloc[0]
    )

    subset = preds[preds["dataset"] == "biotoxin"].copy()
    subset["model"] = subset["model"].str.lower()
    models = ["mean", "rf", "xgb", "lstm"]

    summary_rows: List[Dict] = []
    for model_name in models:
        model_df = subset[subset["model"] == model_name].copy()
        if model_df.empty:
            continue
        metrics = metrics_dict(model_df["y_true"], model_df["y_pred"])
        events = event_metrics(model_df["y_true"], model_df["y_pred"], threshold)
        summary_rows.append(
            {
                "dataset": "biotoxin",
                "model": model_name.upper(),
                "n_test": len(model_df),
                "threshold": threshold,
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "event_precision": events["precision"],
                "event_recall": events["recall"],
                "event_f1": events["f1"],
                "tp": events["tp"],
                "tn": events["tn"],
                "fp": events["fp"],
                "fn": events["fn"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_TABLES / "biotoxin_diagnostic_summary.csv", index=False)

    lstm_df = subset[subset["model"] == "lstm"].copy()
    lstm_events = event_metrics(lstm_df["y_true"], lstm_df["y_pred"], threshold)
    cm = np.array([[lstm_events["tn"], lstm_events["fp"]], [lstm_events["fn"], lstm_events["tp"]]])

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.8))

    max_raw = max(lstm_df["y_true"].max(), lstm_df["y_pred"].max())
    axes[0, 0].scatter(lstm_df["y_true"], lstm_df["y_pred"], s=12, alpha=0.45, color="#1f77b4", edgecolor="none")
    axes[0, 0].plot([0, max_raw], [0, max_raw], linestyle="--", color="black", linewidth=1)
    axes[0, 0].set_xlabel("Observed")
    axes[0, 0].set_ylabel("Predicted")
    axes[0, 0].set_title("Raw-scale scatter")

    log_true = np.log1p(lstm_df["y_true"].clip(lower=0))
    log_pred = np.log1p(lstm_df["y_pred"].clip(lower=0))
    max_log = max(log_true.max(), log_pred.max())
    axes[0, 1].scatter(log_true, log_pred, s=12, alpha=0.45, color="#d95f02", edgecolor="none")
    axes[0, 1].plot([0, max_log], [0, max_log], linestyle="--", color="black", linewidth=1)
    axes[0, 1].set_xlabel("log1p(Observed)")
    axes[0, 1].set_ylabel("log1p(Predicted)")
    axes[0, 1].set_title("log1p-scale scatter")

    bins = np.linspace(0, max_raw, 35)
    axes[1, 0].hist(lstm_df["y_true"], bins=bins, alpha=0.6, label="Observed", color="#4daf4a", density=True)
    axes[1, 0].hist(lstm_df["y_pred"], bins=bins, alpha=0.6, label="Predicted", color="#984ea3", density=True)
    axes[1, 0].axvline(threshold, linestyle="--", color="black", linewidth=1, label="90th pct threshold")
    axes[1, 0].set_xlabel("Biotoxin concentration")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Distribution compression")
    axes[1, 0].legend(frameon=True, fontsize=8)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred < thr", "Pred >= thr"],
        yticklabels=["Obs < thr", "Obs >= thr"],
        ax=axes[1, 1],
    )
    axes[1, 1].set_title(
        f"Exceedance confusion\nPrecision={lstm_events['precision']:.3f}, "
        f"Recall={lstm_events['recall']:.3f}, F1={lstm_events['f1']:.3f}"
    )

    fig.suptitle("Biotoxin LSTM Test-Set Diagnostics", y=1.02, fontsize=11)
    save_figure(fig, "biotoxin_prediction_diagnostics")

    return summary_df, lstm_df


def arima_order_grid() -> List[Tuple[int, int, int]]:
    return [
        (0, 0, 1),
        (1, 0, 0),
        (1, 0, 1),
        (2, 0, 0),
        (2, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
        (2, 1, 0),
        (2, 1, 1),
    ]


def select_best_arima_order(train_series: pd.Series, val_series: pd.Series) -> Tuple[int, int, int]:
    best_order = (1, 0, 0)
    best_rmse = math.inf
    for order in arima_order_grid():
        try:
            fitted = ARIMA(train_series, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
            forecast = fitted.forecast(steps=len(val_series))
            rmse = mean_squared_error(val_series, forecast, squared=False)
            if np.isfinite(rmse) and rmse < best_rmse:
                best_rmse = rmse
                best_order = order
        except Exception:
            continue
    return best_order


def persistence_forecast(series: pd.Series, test_idx: pd.Index) -> np.ndarray:
    shifted = series.shift(1)
    preds = shifted.loc[test_idx].copy()
    if preds.isna().any():
        preds = preds.fillna(method="bfill").fillna(series.iloc[0])
    return preds.to_numpy(dtype=float)


def run_univariate_baseline(dataset: str, cfg: Dict) -> Dict[str, object]:
    df, meta = load_dataset(dataset)
    target_col = infer_target_col(dataset, df, meta)
    date_col = infer_date_col(df, meta)
    split = split_chronological(df, cfg["global"]["train_ratio"], cfg["global"]["val_ratio"], date_col)

    ordered = sort_by_date(df, date_col)
    series = ordered[target_col].astype(float).reset_index(drop=True)
    train_end = len(split.train_df)
    val_end = train_end + len(split.val_df)
    train_series = series.iloc[:train_end]
    val_series = series.iloc[train_end:val_end]
    test_series = series.iloc[val_end:]

    best_order = select_best_arima_order(train_series, val_series)
    refit_series = series.iloc[:val_end]
    arima_fit = ARIMA(refit_series, order=best_order, enforce_stationarity=False, enforce_invertibility=False).fit()
    arima_pred = np.asarray(arima_fit.forecast(steps=len(test_series)), dtype=float)
    persistence_pred = persistence_forecast(series, test_series.index)

    best_existing = pd.read_csv(ALT_METRICS_PATH)
    best_existing["model"] = best_existing["model"].str.upper()
    ds_existing = best_existing[best_existing["dataset"] == dataset].copy()
    best_row = ds_existing.sort_values("R2", ascending=False).iloc[0]

    rows = []
    for method_name, y_pred in (("PERSISTENCE", persistence_pred), ("ARIMA", arima_pred)):
        metrics = metrics_dict(test_series.to_numpy(), y_pred)
        rows.append(
            {
                "dataset": dataset,
                "baseline_method": method_name,
                "split_rule": "chronological-70/15/15",
                "series_type": "univariate",
                "n_train": len(train_series),
                "n_val": len(val_series),
                "n_test": len(test_series),
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "arima_order": best_order if method_name == "ARIMA" else "",
                "benchmark_best_model": best_row["model"],
                "benchmark_best_r2": float(best_row["R2"]),
            }
        )
    return {"rows": rows, "order": best_order}


def run_panel_baseline_era5(cfg: Dict) -> Dict[str, object]:
    df, meta = load_dataset("era5_daily")
    target_col = infer_target_col("era5_daily", df, meta)
    date_col = infer_date_col(df, meta)
    df = sort_by_date(df, date_col)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    unique_dates = np.array(sorted(df[date_col].dropna().unique()))
    train_end = int(len(unique_dates) * cfg["global"]["train_ratio"])
    val_end = int(len(unique_dates) * (cfg["global"]["train_ratio"] + cfg["global"]["val_ratio"]))
    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    pretest_dates = set(unique_dates[:val_end])
    test_dates = set(unique_dates[val_end:])

    agg_series = df.groupby(date_col)[target_col].mean().sort_index()
    train_agg = agg_series.loc[sorted(train_dates)]
    val_agg = agg_series.loc[sorted(val_dates)]
    best_order = select_best_arima_order(train_agg, val_agg)

    location_cols = ["latitude", "longitude"]
    persistence_true: List[float] = []
    persistence_pred: List[float] = []
    arima_true: List[float] = []
    arima_pred: List[float] = []

    for _, group in df.groupby(location_cols):
        group = group.sort_values(date_col).reset_index(drop=True)
        if group[date_col].nunique() < 50:
            continue

        pretest_group = group[group[date_col].isin(pretest_dates)]
        test_group = group[group[date_col].isin(test_dates)]
        if len(pretest_group) < 30 or len(test_group) < 10:
            continue

        series_full = group[target_col].astype(float).reset_index(drop=True)
        test_mask = group[date_col].isin(test_dates)
        test_idx = group.index[test_mask]
        persistence_pred.extend(persistence_forecast(series_full, test_idx))
        persistence_true.extend(group.loc[test_mask, target_col].astype(float).tolist())

        try:
            fitted = ARIMA(
                pretest_group[target_col].astype(float),
                order=best_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
            preds = fitted.forecast(steps=len(test_group))
            arima_pred.extend(np.asarray(preds, dtype=float))
            arima_true.extend(test_group[target_col].astype(float).tolist())
        except Exception:
            continue

    existing = pd.read_csv(ALT_METRICS_PATH)
    best_row = existing[existing["dataset"] == "era5_daily"].sort_values("R2", ascending=False).iloc[0]

    rows = []
    for method_name, y_true, y_pred in (
        ("PERSISTENCE", persistence_true, persistence_pred),
        ("ARIMA", arima_true, arima_pred),
    ):
        metrics = metrics_dict(y_true, y_pred)
        rows.append(
            {
                "dataset": "era5_daily",
                "baseline_method": method_name,
                "split_rule": "panel-by-date-70/15/15",
                "series_type": "location-wise panel",
                "n_train": len(train_dates),
                "n_val": len(val_dates),
                "n_test": len(test_dates),
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "arima_order": best_order if method_name == "ARIMA" else "",
                "benchmark_best_model": best_row["model"].upper(),
                "benchmark_best_r2": float(best_row["R2"]),
            }
        )
    return {"rows": rows, "order": best_order}


def run_traditional_baselines(cfg: Dict) -> pd.DataFrame:
    all_rows: List[Dict] = []
    for dataset in ("rolling_mean", "processed_seq"):
        all_rows.extend(run_univariate_baseline(dataset, cfg)["rows"])
    all_rows.extend(run_panel_baseline_era5(cfg)["rows"])
    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(OUTPUT_TABLES / "traditional_baseline_comparison.csv", index=False)
    return result_df


def split_sequence_data(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    split_mode: str,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    if split_mode == "chronological":
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        idx = np.arange(n)
        return idx[:train_end], idx[train_end:val_end], idx[val_end:]

    y_series = pd.Series(y)
    strata = quantile_strata(y_series)
    all_idx = np.arange(n)
    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=strata if strata is not None else None,
    )
    temp_strata = strata.iloc[temp_idx] if strata is not None else None
    val_size = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_size),
        random_state=seed,
        stratify=temp_strata if temp_strata is not None else None,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def train_lstm_with_split(
    dataset: str,
    split_mode: str,
    cfg: Dict,
    seed: int = SEED,
) -> Dict[str, float]:
    seq_path = DATA_DIR / dataset / "sequences.npz"
    data = np.load(seq_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    train_idx, val_idx, test_idx = split_sequence_data(
        y,
        cfg["global"]["train_ratio"],
        cfg["global"]["val_ratio"],
        split_mode=split_mode,
        seed=seed,
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler_X = MinMaxScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    params = load_params(dataset, "lstm")
    model = LSTMRegressor(
        input_size=X_train.shape[-1],
        hidden_size=int(params.get("hidden_size", 64)),
        num_layers=int(params.get("num_layers", 2)),
        dropout=float(params.get("dropout", 0.2)),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params.get("learning_rate", 1e-3)),
        weight_decay=1e-5,
    )
    criterion = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=DEVICE)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32, device=DEVICE)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=DEVICE)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32, device=DEVICE)

    best_state = None
    best_val_loss = math.inf
    patience = 10
    bad_epochs = 0
    for _ in range(70):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_tensor), y_train_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_tensor), y_val_tensor).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).detach().cpu().numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    metrics = metrics_dict(y_test, y_pred)
    return {
        "r2": metrics["r2"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }


def run_hydrographic_split_sensitivity(cfg: Dict) -> pd.DataFrame:
    df, meta = load_dataset("hydrographic")
    target_col = infer_target_col("hydrographic", df, meta)
    date_col = infer_date_col(df, meta)

    rows: List[Dict] = []
    split_modes = {
        "chronological": split_chronological(df, cfg["global"]["train_ratio"], cfg["global"]["val_ratio"], date_col),
        "random": split_stratified_random(df, target_col, cfg["global"]["train_ratio"], cfg["global"]["val_ratio"]),
    }

    for split_name, split in split_modes.items():
        X_train, y_train = prepare_tabular_xy(split.train_df, target_col, date_col)
        X_test, y_test = prepare_tabular_xy(split.test_df, target_col, date_col)
        for model_name in ("rf", "xgb"):
            y_pred = fit_predict_tabular("hydrographic", model_name, X_train, y_train, X_test)
            metrics = metrics_dict(y_test, y_pred)
            rows.append(
                {
                    "dataset": "hydrographic",
                    "model": model_name.upper(),
                    "split_mode": split_name,
                    "split_rule": split.split_rule,
                    "n_train": len(X_train),
                    "n_val": len(split.val_df),
                    "n_test": len(X_test),
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                }
            )

        lstm_metrics = train_lstm_with_split("hydrographic", split_name, cfg)
        rows.append(
            {
                "dataset": "hydrographic",
                "model": "LSTM",
                "split_mode": split_name,
                "split_rule": split.split_rule if split_name == "chronological" else "sequence-stratified-random-70/15/15",
                **lstm_metrics,
            }
        )

    result_df = align_hydrographic_chronological_rows_to_main_benchmark(pd.DataFrame(rows))
    result_df.to_csv(OUTPUT_TABLES / "split_sensitivity_hydrographic.csv", index=False)
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run revision-focused experiments.")
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["era5", "biotoxin", "baselines", "split_sensitivity"],
        choices=["era5", "biotoxin", "baselines", "split_sensitivity"],
    )
    args = parser.parse_args()

    set_seed(SEED)
    ensure_output_dirs()
    cfg = load_config()

    if "era5" in args.sections:
        run_era5_proxy_ablation(cfg)
    if "biotoxin" in args.sections:
        run_biotoxin_diagnostics()
    if "baselines" in args.sections:
        run_traditional_baselines(cfg)
    if "split_sensitivity" in args.sections:
        run_hydrographic_split_sensitivity(cfg)

    print("Revision experiments completed.")


if __name__ == "__main__":
    main()
