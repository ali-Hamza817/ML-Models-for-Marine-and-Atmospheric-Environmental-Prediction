#!/usr/bin/env python3
"""
Alternative-metrics generator aligned with manuscript split rules (v18).

This script computes complementary metrics (RMSE/NRMSE/NSE + event metrics)
on the held-out test set using the manuscript split protocol:
  - Time-series: chronological 70/15/15 (train/val/test)
  - Cross-sectional: stratified random 70/15/15 (by target bins)

Outputs (same filenames used by export_plos_assets.py):
  - outputs/tables/alternative_metrics_detailed.csv
  - outputs/tables/alternative_metrics_summary.csv
  - outputs/tables/alternative_metrics_predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
import xgboost as xgb
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATHS = [
    REPO_ROOT / "configs" / "config.yaml",
    REPO_ROOT / "code" / "src" / "config.yaml",
]

DEFAULT_DATASETS = [
    "biotoxin",
    "cast",
    "era5_daily",
    "cleaned_data",
    "rolling_mean",
    "processed_seq",
    "hydrographic",
]

# Cross-sectional datasets that should use stratified random splits.
RANDOM_SPLIT_DATASETS = {"cast", "phyto_wide"}

# Legacy dataset-specific feature exclusions (from original pipeline)
LEGACY_DATASET_CONFIG = {
    "biotoxin": {
        "target": "VALUE",
        "exclude_features": ["Date", "date", "time", "Time"],
    },
    "cast": {
        "target": "Bottom_D",
        "exclude_features": [
            "Date", "date", "time", "Time",
            "Lat_Dec", "Lat_Deg", "Lat_Min", "Lat_Hem",
            "Lon_Dec", "Lon_Deg", "Lon_Min", "Lon_Hem",
            "Rpt_Line", "St_Line", "Ac_Line",
            "Rpt_Sta", "St_Station", "Ac_Sta",
            "Sta_ID", "Sta_Code", "Orig_Sta_ID",
            "Cruise_ID", "Cast_ID", "DbSta_ID",
        ],
    },
    "era5_daily": {
        "target": "wind10",
        "exclude_features": ["Date", "date", "time", "Time"],
    },
    "cleaned_data": {
        "target": "G2chla",
        "exclude_features": ["Date", "date", "time", "Time"],
    },
    "rolling_mean": {
        "target": "G2chla",
        "exclude_features": ["Date", "date", "time", "Time"],
    },
    "processed_seq": {
        "target": "G2chla",
        "exclude_features": ["Date", "date", "time", "Time"],
    },
    "hydrographic": {
        "target": "CHLOROPHYLL-a (µg l-1)",
        "exclude_features": [
            "STATION", "LATITUDE (degrees North)", "LONGITUDE (degrees East)",
            "DATUM", "UTC DATE (YYYYMMDD)", "UTC TIME (hhmmss)",
        ],
    },
}


# -------------------------- Deep models --------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_reduction = nn.Linear(input_size, min(32, max(1, input_size // 2)))
        self.input_projection = nn.Linear(min(32, max(1, input_size // 2)), d_model)
        self.pos_encoding = nn.Parameter(torch.randn(100, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.feature_reduction(x)
        x = torch.relu(x)
        x = self.input_projection(x)
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc
        x = self.transformer(x)
        x = self.layer_norm(x.mean(dim=1))
        x = self.dropout(x)
        return self.fc(x)


class TransformerModelOld(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.fc(x)


class DeepModelWrapper:
    def __init__(self, model, scaler_X, scaler_y, device):
        self.model = model.to(device)
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.device = device

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
            preds = self.scaler_y.inverse_transform(outputs.detach().cpu().numpy().reshape(-1, 1)).flatten()
        return preds


def load_deep_model(model_path: Path, model_type: str) -> DeepModelWrapper:
    checkpoint = torch.load(model_path, map_location="cpu")
    model_params = checkpoint.get("model_params", {}).copy()
    model_params.pop("learning_rate", None)

    state_dict = checkpoint.get("model_state_dict", {})
    if model_type == "lstm":
        input_size = checkpoint.get("input_size")
        if input_size is None and "lstm.weight_ih_l0" in state_dict:
            input_size = state_dict["lstm.weight_ih_l0"].shape[1]
        model = LSTMModel(input_size=input_size, **model_params)
    elif model_type == "transformer":
        if "feature_reduction.weight" in state_dict:
            input_size = state_dict["feature_reduction.weight"].shape[1]
            model = TransformerModel(input_size=input_size, **model_params)
        else:
            input_size = state_dict["input_projection.weight"].shape[1]
            model = TransformerModelOld(input_size=input_size, **model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(state_dict)
    return DeepModelWrapper(model, checkpoint["scaler_X"], checkpoint["scaler_y"], device=torch.device("cpu"))


# -------------------------- utilities --------------------------
def load_config() -> Dict:
    for path in CONFIG_PATHS:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("config.yaml not found")


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, train_range: float) -> float:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    if train_range == 0:
        return np.nan
    return rmse / train_range


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - (np.sum((y_true - y_pred) ** 2) / denom)


def event_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0
    )
    return {
        "event_threshold": float(threshold),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
    }


def prepare_legacy_tabular(dataset: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    data_path = REPO_ROOT / "data" / "processed" / dataset / "clean.csv"
    df = pd.read_csv(data_path)
    cfg = LEGACY_DATASET_CONFIG.get(dataset, {})
    target_col = cfg.get("target")
    if target_col not in df.columns:
        # Robust fallback for chlorophyll-like targets with encoding differences.
        chl_candidates = [
            c for c in df.columns
            if ("chlorophyll-a" in c.lower()) or ("chla" in c.lower())
        ]
        if chl_candidates:
            target_col = chl_candidates[0]
        else:
            # Fallback to numeric target
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not len(numeric_cols):
                raise ValueError(f"No numeric target for {dataset}")
            target_col = numeric_cols[-1]

    exclude_cols = set(cfg.get("exclude_features", []))
    exclude_cols.add(target_col)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    y = df[target_col].copy()

    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(y.median())

    # drop constant cols
    constant_cols = [c for c in X.columns if X[c].std() < 1e-10]
    if constant_cols:
        X = X.drop(columns=constant_cols)

    # drop highly correlated cols
    if X.shape[1] > 1:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_remove = set()
        for col in upper.columns:
            high = upper.index[upper[col] > 0.95].tolist()
            for f in high:
                if f not in to_remove:
                    to_remove.add(col)
        if to_remove:
            X = X.drop(columns=list(to_remove))

    return X, y, df


def build_legacy_models(seed: int) -> Dict[str, object]:
    return {
        "rf": RandomForestRegressor(n_estimators=100, random_state=seed),
        "xgb": xgb.XGBRegressor(n_estimators=100, random_state=seed, verbosity=0),
        "svr": SVR(kernel="rbf", C=1.0),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.1),
    }

def stratified_random_split(
    y: pd.Series, test_size: float, val_size: float, seed: int, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_series = pd.Series(y).reset_index(drop=True)
    idx = np.arange(len(y_series))
    try:
        bins = pd.qcut(y_series, q=n_bins, duplicates="drop").astype(str)
        train_idx, temp_idx = train_test_split(
            idx, test_size=test_size + val_size, random_state=seed, stratify=bins
        )
        temp_bins = bins.iloc[temp_idx]
        val_ratio = val_size / (test_size + val_size)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=1 - val_ratio, random_state=seed, stratify=temp_bins
        )
        return train_idx, val_idx, test_idx
    except Exception:
        train_idx, temp_idx = train_test_split(idx, test_size=test_size + val_size, random_state=seed)
        val_ratio = val_size / (test_size + val_size)
        val_idx, test_idx = train_test_split(temp_idx, test_size=1 - val_ratio, random_state=seed)
        return train_idx, val_idx, test_idx


def time_ordered_split(n: int, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def evaluate_legacy_tabular(dataset: str, seed: int, cfg: Dict) -> Tuple[List[Dict], List[Dict], float]:
    X, y, df = prepare_legacy_tabular(dataset)
    train_ratio = float(cfg.get("global", {}).get("train_ratio", 0.70))
    val_ratio = float(cfg.get("global", {}).get("val_ratio", 0.15))
    test_ratio = float(cfg.get("global", {}).get("test_ratio", 0.15))

    date_col = cfg.get("datasets", {}).get(dataset, {}).get("date_col")
    use_random = dataset in RANDOM_SPLIT_DATASETS
    if (not use_random) and date_col in df.columns:
        df_sorted = df.sort_values(by=date_col)
        X = X.loc[df_sorted.index].reset_index(drop=True)
        y = y.loc[df_sorted.index].reset_index(drop=True)
        tr_idx, va_idx, te_idx = time_ordered_split(len(y), train_ratio, val_ratio)
        split_rule = "chronological-70/15/15"
    else:
        tr_idx, va_idx, te_idx = stratified_random_split(y, test_ratio, val_ratio, seed)
        split_rule = "stratified-random-70/15/15"

    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
    dataset_event_threshold = float(np.percentile(y_train, 90))

    models = build_legacy_models(seed)
    detailed_rows: List[Dict] = []
    pred_rows: List[Dict] = []

    for model_name, model in models.items():
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        train_range = float(np.max(y_train) - np.min(y_train))
        ev = event_metrics(y_test.to_numpy(), y_pred, dataset_event_threshold)

        detailed_rows.append({
            "dataset": dataset,
            "model": model_name,
            "split_rule": split_rule,
            "train_n": len(y_train),
            "test_n": len(y_test),
            "R2": float(r2_score(y_test, y_pred)),
            "RMSE": float(mean_squared_error(y_test, y_pred, squared=False)),
            "NRMSE": float(nrmse(y_test.to_numpy(), y_pred, train_range)),
            "NSE": float(nse(y_test.to_numpy(), y_pred)),
            "event_precision": ev["event_precision"],
            "event_recall": ev["event_recall"],
            "event_f1": ev["event_f1"],
            "event_threshold": float(dataset_event_threshold),
        })

        for yt, yp in zip(y_test.to_numpy(), y_pred):
            pred_rows.append({
                "dataset": dataset,
                "model": model_name,
                "split_rule": split_rule,
                "subset": "test",
                "y_true": float(yt),
                "y_pred": float(yp),
            })

    # baseline mean predictor
    y_pred = np.full(len(y_test), y_train.mean())
    train_range = float(np.max(y_train) - np.min(y_train))
    ev = event_metrics(y_test.to_numpy(), y_pred, dataset_event_threshold)
    detailed_rows.append({
        "dataset": dataset,
        "model": "mean",
        "split_rule": split_rule,
        "train_n": len(y_train),
        "test_n": len(y_test),
        "R2": float(r2_score(y_test, y_pred)),
        "RMSE": float(mean_squared_error(y_test, y_pred, squared=False)),
        "NRMSE": float(nrmse(y_test.to_numpy(), y_pred, train_range)),
        "NSE": float(nse(y_test.to_numpy(), y_pred)),
        "event_precision": ev["event_precision"],
        "event_recall": ev["event_recall"],
        "event_f1": ev["event_f1"],
        "event_threshold": float(dataset_event_threshold),
    })
    for yt, yp in zip(y_test.to_numpy(), y_pred):
        pred_rows.append({
            "dataset": dataset,
            "model": "mean",
            "split_rule": split_rule,
            "subset": "test",
            "y_true": float(yt),
            "y_pred": float(yp),
        })

    return detailed_rows, pred_rows, dataset_event_threshold


def evaluate_legacy_deep(dataset: str, model_name: str, seed: int, cfg: Dict) -> Tuple[Dict, List[Dict]]:
    seq_path = REPO_ROOT / "data" / "processed" / dataset / "sequences.npz"
    if not seq_path.exists():
        return {}, []

    data = np.load(seq_path)
    X, y = data["X"], data["y"]
    if len(X) == 0:
        return {}, []

    train_ratio = float(cfg.get("global", {}).get("train_ratio", 0.7))
    val_ratio = float(cfg.get("global", {}).get("val_ratio", 0.15))
    test_ratio = float(cfg.get("global", {}).get("test_ratio", 0.15))
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        test_ratio = 1.0 - train_ratio - val_ratio

    # split rule (match manuscript)
    # manuscript split: chronological unless cross-sectional
    if dataset in RANDOM_SPLIT_DATASETS:
        idx = np.arange(len(y))
        train_idx, val_idx, test_idx = stratified_random_split(
            pd.Series(y), test_ratio, val_ratio, seed
        )
        split_rule = "stratified-random-70/15/15"
    else:
        train_idx, val_idx, test_idx = time_ordered_split(len(y), train_ratio, val_ratio)
        split_rule = "chronological-70/15/15"

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    ckpt_path = REPO_ROOT / "models" / dataset / f"{model_name}.pth"
    if not ckpt_path.exists():
        return {}, []

    wrapper = load_deep_model(ckpt_path, model_name)
    y_pred = wrapper.predict(X_test)

    train_range = float(np.max(y_train) - np.min(y_train))
    threshold = float(np.percentile(y_train, 90))
    ev = event_metrics(y_test, y_pred, threshold)

    detailed = {
        "dataset": dataset,
        "model": model_name,
        "split_rule": split_rule,
        "train_n": len(y_train),
        "test_n": len(y_test),
        "R2": float(r2_score(y_test, y_pred)),
        "RMSE": float(mean_squared_error(y_test, y_pred, squared=False)),
        "NRMSE": float(nrmse(y_test, y_pred, train_range)),
        "NSE": float(nse(y_test, y_pred)),
        "event_precision": ev["event_precision"],
        "event_recall": ev["event_recall"],
        "event_f1": ev["event_f1"],
        "event_threshold": float(threshold),
    }

    pred_rows = [
        {
            "dataset": dataset,
            "model": model_name,
            "split_rule": split_rule,
            "subset": "test",
            "y_true": float(yt),
            "y_pred": float(yp),
        }
        for yt, yp in zip(y_test, y_pred)
    ]

    return detailed, pred_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate legacy alternative metrics aligned with old Table2")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    args = parser.parse_args()

    cfg = load_config()
    seed = int(cfg.get("global", {}).get("random_seed", 42))

    detailed_rows: List[Dict] = []
    pred_rows: List[Dict] = []
    dataset_thresholds: Dict[str, float] = {}

    # Traditional ML (legacy CV)
    for ds in args.datasets:
        rows, preds, threshold = evaluate_legacy_tabular(ds, seed=seed, cfg=cfg)
        detailed_rows.extend(rows)
        pred_rows.extend(preds)
        dataset_thresholds[ds] = float(threshold)

    # Deep models (legacy random split)
    for ds in args.datasets:
        for model_name in ["lstm", "transformer"]:
            detailed, preds = evaluate_legacy_deep(ds, model_name, seed=seed, cfg=cfg)
            if detailed:
                detailed_rows.append(detailed)
                pred_rows.extend(preds)

    detailed_df = pd.DataFrame(detailed_rows)
    pred_df = pd.DataFrame(pred_rows)

    # Enforce one event threshold per dataset and recompute event metrics for all models.
    # This guarantees cross-model comparability in S5.
    for ds, threshold in dataset_thresholds.items():
        ds_mask = detailed_df["dataset"] == ds
        if not ds_mask.any():
            continue
        for model_name in detailed_df.loc[ds_mask, "model"].unique():
            p_mask = (
                (pred_df["dataset"] == ds)
                & (pred_df["model"] == model_name)
                & (pred_df["subset"] == "test")
            )
            preds = pred_df.loc[p_mask, ["y_true", "y_pred"]]
            if preds.empty:
                continue
            ev = event_metrics(
                preds["y_true"].to_numpy(),
                preds["y_pred"].to_numpy(),
                float(threshold),
            )
            row_mask = ds_mask & (detailed_df["model"] == model_name)
            detailed_df.loc[row_mask, "event_threshold"] = float(threshold)
            detailed_df.loc[row_mask, "event_precision"] = ev["event_precision"]
            detailed_df.loc[row_mask, "event_recall"] = ev["event_recall"]
            detailed_df.loc[row_mask, "event_f1"] = ev["event_f1"]

    # Normalize model naming for S5 table
    detailed_df["Model"] = detailed_df["model"].str.upper()
    detailed_df["Dataset"] = detailed_df["dataset"]
    detailed_df["Split Rule"] = detailed_df["split_rule"]

    # Summary table (S5)
    summary_df = detailed_df[[
        "Dataset",
        "Model",
        "Split Rule",
        "train_n",
        "test_n",
        "RMSE",
        "NRMSE",
        "NSE",
        "event_precision",
        "event_recall",
        "event_f1",
        "event_threshold",
        "R2",
    ]].copy()
    summary_df = summary_df.rename(columns={
        "train_n": "Train N",
        "test_n": "Test N",
        "event_precision": "Event Precision (90th pct)",
        "event_recall": "Event Recall (90th pct)",
        "event_f1": "Event F1 (90th pct)",
        "event_threshold": "Event Threshold (train 90th pct)",
        "R2": "R² (test)",
    })

    # Predictions

    out_dir = REPO_ROOT / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    # If partial dataset runs are used, merge with existing outputs.
    detailed_path = out_dir / "alternative_metrics_detailed.csv"
    summary_path = out_dir / "alternative_metrics_summary.csv"
    preds_path = out_dir / "alternative_metrics_predictions.csv"

    if detailed_path.exists():
        existing = pd.read_csv(detailed_path)
        existing = existing[~existing["dataset"].isin(args.datasets)]
        detailed_df = pd.concat([existing, detailed_df], ignore_index=True)

    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[~existing["Dataset"].isin(args.datasets)]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)

    if preds_path.exists():
        existing = pd.read_csv(preds_path)
        existing = existing[~existing["dataset"].isin(args.datasets)]
        pred_df = pd.concat([existing, pred_df], ignore_index=True)

    detailed_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pred_df.to_csv(preds_path, index=False)

    print("✅ Legacy alternative metrics exported:")
    print(f"   - {out_dir / 'alternative_metrics_detailed.csv'}")
    print(f"   - {out_dir / 'alternative_metrics_summary.csv'}")
    print(f"   - {out_dir / 'alternative_metrics_predictions.csv'}")


if __name__ == "__main__":
    main()
