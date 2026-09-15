
"""
Generate complementary metrics (NRMSE, NSE, event metrics) on holdout splits
with a unified split logic and full model coverage.

Outputs:
- outputs/tables/alternative_metrics_detailed.csv
- outputs/tables/alternative_metrics_summary.csv
- outputs/tables/alternative_metrics_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import warnings

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATHS = [
    REPO_ROOT / "configs" / "config.yaml",
    REPO_ROOT / "code" / "src" / "config.yaml",
]
MODELS_DIR = REPO_ROOT / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Focus on the seven datasets retained in main paper tables.
DEFAULT_DATASETS = [
    "biotoxin",
    "cast",
    "era5_daily",
    "cleaned_data",
    "rolling_mean",
    "processed_seq",
    "hydrographic",
]

RANDOM_SPLIT_DATASETS = {"cast", "phyto_wide"}

TARGET_FALLBACKS = {
    "biotoxin": "VALUE",
    "cast": "Bottom_D",
    "era5_daily": "wind10",
    "cleaned_data": "G2chla",
    "rolling_mean": "G2chla",
    "processed_seq": "G2chla",
    "hydrographic": "G2chla",
    "phyto_wide": "Pseudo-nitzschia americana/brasiliana (cells l-1)",
    "phyto_long": "GYMNODINIALES Karlodinium-like",
}


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


class DeepModelWrapper:
    def __init__(self, model, scaler_X, scaler_y, device=DEVICE):
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


def load_config() -> Dict:
    for path in CONFIG_PATHS:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("Could not find config.yaml in expected locations.")


def infer_target_col(dataset: str, df: pd.DataFrame, meta: Optional[Dict]) -> Optional[str]:
    if meta and meta.get("target_col") in df.columns:
        return meta["target_col"]
    mapped = TARGET_FALLBACKS.get(dataset)
    if mapped in df.columns:
        return mapped
    for candidate in ["G2chla", "chla", "target", "y", "VALUE", "wind10"]:
        if candidate in df.columns:
            return candidate
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return numeric_cols[-1] if len(numeric_cols) else None


def infer_date_col(df: pd.DataFrame, meta: Optional[Dict]) -> Optional[str]:
    if meta and meta.get("date_col") in df.columns:
        return meta["date_col"]
    for col in df.columns:
        lower = col.lower()
        if "date" in lower or "time" in lower:
            return col
    return None


def prepare_features(df: pd.DataFrame, target_col: str, date_col: Optional[str]) -> Tuple[pd.DataFrame, pd.Series]:
    exclude = {target_col}
    if date_col:
        exclude.add(date_col)
    for col in df.columns:
        if col.lower() in {"date", "time"}:
            exclude.add(col)
    X = df[[c for c in df.columns if c not in exclude]].select_dtypes(include=[np.number]).copy()
    y = df[target_col].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean(numeric_only=True))
    y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())
    mask = ~(X.isna().any(axis=1) | y.isna())
    return X.loc[mask], y.loc[mask]


def time_split_indices(n: int, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def stratified_random_split(
    df: pd.DataFrame,
    target_col: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[target_col]
    n_bins = min(10, max(2, len(df) // 200))
    try:
        bins = pd.qcut(y, q=n_bins, duplicates="drop")
        stratify = bins.astype(str)
    except Exception:
        stratify = None

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=stratify,
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
    return train_df, val_df, test_df


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, denom: float) -> float:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    if denom == 0:
        return float("nan")
    return float(rmse / denom)


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - (np.sum((y_true - y_pred) ** 2) / denom))


def event_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        average="binary",
        zero_division=0,
    )
    return {
        "event_threshold": float(threshold),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    denom: float,
    include_events: bool,
    event_threshold: Optional[float],
) -> Dict[str, float]:
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "nrmse": nrmse(y_true, y_pred, denom),
        "nse": nse(y_true, y_pred),
    }
    if include_events and event_threshold is not None:
        metrics.update(event_metrics(y_true, y_pred, event_threshold))
    return metrics


def load_model_params(dataset: str, model_name: str) -> Dict:
    params_path = MODELS_DIR / dataset / f"{model_name}_params.json"
    if params_path.exists():
        try:
            return json.loads(params_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def build_sklearn_model(dataset: str, model_name: str, seed: int):
    params = load_model_params(dataset, model_name)
    if model_name == "rf":
        params.setdefault("n_estimators", 300)
        params.setdefault("max_depth", 15)
        params.setdefault("min_samples_split", 5)
        params.setdefault("min_samples_leaf", 2)
        params.setdefault("max_features", "sqrt")
        params["n_estimators"] = int(min(400, params.get("n_estimators", 300)))
        model = RandomForestRegressor(random_state=seed, **params)
    elif model_name == "xgb":
        use_gpu = os.getenv("XGB_USE_GPU", "0") == "1"
        if use_gpu:
            params.setdefault("tree_method", "gpu_hist")
            params.setdefault("predictor", "gpu_predictor")
            params.setdefault("gpu_id", 0)
        params.setdefault("n_estimators", 300)
        params.setdefault("max_depth", 6)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("subsample", 0.9)
        params.setdefault("colsample_bytree", 0.9)
        model = xgb.XGBRegressor(random_state=seed, verbosity=0, **params)
    elif model_name == "svr":
        params.setdefault("C", 10.0)
        params.setdefault("gamma", "scale")
        params.setdefault("epsilon", 0.05)
        model = SVR(**params)
    elif model_name == "ridge":
        params.setdefault("alpha", 1.0)
        model = Ridge(random_state=seed, **params)
    elif model_name == "lasso":
        params.setdefault("alpha", 0.001)
        model = Lasso(random_state=seed, max_iter=5000, **params)
    else:
        raise ValueError(model_name)
    return model


def load_deep_model(dataset: str, model_name: str, input_size: int):
    model_path = MODELS_DIR / dataset / f"{model_name}.pth"
    if not model_path.exists():
        return None
    ckpt = torch.load(model_path, map_location="cpu")
    params = ckpt.get("model_params", {})
    if model_name == "lstm":
        model = LSTMModel(
            input_size=input_size,
            hidden_size=int(params.get("hidden_size", 64)),
            num_layers=int(params.get("num_layers", 2)),
            dropout=float(params.get("dropout", 0.2)),
        )
    else:
        model = TransformerModel(
            input_size=input_size,
            d_model=int(params.get("d_model", 64)),
            nhead=int(params.get("nhead", 4)),
            num_layers=int(params.get("num_layers", 2)),
            dropout=float(params.get("dropout", 0.1)),
        )
    model.load_state_dict(ckpt["model_state_dict"])
    wrapper = DeepModelWrapper(model, ckpt["scaler_X"], ckpt["scaler_y"], device=DEVICE)
    return wrapper


def evaluate_dataset(
    dataset: str,
    meta: Optional[Dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    min_samples: int,
) -> Tuple[List[Dict], List[Dict]]:
    data_path = REPO_ROOT / "data" / "processed" / dataset / "clean.csv"
    if not data_path.exists():
        return [], []

    df = pd.read_csv(data_path)
    target_col = infer_target_col(dataset, df, meta)
    if not target_col or target_col not in df.columns:
        return [], []

    date_col = infer_date_col(df, meta)
    split_rule = "random"

    if dataset in RANDOM_SPLIT_DATASETS:
        train_df, val_df, test_df = stratified_random_split(df, target_col, train_ratio, val_ratio, seed)
    elif date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].notna().sum() > min_samples:
            split_rule = "time-ordered"
            df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
            tr_idx, va_idx, te_idx = time_split_indices(len(df), train_ratio, val_ratio)
            train_df, val_df, test_df = df.iloc[tr_idx], df.iloc[va_idx], df.iloc[te_idx]
        else:
            train_df, val_df, test_df = stratified_random_split(df, target_col, train_ratio, val_ratio, seed)
    else:
        train_df, val_df, test_df = stratified_random_split(df, target_col, train_ratio, val_ratio, seed)

    if len(train_df) < min_samples or len(test_df) < max(50, min_samples // 4):
        return [], []

    X_train, y_train = prepare_features(train_df, target_col, date_col)
    X_val, y_val = prepare_features(val_df, target_col, date_col)
    X_test, y_test = prepare_features(test_df, target_col, date_col)

    if len(X_train) < min_samples or len(X_test) < 50:
        return [], []

    include_events = dataset == "biotoxin"
    event_threshold = None
    if include_events:
        event_threshold = float(np.percentile(y_train.to_numpy(), 90))

    denom = float(np.max(y_train) - np.min(y_train)) if len(y_train) else 0.0

    rows: List[Dict] = []
    preds_rows: List[Dict] = []

    # Baseline: train-mean
    baseline_pred = np.full_like(y_test.to_numpy(), fill_value=float(y_train.mean()), dtype=float)
    base_metrics = compute_metrics(
        y_test.to_numpy(), baseline_pred, denom, include_events=include_events, event_threshold=event_threshold
    )
    rows.append(
        {
            "dataset": dataset,
            "model": "train_mean",
            "split_rule": split_rule,
            "target_col": target_col,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            **base_metrics,
        }
    )
    for yt, yp in zip(y_test.to_numpy(), baseline_pred):
        preds_rows.append(
            {"dataset": dataset, "model": "train_mean", "split_rule": split_rule, "y_true": float(yt), "y_pred": float(yp)}
        )

    # Traditional ML models
    for model_name in ["lasso", "ridge", "rf", "xgb", "svr"]:
        model = build_sklearn_model(dataset, model_name, seed)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        metrics = compute_metrics(y_test.to_numpy(), y_pred, denom, include_events=include_events, event_threshold=event_threshold)
        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "split_rule": split_rule,
                "target_col": target_col,
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "n_test": int(len(X_test)),
                **metrics,
            }
        )
        for yt, yp in zip(y_test.to_numpy(), y_pred):
            preds_rows.append(
                {"dataset": dataset, "model": model_name, "split_rule": split_rule, "y_true": float(yt), "y_pred": float(yp)}
            )

    # Deep models (LSTM/Transformer) if sequence data exists
    seq_path = REPO_ROOT / "data" / "processed" / dataset / "sequences.npz"
    if seq_path.exists():
        data = np.load(seq_path)
        X_seq, y_seq = data["X"], data["y"]
        if len(X_seq) >= min_samples:
            if split_rule == "time-ordered":
                tr_idx, va_idx, te_idx = time_split_indices(len(X_seq), train_ratio, val_ratio)
            else:
                X_temp, X_test_seq, y_temp, y_test_seq = train_test_split(
                    X_seq, y_seq, test_size=(1.0 - train_ratio), random_state=seed
                )
                val_size = val_ratio / (1.0 - train_ratio)
                X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
                    X_temp, y_temp, test_size=(1.0 - val_size), random_state=seed
                )
                tr_idx = np.arange(len(X_train_seq))
                va_idx = np.arange(len(X_train_seq), len(X_train_seq) + len(X_val_seq))
                te_idx = np.arange(len(X_train_seq) + len(X_val_seq), len(X_train_seq) + len(X_val_seq) + len(X_test_seq))
                X_seq = np.concatenate([X_train_seq, X_val_seq, X_test_seq], axis=0)
                y_seq = np.concatenate([y_train_seq, y_val_seq, y_test_seq], axis=0)

            X_train_seq, y_train_seq = X_seq[tr_idx], y_seq[tr_idx]
            X_test_seq, y_test_seq = X_seq[te_idx], y_seq[te_idx]

            if include_events:
                event_threshold = float(np.percentile(y_train_seq, 90))
            denom_seq = float(np.max(y_train_seq) - np.min(y_train_seq)) if len(y_train_seq) else 0.0

            for model_name in ["lstm", "transformer"]:
                wrapper = load_deep_model(dataset, model_name, X_train_seq.shape[-1])
                if wrapper is None:
                    continue
                y_pred_seq = wrapper.predict(X_test_seq)
                metrics = compute_metrics(
                    y_test_seq, y_pred_seq, denom_seq, include_events=include_events, event_threshold=event_threshold
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "split_rule": split_rule,
                        "target_col": target_col,
                        "n_train": int(len(X_train_seq)),
                        "n_val": int(len(y_seq) - len(X_train_seq) - len(X_test_seq)),
                        "n_test": int(len(X_test_seq)),
                        **metrics,
                    }
                )
                for yt, yp in zip(y_test_seq, y_pred_seq):
                    preds_rows.append(
                        {"dataset": dataset, "model": model_name, "split_rule": split_rule, "y_true": float(yt), "y_pred": float(yp)}
                    )

    return rows, preds_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate alternative metric tables.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-samples", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config()
    global_cfg = cfg.get("global", {})
    train_ratio = float(global_cfg.get("train_ratio", 0.7))
    val_ratio = float(global_cfg.get("val_ratio", 0.15))

    rows: List[Dict] = []
    pred_rows: List[Dict] = []
    for dataset in args.datasets:
        meta = cfg.get("datasets", {}).get(dataset, {})
        ds_rows, ds_preds = evaluate_dataset(
            dataset=dataset,
            meta=meta,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=args.seed,
            min_samples=args.min_samples,
        )
        rows.extend(ds_rows)
        pred_rows.extend(ds_preds)

    if not rows:
        raise RuntimeError("No alternative metrics could be generated.")

    out_dir = REPO_ROOT / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = out_dir / "alternative_metrics_detailed.csv"
    summary_path = out_dir / "alternative_metrics_summary.csv"
    preds_path = out_dir / "alternative_metrics_predictions.csv"

    detailed_df = pd.DataFrame(rows)
    detailed_df.to_csv(detailed_path, index=False)

    # SI-friendly summary for all models
    summary_cols = [
        "dataset",
        "model",
        "split_rule",
        "n_train",
        "n_test",
        "rmse",
        "nrmse",
        "nse",
        "event_precision",
        "event_recall",
        "event_f1",
        "event_threshold",
    ]
    summary_df = detailed_df.reindex(columns=summary_cols).copy()
    summary_df = summary_df.rename(
        columns={
            "dataset": "Dataset",
            "model": "Model",
            "split_rule": "Split Rule",
            "n_train": "Train N",
            "n_test": "Test N",
            "rmse": "RMSE",
            "nrmse": "NRMSE",
            "nse": "NSE",
            "event_precision": "Event Precision (90th pct)",
            "event_recall": "Event Recall (90th pct)",
            "event_f1": "Event F1 (90th pct)",
            "event_threshold": "Event Threshold (train 90th pct)",
        }
    )
    summary_df.to_csv(summary_path, index=False)

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(preds_path, index=False)

    print(f"Saved detailed alternative metrics: {detailed_path}")
    print(f"Saved summary alternative metrics: {summary_path}")
    print(f"Saved prediction pairs: {preds_path}")


if __name__ == "__main__":
    main()
