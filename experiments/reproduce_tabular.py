"""Reproduce the paper's tabular rows (MEAN, RIDGE, LASSO, SVR, RF, XGB) for all 7 datasets.

Model settings = build_legacy_models() in the authors' final generator; RobustScaler fit on train.
"""
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

from common import DATASETS, RESULTS, SEED, metrics, tabular_splits


def models():
    return {
        "RF": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=8),
        "XGB": xgb.XGBRegressor(n_estimators=100, random_state=SEED, verbosity=0, n_jobs=8),
        "SVR": SVR(kernel="rbf", C=1.0),
        "RIDGE": Ridge(alpha=1.0),
        "LASSO": Lasso(alpha=0.1),
    }


def run(ds, name):
    X, y, (tr, va, te), rule = tabular_splits(ds)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    if name == "MEAN":
        pred = np.full(len(yte), ytr.mean())
    else:
        sc = RobustScaler().fit(Xtr)
        pred = models()[name].fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
    row = {"dataset": ds, "model": name, "split": rule, "n_features": X.shape[1], "n_train": len(tr), **metrics(yte, pred, ytr)}
    print(f"{ds:14s} {name:6s} R2={row['R2']:.4f} MAE={row['MAE']:.4f}", flush=True)
    return row, pd.DataFrame({"dataset": ds, "model": name, "y_true": yte.to_numpy(), "y_pred": pred})


if __name__ == "__main__":
    dss = sys.argv[1:] or DATASETS
    jobs = [(d, m) for d in dss for m in ["MEAN", "RIDGE", "LASSO", "RF", "XGB", "SVR"]]
    out = Parallel(n_jobs=len(jobs))(delayed(run)(d, m) for d, m in jobs)
    RESULTS.mkdir(exist_ok=True)
    tag = os.getenv("REPRO_TAG", "")  # e.g. "_pinned" for the sklearn 1.5.2 / xgboost 2.1.1 environment
    pd.DataFrame([r for r, _ in out]).to_csv(RESULTS / f"repro_tabular_metrics{tag}.csv", index=False)
    pd.concat([p for _, p in out]).to_csv(RESULTS / f"repro_tabular_predictions{tag}.csv.gz", index=False)
