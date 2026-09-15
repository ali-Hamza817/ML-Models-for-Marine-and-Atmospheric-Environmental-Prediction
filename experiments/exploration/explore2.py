"""Calibration of input representations with one fixed LightGBM (val early stopping)."""
import sys
import lightgbm as lgb
import numpy as np
from sklearn.metrics import r2_score
import features as F

def fit(X, y, split, tag):
    tr, va, te = split
    X = np.asarray(X, float)
    m = lgb.LGBMRegressor(n_estimators=5000, learning_rate=0.02, num_leaves=31, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, verbose=-1, n_jobs=16)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(300, verbose=False)])
    print(f"{tag:40s} nfeat={X.shape[1]:4d} val={r2_score(y[va], m.predict(X[va])):.4f} test={r2_score(y[te], m.predict(X[te])):.4f}", flush=True)

ds = sys.argv[1]
if ds == "era5_daily":
    X, y, split, _ = F.tabular_splits(ds); y = y.to_numpy()
    fit(X, y, split, "era5 raw")
    fit(F.era5_physics(X), y, split, "era5 +physics")
    Xc, _, _, _ = F.era5_context(); fit(Xc, y, split, "era5 +physics+prevday+sameday")
else:
    for hyb in (False, True):
        X, y, split = F.seq_design(ds, hyb); fit(X, y, split, f"{ds} window{'+current-row' if hyb else ''}")
