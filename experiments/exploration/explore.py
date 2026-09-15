"""Quick calibration run: default-ish strong learners, val and test R² side by side."""
import sys
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from common import sequence_splits, tabular_splits

ds, kind = sys.argv[1], sys.argv[2]
if kind == "tab":
    X, y, (tr, va, te), _ = tabular_splits(ds)
    X, y = X.to_numpy(float), y.to_numpy(float)
else:
    S, y, (tr, va, te) = sequence_splits(ds)
    X = S.reshape(len(S), -1)
print(ds, kind, X.shape)
cands = {
    "lgb": lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.02, num_leaves=31, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1, n_jobs=16),
    "xgb": xgb.XGBRegressor(n_estimators=3000, learning_rate=0.02, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                            early_stopping_rounds=200, tree_method="hist", device="cuda:0", verbosity=0),
    "et": ExtraTreesRegressor(500, max_features=0.5, min_samples_leaf=2, n_jobs=32, random_state=42),
    "rf": RandomForestRegressor(500, max_features=0.33, min_samples_leaf=2, n_jobs=32, random_state=42),
}
for n, m in cands.items():
    if n == "lgb":
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
    elif n == "xgb":
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    else:
        m.fit(X[tr], y[tr])
    print(f"  {n:4s} val={r2_score(y[va], m.predict(X[va])):.4f} test={r2_score(y[te], m.predict(X[te])):.4f}", flush=True)
