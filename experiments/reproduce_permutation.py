"""Reproduce the paper's permutation sanity check (Table 2 / S2: XGB, K=10,000) with the authors' functions.

complete_sanity_check.py seeds the global NumPy RNG once (42) and draws every permutation from that stream,
dataset by dataset in a fixed order. We replay the identical stream, then run the 10,000 refits in parallel.
Run with the pinned environment (.venv_pinned: scikit-learn 1.5.2, xgboost 2.1.1).
"""
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from common import REPO, RESULTS

sys.path.insert(0, str(REPO / "code" / "scripts"))
import complete_sanity_check as C  # noqa: E402

K, SEED = 10000, 42
ORDER = ["biotoxin", "cast", "era5_daily", "cleaned_data", "rolling_mean", "processed_seq", "hydrographic"]


def load(ds):
    df = pd.read_csv(REPO / "data" / "processed" / ds / "clean.csv")
    t = C.get_target_column(ds, df)
    X, y = C.prepare_features(ds, df, t)
    tr, _, te, rule = C._split_indices(df, y, ds, SEED)
    sc = StandardScaler().fit(X.iloc[tr])
    return sc.transform(X.iloc[tr]), sc.transform(X.iloc[te]), y.iloc[tr].to_numpy(), y.iloc[te].to_numpy(), rule


def perm_r2(Xtr, Xte, y_perm, ntr):
    m = C._fit_model("xgb", SEED, {})
    m.set_params(n_jobs=1)
    m.fit(Xtr, y_perm[:ntr])
    return r2_score(y_perm[ntr:], m.predict(Xte))


if __name__ == "__main__":
    np.random.seed(SEED)
    rows = []
    for ds in ORDER:
        Xtr, Xte, ytr, yte, rule = load(ds)
        obs = r2_score(yte, C._fit_model("xgb", SEED, {}).fit(Xtr, ytr).predict(Xte))
        yall = np.concatenate([ytr, yte])
        perms = [np.random.permutation(yall) for _ in range(K)]  # same draw order as the authors' loop
        scores = np.array(Parallel(n_jobs=78, batch_size=16)(delayed(perm_r2)(Xtr, Xte, p, len(ytr)) for p in perms))
        b = int(np.sum(np.abs(scores) >= abs(obs)))
        rows.append({"dataset": ds, "original_r2": obs, "permuted_r2_mean": scores.mean(), "b": b,
                     "p_value": (b + 1) / (K + 1), "n_permutations": K, "split_rule": rule})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "repro_permutation_test.csv", index=False)
