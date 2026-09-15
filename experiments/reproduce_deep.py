"""Reproduce the paper's LSTM / Transformer rows by scoring the authors' released checkpoints
on the chronological last-15% of sequences (exactly as generate_alternative_metrics_legacy.py does)."""
import sys

import warnings

import numpy as np
import numpy._core.multiarray as ma
import numpy.dtypes as nd
import pandas as pd
import sklearn.preprocessing as skp
import torch

from common import DEEP_DATASETS, REPO, RESULTS, metrics, sequence_splits, tabular_splits

sys.path.insert(0, str(REPO / "code" / "scripts"))
import generate_alternative_metrics_legacy as L  # noqa: E402  (authors' architecture + loader)

warnings.filterwarnings("ignore")
# Checkpoints pickle sklearn scalers; allowlist exactly those types instead of weights_only=False.
torch.serialization.add_safe_globals(
    [skp.MinMaxScaler, skp.StandardScaler, np.ndarray, np.dtype,
     (ma._reconstruct, "numpy.core.multiarray._reconstruct"), (ma.scalar, "numpy.core.multiarray.scalar")]
    + [getattr(nd, n) for n in dir(nd) if n.endswith("DType")])


def checkpoint_predict(ds, name, X):
    return L.load_deep_model(REPO / "models" / ds / f"{name}.pth", name).predict(X.astype(np.float32))


if __name__ == "__main__":
    rows, preds = [], []
    for ds in DEEP_DATASETS:
        S, y, (tr, va, te) = sequence_splits(ds)
        _, y_tab, (tr_tab, _, _), _ = tabular_splits(ds)
        for name in ["lstm", "transformer"]:
            p = checkpoint_predict(ds, name, S[te])
            # authors enforce the tabular-train 90th percentile as the event threshold for every model
            m = metrics(y[te], p, y_tab.iloc[tr_tab])
            m["NRMSE"] = m["RMSE"] / float(np.ptp(y[tr]))
            rows.append({"dataset": ds, "model": name.upper(), "split": "chronological-70/15/15",
                         "n_train": len(tr), **m})
            preds.append(pd.DataFrame({"dataset": ds, "model": name.upper(), "y_true": y[te], "y_pred": p}))
            print(f"{ds:14s} {name:12s} R2={m['R2']:.4f} MAE={m['MAE']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "repro_deep_metrics.csv", index=False)
    pd.concat(preds).to_csv(RESULTS / "repro_deep_predictions.csv.gz", index=False)
