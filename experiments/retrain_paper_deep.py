"""Leakage-free retraining of the paper's own LSTM / Transformer.

Same architecture classes, same tuned hyperparameters (models/<ds>/<model>_params.json) and the same
final-fit recipe as train_enhanced.py (MinMax X, standardised y, Adam, 200 full-batch MSE epochs on
train+val). Only change: train/val are the chronological first 85% instead of a random 85% that
overlapped the chronological test windows. 5 seeds.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from common import DEEP_DATASETS, REPO, RESULTS, metrics, sequence_splits, tabular_splits

sys.path.insert(0, str(REPO / "code"))
from src.train_enhanced import LSTMModel, TransformerModel  # noqa: E402

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda")
rows, preds = [], []
for ds in DEEP_DATASETS:
    S, y, (tr, va, te) = sequence_splits(ds)
    _, y_tab, (tr_tab, _, _), _ = tabular_splits(ds)
    fit = np.concatenate([tr, va])
    sx = MinMaxScaler().fit(S[tr].reshape(-1, S.shape[2]))  # authors fit scalers on the train part
    sy = StandardScaler().fit(y[tr].reshape(-1, 1))
    Xs = torch.tensor(sx.transform(S.reshape(-1, S.shape[2])).reshape(S.shape), dtype=torch.float32, device=dev)
    ys = torch.tensor(sy.transform(y.reshape(-1, 1)).ravel(), dtype=torch.float32, device=dev)
    for name, cls in [("lstm", LSTMModel), ("transformer", TransformerModel)]:
        p = json.loads((REPO / "models" / ds / f"{name}_params.json").read_text())
        lr = p.pop("learning_rate")
        runs = []
        for seed in range(5):
            torch.manual_seed(seed)
            net = cls(input_size=S.shape[2], **p).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            idx = torch.tensor(fit, device=dev)
            net.train()
            for _ in range(200):
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(Xs[idx]).squeeze(), ys[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                out = net(Xs[torch.tensor(te, device=dev)]).squeeze().cpu().numpy()
            pr = sy.inverse_transform(out.reshape(-1, 1)).ravel()
            m = metrics(y[te], pr, y_tab.iloc[tr_tab])
            runs.append(m["R2"])
            rows.append({"dataset": ds, "model": name.upper() + "_chrono_retrain", "seed": seed, **m})
            preds.append(pd.DataFrame({"dataset": ds, "model": name.upper(), "seed": seed, "y_true": y[te], "y_pred": pr}))
        print(f"{ds:14s} {name:12s} R2 mean={np.mean(runs):.4f} sd={np.std(runs):.4f} runs={np.round(runs, 4)}", flush=True)
pd.DataFrame(rows).to_csv(RESULTS / "paper_deep_chronological_retrain.csv", index=False)
pd.concat(preds).to_csv(RESULTS / "paper_deep_chronological_retrain_predictions.csv.gz", index=False)
