"""Recurrent / convolutional sequence regressors trained without touching the test split:
channel standardisation fit on train, early stopping on validation R², seed ensembles,
optional refit on train+val for the early-stopped epoch count."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score


class SeqNet(nn.Module):
    def __init__(self, n_ch, n_static, kind="gru", hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.kind = kind
        if kind == "cnn":
            self.enc = nn.Sequential(
                nn.Conv1d(n_ch, hidden, 3, padding=1), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4), nn.GELU())
        else:
            rnn = nn.GRU if kind == "gru" else nn.LSTM
            self.enc = rnn(n_ch, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(hidden * 2 + n_static, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, x, s):
        if self.kind == "cnn":
            h = self.enc(x.transpose(1, 2))
            z = torch.cat([h[:, :, -1], h.mean(2)], 1)
        else:
            h, _ = self.enc(x)
            z = torch.cat([h[:, -1], h.mean(1)], 1)
        return self.head(torch.cat([z, s], 1)).squeeze(1)


def _fit_once(S, Z, y, fit_idx, val_idx, seed, kind, epochs, device, hidden, lr, wd, bs, patience=40):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = SeqNet(S.shape[2], Z.shape[1], kind, hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    X = torch.tensor(S, dtype=torch.float32, device=device)
    Zt = torch.tensor(Z, dtype=torch.float32, device=device)
    Y = torch.tensor(y, dtype=torch.float32, device=device)
    best, best_ep, best_state, bad = -np.inf, 0, None, 0
    n_ep = epochs if val_idx is None else 400
    for ep in range(n_ep):
        net.train()
        perm = torch.tensor(np.random.permutation(fit_idx), device=device)
        for b in range(0, len(perm), bs):
            i = perm[b:b + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(X[i], Zt[i]), Y[i])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if val_idx is None:
            continue
        net.eval()
        with torch.no_grad():
            r2 = r2_score(y[val_idx], net(X[val_idx], Zt[val_idx]).cpu().numpy())
        if r2 > best:
            best, best_ep, bad = r2, ep + 1, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(X[b:b + 4096], Zt[b:b + 4096]) for b in range(0, len(y), 4096)]).cpu().numpy()
    return pred, best_ep, best, net


def fit_predict(S, Z, y, split, kind="gru", seeds=5, refit=False, device="cuda", hidden=128, lr=1e-3, wd=1e-4, bs=128,
                return_models=False):
    """Returns predictions for every row (ensemble mean) and the validation R² of the train-only ensemble
    (plus the fitted networks and normalisation constants if return_models)."""
    tr, va, _ = split
    mu, sd = S[tr].reshape(-1, S.shape[2]).mean(0), S[tr].reshape(-1, S.shape[2]).std(0) + 1e-8
    Sn = (S - mu) / sd
    zm, zs = np.zeros(Z.shape[1]), np.ones(Z.shape[1])
    if Z.shape[1]:
        zm, zs = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        Z = (Z - zm) / zs
    ym, ys = y[tr].mean(), y[tr].std()
    yn = (y - ym) / ys
    preds, val_preds, nets = [], [], []
    for seed in range(seeds):
        p, ep, _, net = _fit_once(Sn, Z, yn, tr, va, seed, kind, None, device, hidden, lr, wd, bs)
        val_preds.append(p)
        if refit:
            p, _, _, net = _fit_once(Sn, Z, yn, np.concatenate([tr, va]), None, seed, kind, max(ep, 5), device, hidden, lr, wd, bs)
        preds.append(p)
        nets.append(net)
    val_r2 = r2_score(yn[va], np.mean(val_preds, 0)[va])
    out = np.mean(preds, 0) * ys + ym
    if return_models:
        norm = {"x_mean": mu, "x_std": sd, "z_mean": zm, "z_std": zs, "y_mean": ym, "y_std": ys}
        return out, val_r2, nets, norm
    return out, val_r2
