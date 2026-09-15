"""Tuned tree ensembles for one design matrix.

Protocol (test split never used for any decision):
  * Optuna TPE search per learner, objective = validation R², models fit on train only
  * best config refit (a) on train only, (b) on train+val with boosting rounds scaled by (n_tr+n_va)/n_tr
  * predictions for all rows saved; ensemble weights are chosen later on validation predictions
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import optuna
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score

import features as F
from common import RESULTS, SEED, tabular_splits

optuna.logging.set_verbosity(optuna.logging.WARNING)


def design(name):
    ds, _, rep = name.partition(":")
    if rep in ("window", "hybrid"):
        X, y, split = F.seq_design(ds, rep == "hybrid")
        return X, y, split
    if rep == "context":
        X, y, split, _ = F.era5_context()
    elif rep == "leakfree":
        X, y, split, _ = tabular_splits(ds, F.leakfree_df(ds))
    else:
        X, y, split, _ = tabular_splits(ds)
    return X.to_numpy(float), y.to_numpy(float), split


def lgb_model(p, n):
    return lgb.LGBMRegressor(n_estimators=n, verbose=-1, n_jobs=8, random_state=SEED, subsample_freq=1, **p)


def xgb_model(p, n, gpu):
    return xgb.XGBRegressor(n_estimators=n, tree_method="hist", device=gpu, random_state=SEED, verbosity=0, **p)


def tune(X, y, split, learner, trials, gpu):
    tr, va, _ = split

    def objective(t):
        if learner == "lgb":
            p = dict(learning_rate=t.suggest_float("learning_rate", 0.005, 0.1, log=True),
                     num_leaves=t.suggest_int("num_leaves", 4, 256, log=True),
                     min_child_samples=t.suggest_int("min_child_samples", 3, 200, log=True),
                     colsample_bytree=t.suggest_float("colsample_bytree", 0.2, 1.0),
                     subsample=t.suggest_float("subsample", 0.4, 1.0),
                     reg_lambda=t.suggest_float("reg_lambda", 1e-8, 30, log=True),
                     reg_alpha=t.suggest_float("reg_alpha", 1e-8, 10, log=True),
                     extra_trees=t.suggest_categorical("extra_trees", [False, True]))
            m = lgb_model(p, 6000).fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
                                       callbacks=[lgb.early_stopping(300, verbose=False)])
            t.set_user_attr("n", int(m.best_iteration_ or 6000))
        elif learner == "xgb":
            p = dict(learning_rate=t.suggest_float("learning_rate", 0.005, 0.1, log=True),
                     max_depth=t.suggest_int("max_depth", 2, 10),
                     min_child_weight=t.suggest_float("min_child_weight", 0.5, 100, log=True),
                     colsample_bytree=t.suggest_float("colsample_bytree", 0.2, 1.0),
                     subsample=t.suggest_float("subsample", 0.4, 1.0),
                     reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True),
                     gamma=t.suggest_float("gamma", 1e-8, 1.0, log=True))
            m = xgb_model({**p, "early_stopping_rounds": 300}, 6000, gpu).fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            t.set_user_attr("n", int(m.best_iteration + 1))
        else:
            cls = ExtraTreesRegressor if learner == "et" else RandomForestRegressor
            p = dict(max_features=t.suggest_float("max_features", 0.1, 1.0),
                     min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 30, log=True),
                     max_depth=t.suggest_categorical("max_depth", [None, 8, 16, 24]))
            m = cls(n_estimators=300, n_jobs=16, random_state=SEED, **p).fit(X[tr], y[tr])
            t.set_user_attr("n", 600)
        return r2_score(y[va], m.predict(X[va]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    return study.best_params, study.best_trial.user_attrs["n"], study.best_value


def fit_all(X, y, split, learner, p, n, gpu):
    tr, va, _ = split
    trva = np.concatenate([tr, va])
    if learner in ("lgb", "xgb"):
        make = (lambda k: lgb_model(p, k)) if learner == "lgb" else (lambda k: xgb_model(p, k, gpu))
        a = make(n).fit(X[tr], y[tr])
        b = make(int(round(n * len(trva) / len(tr)))).fit(X[trva], y[trva])
    else:
        cls = ExtraTreesRegressor if learner == "et" else RandomForestRegressor
        a = cls(n_estimators=n, n_jobs=16, random_state=SEED, **p).fit(X[tr], y[tr])
        b = cls(n_estimators=n, n_jobs=16, random_state=SEED, **p).fit(X[trva], y[trva])
    return a, b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("design")  # e.g. rolling_mean, era5_daily:context, hydrographic:hybrid
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--learners", default="lgb,xgb,et,rf")
    ap.add_argument("--gpu", default="cuda:0")
    a = ap.parse_args()
    X, y, split = design(a.design)
    out_dir = RESULTS / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, va, te = split
    for learner in a.learners.split(","):
        trials = a.trials if learner in ("lgb", "xgb") else max(15, a.trials // 3)
        p, n, v = tune(X, y, split, learner, trials, a.gpu)
        ma, mb = fit_all(X, y, split, learner, p, n, a.gpu)
        pa, pb = ma.predict(X), mb.predict(X)
        tag = f"{a.design.replace(':', '__')}__{learner}"
        np.savez(out_dir / f"{tag}.npz", y=y, tr=tr, va=va, te=te, pred_train=pa, pred_trainval=pb)
        (out_dir / f"{tag}.json").write_text(json.dumps({"params": p, "n": n, "val_r2_search": v}, default=str))
        print(f"{a.design:28s} {learner:4s} val={r2_score(y[va], pa[va]):.4f} "
              f"test(train-only)={r2_score(y[te], pa[te]):.4f} test(train+val)={r2_score(y[te], pb[te]):.4f}", flush=True)
