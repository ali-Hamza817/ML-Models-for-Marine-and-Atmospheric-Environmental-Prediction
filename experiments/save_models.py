"""Refit every member of the final ensembles (both fit protocols) and save the weights to models/improved/.
Each refit is checked against the stored predictions that produced REPORT.md."""
import json
import sys

import joblib
import numpy as np
import torch
from sklearn.metrics import r2_score

import finalize as FZ
import improve_trees as IT
import seqnet
from common import ROOT, sequence_splits, tabular_splits

OUT = ROOT / "models" / "improved"
gpu = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
tasks = sys.argv[2:] or list(FZ.POOLS)
for task in tasks:
    pool = FZ.load_pool(FZ.POOLS[task])
    c = next(iter(pool.values()))
    y, va, te = c["y"], c["va"], c["te"]
    w = FZ.caruana({n: v["pred_train"][va] for n, v in pool.items()}, y[va])
    d = OUT / task.replace(" (leak-free features)", "__leakfree")
    d.mkdir(parents=True, exist_ok=True)
    manifest = {"task": task, "ensemble_weights": w, "members": {}}
    for name, wt in w.items():
        stem, kind = name.rsplit("__", 1)
        if kind in ("gru", "lstm", "cnn"):
            ds, rep = stem.split("__")
            S, _, split = sequence_splits(ds)
            Z = tabular_splits(ds)[0].to_numpy(float)[30:30 + len(y)] if rep == "hybrid" else np.zeros((len(y), 0))
            files = {}
            for tag, refit in (("train_only", False), ("train_val", True)):
                p, _, nets, norm = seqnet.fit_predict(S, Z, y, split, kind=kind, seeds=5, refit=refit, device=gpu, return_models=True)
                f = d / f"{name}__{tag}.pt"
                torch.save({"kind": kind, "hidden": 128, "n_ch": S.shape[2], "n_static": Z.shape[1], "norm": norm,
                            "state_dicts": [{k: v.cpu() for k, v in n.state_dict().items()} for n in nets]}, f)
                files[tag] = (f.name, p)
        else:
            X, _, split = IT.design(stem.replace("__", ":"))
            cfg = json.loads((FZ.PRED / f"{name}.json").read_text())
            ma, mb = IT.fit_all(X, y, split, kind, cfg["params"], cfg["n"], gpu)
            files = {}
            for tag, m in (("train_only", ma), ("train_val", mb)):
                f = d / f"{name}__{tag}.joblib"
                joblib.dump(m, f, compress=3)
                files[tag] = (f.name, m.predict(X))
        check = {}
        for tag, key in (("train_only", "pred_train"), ("train_val", "pred_trainval")):
            fname, p = files[tag]
            check[tag] = {"file": fname, "test_R2_refit": r2_score(y[te], p[te]), "test_R2_reported": r2_score(y[te], pool[name][key][te])}
        manifest["members"][name] = {"weight": wt, **check}
        print(task, name, {k: (round(v["test_R2_refit"], 4), round(v["test_R2_reported"], 4)) for k, v in check.items()}, flush=True)
    (d / "ensemble.json").write_text(json.dumps(manifest, indent=2))
