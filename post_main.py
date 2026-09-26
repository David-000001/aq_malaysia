"""Recompute SHAP (all horizons) and LOSO (first horizon) for C.MAIN_CFG without
rerunning the full grid. Used after MAIN_CFG was changed; identical code path to
the corresponding block of run_experiments.py (same tuning, seeds and rows).

    python src/post_main.py
"""
import json
import numpy as np
import pandas as pd

import config as C
from features import build_design, columns_for
from models import fit_xgb
from evaluation import all_metrics
from run_experiments import split_masks

R = C.PATHS["results"]
panel = pd.read_csv(C.PATHS["panel"], parse_dates=["date"])
params = json.load(open(R / "params.json"))
fs = C.MAIN_CFG.split("-")[0]
for h in C.HORIZONS:
    design, groups = build_design(panel, h)
    use = design[design["usable"]].reset_index(drop=True)
    dev, test, val, gap = split_masks(use, h)
    keep_dev = dev & ~gap
    val_in_dev = val[keep_dev]
    cols = columns_for(groups, fs, "op")
    import shap
    Xd = use.loc[keep_dev, cols]
    yd = use.loc[keep_dev, "y"].to_numpy(float)
    m, _ = fit_xgb(Xd.to_numpy(float), yd, use.loc[keep_dev, "date"].reset_index(drop=True), val_in_dev)
    Xt = use.loc[test, cols]
    sample = Xt.sample(min(3000, len(Xt)), random_state=C.SEED)
    sv = shap.TreeExplainer(m).shap_values(sample.to_numpy(float))
    imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    imp.to_csv(R / f"shap_importance_h{h}.csv", header=["mean_abs_shap"])
    np.save(R / f"shap_values_h{h}.npy", sv)
    sample.to_csv(R / f"shap_sample_h{h}.csv", index=False)
    print("SHAP", h, imp.head(5).round(3).to_dict())
    if h != C.HORIZONS[0]:
        continue
    pred = pd.read_csv(R / f"predictions_h{h}.csv", parse_dates=["date"])
    cols_l = [c for c in cols if not c.startswith("stn_")]
    p = params[f"h{h}|{fs}-op|XGB"]
    from xgboost import XGBRegressor
    rows = []
    for sid in use["station_id"].unique():
        tr = keep_dev & (use["station_id"] != sid).to_numpy()
        te = test & (use["station_id"] == sid).to_numpy()
        mm = XGBRegressor(tree_method="hist", random_state=C.SEED, n_jobs=-1, **p)
        mm.fit(use.loc[tr, cols_l].to_numpy(float), use.loc[tr, "y"].to_numpy(float))
        yp = mm.predict(use.loc[te, cols_l].to_numpy(float))
        yt = use.loc[te, "y"].to_numpy(float); ps = use.loc[te, "persist"].to_numpy(float)
        pooled = pred.loc[pred["station_id"] == sid]
        rows.append(dict(station_id=sid, n=int(te.sum()), RMSE_loso=all_metrics(yt, yp, ps)["RMSE"],
                         RMSE_pooled=all_metrics(pooled["y"], pooled[f"{fs}-op|XGB"], pooled["persist"])["RMSE"],
                         RMSE_persist=all_metrics(yt, ps, ps)["RMSE"]))
    pd.DataFrame(rows).to_csv(R / "loso.csv", index=False)
    print(pd.DataFrame(rows).round(2))
