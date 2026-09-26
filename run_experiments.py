"""Run the full experimental grid and write every artefact the paper needs.

    python src/run_experiments.py            # full run on data/processed/panel.csv
    python src/run_experiments.py --fast     # quick sanity run (fewer seeds/iters)
    python src/run_experiments.py --panel data/processed/synthetic_panel.csv --synthetic

Outputs in results/:
    metrics.csv, dm_tests.csv, predictions_h{h}.csv, params.json,
    loso.csv, shap_importance_h{h}.csv, data_summary.csv, meta.json
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import config as C
from features import build_design, columns_for, build_sequences
from models import fit_linear, fit_rf, fit_xgb, fit_predict_lstm
from evaluation import all_metrics, dm_test, block_bootstrap_rmse


def split_masks(use: pd.DataFrame, h: int):
    P = C.PERIOD
    d = use["date"]
    dev_end = pd.Timestamp(P["dev_end"])
    test_start, test_end = pd.Timestamp(P["test_start"]), pd.Timestamp(P["test_end"])
    # purge: target day t+h of a DEV row must also lie in DEV
    dev = (d >= pd.Timestamp(P["start"])) & (d <= dev_end - pd.Timedelta(days=h))
    test = (d >= test_start) & (d <= test_end - pd.Timedelta(days=h))
    val_start = dev_end - pd.DateOffset(months=P["val_months"]) + pd.Timedelta(days=1)
    val = dev & (d >= val_start)
    # purge gap between DEV-train and validation block
    gap = dev & (d < val_start) & (d > val_start - pd.Timedelta(days=h + 1))
    return dev.to_numpy(), test.to_numpy(), val.to_numpy(), gap.to_numpy()


def data_summary(panel: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby("station_id")["pm25"]
    s = pd.DataFrame(dict(coverage=g.apply(lambda x: x.notna().mean()), mean=g.mean(),
                          sd=g.std(), p95=g.quantile(0.95), max=g.max(),
                          pct_exceed=g.apply(lambda x: (x >= C.HAZE_THRESHOLD).mean() * 100)))
    st = pd.read_csv(C.PATHS["stations"]).set_index("station_id")
    return st[["station_name", "state", "type"]].join(s, how="right")


def main(args):
    t0 = time.time()
    res = C.PATHS["results"]
    res.mkdir(parents=True, exist_ok=True)
    if args.fast:
        C.N_ITER_SEARCH = 2
        C.RF_TREES = [60]
        C.LSTM["seeds"] = [0, 1]
        C.LSTM["max_epochs"] = 15
        C.LSTM["patience"] = 4
    if args.horizons:
        C.HORIZONS = [int(x) for x in args.horizons.split(",")]
    if args.configs:
        C.EXPERIMENTS = [tuple(c.split("-")) for c in args.configs.split(",")]
    panel = pd.read_csv(args.panel, parse_dates=["date"])
    data_summary(panel).to_csv(res / "data_summary.csv")

    metrics, dms, params, counts = [], [], {}, {}
    for h in C.HORIZONS:
        design, groups = build_design(panel, h)
        use = design[design["usable"]].reset_index(drop=True)
        dev, test, val, gap = split_masks(use, h)
        keep_dev = dev & ~gap
        val_in_dev = val[keep_dev]
        print(f"\n=== h={h}: dev rows={keep_dev.sum()}, test rows={test.sum()}, "
              f"val rows={val.sum()}")
        counts[f"h{h}"] = dict(dev=int(keep_dev.sum()), test=int(test.sum()),
                               val=int(val.sum()))
        pred = use.loc[test, ["station_id", "date", "y", "persist"]].reset_index(drop=True)

        for fset, mode in C.EXPERIMENTS:
            cols = columns_for(groups, fset, mode)
            if cols is None:
                print(f"skip {fset}-{mode} (no data)")
                continue
            tag = f"{fset}-{mode}"
            Xd = use.loc[keep_dev, cols].to_numpy(float)
            yd = use.loc[keep_dev, "y"].to_numpy(float)
            dd = use.loc[keep_dev, "date"].reset_index(drop=True)
            Xt = use.loc[test, cols].to_numpy(float)

            m, p = fit_linear(Xd, yd, dd)
            pred[f"{tag}|LR"] = m.predict(Xt); params[f"h{h}|{tag}|LR"] = p
            pd.Series(m.named_steps["ridge"].coef_, index=cols, name="std_coef").to_csv(
                res / f"lr_coef_h{h}_{tag}.csv")
            m, p = fit_rf(Xd, yd, dd)
            pred[f"{tag}|RF"] = m.predict(Xt); params[f"h{h}|{tag}|RF"] = p
            m, p = fit_xgb(Xd, yd, dd, val_in_dev)
            pred[f"{tag}|XGB"] = m.predict(Xt); params[f"h{h}|{tag}|XGB"] = p

            MODELS = ["LR", "RF", "XGB"] if args.no_lstm else ["LR", "RF", "XGB", "LSTM"]
            if args.no_lstm:
                seed_rmse = []
            Xs, Xf, dyn_cols, fut_cols = (None,) * 4 if args.no_lstm else \
                build_sequences(panel, design, groups, fset, mode, h)
            if not args.no_lstm:
                pm, per_seed, info = fit_predict_lstm(Xs[keep_dev], Xf[keep_dev], yd, val_in_dev,
                                                      Xs[test], Xf[test], dates_dev=dd,
                                                      tune=not args.no_lstm_tune)
                pred[f"{tag}|LSTM"] = pm; params[f"h{h}|{tag}|LSTM"] = info
                seed_rmse = [float(np.sqrt(np.mean((pred["y"] - s) ** 2))) for s in per_seed]
            for model in MODELS:
                col = f"{tag}|{model}"
                row = dict(h=h, features=fset, met_mode=mode, model=model,
                           **all_metrics(pred["y"], pred[col], pred["persist"]))
                row["RMSE_lo"], row["RMSE_hi"] = block_bootstrap_rmse(pred, col)
                if model == "LSTM":
                    row["RMSE_seed_sd"] = float(np.std(seed_rmse))
                metrics.append(row)
                s, pv = dm_test(pred, col, "persist", h)
                dms.append(dict(h=h, config=tag, a=model, b="Persistence", stat=s, p=pv))
                if model != "XGB":
                    s, pv = dm_test(pred, col, f"{tag}|XGB", h)
                    dms.append(dict(h=h, config=tag, a=model, b="XGB", stat=s, p=pv))
            print(f"  {tag} done ({time.time() - t0:.0f}s)")

        # persistence reference row
        metrics.append(dict(h=h, features="-", met_mode="-", model="Persistence",
                            **all_metrics(pred["y"], pred["persist"], pred["persist"])))
        pred.to_csv(res / f"predictions_h{h}.csv", index=False)

        # ---- SHAP on the richest operational configuration --------------------
        shap_cfg = C.MAIN_CFG.split("-")[0]
        cols = columns_for(groups, shap_cfg, "op")
        try:
            import shap
            Xd = use.loc[keep_dev, cols]
            yd = use.loc[keep_dev, "y"].to_numpy(float)
            m, _ = fit_xgb(Xd.to_numpy(float), yd,
                           use.loc[keep_dev, "date"].reset_index(drop=True), val_in_dev)
            Xt = use.loc[test, cols]
            sample = Xt.sample(min(3000, len(Xt)), random_state=C.SEED)
            sv = shap.TreeExplainer(m).shap_values(sample.to_numpy(float))
            imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
            imp.to_csv(res / f"shap_importance_h{h}.csv", header=["mean_abs_shap"])
            np.save(res / f"shap_values_h{h}.npy", sv)
            sample.to_csv(res / f"shap_sample_h{h}.csv", index=False)
        except Exception as e:                                   # noqa: BLE001
            print("SHAP failed:", e)

        # ---- Leave-one-station-out (XGB, operational, no station dummies) -----
        if C.RUN_LOSO and h == C.HORIZONS[0] and use["station_id"].nunique() < 3:
            print("LOSO skipped: needs >= 3 stations")
        elif C.RUN_LOSO and h == C.HORIZONS[0]:
            cols = [c for c in columns_for(groups, shap_cfg, "op") if not c.startswith("stn_")]
            key = f"h{h}|{shap_cfg}-op|XGB"
            if key not in params:
                print("LOSO skipped: run the", shap_cfg, "-op configuration")
                continue
            p = params[key]
            from xgboost import XGBRegressor
            rows = []
            for sid in use["station_id"].unique():
                tr = keep_dev & (use["station_id"] != sid).to_numpy()
                te = test & (use["station_id"] == sid).to_numpy()
                if te.sum() < 30:
                    continue
                m = XGBRegressor(tree_method="hist", random_state=C.SEED, n_jobs=-1, **p)
                m.fit(use.loc[tr, cols].to_numpy(float), use.loc[tr, "y"].to_numpy(float))
                yp = m.predict(use.loc[te, cols].to_numpy(float))
                yt = use.loc[te, "y"].to_numpy(float)
                ps = use.loc[te, "persist"].to_numpy(float)
                pooled = pred.loc[pred["station_id"] == sid]
                rows.append(dict(station_id=sid, n=int(te.sum()),
                                 RMSE_loso=all_metrics(yt, yp, ps)["RMSE"],
                                 RMSE_pooled=all_metrics(pooled["y"],
                                                         pooled[f"{shap_cfg}-op|XGB"],
                                                         pooled["persist"])["RMSE"],
                                 RMSE_persist=all_metrics(yt, ps, ps)["RMSE"]))
            pd.DataFrame(rows).to_csv(res / "loso.csv", index=False)

    pd.DataFrame(metrics).to_csv(res / "metrics.csv", index=False)
    pd.DataFrame(dms).to_csv(res / "dm_tests.csv", index=False)
    with open(res / "params.json", "w") as f:
        json.dump(params, f, indent=1, default=str)
    with open(res / "meta.json", "w") as f:
        json.dump(dict(synthetic=args.synthetic, fast=args.fast, panel=str(args.panel),
                       source=C.SOURCE, counts=counts,
                       runtime_s=round(time.time() - t0), period=C.PERIOD,
                       horizons=C.HORIZONS, lstm=C.LSTM), f, indent=1)
    print(f"\nfinished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(C.PATHS["panel"]))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no_lstm_tune", action="store_true")
    ap.add_argument("--no_lstm", action="store_true", help="skip the LSTM (sensitivity runs)")
    ap.add_argument("--horizons", help="e.g. 1,3 (default: config.HORIZONS)")
    ap.add_argument("--configs", help="e.g. H-op,HMF-op (default: config.EXPERIMENTS)")
    ap.add_argument("--synthetic", action="store_true",
                    help="marks all outputs as synthetic (tables get a watermark)")
    main(ap.parse_args())
