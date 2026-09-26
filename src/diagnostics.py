"""Post-hoc diagnostics requested in review (no refitting; uses results/*.csv).

    python src/diagnostics.py        # after run_experiments.py

Writes results/diagnostics.json, results/dm_holm.csv, results/mdiag.csv and
paper/tables/numbers_extra.tex (LaTeX macros used in the text).

Contents
  * paired moving-block bootstrap (14-day blocks, 1000 reps) CIs for the skill
    score (1 - RMSE/RMSE_persistence) and for CSI, resampling calendar days so
    that all stations and both forecasts of a day stay together;
  * error conditioned on the FORECAST exceeding the threshold (forecaster's
    dilemma, Lerch et al. 2017), alongside the observed-exceedance conditioning;
  * meteorology-only diagnostics: Pearson r, Murphy MSE decomposition, and skill
    on anomalies from the trailing 30-day mean of observed API;
  * Holm-adjusted DM p-values within the main-configuration family;
  * haze-episode structure (distinct episodes vs station-days);
  * standardised ridge coefficients.
"""
import json

import numpy as np
import pandas as pd

import config as C

R, T = C.PATHS["results"], C.PATHS["tables"]
THR = C.HAZE_THRESHOLD
BLOCK, NREP = 14, 1000


def _main_cfg(m):
    return C.MAIN_CFG


def _block_idx(T_, rng):
    nb = int(np.ceil(T_ / BLOCK))
    starts = rng.integers(0, T_ - BLOCK + 1, nb)
    return (starts[:, None] + np.arange(BLOCK)[None, :]).ravel()[:T_]


def paired_bootstrap(pred, col):
    """CI for skill score and CSI; resample days (blocks), keep stations together."""
    rng = np.random.default_rng(C.SEED)
    g = pred.groupby("date")
    days = list(g.groups.keys())
    y = {d: pred.loc[idx, "y"].to_numpy() for d, idx in g.groups.items()}
    p = {d: pred.loc[idx, col].to_numpy() for d, idx in g.groups.items()}
    q = {d: pred.loc[idx, "persist"].to_numpy() for d, idx in g.groups.items()}
    # per-day sufficient statistics
    se_m = np.array([np.sum((y[d] - p[d]) ** 2) for d in days])
    se_p = np.array([np.sum((y[d] - q[d]) ** 2) for d in days])
    n = np.array([len(y[d]) for d in days])
    hit = np.array([np.sum((y[d] >= THR) & (p[d] >= THR)) for d in days])
    mis = np.array([np.sum((y[d] >= THR) & (p[d] < THR)) for d in days])
    fa = np.array([np.sum((y[d] < THR) & (p[d] >= THR)) for d in days])
    Td = len(days)
    ss, cs = np.empty(NREP), np.empty(NREP)
    for i in range(NREP):
        ix = _block_idx(Td, rng)
        ss[i] = 1 - np.sqrt(se_m[ix].sum() / n[ix].sum()) / np.sqrt(se_p[ix].sum() / n[ix].sum())
        den = hit[ix].sum() + mis[ix].sum() + fa[ix].sum()
        cs[i] = hit[ix].sum() / den if den else np.nan
    return (float(np.percentile(ss, 2.5)), float(np.percentile(ss, 97.5)),
            float(np.nanpercentile(cs, 2.5)), float(np.nanpercentile(cs, 97.5)))


def murphy(y, p):
    y, p = np.asarray(y), np.asarray(p)
    r = np.corrcoef(y, p)[0, 1]
    sy, sp = y.std(), p.std()
    return dict(r=float(r), mse=float(np.mean((y - p) ** 2)),
                bias2=float((p.mean() - y.mean()) ** 2),
                cond=float((sp - r * sy) ** 2),          # conditional bias / scale
                unexpl=float((1 - r ** 2) * sy ** 2))    # irreducible given r


def trailing_baseline(panel, window=30):
    out = []
    for sid, g in panel.groupby("station_id"):
        g = g.sort_values("date")
        b = g["pm25"].rolling(window, min_periods=20).mean()   # up to and incl. issue day
        out.append(pd.DataFrame({"station_id": sid, "date": g["date"], "base": b}))
    return pd.concat(out)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def episodes(pred):
    """Distinct unhealthy episodes: runs of target days with >=1 station >= THR,
    merged when separated by <= 2 days."""
    tday = pd.to_datetime(pred["date"]) + pd.to_timedelta(1, "D")
    ex = sorted(set(tday[pred["y"] >= THR]))
    eps, cur = [], []
    for d in ex:
        if cur and (d - cur[-1]).days > 3:
            eps.append(cur); cur = []
        cur.append(d)
    if cur:
        eps.append(cur)
    counts = []
    for e in eps:
        s, t = e[0], e[-1]
        mask = (tday >= s) & (tday <= t) & (pred["y"] >= THR)
        counts.append(dict(start=str(s.date()), end=str(t.date()), days=len(e),
                           station_days=int(mask.sum())))
    return counts


def fmt(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"$-${abs(x):.{d}f}" if x < 0 else f"{x:.{d}f}"


def main():
    m = pd.read_csv(R / "metrics.csv")
    dm = pd.read_csv(R / "dm_tests.csv")
    panel = pd.read_csv(C.PATHS["panel"], parse_dates=["date"])
    cfg = _main_cfg(m)
    out, macros, mrows = {"main_cfg": cfg}, {}, []
    base = trailing_baseline(panel)
    for h in C.HORIZONS:
        pred = pd.read_csv(R / f"predictions_h{h}.csv", parse_dates=["date"])
        pred = pred.merge(base, on=["station_id", "date"], how="left")
        # --- bootstrap CIs for the main configuration -------------------------
        for mod in ["LR", "RF", "XGB", "LSTM"]:
            col = f"{cfg}|{mod}"
            s_lo, s_hi, c_lo, c_hi = paired_bootstrap(pred, col)
            fc = pred[col] >= THR
            out[f"h{h}|{mod}"] = dict(skill_lo=s_lo, skill_hi=s_hi, csi_lo=c_lo, csi_hi=c_hi,
                                      n_fc_exceed=int(fc.sum()),
                                      rmse_fc_exceed=float(np.sqrt(np.mean((pred.y[fc] - pred[col][fc]) ** 2))) if fc.any() else None,
                                      bias_fc_exceed=float(np.mean(pred[col][fc] - pred.y[fc])) if fc.any() else None)
        # persistence CSI CI and forecast-conditioned error
        pred["_p"] = pred["persist"]
        _, _, c_lo, c_hi = paired_bootstrap(pred.assign(persist=pred["persist"]), "persist")
        fc = pred["persist"] >= THR
        out[f"h{h}|Persistence"] = dict(csi_lo=c_lo, csi_hi=c_hi, n_fc_exceed=int(fc.sum()),
                                        rmse_fc_exceed=float(np.sqrt(np.mean((pred.y[fc] - pred.persist[fc]) ** 2))) if fc.any() else None)
        # --- meteorology-only diagnostics -------------------------------------
        ok = pred["base"].notna()
        for tag in ["M-op", "M-oracle", "H-op", cfg]:
            for mod in ["LR", "RF", "XGB", "LSTM"]:
                col = f"{tag}|{mod}"
                if col not in pred:
                    continue
                md = murphy(pred.y, pred[col])
                ya, pa = pred.y[ok] - pred.base[ok], pred[col][ok] - pred.base[ok]
                md.update(h=h, config=tag, model=mod,
                          anom_r=float(np.corrcoef(ya, pa)[0, 1]),
                          anom_r2=float(1 - np.sum((ya - pa) ** 2) / np.sum((ya - ya.mean()) ** 2)))
                mrows.append(md)
        ya, pa = pred.y[ok] - pred.base[ok], pred.persist[ok] - pred.base[ok]
        mrows.append(dict(h=h, config="-", model="Persistence", **murphy(pred.y, pred.persist),
                          anom_r=float(np.corrcoef(ya, pa)[0, 1]),
                          anom_r2=float(1 - np.sum((ya - pa) ** 2) / np.sum((ya - ya.mean()) ** 2))))
        out[f"h{h}|episodes"] = episodes(pred)
    md = pd.DataFrame(mrows)
    md.to_csv(R / "mdiag.csv", index=False)

    # --- Holm within the main-configuration DM family ---------------------------
    fam = dm[dm.config == cfg].copy()
    fam["p_holm"] = holm(fam["p"].to_numpy())
    fam.to_csv(R / "dm_holm.csv", index=False)
    out["holm_sig_vs_persist"] = int(((fam.b == "Persistence") & (fam.p_holm < 0.05) & (fam.stat < 0)).sum())
    out["raw_sig_vs_persist"] = int(((fam.b == "Persistence") & (fam.p < 0.05) & (fam.stat < 0)).sum())

    # --- level shift DEV vs TEST --------------------------------------------------
    P = C.PERIOD
    dev = panel[(panel.date >= P["start"]) & (panel.date <= P["dev_end"])]["pm25"]
    tst = panel[(panel.date >= P["test_start"]) & (panel.date <= P["test_end"])]["pm25"]
    out["dev_mean"], out["test_mean"] = float(dev.mean()), float(tst.mean())

    # --- LR coefficients -----------------------------------------------------------
    hA = C.HORIZONS[0]
    cf = pd.read_csv(R / f"lr_coef_h{hA}_{cfg}.csv", index_col=0)["std_coef"]
    cf = cf[~cf.index.str.startswith("stn_")]
    top = cf.reindex(cf.abs().sort_values(ascending=False).index).head(8)
    out["lr_coef_top"] = {k: float(v) for k, v in top.items()}
    json.dump(out, open(R / "diagnostics.json", "w"), indent=1, default=str)

    # --- macros ---------------------------------------------------------------------
    hB = C.HORIZONS[-1]
    def best_m(h, tag):
        sub = m[(m.h == h) & (m.features + "-" + m.met_mode == tag)]
        return sub.loc[sub.RMSE.idxmin(), "model"]
    for h, s in ((hA, "a"), (hB, "b")):
        bm = best_m(h, "M-op")
        r = md[(md.h == h) & (md.config == "M-op") & (md.model == bm)].iloc[0]
        rp = md[(md.h == h) & (md.model == "Persistence")].iloc[0]
        macros[f"MonlyR{s}"] = fmt(r.r)
        macros[f"MonlyBiasShare{s}"] = fmt(100 * r.bias2 / r.mse, 0)
        macros[f"MonlyAnomR{s}"] = fmt(r.anom_r)
        macros[f"MonlyAnomRtwo{s}"] = fmt(r.anom_r2)
        macros[f"PersistAnomRtwo{s}"] = fmt(rp.anom_r2)
        bh = best_m(h, cfg)
        rh = md[(md.h == h) & (md.config == cfg) & (md.model == bh)].iloc[0]
        macros[f"BestAnomRtwo{s}"] = fmt(rh.anom_r2)
        d = out[f"h{h}|{bh}"]
        macros[f"BestSkillLo{s}"] = fmt(100 * d["skill_lo"], 1)
        macros[f"FcExceedN{s}"] = str(d["n_fc_exceed"])
        macros[f"FcExceedRMSE{s}"] = fmt(d["rmse_fc_exceed"]) if d["rmse_fc_exceed"] is not None else "--"
        macros[f"FcExceedBias{s}"] = fmt(d["bias_fc_exceed"], 1) if d["bias_fc_exceed"] is not None else "--"
        pp = out[f"h{h}|Persistence"]
        macros[f"PersistFcExceedRMSE{s}"] = fmt(pp["rmse_fc_exceed"]) if pp["rmse_fc_exceed"] is not None else "--"
        macros[f"BestSkillHi{s}"] = fmt(100 * d["skill_hi"], 1)
    macros["DevMean"], macros["TestMean"] = fmt(out["dev_mean"], 1), fmt(out["test_mean"], 1)
    macros["NholmBeat"] = str(out["holm_sig_vs_persist"])
    ep = out[f"h{hA}|episodes"]
    macros["NEpisodes"] = str(len(ep))
    big = max(ep, key=lambda e: e["station_days"]) if ep else None
    macros["BigEpisodeN"] = str(big["station_days"]) if big else "--"
    macros["BigEpisodeSpan"] = (f"{pd.Timestamp(big['start']):%-d~%B}--{pd.Timestamp(big['end']):%-d~%B~%Y}"
                                if big else "--")
    lines = ["% AUTO-GENERATED by src/diagnostics.py -- do not edit"]
    lines += ["\\newcommand{\\" + k + "}{" + v + "\\xspace}" for k, v in macros.items()]
    (T / "numbers_extra.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(macros, indent=1))


if __name__ == "__main__":
    main()
