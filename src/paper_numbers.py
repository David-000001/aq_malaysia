"""Compute every number cited in the paper TEXT and write them as LaTeX macros
to paper/tables/numbers.tex. The manuscript uses these macros instead of typed
numbers, so the text, the tables and the results can never disagree.

Called automatically by report.py. Placeholder mode defines each macro as a red
[name] so the paper compiles before any run.
"""
import json

import numpy as np
import pandas as pd

import config as C

R, T = C.PATHS["results"], C.PATHS["tables"]

MACROS = [
    "NStations", "StudyPeriod", "DevPeriod", "TestPeriod", "NDevRows", "NTestRows",
    "PersistRMSEa", "PersistRMSEb", "BestModelA", "BestModelB", "BestRMSEa", "BestRMSEb",
    "BestSkillA", "BestSkillB", "MonlyRMSEa", "MonlyRtwoA", "MonlyVsPersistA",
    "MetGainA", "MetGainB", "MetGainTrend", "OracleGainA", "OracleGainB",
    "HazeN", "HazeRMSEa", "HazeBiasA", "HazeBiasWord", "PODa", "FARa", "CSIa", "CSIpersistA",
    "FireGainAllA", "FireGainHazeA", "FireWord",
    "DMlstmxgbA", "DMlstmxgbPa", "LSTMverdictA", "LSTMseedSDa",
    "NbeatPersist", "NcompPersist",
    "LOSOdelta", "LOSOmin", "LOSOmax", "LOSObeat", "LOSOn",
    "APIpreBreak", "APIpostBreak", "HazeThr",
]


def _fmt(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"$-${abs(x):.{d}f}" if x < 0 else f"{x:.{d}f}"


def _best(m, h, fs, mode):
    sub = m[(m.h == h) & (m.features == fs) & (m.met_mode == mode)]
    return None if sub.empty else sub.loc[sub.RMSE.idxmin()]


def _pct_drop(a, b):
    """% RMSE reduction going from row a to row b."""
    # computed from the 2-dp RMSEs printed in the tables, so readers can recompute
    return None if a is None or b is None else 100 * (1 - round(b.RMSE, 2) / round(a.RMSE, 2))


def compute():
    m = pd.read_csv(R / "metrics.csv")
    meta = json.load(open(R / "meta.json"))
    dm = pd.read_csv(R / "dm_tests.csv")
    ds = pd.read_csv(R / "data_summary.csv")
    hA, hB = C.HORIZONS[0], C.HORIZONS[-1]
    P = meta["period"]
    fire = ((m.features == "HMF") & (m.met_mode == "op")).any()
    main_fs = C.MAIN_CFG.split("-")[0]
    v = {}
    v["NStations"] = str(len(ds))
    v["StudyPeriod"] = f"{P['start'][:4]}--{P['test_end'][:4]}"
    v["DevPeriod"] = f"{P['start'][:4]}--{P['dev_end'][:4]}"
    ts, te = P["test_start"][:4], P["test_end"][:4]
    v["TestPeriod"] = ts if ts == te else f"{ts}--{te}"
    v["NDevRows"] = f"{meta['counts'][f'h{hA}']['dev']:,}"
    v["NTestRows"] = f"{meta['counts'][f'h{hA}']['test']:,}"

    pers = {h: m[(m.h == h) & (m.model == "Persistence")].iloc[0] for h in (hA, hB)}
    v["PersistRMSEa"], v["PersistRMSEb"] = _fmt(pers[hA].RMSE), _fmt(pers[hB].RMSE)
    bA, bB = _best(m, hA, main_fs, "op"), _best(m, hB, main_fs, "op")
    v["BestModelA"], v["BestModelB"] = bA.model, bB.model
    v["BestRMSEa"], v["BestRMSEb"] = _fmt(bA.RMSE), _fmt(bB.RMSE)
    v["BestSkillA"], v["BestSkillB"] = _fmt(100 * bA.Skill, 1), _fmt(100 * bB.Skill, 1)

    mo = _best(m, hA, "M", "op")
    v["MonlyRMSEa"] = _fmt(mo.RMSE) if mo is not None else "--"
    v["MonlyRtwoA"] = _fmt(mo.R2) if mo is not None else "--"
    v["MonlyVsPersistA"] = "--" if mo is None else (
        "lower than" if mo.RMSE < pers[hA].RMSE else "higher than")

    gA = _pct_drop(_best(m, hA, "H", "op"), _best(m, hA, "HM", "op"))
    gB = _pct_drop(_best(m, hB, "H", "op"), _best(m, hB, "HM", "op"))
    v["MetGainA"], v["MetGainB"] = _fmt(gA, 1), _fmt(gB, 1)
    v["MetGainTrend"] = "--" if gA is None or gB is None else ("grows" if gB > gA else "shrinks")
    v["OracleGainA"] = _fmt(_pct_drop(_best(m, hA, "HM", "op"), _best(m, hA, "HM", "oracle")), 1)
    v["OracleGainB"] = _fmt(_pct_drop(_best(m, hB, "HM", "op"), _best(m, hB, "HM", "oracle")), 1)

    v["HazeN"] = str(int(bA.n_events))
    v["HazeRMSEa"], v["HazeBiasA"] = _fmt(bA.RMSE_haze), _fmt(bA.Bias_haze, 1)
    v["HazeBiasWord"] = "under-predicted" if bA.Bias_haze < 0 else "over-predicted"
    v["PODa"], v["FARa"], v["CSIa"] = _fmt(bA.POD), _fmt(bA.FAR), _fmt(bA.CSI)
    v["CSIpersistA"] = _fmt(pers[hA].CSI)

    if fire:
        hm = m[(m.h == hA) & (m.features == "HM") & (m.met_mode == "op") & (m.model == bA.model)]
        hmf = m[(m.h == hA) & (m.features == "HMF") & (m.met_mode == "op") & (m.model == bA.model)]
        if len(hm) and len(hmf):
            hm, bA_f = hm.iloc[0], hmf.iloc[0]
            bA_save, bA = bA, bA_f
            fa = 100 * (1 - round(bA.RMSE, 2) / round(hm.RMSE, 2))
            fh = 100 * (1 - round(bA.RMSE_haze, 2) / round(hm.RMSE_haze, 2))
            v["FireGainAllA"], v["FireGainHazeA"] = _fmt(fa, 1), _fmt(fh, 1)
            v["FireWord"] = "reduced" if fh > 0 else "increased"
            bA = bA_save
    for k in ("FireGainAllA", "FireGainHazeA", "FireWord"):
        v.setdefault(k, "--")

    cfg = f"{main_fs}-op"
    d = dm[(dm.h == hA) & (dm.config == cfg) & (dm.a == "LSTM") & (dm.b == "XGB")]
    if len(d):
        s, p = d.iloc[0].stat, d.iloc[0].p
        v["DMlstmxgbA"], v["DMlstmxgbPa"] = _fmt(s), _fmt(p, 3)
        v["LSTMverdictA"] = ("not significantly different from XGBoost" if p >= 0.05 else
                             "significantly more accurate than XGBoost" if s < 0 else
                             "significantly less accurate than XGBoost")
    else:
        v["DMlstmxgbA"] = v["DMlstmxgbPa"] = v["LSTMverdictA"] = "--"
    lr = m[(m.h == hA) & (m.features == main_fs) & (m.met_mode == "op") & (m.model == "LSTM")]
    v["LSTMseedSDa"] = _fmt(lr.iloc[0].RMSE_seed_sd) if len(lr) and "RMSE_seed_sd" in lr else "--"

    dp = dm[(dm.config == cfg) & (dm.b == "Persistence")]
    v["NbeatPersist"] = str(int(((dp.p < 0.05) & (dp.stat < 0)).sum()))
    v["NcompPersist"] = str(len(dp))

    lf = R / "loso.csv"
    if lf.exists():
        lo = pd.read_csv(lf)
        dl = lo.RMSE_loso - lo.RMSE_pooled
        v["LOSOdelta"], v["LOSOmin"], v["LOSOmax"] = _fmt(dl.mean()), _fmt(dl.min()), _fmt(dl.max())
        v["LOSObeat"], v["LOSOn"] = str(int((lo.RMSE_loso < lo.RMSE_persist).sum())), str(len(lo))
    for k in ("LOSOdelta", "LOSOmin", "LOSOmax", "LOSObeat", "LOSOn"):
        v.setdefault(k, "--")
    bf = R / "api_break.json"
    if bf.exists():
        b = json.load(open(bf))
        v["APIpreBreak"], v["APIpostBreak"] = f"{b['pre']:.0f}", f"{b['post']:.0f}"
    else:
        v["APIpreBreak"] = v["APIpostBreak"] = "--"
    v["HazeThr"] = f"{C.HAZE_THRESHOLD:.0f}"
    return v


def write(placeholder=False, synthetic=False):
    T.mkdir(parents=True, exist_ok=True)
    if placeholder:
        vals = {k: f"\\res{{{k}}}" for k in MACROS}
    else:
        vals = compute()
        if synthetic:
            vals = {k: f"\\textcolor{{red}}{{{x}\\textsuperscript{{SYN}}}}" for k, x in vals.items()}
    lines = ["% AUTO-GENERATED by src/paper_numbers.py -- do not edit"]
    for k in MACROS:
        val = vals.get(k, "\\res{" + k + "}")
        lines.append("\\newcommand{\\" + k + "}{" + str(val) + "\\xspace}")
    (T / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("wrote", T / "numbers.tex")
    return vals


if __name__ == "__main__":
    import sys
    write(placeholder="--placeholder" in sys.argv)
