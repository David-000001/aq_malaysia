"""Turn results/*.csv into the LaTeX tables and PDF figures the paper \\input's.

    python src/report.py                 # after run_experiments.py
    python src/report.py --placeholder   # empty tables so the paper compiles pre-run

Nothing in the paper's tables is typed by hand. If the numbers change, rerun.
"""
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import paper_numbers

T, F, R = C.PATHS["tables"], C.PATHS["figures"], C.PATHS["results"]
MODELS = ["Persistence", "LR", "RF", "XGB", "LSTM"]
UNIT_NAME = C.UNIT_TEX if C.UNIT_TEX else "API units"
UNIT_SP = ("\\," + C.UNIT_TEX) if C.UNIT_TEX else ""
DIAG = {}
CFG_ORDER = ["H-op", "M-op", "M-oracle", "HM-op", "HM-oracle", "HMF-op", "HMF-oracle"]
CFG_LABEL = {"H-op": "H", "M-op": "M", "M-oracle": "M$^\\ast$", "HM-op": "H+M",
             "HM-oracle": "H+M$^\\ast$", "HMF-op": "H+M+F", "HMF-oracle": "H+M$^\\ast$+F"}


def wm(synth):
    return "\\textcolor{red}{\\textbf{[SYNTHETIC SMOKE TEST -- NOT RESULTS]}} " if synth else ""


def f(x, d=2):
    if pd.isna(x):
        return "--"
    return f"$-${abs(x):.{d}f}" if round(x, d) < 0 else f"{abs(x) if round(x, d) == 0 else x:.{d}f}"


def write(name, body):
    T.mkdir(parents=True, exist_ok=True)
    (T / name).write_text(body)
    print("wrote", T / name)


# ---------------------------------------------------------------------------
def tab_data(synth):
    s = pd.read_csv(R / "data_summary.csv")
    cov = ("share of days with a valid daily value" if C.SOURCE == "aqicn" else
           f"share of days meeting the {C.MIN_HOURS_PER_DAY}-hour completeness rule")
    lines = [f"{r.station_name} & {r.type[0].upper()} & {f(r.coverage*100,0)} & "
             f"{f(r['mean'],1)} & {f(r.sd,1)} & {f(r.p95,0)} & {f(r['max'],0)} & "
             f"{f(r.pct_exceed,1)} \\\\" for _, r in s.iterrows()]
    write("tab_data.tex", f"""\\begin{{table}}[t]\\centering
\\caption{{{wm(synth)}Stations and daily {C.TARGET_TEX} statistics
({C.PERIOD['start'][:4]}--{C.PERIOD['test_end'][:4]}). U/S = urban/suburban; Cov.\\ = {cov}; Exc.\\ = share of days $\\geq${C.HAZE_THRESHOLD:.0f}{UNIT_SP}.}}
\\label{{tab:data}}\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{llrrrrrr}}\\toprule
Station & & Cov.\\,\\% & Mean & SD & P95 & Max & Exc.\\,\\%\\\\\\midrule
{chr(10).join(lines)}
\\bottomrule\\end{{tabular}}\\end{{table}}
""")


def main_cfg(m):
    return C.MAIN_CFG


def tab_main(m, synth):
    cfg = main_cfg(m)
    fs, mode = cfg.split("-")
    body = []
    for h in C.HORIZONS:
        body.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{Horizon $h={h}$ day(s)}}}}\\\\")
        sub = m[(m.h == h) & (((m.features == fs) & (m.met_mode == mode)) |
                              (m.model == "Persistence"))].set_index("model")
        best = sub["RMSE"].idxmin()
        for mod in MODELS:
            if mod not in sub.index:
                continue
            r = sub.loc[mod]
            rm = f(r.RMSE)
            rm = f"\\textbf{{{rm}}}" if mod == best else rm
            dg = DIAG.get(f"h{h}|{mod}", {})
            sk = (f"{f(100*r.Skill,1)} [{f(100*dg['skill_lo'],1)}, {f(100*dg['skill_hi'],1)}]"
                  if mod != "Persistence" and dg else "0")
            cs = (f"{f(r.CSI,2)} [{f(dg['csi_lo'],2)}, {f(dg['csi_hi'],2)}]" if dg and pd.notna(r.CSI)
                  else f(r.CSI, 2))
            sd = f" $\\pm${f(r.RMSE_seed_sd)}" if mod == "LSTM" and "RMSE_seed_sd" in r \
                and pd.notna(r.RMSE_seed_sd) else ""
            body.append(f"{mod} & {rm}{sd} & {f(r.MAE)} & {f(r.R2,3)} & {sk} & "
                        f"{f(r.RMSE_haze)} & {cs}\\\\")
        body.append("\\midrule")
    body[-1] = "\\bottomrule"
    write("tab_main.tex", f"""\\begin{{table*}}[t]\\centering
\\caption{{{wm(synth)}Test-period accuracy with the operational feature set ({CFG_LABEL[cfg]}).
RMSE/MAE in {UNIT_NAME}; Skill $=100\\,(1-\\mathrm{{RMSE}}/\\mathrm{{RMSE}}_{{\\text{{persistence}}}})$ in \\%;
brackets: 95\\,\\% CI from a paired 14-day moving-block bootstrap over days (all stations of a day
resampled together);
RMSE$_\\text{{haze}}$ on days with observed {C.TARGET_TEX} $\\geq{C.HAZE_THRESHOLD:.0f}$;
CSI = critical success index for exceedance of that threshold. LSTM: mean of
{len(json.load(open(R/'meta.json'))['lstm']['seeds'])}-seed ensemble, $\\pm$ = SD of single-seed RMSE.}}
\\label{{tab:main}}\\small
\\begin{{tabular}}{{lrrrcrc}}\\toprule
Model & RMSE & MAE & $R^2$ & Skill (\\%) & RMSE$_\\text{{haze}}$ & CSI\\\\\\midrule
{chr(10).join(body)}
\\end{{tabular}}\\end{{table*}}
""")


def tab_ablation(m, synth):
    rows = []
    for cfg in CFG_ORDER:
        fs, mode = cfg.split("-")
        cells = []
        for h in C.HORIZONS:
            sub = m[(m.h == h) & (m.features == fs) & (m.met_mode == mode)].set_index("model")
            if sub.empty:
                cells += ["--"] * 4
                continue
            best = sub["RMSE"].idxmin()
            for mod in ["LR", "RF", "XGB", "LSTM"]:
                v = f(sub.loc[mod, "RMSE"]) if mod in sub.index else "--"
                cells.append(f"\\textbf{{{v}}}" if mod == best else v)
        if any(c != "--" for c in cells):
            rows.append(f"{CFG_LABEL[cfg]} & " + " & ".join(cells) + "\\\\")
    pers = [f(m[(m.h == h) & (m.model == "Persistence")]["RMSE"].iloc[0]) for h in C.HORIZONS]
    rows.append("\\midrule Persistence & " + " & ".join(
        f"\\multicolumn{{4}}{{c}}{{{p}}}" for p in pers) + "\\\\")
    hdr = " & ".join(f"\\multicolumn{{4}}{{c}}{{$h={h}$}}" for h in C.HORIZONS)
    cm = " ".join(f"\\cmidrule(lr){{{2+4*i}-{5+4*i}}}" for i in range(len(C.HORIZONS)))
    write("tab_ablation.tex", f"""\\begin{{table}}[t]\\centering
\\caption{{{wm(synth)}Feature-set ablation: test RMSE ({UNIT_NAME}). H = {C.TARGET_TEX}
history, M = meteorology up to issue day, M$^\\ast$ = M plus target-day reanalysis
meteorology (perfect-forecast upper bound){", F = regional fire hotspots" if (m.features == "HMF").any() else ""}. Bold = best per row.}}
\\label{{tab:ablation}}\\scriptsize\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{l{'rrrr' * len(C.HORIZONS)}}}\\toprule
 & {hdr}\\\\ {cm}
Features & {' & '.join(['LR & RF & XGB & LSTM'] * len(C.HORIZONS))}\\\\\\midrule
{chr(10).join(rows)}
\\bottomrule\\end{{tabular}}\\end{{table}}
""")


def tab_dm(synth):
    m = pd.read_csv(R / "metrics.csv")
    cfg = main_cfg(m)
    d = pd.read_csv(R / "dm_holm.csv")
    rows = []
    for _, r in d.iterrows():
        star = "$^{**}$" if r.p_holm < 0.01 else "$^{*}$" if r.p_holm < 0.05 else ""
        rows.append(f"{r.h} & {r.a} vs {r.b} & {f(r.stat)} & {f(r.p,3)} & {f(r.p_holm,3)}{star}\\\\")
    write("tab_dm.tex", f"""\\begin{{table}}[t]\\centering
\\caption{{{wm(synth)}Diebold--Mariano tests (squared-error loss, HLN correction, station-averaged
daily loss differential) for configuration {CFG_LABEL[cfg]}. Negative statistic: first model
more accurate. $p_\\text{{H}}$: Holm-adjusted over the {len(d)} tests of the table;
$^{{*}}p_\\text{{H}}<0.05$, $^{{**}}p_\\text{{H}}<0.01$.}}
\\label{{tab:dm}}\\small
\\begin{{tabular}}{{clrrr}}\\toprule
$h$ & Comparison & DM & $p$ & $p_\\text{{H}}$\\\\\\midrule
{chr(10).join(rows)}
\\bottomrule\\end{{tabular}}\\end{{table}}
""")


def tab_loso(synth):
    p = R / "loso.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    st = pd.read_csv(C.PATHS["stations"]).set_index("station_id")["station_name"]
    rows = [f"{st.get(r.station_id, r.station_id)} & {r.n} & {f(r.RMSE_pooled)} & "
            f"{f(r.RMSE_loso)} & {f(r.RMSE_persist)}\\\\" for _, r in d.iterrows()]
    rows.append(f"\\midrule Mean & -- & {f(d.RMSE_pooled.mean())} & {f(d.RMSE_loso.mean())} & "
                f"{f(d.RMSE_persist.mean())}\\\\")
    write("tab_loso.tex", f"""\\begin{{table}}[t]\\centering
\\caption{{{wm(synth)}Spatial transfer ($h={C.HORIZONS[0]}$, XGBoost, operational features):
pooled model vs.\\ leave-one-station-out (LOSO) model that never saw the target station
during training. Test RMSE in {UNIT_NAME}.}}
\\label{{tab:loso}}\\small
\\begin{{tabular}}{{lrrrr}}\\toprule
Station & $n$ & Pooled & LOSO & Persist.\\\\\\midrule
{chr(10).join(rows)}
\\bottomrule\\end{{tabular}}\\end{{table}}
""")


# ---------------------------------------------------------------------------
def fig_timeseries(m, synth):
    h = C.HORIZONS[0]
    p = pd.read_csv(R / f"predictions_h{h}.csv", parse_dates=["date"])
    cfg = main_cfg(m)
    sid = (p.assign(e=p.y >= C.HAZE_THRESHOLD).groupby("station_id")["e"].sum().idxmax())
    q = p[p.station_id == sid].set_index("date").sort_index()
    # 120-day window centred on the worst test-period day
    peak = q["y"].idxmax()
    q = q.loc[peak - pd.Timedelta(days=60): peak + pd.Timedelta(days=60)]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(q.index, q.y, "k-", lw=1.4, label="Observed")
    ax.plot(q.index, q.persist, color="0.6", lw=1, ls="--", label="Persistence")
    for mod, c in [("XGB", "tab:blue"), ("LSTM", "tab:red"), ("LR", "tab:green")]:
        ax.plot(q.index, q[f"{cfg}|{mod}"], color=c, lw=1, label=mod)
    ax.axhline(C.HAZE_THRESHOLD, color="tab:orange", lw=0.8, ls=":")
    ax.set_ylabel(C.TARGET_TEX + (f" ({C.UNIT_PLAIN})" if C.UNIT_PLAIN else ""))
    ax.legend(ncol=1, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    st = pd.read_csv(C.PATHS["stations"]).set_index("station_id")["station_name"]
    ax.set_title(f"{st.get(sid, sid)}, h={h} day", fontsize=8)
    if synth:
        ax.text(0.5, 0.5, "SYNTHETIC", transform=ax.transAxes, fontsize=40, color="red",
                alpha=0.25, ha="center", va="center", rotation=15)
    fig.autofmt_xdate()
    fig.tight_layout()
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / "fig_timeseries.pdf")
    plt.close(fig)


def fig_shap(synth):
    h = C.HORIZONS[0]
    p = R / f"shap_importance_h{h}.csv"
    if not p.exists():
        return
    s = pd.read_csv(p, index_col=0)["mean_abs_shap"]
    s = s[~s.index.str.startswith("stn_")].head(15)[::-1]
    if C.TARGET_MODE == "api":
        s.index = s.index.str.replace("pm_", "api_", regex=False)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.barh(s.index, s.values, color="tab:blue")
    ax.set_xlabel("mean |SHAP|" + (f" ({C.UNIT_PLAIN})" if C.UNIT_PLAIN else f" ({C.TARGET_TEX} units)"))
    ax.tick_params(axis="y", labelsize=7)
    if synth:
        ax.text(0.5, 0.5, "SYNTHETIC", transform=ax.transAxes, fontsize=28, color="red",
                alpha=0.25, ha="center", va="center", rotation=30)
    fig.tight_layout()
    fig.savefig(F / "fig_shap.pdf")
    plt.close(fig)


def fig_ablation(m, synth):
    fig, axes = plt.subplots(1, len(C.HORIZONS), figsize=(7, 2.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, C.HORIZONS):
        cfgs = [c for c in CFG_ORDER if ((m.h == h) & (m.features == c.split("-")[0]) &
                                         (m.met_mode == c.split("-")[1])).any()]
        x = np.arange(len(cfgs))
        for i, mod in enumerate(["LR", "RF", "XGB", "LSTM"]):
            v = [m[(m.h == h) & (m.features == c.split("-")[0]) & (m.met_mode == c.split("-")[1])
                   & (m.model == mod)]["RMSE"].iloc[0] for c in cfgs]
            ax.bar(x + (i - 1.5) * 0.2, v, 0.2, label=mod)
        ax.axhline(m[(m.h == h) & (m.model == "Persistence")]["RMSE"].iloc[0], color="k",
                   ls="--", lw=0.8, label="Persistence")
        ax.set_xticks(x)
        ax.set_xticklabels([CFG_LABEL[c].replace("$^\\ast$", "*") for c in cfgs],
                           fontsize=6, rotation=30)
        ax.set_title(f"h={h}", fontsize=8)
        ax.set_ylabel("RMSE", fontsize=7)
        if synth:
            ax.text(0.5, 0.5, "SYNTHETIC", transform=ax.transAxes, fontsize=22, color="red",
                    alpha=0.25, ha="center", va="center", rotation=20)
    axes[0].legend(fontsize=6, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.08), frameon=False)
    fig.tight_layout()
    fig.savefig(F / "fig_ablation.pdf")
    plt.close(fig)


def fig_break():
    """Monthly median daily API across stations, 2016-2019 (archived series)."""
    import glob
    fr = []
    for fpath in glob.glob(str(C.ROOT / "data/raw/aqicn/*.csv")):
        x = pd.read_csv(fpath, skipinitialspace=True)
        x.columns = [c.strip().lower() for c in x.columns]
        if "aqi" not in x:
            continue
        x["date"] = pd.to_datetime(x["date"].astype(str).str.strip(), errors="coerce") + \
            pd.Timedelta(days=C.AQICN_DATE_SHIFT)
        x["aqi"] = pd.to_numeric(x["aqi"].astype(str).str.strip(), errors="coerce")
        fr.append(x.set_index("date")["aqi"].rename(fpath))
    if not fr:
        return
    w = pd.concat(fr, axis=1).sort_index()["2016-01-01":"2019-12-31"]
    mon = w.resample("MS").median().median(axis=1)
    n = w.resample("MS").count().max(axis=1)
    mon[n < 10] = np.nan
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.plot(mon.index, mon.values, "k.-", lw=1, ms=3)
    ax.axvline(pd.Timestamp("2018-08-16"), color="tab:red", lw=0.9, ls="--")
    ax.axvspan(pd.Timestamp("2017-01-01"), pd.Timestamp("2017-12-31"), color="0.85", lw=0)
    ax.text(pd.Timestamp("2017-07-01"), ax.get_ylim()[1] * 0.97, "PM$_{2.5}$ in API\n(DOE: 2017)",
            ha="center", va="top", fontsize=6)
    ax.text(pd.Timestamp("2018-07-20"), ax.get_ylim()[1] * 0.97, "step,\nmid-Aug\n2018",
            color="tab:red", fontsize=6, va="top", ha="right")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("monthly median API", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(F / "fig_break.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
PLACEHOLDER = """\\begin{table}[t]\\centering
\\caption{\\textcolor{red}{PLACEHOLDER -- run \\texttt{src/run\\_experiments.py} then
\\texttt{src/report.py}; this file is overwritten automatically.}}\\label{%s}
\\begin{tabular}{c}\\toprule Results pending\\\\\\bottomrule\\end{tabular}\\end{table}
"""


def placeholders():
    for name, lab in [("tab_data.tex", "tab:data"), ("tab_main.tex", "tab:main"),
                      ("tab_ablation.tex", "tab:ablation"), ("tab_dm.tex", "tab:dm"),
                      ("tab_loso.tex", "tab:loso")]:
        write(name, PLACEHOLDER % lab)
    F.mkdir(parents=True, exist_ok=True)
    for name, size in [("fig_timeseries.pdf", (7, 2.6)), ("fig_shap.pdf", (3.4, 3.4)),
                       ("fig_ablation.pdf", (7, 2.4))]:
        fig, ax = plt.subplots(figsize=size)
        ax.text(0.5, 0.5, "Figure generated by src/report.py", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(F / name)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--placeholder", action="store_true")
    a = ap.parse_args()
    if a.placeholder:
        placeholders()
        paper_numbers.write(placeholder=True)
    else:
        meta = json.load(open(R / "meta.json"))
        synth = bool(meta.get("synthetic"))
        m = pd.read_csv(R / "metrics.csv")
        import diagnostics
        diagnostics.main()
        DIAG.update(json.load(open(R / "diagnostics.json")))
        tab_data(synth); tab_main(m, synth); tab_ablation(m, synth)
        fig_break()
        tab_dm(synth); tab_loso(synth)
        fig_timeseries(m, synth); fig_shap(synth); fig_ablation(m, synth)
        paper_numbers.write(synthetic=synth)
        if synth:
            print("\n!!! SYNTHETIC RUN: tables/figures are watermarked. Do NOT submit them.")
