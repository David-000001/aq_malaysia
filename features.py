"""Leakage-safe feature construction.

Conventions (issue day t, target day t+h):
  * Everything in a feature row is known at the end of day t, EXCEPT the
    '*_tgt' meteorology columns, which exist only in met_mode='oracle' and
    stand in for a perfect numerical weather forecast of day t+h.
  * Forward-fill is causal and applied to features only; targets are never
    filled or interpolated.
  * All experiment configurations are evaluated on the SAME rows (the
    intersection of rows that every configuration can use), so differences
    between feature sets cannot come from different test samples.
"""
import numpy as np
import pandas as pd

from config import PM_LAGS, MET_LAGS, ROLL_WINDOWS, FFILL_LIMIT_DAYS, FIRE_REGIONS, LSTM

# "blh" (ERA5 boundary-layer height) is NOT used: the Open-Meteo ERA5 archive has no BLH
# for 2024-01-02..2024-06-30 (checked for era5, era5_seamless, best_match), which would
# silently drop half of the test year from the common evaluation rows.
MET_BASE = ["temp", "rh", "rain_log", "pres", "ws", "u10", "v10"]


def _cyc(d: pd.Series, prefix: str) -> pd.DataFrame:
    doy = d.dt.dayofyear
    return pd.DataFrame({f"{prefix}_doy_sin": np.sin(2 * np.pi * doy / 365.25),
                         f"{prefix}_doy_cos": np.cos(2 * np.pi * doy / 365.25),
                         f"{prefix}_dow": d.dt.dayofweek.astype(float)}, index=d.index)


def build_design(panel: pd.DataFrame, h: int):
    """Return (frame, groups). frame has one row per station-issue-day."""
    met_vars = [v for v in MET_BASE if v in panel.columns]
    fire_vars = [f"fire_{r}" for r in FIRE_REGIONS if f"fire_{r}" in panel.columns]
    out = []
    for sid, g in panel.groupby("station_id", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        f = pd.DataFrame({"station_id": sid, "date": g["date"]})
        pm = g["pm25"].ffill(limit=FFILL_LIMIT_DAYS)
        f["y"] = g["pm25"].shift(-h)                       # raw target, never filled
        f["persist"] = pm                                   # persistence forecast
        # --- H: history of the target -----------------------------------
        for k in PM_LAGS:
            f[f"pm_lag{k}"] = pm.shift(k)
        for w in ROLL_WINDOWS:
            f[f"pm_rmean{w}"] = pm.rolling(w, min_periods=w - 1).mean()
        f["pm_rstd7"] = pm.rolling(7, min_periods=6).std()
        f["pm_diff1"] = pm - pm.shift(1)
        # --- M: meteorology up to day t ------------------------------------
        met = g[met_vars].ffill(limit=FFILL_LIMIT_DAYS)
        for k in MET_LAGS:
            for v in met_vars:
                f[f"{v}_lag{k}"] = met[v].shift(k)
        f["rain3_log"] = np.log1p(g["rain"].ffill(limit=FFILL_LIMIT_DAYS).rolling(3).sum())
        # --- oracle meteorology on the target day -------------------------------
        for v in met_vars:
            f[f"{v}_tgt"] = met[v].shift(-h)
        # --- F: regional fire activity ----------------------------------------
        for v in fire_vars:
            fv = g[v].fillna(0)
            f[f"{v}_lag0"] = np.log1p(fv)
            f[f"{v}_lag1"] = np.log1p(fv.shift(1))
            f[f"{v}_sum7"] = np.log1p(fv.rolling(7).sum())
        # --- calendar of the TARGET day (known in advance) -----------------
        f = pd.concat([f, _cyc(g["date"] + pd.Timedelta(days=h), "tgt")], axis=1)
        # --- sequence completeness flag for the LSTM ------------------------
        seq_cols = pm.to_frame().join(met)
        ok = seq_cols.notna().all(axis=1).astype(int)
        f["seq_ok"] = ok.rolling(LSTM["window_max"]).sum().eq(LSTM["window_max"])
        out.append(f)
    df = pd.concat(out, ignore_index=True)
    dummies = pd.get_dummies(df["station_id"], prefix="stn", dtype=float)
    df = pd.concat([df, dummies], axis=1)

    groups = dict(
        H=[c for c in df if c.startswith("pm_")],
        M=[f"{v}_lag{k}" for k in MET_LAGS for v in met_vars] + ["rain3_log"],
        F=[c for c in df if c.startswith("fire_")],
        ORACLE=[f"{v}_tgt" for v in met_vars],
        ALWAYS=["tgt_doy_sin", "tgt_doy_cos", "tgt_dow"] + list(dummies.columns),
        met_vars=met_vars, fire_vars=fire_vars,
    )
    # Common evaluation rows: every configuration must be computable.
    need = groups["H"] + groups["M"] + groups["F"] + groups["ORACLE"] + ["y", "persist"]
    df["usable"] = df[need].notna().all(axis=1) & df["seq_ok"]
    return df, groups


def columns_for(groups, fset: str, mode: str):
    cols = list(groups["ALWAYS"])
    if "H" in fset:
        cols += groups["H"]
    if "M" in fset:
        cols += groups["M"]
    if "F" in fset:
        if not groups["F"]:
            return None                         # no fire data -> skip config
        cols += groups["F"]
    if mode == "oracle":
        if "M" not in fset:
            return None
        cols += groups["ORACLE"]
    return cols


# ---------------------------------------------------------------------------
# Sequence tensors for the LSTM
# ---------------------------------------------------------------------------
def build_sequences(panel: pd.DataFrame, design: pd.DataFrame, groups, fset, mode, h):
    """Return X_seq (N, L, D_dyn), X_fut (N, D_fut) aligned with design rows
    where design['usable'] is True (same order as design[design.usable])."""
    L = LSTM["window_max"]
    met_vars, fire_vars = groups["met_vars"], groups["fire_vars"]
    dyn_cols = ["doy_sin", "doy_cos"]
    if "H" in fset:
        dyn_cols = ["pm"] + dyn_cols
    if "M" in fset:
        dyn_cols += met_vars
    if "F" in fset:
        dyn_cols += fire_vars
    fut_cols = ["tgt_doy_sin", "tgt_doy_cos", "tgt_dow"] + \
        [c for c in groups["ALWAYS"] if c.startswith("stn_")]
    if mode == "oracle":
        fut_cols += groups["ORACLE"]

    seqs, futs = [], []
    use = design[design["usable"]]
    for sid, g in panel.groupby("station_id", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        dyn = pd.DataFrame({"pm": g["pm25"].ffill(limit=FFILL_LIMIT_DAYS)})
        doy = g["date"].dt.dayofyear
        dyn["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        dyn["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        for v in met_vars:
            dyn[v] = g[v].ffill(limit=FFILL_LIMIT_DAYS)
        for v in fire_vars:
            dyn[v] = np.log1p(g[v].fillna(0))
        arr = dyn[dyn_cols].to_numpy(dtype=np.float32)
        pos = pd.Series(np.arange(len(g)), index=g["date"])
        u = use[use["station_id"] == sid]
        ends = pos.loc[u["date"]].to_numpy()
        idx = ends[:, None] + np.arange(-L + 1, 1)[None, :]
        seqs.append(arr[idx])
        futs.append(u[fut_cols].to_numpy(dtype=np.float32))
    # re-order to match `use` order (use is sorted by station then date already)
    return np.concatenate(seqs), np.concatenate(futs), dyn_cols, fut_cols
