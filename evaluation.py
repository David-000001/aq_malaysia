"""Evaluation: point accuracy, skill vs persistence, haze-episode skill,
exceedance detection, Diebold-Mariano (HLN-corrected) and block-bootstrap CIs."""
import numpy as np
import pandas as pd
from scipy import stats

from config import HAZE_THRESHOLD, SEED


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))


def exceedance(y, p, thr=HAZE_THRESHOLD):
    """Categorical skill for 'will the 24-h standard be exceeded?'"""
    obs, fc = y >= thr, p >= thr
    hits = np.sum(obs & fc)
    miss = np.sum(obs & ~fc)
    fa = np.sum(~obs & fc)
    pod = hits / (hits + miss) if hits + miss else np.nan
    far = fa / (hits + fa) if hits + fa else np.nan
    csi = hits / (hits + miss + fa) if hits + miss + fa else np.nan
    return dict(POD=pod, FAR=far, CSI=csi, n_events=int(obs.sum()))


def all_metrics(y, p, persist):
    y, p, persist = map(np.asarray, (y, p, persist))
    m = dict(RMSE=rmse(y, p), MAE=mae(y, p), R2=r2(y, p))
    m["Skill"] = 1 - m["RMSE"] / rmse(y, persist)          # >0 beats persistence
    haze = y >= HAZE_THRESHOLD
    m["RMSE_haze"] = rmse(y[haze], p[haze]) if haze.any() else np.nan
    m["Bias_haze"] = float(np.mean(p[haze] - y[haze])) if haze.any() else np.nan
    m.update(exceedance(y, p))
    return m


def dm_test(df: pd.DataFrame, col_a: str, col_b: str, h: int):
    """Diebold-Mariano with Harvey-Leybourne-Newbold small-sample correction.
    Loss = squared error, averaged over stations per calendar day so that the
    test statistic is computed on one time series (cross-sectional averaging
    avoids treating correlated stations as independent evidence).
    Negative statistic => model A has lower loss than model B."""
    d = ((df["y"] - df[col_a]) ** 2 - (df["y"] - df[col_b]) ** 2)
    d = d.groupby(df["date"]).mean().to_numpy()
    T = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma = [np.sum(dc[k:] * dc[:T - k]) / T for k in range(h)]
    var = gamma[0] + 2 * sum(gamma[1:])
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p = 2 * stats.t.sf(np.abs(stat), df=T - 1)
    return float(stat), float(p)


def block_bootstrap_rmse(df: pd.DataFrame, col: str, block: int = 14, n: int = 1000):
    """95 % CI for RMSE by moving-block bootstrap over calendar days."""
    rng = np.random.default_rng(SEED)
    se = ((df["y"] - df[col]) ** 2).groupby(df["date"]).agg(["sum", "count"])
    s, c = se["sum"].to_numpy(), se["count"].to_numpy()
    T = len(s)
    nb = int(np.ceil(T / block))
    vals = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, T - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]
        vals[i] = np.sqrt(s[idx].sum() / c[idx].sum())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
