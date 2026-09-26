"""Assemble the station-day panel: PM2.5 + ERA5 meteorology + fire counts.

Accepted PM2.5 inputs
---------------------
1) --format long (recommended; DOE data obtained on request are easily
   reshaped to this):   station_id, datetime, pm25
   'datetime' may be hourly (preferred) or daily. Hourly data are averaged to
   local calendar days only if >= MIN_HOURS_PER_DAY valid hours exist.

2) --format aqicn --aqicn_dir DIR : one CSV per station (DIR/<station_id>.csv)
   downloaded from the World Air Quality Index historical-data platform.
   Those files contain *US-EPA AQI sub-index* values (daily medians), NOT
   concentrations. They are converted back to ug/m3 with the inverse of the
   pre-2024 US-EPA PM2.5 breakpoints. Treat this path as second-best and
   state it in the paper if you use it.

Usage:
    python src/build_panel.py --format long --pm data/raw/pm25.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from config import PATHS, PERIOD, MIN_HOURS_PER_DAY, TARGET_MODE

# (C_lo, C_hi, I_lo, I_hi): pre-2024 US-EPA PM2.5 breakpoints
EPA_PM25_BP = [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
               (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
               (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]


def aqi_to_conc(aqi):
    aqi = np.asarray(aqi, dtype=float)
    out = np.full_like(aqi, np.nan)
    for c_lo, c_hi, i_lo, i_hi in EPA_PM25_BP:
        m = (aqi >= i_lo) & (aqi <= i_hi)
        out[m] = c_lo + (aqi[m] - i_lo) * (c_hi - c_lo) / (i_hi - i_lo)
    return out


def load_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = "datetime" if "datetime" in df else "date"
    df["t"] = pd.to_datetime(df[tcol])
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df.loc[(df["pm25"] < 0) | (df["pm25"] > 1000), "pm25"] = np.nan   # sensor faults
    df["date"] = df["t"].dt.floor("D")
    per_day = df.groupby(["station_id", "date"])["t"].transform("size")
    hourly = per_day.max() > 1
    if hourly:
        g = df.groupby(["station_id", "date"])["pm25"]
        daily = g.mean().to_frame("pm25")
        daily["n_valid"] = g.count()
        daily.loc[daily["n_valid"] < MIN_HOURS_PER_DAY, "pm25"] = np.nan
        daily = daily.drop(columns="n_valid").reset_index()
    else:
        daily = df[["station_id", "date", "pm25"]]
    return daily


def _slug(x: str) -> str:
    return "".join(ch for ch in str(x).lower() if ch.isalnum())


def _find_aqicn_file(d: Path, sid: str, name: str):
    """aqicn names downloads like 'cheras,-kuala-lumpur, malaysia-air-quality.csv'.
    Accept either <station_id>.csv or any CSV whose name contains the station name."""
    exact = d / f"{sid}.csv"
    if exact.exists():
        return exact
    key = _slug(name)
    hits = [f for f in d.glob("*.csv") if key and key in _slug(f.stem)]
    if len(hits) > 1:
        print(f"  {sid}: several files match '{name}': {[h.name for h in hits]} -> "
              f"using {sorted(hits, key=lambda f: len(f.name))[0].name}; rename to {sid}.csv to force")
    return sorted(hits, key=lambda f: len(f.name))[0] if hits else None


def load_aqicn(d: Path, stations: pd.DataFrame) -> pd.DataFrame:
    """WAQI historical CSV: columns 'date, pm25, pm10, o3, no2, so2, co' (leading
    spaces, blanks for missing). Values are the daily MEDIAN of hourly US-EPA
    individual AQI. The EPA sub-index is monotone in concentration, so inverting
    the median AQI gives the median concentration (up to integer rounding)."""
    frames = []
    for _, st in stations.iterrows():
        f = _find_aqicn_file(d, st.station_id, st.station_name)
        if f is None:
            print(f"  MISSING file for {st.station_id} ({st.station_name}) in {d}")
            continue
        x = pd.read_csv(f, skipinitialspace=True)
        x.columns = [c.strip().lower() for c in x.columns]
        if TARGET_MODE == "pm25" and "pm25" not in x:
            print(f"  {f.name}: no pm25 column -> skipped")
            continue
        x["date"] = pd.to_datetime(x["date"].astype(str).str.strip(), errors="coerce")
        x["date"] = x["date"] + pd.Timedelta(days=C.AQICN_DATE_SHIFT)
        if TARGET_MODE == "api":
            # target = Malaysian API series (kept in the internal 'pm25' column name,
            # which the rest of the pipeline treats simply as "the target series")
            if "aqi" not in x:
                print(f"  {f.name}: no aqi column -> skipped")
                continue
            x["pm25"] = pd.to_numeric(x["aqi"].astype(str).str.strip(), errors="coerce")
        else:
            aqi = pd.to_numeric(x["pm25"].astype(str).str.strip(), errors="coerce")
            x["pm25"] = aqi_to_conc(aqi)
        x = x.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
        x["station_id"] = st.station_id
        print(f"  {st.station_id:<14} <- {f.name}  ({x['pm25'].notna().sum()} valid days, "
              f"{x['date'].min().date()} .. {x['date'].max().date()})")
        frames.append(x[["station_id", "date", "pm25"]])
    if not frames:
        raise SystemExit("no aqicn files matched; check --aqicn_dir and stations.csv names")
    out = pd.concat(frames, ignore_index=True)
    if TARGET_MODE == "api":
        record_api_break(out)
    return out


def record_api_break(df: pd.DataFrame):
    """Median API before (Jan-Jul 2018) and after (Sep-Dec 2018) DOE added PM2.5."""
    import json
    pre = df[(df.date >= "2018-01-01") & (df.date <= "2018-07-31")].groupby("station_id")["pm25"].median()
    post = df[(df.date >= "2018-09-01") & (df.date <= "2018-12-31")].groupby("station_id")["pm25"].median()
    res = dict(pre=float(pre.median()), post=float(post.median()), n=int(len(pre.dropna())))
    PATHS["results"].mkdir(parents=True, exist_ok=True)
    json.dump(res, open(PATHS["results"] / "api_break.json", "w"))
    print(f"  API break check: median Jan-Jul 2018 = {res['pre']:.0f}, Sep-Dec 2018 = {res['post']:.0f}")


def assemble(pm: pd.DataFrame) -> pd.DataFrame:
    st = pd.read_csv(PATHS["stations"])
    idx = pd.date_range(PERIOD["start"], PERIOD["test_end"], freq="D")
    frames = []
    for sid in st["station_id"]:
        base = pd.DataFrame({"date": idx, "station_id": sid})
        p = pm[pm["station_id"] == sid][["date", "pm25"]]
        base = base.merge(p, on="date", how="left")
        mf = PATHS["meteo_dir"] / f"{sid}.csv"
        if not mf.exists():
            raise FileNotFoundError(f"{mf}: run fetch_meteo.py first")
        met = pd.read_csv(mf, parse_dates=["date"])
        base = base.merge(met, on="date", how="left")
        frames.append(base)
    panel = pd.concat(frames, ignore_index=True)
    if PATHS["firms_daily"].exists():
        fire = pd.read_csv(PATHS["firms_daily"], parse_dates=["date"])
        panel = panel.merge(fire, on="date", how="left")
    else:
        print("WARNING: no FIRMS file -> HMF experiments will be skipped")
    return panel.sort_values(["station_id", "date"]).reset_index(drop=True)


def report_coverage(panel):
    cov = panel.groupby("station_id")["pm25"].apply(lambda s: s.notna().mean())
    print("Target daily coverage per station:\n", cov.round(3).to_string())
    bad = cov[cov < 0.6]
    if len(bad):
        print(f"\nWARNING: {list(bad.index)} have <60 % coverage; consider dropping them.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["long", "aqicn"], default="long")
    ap.add_argument("--pm", type=Path, default=PATHS["pm_raw"])
    ap.add_argument("--aqicn_dir", type=Path)
    a = ap.parse_args()
    st = pd.read_csv(PATHS["stations"])
    pm = load_long(a.pm) if a.format == "long" else load_aqicn(a.aqicn_dir, st)
    panel = assemble(pm)
    PATHS["panel"].parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PATHS["panel"], index=False)
    report_coverage(panel)
    print(f"\nwrote {PATHS['panel']}  shape={panel.shape}")
