"""Daily VIIRS active-fire counts over Sumatra and Borneo (NASA FIRMS).

Two ways to get the data:

  A) API (needs a free MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/):
        FIRMS_MAP_KEY=xxxx python src/fetch_firms.py --api
  B) Archive CSVs you downloaded manually from the FIRMS "Archive Download"
     page (e.g. yearly VIIRS S-NPP files for Indonesia and Malaysia):
        python src/fetch_firms.py --csv path/to/*.csv

Output: data/raw/firms_daily_counts.csv with columns
        date, fire_sumatra, fire_borneo
Low-confidence VIIRS detections ('l') are discarded.
"""
import argparse
import glob
import io
import os
import time

import pandas as pd
import requests

from config import PATHS, PERIOD, FIRE_REGIONS

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{bbox}/{days}/{date}"


def _count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "confidence" in df:
        df = df[df["confidence"].astype(str).str.lower() != "l"]
    df["date"] = pd.to_datetime(df["acq_date"])
    out = {}
    for name, (x0, y0, x1, y1) in FIRE_REGIONS.items():
        m = df["longitude"].between(x0, x1) & df["latitude"].between(y0, y1)
        out[f"fire_{name}"] = df[m].groupby("date").size()
    res = pd.DataFrame(out)
    idx = pd.date_range(PERIOD["start"], PERIOD["test_end"], freq="D", name="date")
    return res.reindex(idx).fillna(0).astype(int).reset_index()


def from_api(key: str, source: str = "VIIRS_SNPP_SP", chunk: int = 5):   # API max = 5 days
    frames = []
    for name, (x0, y0, x1, y1) in FIRE_REGIONS.items():
        bbox = f"{x0},{y0},{x1},{y1}"
        d = pd.Timestamp(PERIOD["start"])
        end = pd.Timestamp(PERIOD["test_end"])
        while d <= end:
            url = API.format(key=key, src=source, bbox=bbox, days=chunk,
                             date=d.strftime("%Y-%m-%d"))
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            if r.text.strip() and not r.text.startswith("Invalid"):
                frames.append(pd.read_csv(io.StringIO(r.text)))
            print(f"{name} {d.date()} ok")
            d += pd.Timedelta(days=chunk)
            time.sleep(0.6)
    return _count(pd.concat(frames, ignore_index=True))


def from_csv(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    df = pd.concat([pd.read_csv(f, usecols=lambda c: c in
                    {"latitude", "longitude", "acq_date", "confidence"})
                    for f in files], ignore_index=True)
    return _count(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--csv", type=str)
    a = ap.parse_args()
    if a.api:
        res = from_api(os.environ["FIRMS_MAP_KEY"])
    elif a.csv:
        res = from_csv(a.csv)
    else:
        raise SystemExit("use --api or --csv <glob>")
    res.to_csv(PATHS["firms_daily"], index=False)
    print(res.describe())
