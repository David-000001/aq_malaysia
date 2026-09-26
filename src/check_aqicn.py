"""Check downloaded aqicn CSVs BEFORE running anything.

    python src/check_aqicn.py data/raw/aqicn

For every file: days with data per year for the 'pm25' column (US-EPA PM2.5
sub-index) and the 'aqi' column (on Malaysian DOE feeds this tracks the
official Malaysian API). Tells you which target the data can support.
"""
import sys
from pathlib import Path

import pandas as pd

YEARS = range(2018, 2025)


def load(f):
    x = pd.read_csv(f, skipinitialspace=True)
    x.columns = [c.strip().lower() for c in x.columns]
    x["date"] = pd.to_datetime(x["date"].astype(str).str.strip(), errors="coerce")
    x = x.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
    for c in ("pm25", "aqi"):
        x[c] = pd.to_numeric(x[c].astype(str).str.strip(), errors="coerce") if c in x else float("nan")
    return x


def main(d):
    rows = []
    for f in sorted(Path(d).glob("*.csv")):
        x = load(f)
        r = {"file": f.name[:40]}
        for c in ("pm25", "aqi"):
            s = x[c].dropna()
            for y in YEARS:
                r[f"{c}_{y}"] = int((s.index.year == y).sum())
            full = pd.date_range("2018-01-01", "2023-12-31")
            r[f"{c}_cov18_23"] = round(s.reindex(full).notna().mean(), 2)
        rows.append(r)
    t = pd.DataFrame(rows).set_index("file")
    pd.set_option("display.width", 250)
    for c in ("pm25", "aqi"):
        print(f"\n=== days with '{c}' data per year ===")
        print(t[[k for k in t if k.startswith(c + "_")]].to_string())
    ok_pm = (t["pm25_cov18_23"] >= 0.6).sum()
    ok_api = (t["aqi_cov18_23"] >= 0.6).sum()
    print(f"\nstations usable (>=60% of 2018-2023): pm25 -> {ok_pm}, aqi/API -> {ok_api} of {len(t)}")
    if ok_pm >= 5:
        print("RECOMMENDATION: target pm25   (AQ_TARGET=pm25, default)")
    elif ok_api >= 5:
        print("RECOMMENDATION: target API    (set AQ_TARGET=api)")
    else:
        print("RECOMMENDATION: not enough stations yet -- download more (need >= 5, ideally 8-15)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/aqicn")
