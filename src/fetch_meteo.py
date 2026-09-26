"""Download ERA5 reanalysis meteorology for each station via the Open-Meteo
historical archive API and aggregate hourly values to local (MYT) daily values.

Usage:  python src/fetch_meteo.py
Output: data/raw/meteo/<station_id>.csv  (one row per local calendar day)

Why ERA5 instead of MetMalaysia station data: it is free, gap-free, has
a documented provenance (Hersbach et al., 2020) and gives every AQ station the
same variables. The trade-off (0.25 deg ~ 28 km grid) is stated as a
limitation in the paper.
"""
import time
import numpy as np
import pandas as pd
import requests

from config import PATHS, PERIOD

URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = ["temperature_2m", "relative_humidity_2m", "precipitation",
          "surface_pressure", "wind_speed_10m", "wind_direction_10m",
          "boundary_layer_height"]


def _request(lat, lon, start, end, variables):
    params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                  hourly=",".join(variables), timezone="Asia/Kuala_Lumpur",
                  wind_speed_unit="ms", models="era5")
    for attempt in range(5):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 400:            # unsupported variable -> caller retries
            raise ValueError(r.text)
        time.sleep(5 * (attempt + 1))       # rate limit / transient
    r.raise_for_status()


def fetch_station(lat, lon, start, end):
    variables = list(HOURLY)
    try:
        js = _request(lat, lon, start, end, variables)
    except ValueError:
        variables.remove("boundary_layer_height")
        print("  boundary_layer_height unavailable -> continuing without it")
        js = _request(lat, lon, start, end, variables)
    h = pd.DataFrame(js["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    return h


def hourly_to_daily(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    # Wind direction is circular: average vector components, never degrees.
    rad = np.deg2rad(h["wind_direction_10m"])
    h["u10"] = -h["wind_speed_10m"] * np.sin(rad)   # +u = wind blowing towards east
    h["v10"] = -h["wind_speed_10m"] * np.cos(rad)   # +v = wind blowing towards north
    h["date"] = h["time"].dt.floor("D")
    agg = dict(temperature_2m="mean", relative_humidity_2m="mean",
               precipitation="sum", surface_pressure="mean",
               wind_speed_10m="mean", u10="mean", v10="mean")
    if "boundary_layer_height" in h:
        agg["boundary_layer_height"] = "mean"
    d = h.groupby("date").agg(agg)
    d = d.rename(columns=dict(temperature_2m="temp", relative_humidity_2m="rh",
                              precipitation="rain", surface_pressure="pres",
                              wind_speed_10m="ws", boundary_layer_height="blh"))
    d["rain_log"] = np.log1p(d["rain"])
    return d.reset_index()


def main():
    st = pd.read_csv(PATHS["stations"])
    PATHS["meteo_dir"].mkdir(parents=True, exist_ok=True)
    start, end = PERIOD["start"], PERIOD["test_end"]
    for _, s in st.iterrows():
        out = PATHS["meteo_dir"] / f"{s.station_id}.csv"
        if out.exists():
            print(f"skip {s.station_id} (exists)")
            continue
        print(f"fetching {s.station_id} ({s.lat}, {s.lon})")
        daily = hourly_to_daily(fetch_station(s.lat, s.lon, start, end))
        daily.to_csv(out, index=False)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
