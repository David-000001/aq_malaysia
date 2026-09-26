"""SYNTHETIC data for pipeline smoke tests ONLY.

Never report numbers produced from this file. run_experiments.py --synthetic
marks meta.json, and report.py stamps every table/figure with a watermark.
"""
import numpy as np
import pandas as pd

from config import PATHS, PERIOD, SEED


def make(n_stations=4):
    rng = np.random.default_rng(SEED)
    st = pd.read_csv(PATHS["stations"]).head(n_stations)
    dates = pd.date_range(PERIOD["start"], PERIOD["test_end"], freq="D")
    T = len(dates)
    doy = dates.dayofyear.to_numpy()
    sw_monsoon = ((doy > 150) & (doy < 280)).astype(float)
    # regional fires: seasonal + strong years
    fire_s = rng.poisson(20 + 400 * sw_monsoon * np.isin(dates.year, [2019, 2023]) *
                         np.exp(-((doy - 255) / 20.0) ** 2) * 5 + 30 * sw_monsoon)
    fire_b = rng.poisson(15 + 300 * np.isin(dates.year, [2019, 2023]) *
                         np.exp(-((doy - 260) / 18.0) ** 2) * 5)
    rows = []
    for k, s in st.iterrows():
        temp = 27.5 + 0.8 * np.sin(2 * np.pi * (doy - 120) / 365) + rng.normal(0, .6, T)
        rain = rng.gamma(0.5, 12, T) * (rng.random(T) < 0.55 - 0.2 * sw_monsoon)
        rh = np.clip(80 + 0.4 * rain - 3 * sw_monsoon + rng.normal(0, 3, T), 50, 99)
        u = -2.0 * (1 - sw_monsoon) + 1.8 * sw_monsoon + rng.normal(0, 1, T)
        v = -1.5 * (1 - sw_monsoon) + 1.2 * sw_monsoon + rng.normal(0, 1, T)
        ws = np.hypot(u, v)
        blh = 700 + 60 * ws + 15 * (temp - 27) - 3 * rain + rng.normal(0, 60, T)
        pres = 1008 + rng.normal(0, 1.5, T)
        smoke = 0.03 * np.convolve(fire_s, [0.2, 0.5, 0.3], "same") * np.clip(u, 0, None)
        pm = np.empty(T)
        pm[0] = 20
        for t in range(1, T):
            mean_t = 18 + 3 * k - 0.25 * rain[t] - 0.004 * (blh[t] - 800) + smoke[t]
            pm[t] = 0.6 * pm[t - 1] + 0.4 * mean_t + rng.normal(0, 3.5)
        pm = np.clip(pm, 2, None)
        miss = rng.random(T) < 0.04
        pm[miss] = np.nan
        rows.append(pd.DataFrame(dict(date=dates, station_id=s.station_id, pm25=pm,
                                      temp=temp, rh=rh, rain=rain, pres=pres, ws=ws,
                                      u10=u, v10=v, blh=blh, rain_log=np.log1p(rain),
                                      fire_sumatra=fire_s, fire_borneo=fire_b)))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    out = PATHS["panel"].parent / "synthetic_panel.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(out, index=False)
    print("wrote", out)
