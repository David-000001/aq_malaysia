# How far ahead can ML forecast the Klang Valley's Air Pollutant Index?

This is the code for the paper *"How Far Ahead Can Machine Learning Forecast the Klang Valley's Air
Pollutant Index? A Leakage-Controlled Benchmark Against Persistence"*.

The paper benchmarks ridge regression, random forest, XGBoost and a tuned LSTM for 1- and
3-day-ahead daily Air Pollutant Index (API) forecasts at five DOE stations: Cheras, Batu Muda,
Petaling Jaya, Shah Alam and Banting. The study runs 2018-09 to 2024-12, and 2023–2024 is the
held-out test period. Predictors are API history, ERA5 meteorology and VIIRS fire counts.

## Reproduce

```bash
pip install -r requirements.txt
# 1. WAQI station CSVs from https://aqicn.org/historical/  ->  data/raw/aqicn/
#    (one file per station in stations.csv; keep the downloaded file names)
# 2. optional, for the fire features: free NASA FIRMS key
#    export FIRMS_MAP_KEY=...        (Windows: set FIRMS_MAP_KEY=...)
python src/run_all.py --aqicn_dir data/raw/aqicn --quick   # check run, a few minutes
python src/run_all.py --aqicn_dir data/raw/aqicn           # full run, ~3 h on 2 CPUs
```

`run_all.py` does the whole pipeline: data check, ERA5 download (Open-Meteo, `models=era5`),
FIRMS fire counts, panel, experiments, diagnostics, and then the tables, figures and numbers.
Raw data is not redistributed here, because of the WAQI terms.

## Layout

| Path | Contents |
|---|---|
| `src/config.py` | Every setting: period, horizons, feature sets, model search spaces, `MAIN_CFG` |
| `src/build_panel.py` | Station-day panel. WAQI `aqi` is re-dated by `AQ_DATE_SHIFT` (default +1 day, see below) |
| `src/features.py` | Leakage-safe features. Every configuration is scored on the same rows |
| `src/models.py` | Ridge, RF, XGBoost and LSTM. All four get the same expanding-window random search (15 candidates) |
| `src/run_experiments.py` | Full grid (7 feature configs × 2 horizons), DM tests, LOSO, SHAP |
| `src/diagnostics.py` | Paired block-bootstrap CIs, Holm correction, haze-event scores, M-only decomposition |
| `src/report.py` | LaTeX tables, figures and the numbers quoted in the text |
| `results/` | Outputs of the paper run (`metrics.csv`, `dm_holm.csv`, `loso.csv`, …) |
| `results/sensitivity_shift0/` | Same run without the date correction (`AQ_DATE_SHIFT=0`) |

## Notes on the data

- **Target.** Before 2025, the WAQI Malaysian station files have only an `aqi` column. We checked
  it against DOE's official hourly API (APIMS, 17–23 Sep 2026, 27 station-days, mean absolute
  difference 0.7). It is the daily mean of the official API, but **stamped one day early**.
  The pipeline shifts it by +1 day. `AQ_DATE_SHIFT=0` reproduces the original dates.
- **Level shift.** The archived series steps up in mid-August 2018. This does not match the
  documented inclusion of PM2.5 in the API (2017), so the study starts on 1 Sep 2018.
- **Boundary-layer height** is not used. The Open-Meteo ERA5 archive has no BLH for Jan–Jun 2024.

## Main results (test 2023–2024, weather + API history)

| | Persistence RMSE | Best model | RMSE | Skill vs persistence |
|---|---|---|---|---|
| 1 day ahead | 9.35 | Ridge regression | 8.44 | 9.8 % (only one significant after Holm) |
| 3 days ahead | 13.40 | Ridge regression | 11.43 | 14.7 % |

- Meteorology alone has almost no skill.
- Fire counts make the tree models worse.
- On unhealthy days (API ≥ 101), no model beats persistence.

See the paper for details.
