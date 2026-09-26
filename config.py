"""Central configuration. Edit here, not inside the modules.

Every number reported in the paper must be traceable to a run of
run_experiments.py with this file under version control.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATHS = dict(
    stations=ROOT / "stations.csv",
    pm_raw=ROOT / "data/raw/pm25_hourly_or_daily.csv",   # your DOE / AQICN export
    meteo_dir=ROOT / "data/raw/meteo",
    firms_daily=ROOT / "data/raw/firms_daily_counts.csv",
    panel=ROOT / "data/processed/panel.csv",
    results=ROOT / "results",
    tables=ROOT / "paper/tables",
    figures=ROOT / "paper/figures",
)

# ---- Study period and chronological split --------------------------------
# DEV = model development (tuning by expanding-window CV + early stopping).
# TEST = strictly held out, touched once. Chosen to contain the Sept-Oct 2023
# transboundary haze season so that episode skill can be evaluated.
PERIOD = dict(
    start="2018-01-01",
    dev_end="2022-12-31",
    test_start="2023-01-01",
    test_end="2024-12-31",
    val_months=12,          # last N months of DEV used for early stopping
)

# ---- Data source profile -------------------------------------------------
# "aqicn": WAQI historical platform (daily medians of DOE data, same-day download).
#          Test period stops at 2023-12-31 so the US-EPA PM2.5 breakpoint revision
#          of 2024 cannot contaminate the AQI -> concentration inversion.
# "doe"  : hourly data obtained directly from DOE (2018-2024).
SOURCE = os.environ.get("AQ_SOURCE", "aqicn")
# WAQI labels each daily 'aqi' value one day early: the value stamped d equals the
# mean of DOE's 24 hourly API values on local day d+1 (validated against APIMS,
# 17-23 Sep 2026, 27 station-days, MAE 0.7). Shift = +1 re-labels it to the day
# it describes. AQ_DATE_SHIFT=0 reproduces the original (uncorrected) alignment.
AQICN_DATE_SHIFT = int(os.environ.get("AQ_DATE_SHIFT", "1"))
# Target series:
#  "pm25": daily PM2.5 concentration (ug/m3), recovered from the aqicn pm25 column
#  "api" : the aqicn 'aqi' column, which on Malaysian DOE feeds tracks the official
#          Malaysian API. Use when the pm25 column has no pre-2025 history.
TARGET_MODE = os.environ.get("AQ_TARGET", "api")
if TARGET_MODE == "api":
    # DOE added PM2.5 to the API in August 2018: the station series jump by ~70 %
    # that month (verified in the data). Start after the break.
    PERIOD["start"] = "2018-09-01"
    PERIOD["test_end"] = "2024-12-31"
    TARGET_TEX, UNIT_TEX, UNIT_PLAIN = "API", "", ""
else:
    if SOURCE == "aqicn":
        PERIOD["test_end"] = "2023-12-31"   # avoid the 2024 US-EPA breakpoint change
    TARGET_TEX, UNIT_TEX, UNIT_PLAIN = r"PM$_{2.5}$", r"$\mu$g\,m$^{-3}$", r"$\mu$g m$^{-3}$"

# ---- Forecasting problem --------------------------------------------------
TARGET = "pm25"             # daily mean PM2.5 (ug/m3). API is derived, see paper Sec. III-A
HORIZONS = [1, 3]           # days ahead
MIN_HOURS_PER_DAY = 18      # 75 % completeness rule for daily means
FFILL_LIMIT_DAYS = 2        # causal forward fill for FEATURES only (never targets)
HAZE_THRESHOLD = 35.0       # NMAAQS 2020 24-h PM2.5 standard (ug/m3)
if os.environ.get("AQ_TARGET", "api") == "api":
    HAZE_THRESHOLD = 101.0  # API >= 101 = "unhealthy" band (DOE)

# ---- Feature sets (ablation) ---------------------------------------------
#  H   : historical PM2.5 only
#  M   : meteorology only
#  HM  : historical + meteorology
#  HMF : historical + meteorology + regional fire hotspots
# met_mode:
#  "op"     : meteorology observed up to issue day t only (operational)
#  "oracle" : additionally meteorology on target day t+h taken from reanalysis
#             (upper bound for a perfect NWP forecast; must be labelled as such)
EXPERIMENTS = [
    ("H", "op"),
    ("M", "op"), ("M", "oracle"),
    ("HM", "op"), ("HM", "oracle"),
    ("HMF", "op"), ("HMF", "oracle"),
]

PM_LAGS = [0, 1, 2, 3, 6]   # days before issue day t (0 = day t itself)
MET_LAGS = [0, 1]
ROLL_WINDOWS = [3, 7]
FIRE_REGIONS = {            # lon_min, lat_min, lon_max, lat_max
    "sumatra": (95.0, -6.0, 106.5, 6.0),
    "borneo": (108.5, -4.5, 119.5, 7.5),
}

# ---- Models ---------------------------------------------------------------
SEED = 42
CV_FOLDS = 3                 # expanding-window folds inside DEV
N_ITER_SEARCH = 15           # randomized search budget per tree model/config
RF_TREES = [300, 500]        # candidate forest sizes
LSTM = dict(
    window=14, hidden=64, layers=1, dropout=0.2, lr=1e-3,     # defaults; tuned below
    batch=256, max_epochs=150, patience=15, seeds=[0, 1, 2, 3, 4],
    window_max=21,           # sequences are built with this length; the tuned window
                             # uses the last L steps, so every L sees the SAME rows
)
# LSTM hyper-parameter search: same budget and the same expanding-window folds as
# RF / XGBoost (N_ITER_SEARCH random candidates, one seed per candidate).
LSTM_SPACE = dict(window=[7, 14, 21], hidden=[32, 64, 128], dropout=[0.0, 0.2, 0.4],
                  lr=[3e-4, 1e-3, 3e-3])
# Reference ("main") configuration for Table III, DM tests, SHAP and LOSO.
# H+M rather than H+M+F: fire features degraded both tree ensembles on
# unhealthy days (see paper, Sec. V-D), so H+M is the cleaner learner comparison;
# every H+M+F result is still reported in the ablation table.
MAIN_CFG = os.environ.get("AQ_MAIN_CFG", "HM-op")
RUN_LOSO = True              # leave-one-station-out generalisation (XGBoost)
