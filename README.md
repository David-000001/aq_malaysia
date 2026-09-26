# >>> YOUR NEXT STEPS (data is already included) <<<
Your 5 Klang Valley station files (Cheras, Batu Muda, Petaling Jaya, Shah Alam,
Banting) are already in `data/raw/aqicn/`, and `stations.csv` is set up.
The "Kuala Lumpur" city feed is deliberately excluded (composite of Cheras +
Batu Muda). Target = daily API (default).

```bash
pip install -r requirements.txt
# optional, recommended: free key from https://firms.modaps.eosdis.nasa.gov/api/
export FIRMS_MAP_KEY=yourkey          # Windows: set FIRMS_MAP_KEY=yourkey
python src/run_all.py --aqicn_dir data/raw/aqicn --quick   # check, ~15-30 min
python src/run_all.py --aqicn_dir data/raw/aqicn           # paper run, ~1-3 h
```
Then upload `results/`, `paper/tables/`, `paper/figures/` to Claude to write the
remaining red interpretation sentences. Do NOT push `data/raw/` to GitHub
(.gitignore already excludes it).

# ML-Based Prediction of Air Quality in Malaysia — paper + reproducible pipeline

Everything the paper reports is produced by the scripts in `src/`. Tables and
figures in `paper/` are **generated**, never typed by hand. The manuscript ships
with red `[...]` placeholders wherever a number or a data-dependent claim goes.

## 0. Install
```bash
pip install -r requirements.txt
```

## FAST ROUTE (same day, no DOE request) — recommended for a near deadline

1. **Download PM2.5 history (≈30 min).** Go to https://aqicn.org/historical/ ,
   search each DOE station (e.g. "Cheras, Kuala Lumpur"), and download its CSV
   (the site asks for a short form; state academic use). Save all files into
   `data/raw/aqicn/` without renaming — files are matched to `stations.csv`
   by station name. Download ONLY from aqicn.org itself; ignore third-party
   "AQI decoder" tools.
2. **Edit `stations.csv`** to the stations you downloaded, with coordinates
   from the station pages. Keep ~8–15 stations that have PM2.5 back to 2018.
3. **Optional but recommended: FIRMS key** (free, emailed instantly):
   https://firms.modaps.eosdis.nasa.gov/api/  → `export FIRMS_MAP_KEY=...`
   (Windows: `set FIRMS_MAP_KEY=...`).
3b. **Check your files first:** `python src/check_aqicn.py data/raw/aqicn`
   shows, per station and year, how many days have `pm25` and `aqi` data and
   recommends the target. DEFAULT IS NOW `AQ_TARGET=api` (daily Malaysian API,
   study period from 1 Sep 2018 because DOE added PM2.5 to the API in Aug 2018,
   test 2023-2024, unhealthy = API >= 101). The aqicn Malaysian feeds checked so
   far have no pm25 history before 2025. The API paper is `paper/main.tex`; the
   PM2.5 version is kept as `paper/main_pm25.tex` (use with AQ_TARGET=pm25).
4. **One command:**
   ```bash
   python src/run_all.py --aqicn_dir data/raw/aqicn --quick   # check (~15-30 min)
   python src/run_all.py --aqicn_dir data/raw/aqicn           # paper run (~2-4 h CPU)
   ```
5. Compile `paper/main.tex`. Every number in the abstract, results, discussion
   and conclusion is filled automatically from `paper/tables/numbers.tex`.
   What remains red are ~10 short interpretation sentences you must write
   after looking at the tables/figures (`grep -n "res{" paper/main.tex`).

Data profile: `AQ_SOURCE=aqicn` (default) uses 2018–2022 for development and
2023 for testing (avoids the 2024 US-EPA breakpoint change). If you later get
hourly DOE data, set `AQ_SOURCE=doe`, use `--pm`, and swap the Data paragraph
(an alternative text is in a comment in main.tex).

## 1. (Slow route) PM2.5 data directly from DOE
## 2. Meteorology (ERA5, free, no key)
```bash
cd src && python fetch_meteo.py
```

## 3. Fire hotspots (NASA FIRMS, free MAP_KEY)
```bash
FIRMS_MAP_KEY=your_key python fetch_firms.py --api
# or: python fetch_firms.py --csv "path/to/firms_archive/*.csv"
```
Skip this and the H+M+F configurations are dropped automatically.

## 4. Build the panel and check coverage
```bash
python build_panel.py --format long --pm ../data/raw/pm25_hourly_or_daily.csv
```
Drop stations with < 60 % daily coverage (it prints a warning).

## 5. Run
```bash
python run_experiments.py --fast      # sanity check first (minutes)
python run_experiments.py             # full run (hours on a laptop CPU)
python report.py                      # writes paper/tables/*.tex and paper/figures/*.pdf
```
Subsets: `--horizons 1 --configs H-op,HM-op,HMF-op`.

Smoke test without real data (outputs are watermarked SYNTHETIC):
```bash
python synthetic.py && python run_experiments.py --panel ../data/processed/synthetic_panel.csv --synthetic --fast
```

## 6. Finish the paper
1. Upload `paper/` to Overleaf (IEEEtran is built in) or compile locally.
2. Fill every `\res{...}` from `paper/tables/*.tex` and `results/*.csv`.
3. Rewrite any placeholder sentence the data contradict. Do not bend results to the draft.
4. `grep -n "\\res{" paper/main.tex` must return nothing.
5. Add 3–5 recent Malaysian ML/air-quality papers to Related Work (see TODO).
6. Verify every reference in `references.bib` against the publisher page.
7. Push `src/` to GitHub and put the URL in "Data and Code Availability".
8. Check the venue page limit; cut Sec. V-F (LOSO) or the DM table first if over.

## Design decisions reviewers will ask about
| Decision | Why |
|---|---|
| Target = daily PM2.5, not API | API is a max of piecewise sub-indices and changed definition when PM2.5 was added (~2017–18) |
| Chronological split + purge | Shuffled splits on autocorrelated series inflate accuracy |
| Persistence baseline + skill score | Without it, high R² means nothing |
| "oracle" meteorology labelled separately | Target-day reanalysis = perfect-forecast upper bound, not deployable |
| Identical test rows for all configs | Differences can't come from different samples |
| DM test on station-averaged daily loss | Correlated stations are not independent evidence |
| 5-seed LSTM ensemble + seed SD | Single LSTM runs are seed-sensitive |

## Run log — 24 Sep 2026 (revision after review; this supersedes the first run)
- **Target validated.** The WAQI `aqi` column (the only column before 2025) is the daily mean of
  DOE's official hourly API, **dated one day early**. Checked against APIMS hourly values for
  17–23 Sep 2026: 27 station-days, MAE 0.7. `build_panel.py` now re-dates it by +1 day
  (`AQ_DATE_SHIFT`, default 1; set it to 0 to reproduce the old alignment).
- **The 2018 break is not the PM2.5 inclusion.** DOE's own document says PM2.5 entered the API
  in 2017. The archive steps up in mid-Aug 2018, so the paper now calls it an undocumented level
  shift in the archive. Figure: `paper/figures/fig_break.pdf`.
- **Fire features.** VIIRS S-NPP SP counts (FIRMS area API, 5-day chunks, low confidence
  dropped) are in `data/raw/firms_daily_counts.csv`. `fetch_firms.py` now uses 5-day chunks
  (the API limit).
- **LSTM is now tuned.** It gets the same 15-candidate random search and the same folds as
  RF/XGBoost (`LSTM_SPACE` in `config.py`). Sequences are built with 21 days, so every window
  is scored on the same rows.
- **Reference config is H+M** (`MAIN_CFG`). Fire features degraded RF/XGBoost; all H+M+F
  results are still in Table III. `post_main.py` recomputes SHAP/LOSO for `MAIN_CFG`.
- **New analysis:** `diagnostics.py` computes paired block-bootstrap CIs (skill, CSI),
  Holm-adjusted DM p-values, forecast-conditioned haze errors, the M-only MSE decomposition
  and anomaly skill, haze episodes and ridge coefficients. Its macros go to
  `paper/tables/numbers_extra.tex`.
- **Sensitivity run** (old date alignment, LR/RF/XGB): `run_experiments.py --no_lstm`. It
  lives in a separate copy of the project; its numbers are quoted in the paper's Sec. V-G.
- Full run: about 2.7 h on 2 CPUs. The paper is 8 pages. Only 3 red placeholders remain:
  email, repository URL, supervisor.
- `paper/main_api.tex` is the old pre-results draft, kept for reference.
