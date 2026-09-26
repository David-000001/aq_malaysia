"""One command from raw downloads to a filled paper.

    python src/run_all.py --aqicn_dir data/raw/aqicn            # full (paper) run
    python src/run_all.py --aqicn_dir data/raw/aqicn --quick    # ~10x faster, for checking

Steps: ERA5 meteorology -> FIRMS fires (if FIRMS_MAP_KEY set) -> panel ->
experiments -> tables/figures/numbers. Steps whose outputs exist are skipped
(delete the file to force). Logs go to results/run_all.log.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402


def step(name, args):
    print(f"\n>>> {name}\n    {' '.join(args)}", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, *args], cwd=HERE)
    if r.returncode:
        raise SystemExit(f"step '{name}' failed (exit {r.returncode})")
    print(f"    done in {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aqicn_dir", type=Path, help="folder with WAQI historical CSVs")
    ap.add_argument("--pm", type=Path, help="DOE long-format CSV (instead of aqicn)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if not a.aqicn_dir and not a.pm:
        raise SystemExit("give --aqicn_dir or --pm")

    if a.aqicn_dir:
        step("data check", ["check_aqicn.py", str(a.aqicn_dir.resolve())])
    print(f"\n>>> target: {C.TARGET_MODE}  (change with AQ_TARGET=pm25|api)")
    step("meteorology (ERA5 via Open-Meteo)", ["fetch_meteo.py"])
    if os.environ.get("FIRMS_MAP_KEY") and not C.PATHS["firms_daily"].exists():
        step("fire hotspots (NASA FIRMS)", ["fetch_firms.py", "--api"])
    elif not C.PATHS["firms_daily"].exists():
        print("\n>>> FIRMS_MAP_KEY not set -> fire features skipped (H+M+F rows omitted)")
    if a.aqicn_dir:
        step("panel", ["build_panel.py", "--format", "aqicn", "--aqicn_dir", str(a.aqicn_dir.resolve())])
    else:
        step("panel", ["build_panel.py", "--format", "long", "--pm", str(a.pm.resolve())])
    step("experiments", ["run_experiments.py"] + (["--fast"] if a.quick else []))
    step("tables, figures, numbers", ["report.py"])
    print("\nALL DONE. Compile paper/main.tex; remaining red [..] are interpretation "
          "sentences to write yourself (grep -n 'res{' paper/main.tex).")


if __name__ == "__main__":
    main()
