"""
CLI for actigraphy sleep-wake metrics — Step 2.

Usage
-----
    python -m cli read <input_file.csv> [--measures GROUP] [--verbose]

Measure groups
--------------
    nonparametric   IS, IV, aggregate M10 / L5
    periodogram     Enright A_p and chi-square Q_p for periods 14-34 h
    temp_metrics    Per-day M10 / L5 (noon-to-noon window)

    Omit --measures to run all three groups.

Examples
--------
    python -m cli read data.csv
    python -m cli read data.csv --measures nonparametric --verbose
    python -m cli read data.csv --measures periodogram
    python -m cli read data.csv --measures temp_metrics --verbose

Outputs (written to step2/outputs/)
------------------------------------
    <stem>_nonparametric.csv  scalar non-parametric measures (IS, IV, M10, L5)
    <stem>_daily.csv          per-day M10 / L5 table
    <stem>_periodogram.csv    Enright + chi-square periodogram tables
    <stem>_report.pdf         multi-page PDF: heatmap, metrics table,
                              per-day table, periodogram plots
"""
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import importlib.util

import pandas as pd
import numpy as np

# ── locate this file's directory and ensure it is on sys.path ────────────────
_pkg = os.path.dirname(os.path.abspath(__file__))
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)

# ── load our local io.py via importlib to avoid shadowing stdlib 'io' ────────
def _load_local(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_pkg, relpath))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_io = _load_local("_acti_io", "io.py")
readParticipantCSV = _io.readParticipantCSV

from preprocess   import timeBin, aggregateHours, dayByHourMatrix
from nonparametric  import (interdailyStability, intradailyVariability,
                             compute_M10, compute_L5)
from periodogram    import (periodogram_Enright, periodogram_ChiSquare,
                             add_pValue_ChiSquare)
from visualize      import activityHeatmap, generate_pdf_report
from temp_metrics   import compute_daily_metrics

# ── valid measure group names ─────────────────────────────────────────────────
MEASURE_GROUPS = ('nonparametric', 'periodogram', 'temp_metrics')


# ── argument parser ───────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog='cli',
        description='Compute sleep-wake regularity metrics from Actigraph CSV data.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    rp = sub.add_parser('read', help='Read a CSV file and compute metrics.')
    rp.add_argument('input_file',
                    help='Path to the Actigraph CSV file (60-second epochs)')
    rp.add_argument('--measures',
                    choices=MEASURE_GROUPS,
                    default=None,
                    metavar='GROUP',
                    help='nonparametric | periodogram | temp_metrics  (omit to run all)')
    rp.add_argument('--verbose', action='store_true',
                    help='Print progress and computed values to stdout')
    return parser


# ── helpers ───────────────────────────────────────────────────────────────────

def vprint(flag, *args, **kwargs):
    if flag:
        print(*args, **kwargs)


# ── main read command ─────────────────────────────────────────────────────────

def run_read(args):
    input_path = Path(args.input_file).resolve()
    v = args.verbose

    run_nonparametric = args.measures in (None, 'nonparametric')
    run_periodogram   = args.measures in (None, 'periodogram')
    run_temp_metrics  = args.measures in (None, 'temp_metrics')

    outputs_dir = Path(_pkg) / 'outputs'
    outputs_dir.mkdir(exist_ok=True)
    stem = input_path.stem

    # ── 1. load ───────────────────────────────────────────────────────────────
    vprint(v, f"[1/4] Reading: {input_path}")
    d = readParticipantCSV(str(input_path.parent), input_path.name)

    delta = timeBin(d)
    vprint(v, f"      Rows: {len(d):,}  |  epoch: {delta:.1f} min  |  "
              f"{d['Time'].iloc[0]}  to  {d['Time'].iloc[-1]}")

    if delta > 60:
        sys.exit("ERROR: epoch interval > 60 min — too coarse for hourly aggregation.")

    # ── 2. aggregate to hourly means and build day×hour matrix ────────────────
    vprint(v, "[2/4] Aggregating to hourly means and building day×hour matrix...")
    hourly     = aggregateHours(d)
    mat, dates = dayByHourMatrix(hourly)
    vprint(v, f"      Matrix: {mat.shape[0]} days × {mat.shape[1]} hours  "
              f"({np.isnan(mat).sum()} NaN cells)")

    # ── 3. compute metrics ────────────────────────────────────────────────────
    results      = {}   # scalar metrics   -> nonparametric CSV + PDF summary table
    daily_rows   = []   # per-day metrics  -> daily CSV + PDF daily table
    pgrams       = {}   # periodogram DFs  -> periodogram CSV + PDF plots

    if run_nonparametric:
        vprint(v, "[3/4] Computing non-parametric metrics...")

        # ---- aggregate (whole-recording) measures ----
        IS  = interdailyStability(mat)
        IV  = intradailyVariability(mat)
        m10 = compute_M10(mat)
        l5  = compute_L5(mat)

        results = {
            'IS':                IS,
            'IV':                IV,
            'M10_hour':          m10['M10_hour'],
            'M10_mean_activity': m10['M10_mean_activity'],
            'L5_hour':           l5['L5_hour'],
            'L5_mean_activity':  l5['L5_mean_activity'],
        }

        if v:
            print(f"      IS   = {IS:.4f}  (0 = no stability, 1 = perfect)")
            print(f"      IV   = {IV:.4f}  (0 = sinusoidal, ~2 = pure noise)")
            print(f"      M10  onset {int(m10['M10_hour']):02d}:00,  "
                  f"mean activity = {m10['M10_mean_activity']:.4f}")
            print(f"      L5   onset {int(l5['L5_hour']):02d}:00,  "
                  f"mean activity = {l5['L5_mean_activity']:.4f}")


    if run_temp_metrics:
        vprint(v, "[3/4] Computing per-day M10/L5 (temp_metrics)...")
        # Delegated to temp_metrics.compute_daily_metrics which calls
        # oneDay_M10 / oneDay_L5 for each date in the recording.
        # Convention: window starts at noon (hour 12) so the sleep period
        # (around midnight) is centred within the 24-hour window.
        vprint(v, f"      {len(dates)} days, noon-to-noon window (start_hour=12)...")
        daily_rows = compute_daily_metrics(mat, dates,
                                           start_hour=12, verbose=v)

    if run_periodogram:
        vprint(v, "[3/4] Computing periodograms (periods 14 – 34 h)...")

        pg_enright = periodogram_Enright(mat)
        pg_chi     = add_pValue_ChiSquare(periodogram_ChiSquare(mat))

        pgrams['enright']   = pg_enright
        pgrams['chisquare'] = pg_chi

        if v:
            peak_e = pg_enright.loc[pg_enright['Value'].idxmax()]
            peak_c = pg_chi.loc[pg_chi['Value'].idxmax()]
            print(f"      Enright  peak: period = {int(peak_e['Period'])} h, "
                  f"A_p = {peak_e['Value']:.4f}")
            print(f"      Chi-sq   peak: period = {int(peak_c['Period'])} h, "
                  f"Q_p = {peak_c['Value']:.4f}")

    # ── 4. save outputs ───────────────────────────────────────────────────────
    vprint(v, "[4/4] Writing outputs...")

    if results:
        row = {
            'fname':                input_path.name,
            'processing_timestamp': datetime.now().isoformat(timespec='seconds'),
            **results,
        }
        csv_path = outputs_dir / f"{stem}_nonparametric.csv"
        pd.DataFrame([row]).to_csv(csv_path, index=False)
        vprint(v, f"      Nonparametric CSV  -> {csv_path}")

    if daily_rows:
        daily_df = pd.DataFrame(daily_rows)
        daily_df.insert(0, 'fname', input_path.name)
        daily_csv = outputs_dir / f"{stem}_daily.csv"
        daily_df.to_csv(daily_csv, index=False)
        vprint(v, f"      Daily CSV          -> {daily_csv}")

    if pgrams:
        # merge Enright + chi-square into a single periodogram table
        pg_out = pgrams.get('enright', pd.DataFrame()).rename(
            columns={'Value': 'Enright_Value'})
        if 'chisquare' in pgrams:
            cs = pgrams['chisquare'][['Period', 'Value', 'log_Pvalue']].rename(
                columns={'Value': 'ChiSq_Qp'})
            pg_out = (pg_out.merge(cs, on='Period')
                      if not pg_out.empty else cs)
        pg_csv = outputs_dir / f"{stem}_periodogram.csv"
        pg_out.to_csv(pg_csv, index=False)
        vprint(v, f"      Periodogram CSV    -> {pg_csv}")

    pdf_path = outputs_dir / f"{stem}_report.pdf"
    generate_pdf_report(
        pdf_path=str(pdf_path),
        mat=mat,
        dates=dates,
        fname=input_path.name,
        results=results,
        daily_results=daily_rows,
        pgrams=pgrams,
    )
    vprint(v, f"      PDF report         -> {pdf_path}")

    if not v:
        print(f"Done. Outputs written to: {outputs_dir}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()
    if args.command == 'read':
        run_read(args)


if __name__ == '__main__':
    main()
