# temp_metrics.py
# Temporary / experimental metrics module.
#
# Owns the per-day M10 / L5 functions (oneDay_M10, oneDay_L5, oneDay_Filter,
# offsetTime) and the extraction helpers they depend on.  Also exposes
# compute_daily_metrics(), which is the single entry point called by cli.py.
#
# The aggregate (whole-recording) M10 and L5 remain in nonparametric.py as
# compute_M10 / compute_L5.  The functions here share the same rolling-mean
# filter infrastructure (LOCF_NA, _filterHour_ForwardBack) via import.
#
# compute_daily_metrics is called by cli.py when --measures nonparametric is
# selected (or when no --measures flag is given).  The results feed into both
# the daily CSV output (<stem>_daily.csv) and the per-day table page in the
# PDF report.

import os
import warnings
import numpy as np
import pandas as pd

# shared filter primitives live in nonparametric.py
from nonparametric import LOCF_NA, _filterHour_ForwardBack


# ── Matrix extraction helpers ──────────────────────────────────────────────────

def _extractHoursFrom(mat, dates, idx_date, start_hour, duration):
    # Extracts a consecutive run of `duration` hours from the day-by-hour matrix
    # starting at clock-hour `start_hour` on the day identified by `idx_date`,
    # wrapping into the next day if the window crosses midnight.
    #
    # Equivalent to R's extractHoursFrom() used by oneDay_M10.
    #
    # Arguments:
    #   mat        — numpy array [days x 24] of mean hourly Activity
    #   dates      — array-like of date labels, length = number of rows in mat
    #                (e.g. the second return value of dayByHourMatrix())
    #   idx_date   — the date label for the day of interest (must match an entry
    #                in `dates`)
    #   start_hour — first clock-hour of the extraction window (0..23)
    #   duration   — number of consecutive hours to extract (e.g. 24 for L5,
    #                or a partial window for M10)
    #
    # Returns:
    #   pandas Series of length `duration` whose index holds the clock-hour
    #   labels (wrapping modulo 24), with NaN where data are absent.
    #
    # Input column (raw CSV)  : SVM_sum  — signal vector magnitude sum per 60-s epoch
    # Column after io.py      : SVM      (SVM_sum renamed)
    # Column after aggregation: Activity — mean(SVM) per calendar hour
    # Matrix input            : mat[d, h] = Activity for day d, hour h
    #                           shape (D, 24), NaN for hours with no recording

    dates = list(dates)
    try:
        day_idx = dates.index(idx_date)
    except ValueError:
        return pd.Series(np.full(duration, np.nan),
                         index=[(start_hour + i) % 24 for i in range(duration)])

    values = []
    hours  = []
    for i in range(duration):
        clock_hour = (start_hour + i) % 24
        # advance to the next day's row when the window wraps past midnight
        d_offset   = (start_hour + i) // 24
        d_idx      = day_idx + d_offset
        if d_idx < mat.shape[0]:
            values.append(mat[d_idx, clock_hour])
        else:
            values.append(np.nan)
        hours.append(clock_hour)

    return pd.Series(values, index=hours, dtype=float)


def _extract24HoursFrom(mat, dates, idx_date, start_hour):
    # Convenience wrapper: extracts exactly 24 consecutive hours starting at
    # `start_hour` on `idx_date`, wrapping into the next day as needed.
    #
    # Equivalent to R's extract24HoursFrom() used by oneDay_L5.
    #
    # Arguments: see _extractHoursFrom.
    # Returns:   pandas Series of length 24 with clock-hour index.
    return _extractHoursFrom(mat, dates, idx_date, start_hour, duration=24)


# ── Per-day public API ─────────────────────────────────────────────────────────

def oneDay_Filter(X_h, filter_hour, maximum=True):
    # Compute either the most-active (M10-style) or least-active (L5-style)
    # window for a single day's hourly activity vector.
    #
    # Equivalent to R's oneDay_Filter().
    #
    # Sanity check: if all non-NaN values are exactly 0.0 (e.g. device removed
    # all day) the function returns NaN sentinels rather than a spurious result.
    #
    # Arguments:
    #   X_h         — pandas Series of hourly mean Activity values with
    #                 clock-hour index, as returned by _extractHoursFrom /
    #                 _extract24HoursFrom.
    #   filter_hour — rolling-mean window width passed to _filterHour_ForwardBack.
    #                 Use 11 for M10 (centres ±5 h), 6 for L5 (centres ±3 h).
    #   maximum     — if True (default), find the hour with the HIGHEST filtered
    #                 mean (M10); if False, find the hour with the LOWEST (L5).
    #
    # Returns dict with keys:
    #   'hour'            — clock-hour of the identified window midpoint (int),
    #                       or NaN if the sanity check triggered
    #   'mean_activity'   — filtered rolling-mean value at that hour (float / NaN)
    #   'actual_activity' — raw (unfiltered) hourly mean at that hour (float / NaN)
    #   'filter_output'   — the full DataFrame from _filterHour_ForwardBack, or
    #                       None when the sanity check triggered
    #
    # Input column (raw CSV)  : SVM_sum  — signal vector magnitude sum per 60-s epoch
    # Column after io.py      : SVM      (SVM_sum renamed)
    # Column after aggregation: Activity — mean(SVM) per calendar hour
    # Series input            : X_h[h]  = Activity for clock-hour h on one day,
    #                           index = clock-hour labels (0..23, may wrap)

    vals = np.asarray(X_h, dtype=float)

    # sanity check: device not worn / no activity recorded all day
    non_nan = vals[~np.isnan(vals)]
    if len(non_nan) > 0 and np.all(non_nan == 0.0):
        return {
            'hour':            np.nan,
            'mean_activity':   np.nan,
            'actual_activity': np.nan,
            'filter_output':   None,
        }

    # fill NaN gaps before applying the rolling mean filter.
    # Edge days (the first and last day of a recording) often have NaN for
    # the hours outside the recorded window (e.g. the last day's noon-to-noon
    # window extends into a "next day" row that does not exist in the matrix).
    # np.convolve propagates NaN through every window that contains one, so we
    # fill with LOCF then backward-fill any remaining leading NaN before
    # filtering.  This matches the R approach of assuming "no change from last
    # recorded hour" for missing observations at the edges.
    if np.any(np.isnan(vals)):
        filled = pd.Series(LOCF_NA(vals)).ffill().bfill().to_numpy()
        X_h = (pd.Series(filled, index=X_h.index)
               if isinstance(X_h, pd.Series) else filled)

    df  = _filterHour_ForwardBack(X_h, filter_hour)
    df2 = df.copy()   # preserve the full output before sorting

    if maximum:
        df = df.sort_values(by='filterHourMean', ascending=False)
    else:
        df = df.sort_values(by='filterHourMean', ascending=True)

    return {
        'hour':            int(df.iloc[0]['Hour']),
        'mean_activity':   df.iloc[0]['filterHourMean'],
        'actual_activity': df.iloc[0]['HourMean'],
        'filter_output':   df2,
    }


def oneDay_M10(mat, dates, idx_date, start_hour, duration):
    # Compute M10 for a single recording day.
    #
    # Equivalent to R's oneDay_M10().
    #
    # Extracts `duration` consecutive hours starting at `start_hour` on
    # `idx_date` (wrapping across midnight if needed), applies the bidirectional
    # 11-hour rolling mean, and returns the hour with the highest mean activity.
    #
    # Window width 11 centres the reported onset hour as the midpoint of the
    # 10-hour most-active window (±5 hours), matching the R implementation.
    #
    # Arguments:
    #   mat        — numpy array [days x 24] of mean hourly Activity
    #   dates      — array-like of date labels (same length as mat rows)
    #   idx_date   — date label for the day of interest
    #   start_hour — clock-hour at which the extraction window begins (0..23)
    #   duration   — number of hours to extract (typically 24 for a full day)
    #
    # Returns dict with keys:
    #   'M10_hour'            — clock-hour of the M10 midpoint (int / NaN)
    #   'M10_actual_activity' — raw hourly mean at that hour
    #   'M10_mean_activity'   — filtered rolling-mean value at that hour
    #   'M10_date'            — the date label supplied as idx_date
    #   'filtered_data'       — full filter DataFrame (for diagnostics / plotting)
    #
    # Input column (raw CSV)  : SVM_sum  — signal vector magnitude sum per 60-s epoch
    # Column after io.py      : SVM      (SVM_sum renamed)
    # Column after aggregation: Activity — mean(SVM) per calendar hour
    # Matrix input            : mat[d, h] = Activity for day d, hour h

    X_h = _extractHoursFrom(mat, dates, idx_date, start_hour, duration)
    res = oneDay_Filter(X_h, filter_hour=11, maximum=True)

    # if no valid activity value, discard the reported hour too
    if _isnan(res['mean_activity']):
        res['hour'] = np.nan

    return {
        'M10_hour':            res['hour'],
        'M10_actual_activity': res['actual_activity'],
        'M10_mean_activity':   res['mean_activity'],
        'M10_date':            idx_date,
        'filtered_data':       res['filter_output'],
    }


def oneDay_L5(mat, dates, idx_date, start_hour):
    # Compute L5 for a single recording day.
    #
    # Equivalent to R's oneDay_L5().
    #
    # Extracts 24 consecutive hours starting at `start_hour` on `idx_date`,
    # applies the bidirectional 6-hour rolling mean, and identifies the
    # least-active 5-hour window midpoint.
    #
    # Window width 6 centres the reported onset hour as the midpoint of the
    # 5-hour rest window (±3 hours), matching the R implementation.
    #
    # Special case — tied minima at zero:
    #   When several consecutive hours all have zero activity (e.g. full nights
    #   of rest), the rolling mean is zero for a run of hours and there is no
    #   unique minimum.  In this case the midpoint of the run is returned,
    #   matching R:
    #     nr <- nrow(df)                      # number of tied minima
    #     return.hour.idx <- floor(nr/2) + 1  # 1-based R index -> midpoint
    #   In Python (0-based):
    #     return_idx = int(np.floor(nr / 2))  # equivalent midpoint
    #
    # Arguments:
    #   mat        — numpy array [days x 24] of mean hourly Activity
    #   dates      — array-like of date labels (same length as mat rows)
    #   idx_date   — date label for the day of interest
    #   start_hour — clock-hour at which the 24-hour extraction window begins
    #
    # Returns dict with keys:
    #   'L5_hour'            — clock-hour of the L5 midpoint (int / NaN)
    #   'L5_mean_activity'   — filtered rolling-mean value at that hour
    #   'L5_actual_activity' — raw hourly mean at that hour
    #   'L5_date'            — the date label supplied as idx_date
    #   'filtered_data'      — full filter DataFrame (for diagnostics / plotting)
    #
    # Input column (raw CSV)  : SVM_sum  — signal vector magnitude sum per 60-s epoch
    # Column after io.py      : SVM      (SVM_sum renamed)
    # Column after aggregation: Activity — mean(SVM) per calendar hour
    # Matrix input            : mat[d, h] = Activity for day d, hour h

    X_h = _extract24HoursFrom(mat, dates, idx_date, start_hour)

    # fill NaN gaps before filtering — same rationale as oneDay_Filter:
    # edge days have NaN for hours outside the recording window, which would
    # propagate through np.convolve and leave idx_minima empty.
    vals = X_h.to_numpy(dtype=float)
    if np.any(np.isnan(vals)):
        filled = pd.Series(LOCF_NA(vals)).ffill().bfill().to_numpy()
        X_h = pd.Series(filled, index=X_h.index)

    df  = _filterHour_ForwardBack(X_h, n=6)
    df2 = df.copy()   # preserve unsorted for filter_output

    min_val = df['filterHourMean'].min()

    # guard: if the entire filter output is still NaN (completely empty day
    # with no valid data even after fill), return NaN sentinels rather than
    # raising an IndexError.
    if pd.isna(min_val):
        return {
            'L5_hour':            np.nan,
            'L5_mean_activity':   np.nan,
            'L5_actual_activity': np.nan,
            'L5_date':            idx_date,
            'filtered_data':      df2,
        }

    idx_minima = df.index[df['filterHourMean'] == min_val].tolist()

    nr = len(idx_minima)
    if nr > 1:
        # multiple tied minima — return the midpoint of the run
        # R:  return.hour.idx <- floor(nr/2) + 1  (1-based)
        # Python equivalent (0-based):
        return_idx = int(np.floor(nr / 2))
    else:
        return_idx = 0

    chosen_row = df.loc[idx_minima[return_idx]]

    return {
        'L5_hour':            int(chosen_row['Hour']),
        'L5_mean_activity':   chosen_row['filterHourMean'],
        'L5_actual_activity': chosen_row['HourMean'],
        'L5_date':            idx_date,
        'filtered_data':      df2,
    }


def offsetTime(start, end):
    # Computes the forward offset in hours from a start clock-hour to an end
    # clock-hour on a 24-hour circular clock.
    #
    # Equivalent to R's offsetTime():
    #   return( ((24 - start) + end) %% 24 )
    #
    # Examples:
    #   offsetTime(15, 2)  ->  11  (3 pm to 2 am the next morning = 11 hours)
    #   offsetTime(0, 6)   ->   6  (midnight to 6 am)
    #   offsetTime(22, 22) ->   0  (same time, no offset)
    #
    # Arguments:
    #   start — clock-hour of the reference time  (0..23)
    #   end   — clock-hour of the target time     (0..23)
    #
    # Returns: integer offset in hours (0..23)
    return ((24 - start) + end) % 24


# ── Per-day batch wrapper ──────────────────────────────────────────────────────

def compute_daily_metrics(mat, dates, start_hour=12, verbose=False):
    # Compute per-day M10 and L5 metrics for every date in the recording.
    #
    # Uses a noon-to-noon window (start_hour=12, duration=24 h) so the sleep
    # period (around midnight) falls in the centre of each day's analysis
    # window.  This is the standard actigraphy convention used by the R
    # implementation.
    #
    # If a day's data are incomplete (e.g. the first or last day of the
    # recording) the per-day functions apply LOCF then back-fill to handle the
    # missing hours before filtering.  If a day is entirely absent (all NaN)
    # the returned hour and activity values will be None / NaN.
    #
    # Arguments:
    #   mat        — numpy array [days x 24] of mean hourly Activity
    #                (second return value of dayByHourMatrix() in preprocess.py)
    #   dates      — array-like of date labels, one per row of mat
    #                (first return value of dayByHourMatrix())
    #   start_hour — clock-hour at which each day's window begins (default 12)
    #   verbose    — if True, print one line per day showing M10 and L5 onsets
    #
    # Returns:
    #   list of dicts, one per date, with keys:
    #     'date'                 — date label (string)
    #     'M10_hour'             — most-active 10-h window midpoint (int or None)
    #     'M10_mean_activity'    — filtered rolling mean at M10 onset
    #     'M10_actual_activity'  — raw hourly mean at M10 onset hour
    #     'L5_hour'              — least-active 5-h window midpoint (int or None)
    #     'L5_mean_activity'     — filtered rolling mean at L5 onset
    #     'L5_actual_activity'   — raw hourly mean at L5 onset hour
    #
    # Input column (raw CSV)  : SVM_sum  — signal vector magnitude sum per 60-s epoch
    # Column after io.py      : SVM      (SVM_sum renamed)
    # Column after aggregation: Activity — mean(SVM) per calendar hour
    # Matrix input            : mat[d, h] = Activity for day d, hour h
    #                           shape (D, 24), NaN for hours with no recording

    rows = []
    for idx_date in dates:
        r_m10 = oneDay_M10(mat, dates, idx_date,
                           start_hour=start_hour, duration=24)
        r_l5  = oneDay_L5(mat, dates, idx_date,
                          start_hour=start_hour)

        m10_h = r_m10['M10_hour']
        l5_h  = r_l5['L5_hour']

        rows.append({
            'date':                str(idx_date),
            'M10_hour':            int(m10_h) if not _isnan(m10_h) else None,
            'M10_mean_activity':   r_m10['M10_mean_activity'],
            'M10_actual_activity': r_m10['M10_actual_activity'],
            'L5_hour':             int(l5_h)  if not _isnan(l5_h)  else None,
            'L5_mean_activity':    r_l5['L5_mean_activity'],
            'L5_actual_activity':  r_l5['L5_actual_activity'],
        })

        if verbose:
            m10_str = f"{int(m10_h):02d}:00" if not _isnan(m10_h) else "  N/A"
            l5_str  = f"{int(l5_h):02d}:00"  if not _isnan(l5_h)  else "  N/A"
            print(f"        {str(idx_date)}  M10={m10_str}  L5={l5_str}")

    return rows


def _isnan(x):
    """Return True if x is NaN or None (safe for int / float / None)."""
    if x is None:
        return True
    try:
        return np.isnan(x)
    except (TypeError, ValueError):
        return False


# ── GGIR / SRI stubs ───────────────────────────────────────────────────────────
# Placeholder wrappers for future integration with the GGIR R package
# and the Sleep Regularity Index.  Not yet implemented.

def GGIR_from_csv(dsdir, alloutdir=None, outputdir=None, tz="UTC"):
    """
    Wrapper function to execute GGIR R script from Python.
    GGIR is a large R package and a direct Python translation is not feasible.
    This function assumes you have an R script named `run_ggir.R` that contains
    the GGIR call.
    """
    if not dsdir:
        raise ValueError("Specify directory containing down-sampled .csv files")

    if not alloutdir:
        alloutdir = os.path.dirname(dsdir)

    if not outputdir:
        outputdir = os.path.join(alloutdir, "GGIR_output")
    os.makedirs(outputdir, exist_ok=True)

    warnings.warn("GGIR_from_csv is a wrapper and requires an R installation "
                  "with the GGIR package.")


def SRI_from_GGIR(outputdir, nwdir=None, use_naps=True, use_WASO=True):
    """
    Calculate Sleep Regularity Index (SRI) from GGIR Output.
    This function processes the output files generated by GGIR.
    """
    if not outputdir:
        raise ValueError("Specify directory containing GGIR output")

    warnings.warn("SRI_from_GGIR requires reading specific GGIR output structures.")
