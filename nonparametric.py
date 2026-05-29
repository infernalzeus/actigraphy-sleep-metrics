# Non-parametric summary statistics for actigraphy data.
# Designed to be implemented without the need for periodogram functions.
# Python translation of non-parametric-measures.R

import numpy as np
import pandas as pd


def interdailyStability(mat_df, P=24):
    # Computes inter-daily stability (IS) — default period of 24 hours.
    # See W. Witting, I.H. Kwa, P. Eikelenboom, M. Mirmiran, D.F. Swaab,
    #   Alterations in the circadian rest-activity rhythm in aging and
    #   Alzheimer's disease, Biological Psychiatry, Volume 27, Issue 6, 1990,
    #   pp. 563-572
    #
    # IS quantifies how consistent the 24-hour activity pattern is across days.
    # IS = 0  ->  no day-to-day regularity (pure noise)
    # IS = 1  ->  identical 24-hour pattern on every recorded day
    #
    # Input column (raw CSV)  : SVM_sum  — signal vector magnitude sum per 60-s epoch
    # Column after io.py      : SVM      (SVM_sum renamed)
    # Column after aggregation: Activity — mean(SVM) per calendar hour
    # Matrix input            : mat_df[d, h] = Activity for day d, hour h
    #                           shape (D, 24), NaN for hours with no recording
    #         P               — period length in hours (default 24)
    #
    # The matrix is first flattened row-by-row (day-major order) to produce X_v,
    # then padded with NaN to fill a complete rectangular matrix of P columns.
    #
    # Formula:
    #
    #          N * sum_h( (X_h - X_bar)^2 )
    #   IS = ─────────────────────────────────
    #          P * sum_i( (X_i - X_bar)^2 )
    #
    # where:
    #   N      = number of valid (non-NaN) hourly observations in the full series
    #   P      = 24  (period length, columns of the reshaped matrix)
    #   X_h    = mean activity for hour h, averaged across all days  (column means)
    #   X_bar  = grand mean of all valid hourly values
    #   X_i    = individual hourly value at position i in the flattened series

    # flatten the [days x hours] matrix row-by-row: day1_h0, day1_h1, ..., day2_h0, ...
    X_v = mat_df.flatten()
    N   = len(X_v)
    K   = N / P

    # pad the end with NaN so the series fills a complete K x P rectangle
    pad_len = int(np.ceil(K) * P - N)
    if pad_len > 0:
        X_v = np.pad(X_v, (0, pad_len), constant_values=np.nan)

    # reshape to [K rows x P columns] to compute column means X_h
    X_m   = X_v.reshape((-1, P))
    X_bar = np.nanmean(X_m)
    X_h   = np.nanmean(X_m, axis=0)   # mean activity for each hour across days

    # N_included: number of valid data points before padding
    N_included = N - np.isnan(X_v[:N]).sum()

    numer = N_included * np.nansum((X_h - X_bar) ** 2)
    denom = P          * np.nansum((X_v - X_bar) ** 2)

    return numer / denom


def LOCF_NA(x):
    # Last Observation Carried Forward — fills internal NaN gaps by repeating
    # the last valid value.  Leading NaNs (before the first valid observation)
    # are left as NaN and removed by the caller.
    # Equivalent to R: LOCF_NA <- function(x) { v <- !is.na(x); c(NA, x[v])[cumsum(v)+1] }
    mask = np.isnan(x)
    if not np.any(mask):
        return x
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx].copy()
    # restore any leading NaNs that existed before the first valid observation
    if mask[0]:
        first_valid = np.argmax(~mask)
        out[:first_valid] = np.nan
    return out


def intradailyVariability(mat_df, LOCF=True):
    # Computes intra-daily variability (IV).
    # See W. Witting, I.H. Kwa, P. Eikelenboom, M. Mirmiran, D.F. Swaab,
    #   Alterations in the circadian rest-activity rhythm in aging and
    #   Alzheimer's disease, Biological Psychiatry, Volume 27, Issue 6, 1990,
    #   pp. 563-572
    #
    # IV quantifies within-day fragmentation of the rest-activity rhythm:
    # how often and how sharply activity alternates between high and low values.
    # IV ~ 0   ->  smooth, sinusoidal rhythm (low fragmentation)
    # IV ~ 2   ->  pure random noise (maximum fragmentation)
    #
    # Input column (raw CSV)  : SVM_sum  — signal vector magnitude sum per 60-s epoch
    # Column after io.py      : SVM      (SVM_sum renamed)
    # Column after aggregation: Activity — mean(SVM) per calendar hour
    # Matrix input            : mat_df[d, h] = Activity for day d, hour h
    #                           shape (D, 24), NaN for hours with no recording
    #         LOCF            — if True, fill internal NaN gaps before computing the
    #                           first difference (default True, matching R behaviour)
    #
    # NOTE: because recordings rarely start and end exactly at midnight,
    # the first and last days often have missing hours (NaN).  LOCF fills
    # internal gaps (e.g. a missing hour mid-recording) by assuming no change
    # from the last recorded hour.  Leading/trailing NaNs are then stripped.
    #
    # Formula (Eq. 2 of Witting et al., 1990):
    #
    #          N * sum_i( (X_{i+1} - X_i)^2 )
    #   IV = ───────────────────────────────────────
    #          (N - 1) * sum_i( (X_i - X_bar)^2 )
    #
    # where:
    #   N       = number of valid hourly values after LOCF and NaN removal
    #   X_i     = activity at hour i in the flattened, LOCF-filled series
    #   X_{i+1} = activity at the next hour  (numerator is the squared first difference)
    #   X_bar   = mean of the cleaned series

    X_v = mat_df.flatten()

    if LOCF:
        # replace internal NaN values with last valid observation
        X_v = LOCF_NA(X_v)

    # remove remaining NaNs (leading/trailing only after LOCF)
    X_v = X_v[~np.isnan(X_v)]

    if len(X_v) == 0:
        return np.nan

    X_bar = np.mean(X_v)
    N     = len(X_v)

    # numerical first derivative of the time series
    dX_v = np.diff(X_v)

    # numerator / denominator of Eq. 2 in Witting et al., 1990
    numer = N       * np.sum(dX_v ** 2)
    denom = (N - 1) * np.sum((X_bar - X_v) ** 2)

    return numer / denom


# ── Rolling-mean filter helpers ────────────────────────────────────────────────

def _filterHour(x, n):
    # One-sided circular rolling mean of width n applied to the values of x.
    # Equivalent to R: stats::filter(x, rep(1/n, n), method="convolution",
    #                                circular=TRUE, sides=1)
    # The series is treated as circular (hourly means wrap at midnight).
    # Pads n-1 wrapped values at the END so a valid convolution of length
    # len(x) is produced with a rightward (backward-looking) lag.
    #
    # Arguments:
    #   x — array-like of length L (hourly mean Activity values)
    #   n — window width in hours
    #
    # Returns: numpy array of length L
    vals    = np.asarray(x, dtype=float)
    weights = np.ones(n) / n
    return np.convolve(np.pad(vals, (0, n - 1), mode='wrap'), weights, mode='valid')


def _filterHour_ForwardBack(x, n):
    # Bidirectional rolling mean — applies the one-sided filter in both
    # directions and averages the two passes to remove phase shift.
    #
    # Equivalent to R's filterHour_ForwardBack():
    #   f.x.1 <- rev( filterHour( rev(x), n ) )   # backward pass
    #   f.x.2 <- filterHour( x, n )               # forward pass
    #   f.x   <- 0.5 * (f.x.1 + f.x.2)
    #
    # Arguments:
    #   x — pandas Series whose index holds the clock-hour labels (e.g. 0..23),
    #       OR a plain numpy array/list (hours will be labelled 0..len(x)-1)
    #   n — window width in hours
    #
    # The forward pass  f2[i] uses hours  i, i+1, ..., i+n-1  (circular)
    # The backward pass f1[i] uses hours  i-n+1, ..., i-1, i  (circular)
    # Their average  f[i] = 0.5*(f1[i] + f2[i])  is a symmetric n-hour
    # rolling mean centred on hour i, matching the R implementation which
    # uses Hour = as.numeric(names(x)) to preserve the original hour indices.
    if isinstance(x, pd.Series):
        hours = x.index.to_numpy(dtype=float)
        vals  = x.to_numpy(dtype=float)
    else:
        vals  = np.asarray(x, dtype=float)
        hours = np.arange(len(vals), dtype=float)

    f_x_1 = _filterHour(vals[::-1], n)[::-1]   # backward pass then un-reverse
    f_x_2 = _filterHour(vals, n)               # forward pass
    f_x   = 0.5 * (f_x_1 + f_x_2)

    return pd.DataFrame({
        'Hour':           hours,
        'HourMean':       vals,
        'filterHourMean': f_x,
    })


# ── Multi-day (aggregate) M10 / L5 ────────────────────────────────────────────

def compute_M10(X_m):
    # Computes M10: the onset hour and mean activity of the most active
    # 10-consecutive-hour window, averaged across all recording days.
    # See E.J. van Someren, D.F. Swaab, C.C. Colenda, W. Cohen,
    #   W.V. McCall, P.B. Rosenquist, Bright light therapy: improved
    #   sensitivity to its effects on rest-activity rhythms in Alzheimer
    #   patients by application of nonparametric methods.
    #   Chronobiology International, 16(4), 1999, pp. 505-518.
    #
    # Input column (raw CSV)  : SVM_sum  — signal vector magnitude sum per 60-s epoch
    # Column after io.py      : SVM      (SVM_sum renamed)
    # Column after aggregation: Activity — mean(SVM) per calendar hour
    # Matrix input            : X_m[d, h] = Activity for day d, hour h
    #                           shape (D, 24), NaN for hours with no recording
    #
    # Algorithm:
    #   1. Compute X_h = mean(X_m[:, h]) — mean Activity at each hour across all days
    #   2. Apply the bidirectional 11-hour rolling mean to X_h (circular).
    #      Window width 11 is used so that the identified hour is the exact
    #      midpoint of a 10-hour active window (±5 hours either side),
    #      matching the R implementation which uses filterHour_ForwardBack(X_h, 11).
    #   3. M10 onset = the hour h with the HIGHEST rolling mean value
    #   4. M10 mean  = that highest rolling mean value

    X_h = np.nanmean(X_m, axis=0)              # mean Activity per hour across days
    # Pass as a Series with clock-hour index so _filterHour_ForwardBack
    # preserves the original hour labels in the returned DataFrame
    X_h_series = pd.Series(X_h, index=np.arange(24))
    df  = _filterHour_ForwardBack(X_h_series, 11)

    df_sorted = df.sort_values(by='filterHourMean', ascending=False)

    return {
        'M10_hour':          int(df_sorted.iloc[0]['Hour']),
        'M10_mean_activity': df_sorted.iloc[0]['filterHourMean'],
    }


def compute_L5(X_m):
    # Computes L5: the onset hour and mean activity of the least active
    # 5-consecutive-hour window, averaged across all recording days.
    # See van Someren et al. (1999) — same reference as compute_M10.
    #
    # Input column (raw CSV)  : SVM_sum  — signal vector magnitude sum per 60-s epoch
    # Column after io.py      : SVM      (SVM_sum renamed)
    # Column after aggregation: Activity — mean(SVM) per calendar hour
    # Matrix input            : X_m[d, h] = Activity for day d, hour h
    #                           shape (D, 24), NaN for hours with no recording
    #
    # Algorithm:
    #   1. Compute X_h = mean(X_m[:, h]) — mean Activity at each hour across all days
    #   2. Apply the bidirectional 6-hour rolling mean to X_h (circular).
    #      Window width 6 is used so that the identified hour is the exact
    #      midpoint of a 5-hour rest window (±3 hours either side),
    #      matching the R implementation which uses filterHour_ForwardBack(X_h, 6).
    #   3. L5 onset = the hour h with the LOWEST rolling mean value
    #   4. L5 mean  = that lowest rolling mean value
    #
    # L5 typically identifies the rest/sleep window (lowest activity) and
    # its onset hour can serve as a proxy for habitual sleep time.

    X_h = np.nanmean(X_m, axis=0)              # mean Activity per hour across days
    X_h_series = pd.Series(X_h, index=np.arange(24))
    df  = _filterHour_ForwardBack(X_h_series, 6)

    df_sorted = df.sort_values(by='filterHourMean', ascending=True)

    return {
        'L5_hour':          int(df_sorted.iloc[0]['Hour']),
        'L5_mean_activity': df_sorted.iloc[0]['filterHourMean'],
    }
