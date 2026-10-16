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


def intradailyVariability(mat_df, LOCF=False):
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

    finite = np.isfinite(X_v)
    valid_positions = np.flatnonzero(finite)
    if valid_positions.size:
        X_v = X_v[valid_positions[0]:valid_positions[-1] + 1]
        finite = np.isfinite(X_v)
    valid_values = X_v[finite]
    if len(valid_values) < 2:
        return np.nan

    X_bar = np.mean(valid_values)
    N = len(valid_values)
    if LOCF:
        X_v = X_v[finite]
        dX_v = np.diff(X_v)
    else:
        # Witting's equation assumes one uninterrupted sequence. Partial first
        # and last days are trimmed; internal gaps return NA rather than being
        # imputed or joined across.
        if not finite.all():
            return np.nan
        dX_v = np.diff(X_v)

    # numerator / denominator of Eq. 2 in Witting et al., 1990
    numer = N       * np.sum(dX_v ** 2)
    denom = (N - 1) * np.sum((X_bar - valid_values) ** 2)
    return numer / denom if denom > 0 else np.nan


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


# ── Canonical minute-level aggregate Mx / Lx ─────────────────────────────────

def _clock_minute_profile(data, excluded_dates=()):
    """Average activity for each local clock minute, without imputation."""
    required = {'Time', 'SVM'}
    if not required.issubset(data.columns):
        raise ValueError("Mx/Lx requires Time and SVM columns.")
    frame = data[['Time', 'SVM']].dropna(subset=['Time']).copy()
    if excluded_dates:
        frame = frame[~frame['Time'].dt.date.isin(set(excluded_dates))]
    frame['minute'] = frame['Time'].dt.floor('min')
    minute = frame.groupby('minute', sort=True)['SVM'].mean()
    minute_of_day = minute.index.hour * 60 + minute.index.minute
    return minute.groupby(minute_of_day).mean().reindex(range(1440)).to_numpy(float)


def _circular_window_means(profile, window_minutes, resolution_minutes):
    if not 1 <= window_minutes <= len(profile):
        raise ValueError("window must be between 1 minute and 24 hours.")
    if resolution_minutes < 1:
        raise ValueError("resolution_minutes must be positive.")
    extended = np.concatenate([profile, profile[:window_minutes - 1]])
    finite = np.isfinite(extended)
    values = np.where(finite, extended, 0.0)
    sums = np.r_[0.0, np.cumsum(values)]
    counts = np.r_[0, np.cumsum(finite.astype(np.int64))]
    starts = np.arange(0, len(profile), resolution_minutes, dtype=int)
    window_sums = sums[starts + window_minutes] - sums[starts]
    window_counts = counts[starts + window_minutes] - counts[starts]
    means = np.full(len(starts), np.nan, dtype=float)
    complete = window_counts == window_minutes
    means[complete] = window_sums[complete] / window_minutes
    return starts, means


def _format_clock_minute(value):
    return f"{value // 60:02d}:{value % 60:02d}"


def compute_mxlx(data, m_window_hours=10.0, l_window_hours=5.0,
                  resolution_minutes=1, excluded_dates=()):
    """Canonical nparACT-style Mx/Lx on an average 24-hour minute profile.

    Windows are exact, contiguous and circular. A candidate is valid only when
    every clock minute in it is observed on at least one non-excluded day.
    """
    m_minutes = int(round(float(m_window_hours) * 60))
    l_minutes = int(round(float(l_window_hours) * 60))
    if not np.isclose(m_minutes, float(m_window_hours) * 60):
        raise ValueError("m_window_hours must resolve to a whole number of minutes.")
    if not np.isclose(l_minutes, float(l_window_hours) * 60):
        raise ValueError("l_window_hours must resolve to a whole number of minutes.")
    profile = _clock_minute_profile(data, excluded_dates=excluded_dates)
    m_starts, m_means = _circular_window_means(profile, m_minutes, resolution_minutes)
    l_starts, l_means = _circular_window_means(profile, l_minutes, resolution_minutes)
    m_valid = np.isfinite(m_means)
    l_valid = np.isfinite(l_means)
    m_index = int(np.nanargmax(m_means)) if m_valid.any() else None
    l_index = int(np.nanargmin(l_means)) if l_valid.any() else None
    return {
        'M10_onset_time': _format_clock_minute(int(m_starts[m_index])) if m_index is not None else None,
        'M10_mean_activity': float(m_means[m_index]) if m_index is not None else np.nan,
        'L5_onset_time': _format_clock_minute(int(l_starts[l_index])) if l_index is not None else None,
        'L5_mean_activity': float(l_means[l_index]) if l_index is not None else np.nan,
        'm_window_hours': float(m_window_hours),
        'l_window_hours': float(l_window_hours),
        'mxlx_resolution_minutes': int(resolution_minutes),
        'mxlx_complete_profile': bool(np.isfinite(profile).all()),
    }
