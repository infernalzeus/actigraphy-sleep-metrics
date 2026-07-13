# sri.py — Sleep Regularity Index (SRI)
#
# Computes the Sleep Regularity Index directly from the epoch-level SVM series
# that the pipeline already reads (readParticipantCSV in io.py) — NO GGIR / R
# dependency is required.
#
# SRI measures the probability that any two epochs 24 hours apart share the same
# sleep/wake state, rescaled to a percentage:
#
#     SRI = 100  -> perfectly regular (state at time t always equals state at t+24h)
#     SRI = 0    -> no better than chance
#     SRI = -100 -> perfectly irregular (state always flips 24h later)
#
# Reference:
#   Phillips A.J.K. et al. (2017). Irregular sleep/wake patterns are associated
#   with poorer academic performance and delayed circadian and sleep/wake timing.
#   Scientific Reports, 7, 3216.
#   Formula as summarised in Fischer et al. (2021), PMC8503839:
#
#            -100 + 200/(M(N-1)) * sum_{j=1}^{M} sum_{i=1}^{N-1} delta(s_{i,j}, s_{i+1,j})
#
#   where  s_{i,j} in {0 (sleep), 1 (wake)} is the state at epoch j on day i,
#          M = epochs per day, N = number of days, and
#          delta(a, b) = 1 if a == b else 0.
#
# NOTE ON RELATION TO OTHER METRICS:
#   SRI is NOT a function of M10 / L5 and shares no inputs with them.
#   M10 / L5 describe the amplitude and timing of the *average* activity day
#   (continuous SVM, collapsed across days). SRI describes day-to-day
#   *reproducibility* of the *binary* sleep/wake state. SRI is the close cousin
#   of Interdaily Stability (IS) — both quantify regularity — but IS compares
#   each day to the grand-average profile using continuous activity, whereas SRI
#   compares each day to the *next* day using binary states.

import warnings
import numpy as np
import pandas as pd


# ── Sleep/wake classification ───────────────────────────────────────────────────

# Cole-Kripke (1992) automatic sleep/wake scoring for 1-minute epochs.
#   D_t = P * ( 106*A_{t-4} + 54*A_{t-3} + 58*A_{t-2} + 76*A_{t-1}
#             + 230*A_t + 74*A_{t+1} + 67*A_{t+2} )
#   epoch t is scored SLEEP if D_t < 1, otherwise WAKE.
# See: Cole R.J., Kripke D.F., Gruen W., Mullaney D.J., Gillin J.C. (1992).
#   Automatic sleep/wake identification from wrist activity. Sleep, 15(5), 461-469.
_CK_WEIGHTS   = np.array([106, 54, 58, 76, 230, 74, 67], dtype=float)  # A-4 .. A+2
_CK_OFFSETS   = np.array([-4, -3, -2, -1, 0, 1, 2])
_CK_SCALE     = 0.001   # the "P" scale factor in the original formula
_CK_THRESHOLD = 1.0     # D < threshold -> sleep


def cole_kripke(activity, scale=_CK_SCALE, threshold=_CK_THRESHOLD):
    # Classify each 1-minute epoch as sleep (0) or wake (1) using the
    # Cole-Kripke weighted moving window.
    #
    # Arguments:
    #   activity  — 1-D array-like of per-epoch activity (the SVM column).
    #               NaN entries (gaps / non-wear) are preserved as NaN in the
    #               output so they can be excluded from the SRI comparison.
    #   scale     — the P scale factor. The original constants were derived for
    #               zero-crossing / PIM activity counts; SVM_sum is on a
    #               different scale, so this may need calibration for a given
    #               device (see the CALIBRATION note in compute_SRI).
    #   threshold — decision boundary on the weighted sum D (default 1.0).
    #
    # Returns: float numpy array, 0.0 = sleep, 1.0 = wake, NaN where input NaN.
    a       = np.asarray(activity, dtype=float)
    nanmask = np.isnan(a)
    a0      = np.nan_to_num(a, nan=0.0)
    n       = a0.shape[0]

    D = np.zeros(n, dtype=float)
    for w, off in zip(_CK_WEIGHTS, _CK_OFFSETS):
        shifted = np.zeros(n, dtype=float)
        if off < 0:            # shifted[t] = a0[t + off]   (a past epoch)
            shifted[-off:] = a0[:n + off]
        elif off > 0:          # shifted[t] = a0[t + off]   (a future epoch)
            shifted[:n - off] = a0[off:]
        else:
            shifted = a0
        D += w * shifted
    D *= scale

    state = (D >= threshold).astype(float)   # 1 = wake, 0 = sleep
    state[nanmask] = np.nan
    return state


# ── Epoch-by-day alignment ──────────────────────────────────────────────────────

def _dayByEpochMatrix(states, index, epoch_sec):
    # Reshape a flat, regularly-spaced sleep/wake series into a
    # [days x epochs-per-day] matrix aligned by *clock time of day*, so that
    # column j is always the same time of day on every row.
    #
    # Rows are reindexed to a complete run of consecutive calendar dates, so that
    # adjacent rows are guaranteed to be consecutive days (a fully-missing day
    # becomes an all-NaN row and is naturally excluded from the SRI comparison).
    #
    # Arguments:
    #   states    — 1-D array of 0/1/NaN states, one per epoch
    #   index     — DatetimeIndex aligned to `states` (regular epoch grid)
    #   epoch_sec — epoch length in seconds
    #
    # Returns: (mat, dates)
    #   mat   — numpy array [D x M], M = 86400 / epoch_sec, NaN where no data
    #   dates — DatetimeIndex of the D row dates (consecutive calendar days)
    M           = 86400 // epoch_sec
    sec_of_day  = index.hour * 3600 + index.minute * 60 + index.second
    epoch_of_day = (sec_of_day // epoch_sec).astype(int)     # 0 .. M-1
    dates        = index.normalize()                         # midnight of each day

    dfm = pd.DataFrame({'date': dates, 'eod': epoch_of_day, 'state': states})
    mat_df = dfm.pivot_table(index='date', columns='eod',
                             values='state', aggfunc='first')

    # guarantee every epoch column exists, and every calendar day is a row
    full_dates = pd.date_range(mat_df.index.min(), mat_df.index.max(), freq='D')
    mat_df = mat_df.reindex(index=full_dates, columns=range(M))

    return mat_df.values, mat_df.index


# ── Public entry point ──────────────────────────────────────────────────────────

def compute_SRI(d, epoch_sec=60, scale=_CK_SCALE, threshold=_CK_THRESHOLD,
                min_pairs=1, normalize_per_second=True, verbose=False):
    # Compute the Sleep Regularity Index from a raw epoch DataFrame.
    #
    # This uses the SAME data the rest of the pipeline already collects — the
    # per-epoch SVM series read by readParticipantCSV — so no GGIR output is
    # needed. The steps are:
    #   1. Put SVM on a complete, regular epoch grid (gaps -> NaN).
    #   2. Score each epoch sleep(0)/wake(1) with Cole-Kripke.
    #   3. Align into a [days x epochs-per-day] matrix by clock time.
    #   4. Compare each day to the next (same clock epoch) and count matches:
    #          SRI = 200 * (matching epoch-pairs / valid epoch-pairs) - 100
    #      Epoch-pairs where either day is missing (NaN) are excluded, and the
    #      denominator is reduced accordingly.
    #
    # Arguments:
    #   d         — epoch DataFrame with columns 'Time' (datetime) and 'SVM'
    #               (output of readParticipantCSV; SVM_sum already renamed to SVM)
    #   epoch_sec — epoch length in seconds (default 60). Cole-Kripke's weights
    #               are defined for 1-minute epochs; a warning is issued if this
    #               is not 60.
    #   scale, threshold — Cole-Kripke parameters (see cole_kripke).
    #   min_pairs — minimum number of valid epoch-pairs required to return a
    #               value (otherwise SRI is NaN).
    #   verbose   — print a one-line summary.
    #
    # CALIBRATION: the Cole-Kripke constants were validated on activity counts
    # from older devices, not on Actigraph SVM_sum. The *shape* of the sleep/wake
    # series (and therefore SRI) is robust to the exact scale, but if the scored
    # %-sleep looks implausible for your device, tune `scale` (or `threshold`)
    # against a night of known sleep timing.
    #
    # Returns a dict:
    #   'SRI'                 — the index, in [-100, 100] (NaN if too little data)
    #   'n_days'              — number of calendar days spanned
    #   'n_valid_epoch_pairs' — epoch-pairs actually compared
    #   'pct_epochs_sleep'    — % of scored epochs classified as sleep
    #   'epoch_sec'           — epoch length used
    #   'classifier'          — 'Cole-Kripke'
    #   'ck_scale', 'ck_threshold' — parameters used
    #   'states_matrix'       — [days x epochs] binary matrix (for plotting)
    #   'dates'               — DatetimeIndex of the matrix rows
    if 'SVM' not in d.columns:
        raise ValueError("compute_SRI: input DataFrame has no 'SVM' column "
                         "(expected readParticipantCSV output).")
    if epoch_sec != 60:
        warnings.warn(f"compute_SRI: epoch_sec={epoch_sec}s but Cole-Kripke "
                      "weights are defined for 60-s epochs; results may be "
                      "unreliable.")

    # ── 1. regular epoch grid ──────────────────────────────────────────────────
    s = (d[['Time', 'SVM']]
         .dropna(subset=['Time'])
         .sort_values('Time')
         .set_index('Time')['SVM'])
    s = s[~s.index.duplicated(keep='first')]
    full_index = pd.date_range(s.index[0], s.index[-1], freq=f'{epoch_sec}s')
    s = s.reindex(full_index)     # missing epochs -> NaN

    # ── 2. sleep/wake scoring ──────────────────────────────────────────────────
    # SVM_sum is a SUM over every sample in the epoch, so its magnitude scales
    # with epoch length and sits far above the per-minute activity-count range
    # Cole-Kripke's constants were tuned for (where sleep sits near 0). Left raw,
    # every epoch scores WAKE. Normalise to a per-second rate (SVM_sum / epoch_sec)
    # so the default scale/threshold separate sleep from wake for GENEActiv SVM_sum.
    activity = (s.values / float(epoch_sec)) if normalize_per_second else s.values
    states = cole_kripke(activity, scale=scale, threshold=threshold)

    # ── 3. align to [days x epochs] ────────────────────────────────────────────
    mat, dates = _dayByEpochMatrix(states, s.index, epoch_sec)

    # ── 4. consecutive-day comparison ──────────────────────────────────────────
    today, tomorrow = mat[:-1], mat[1:]
    valid   = ~np.isnan(today) & ~np.isnan(tomorrow)
    matches = valid & (today == tomorrow)
    n_pairs = int(valid.sum())

    if n_pairs < min_pairs:
        sri = np.nan
    else:
        sri = 200.0 * matches.sum() / n_pairs - 100.0

    scored = states[~np.isnan(states)]
    pct_sleep = float((scored == 0).mean() * 100.0) if scored.size else np.nan

    result = {
        'SRI':                 sri,
        'n_days':              int(mat.shape[0]),
        'n_valid_epoch_pairs': n_pairs,
        'pct_epochs_sleep':    pct_sleep,
        'epoch_sec':           epoch_sec,
        'classifier':          'Cole-Kripke',
        'ck_scale':            scale,
        'ck_threshold':        threshold,
        'activity_normalization': ('SVM_sum/epoch_sec (per-second)'
                                   if normalize_per_second else 'raw SVM_sum'),
        'states_matrix':       mat,
        'dates':               dates,
    }

    if verbose:
        sri_str = f"{sri:.2f}" if not np.isnan(sri) else "N/A"
        print(f"      SRI  = {sri_str}  "
              f"({mat.shape[0]} days, {n_pairs:,} epoch-pairs, "
              f"{pct_sleep:.1f}% epochs scored sleep)")

    return result
