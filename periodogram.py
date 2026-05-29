# Implementations of different periodogram functions.
# See Sokolove and Bushell (1978) for the implementation of Enright (1965).
#
# NOTE: these functions are coded for clarity so they transparently match
# the computations in the Sokolove & Bushell paper.  They could be written
# more concisely with vectorisation, but the explicit form makes it easier
# to verify correctness against the published equations.
# Python translation of periodogram-functions.R

import numpy as np
import pandas as pd
from scipy.stats import chi2


# ── Enright Periodogram ────────────────────────────────────────────────────────
# Reference:
#   Sokolove, P.G., & Bushell, W.N. (1978). The chi square periodogram: its
#     utility for analysis of circadian rhythms.
#     Journal of Theoretical Biology, 72(1), 131-160.


def computeMean_X_h_p(X, h, p, K, na_rm=True):
    # Computes the mean activity for phase bin h within a hypothesised period p.
    # Implements Eq. 1 of Sokolove & Bushell (1978).
    #
    # The vectorised activity series X is indexed at positions h, h+p, h+2p, ...
    # This selects all observations that fall at the same relative phase within
    # each hypothesised p-hour cycle.  For example, with p=24 and h=0 this
    # selects midnight of every day; with h=12 it selects noon of every day.
    #
    # Arguments:
    #   X    — vectorised [day x hour] activity matrix  (row-major order)
    #   h    — phase bin index  (0 to P-1)
    #   p    — hypothesised period in hours  (e.g. 24)
    #   K    — number of complete cycles to use  (= number of days)
    #   na_rm — if True, skip NaN values and adjust the count accordingly
    #           (handles partial first/last days where some hours are missing)
    #
    # Returns: scalar mean of X at indices  h, h+p, h+2p, ..., h+(K-1)*p

    j       = np.arange(K)
    indices = h + j * p
    indices = indices[indices < len(X)]
    X_h_p   = X[indices]

    if na_rm:
        X_h_p = X_h_p[~np.isnan(X_h_p)]

    this_K = len(X_h_p)
    if this_K == 0:
        return np.nan

    # (1/K) * sum  — Eq. 1 of Sokolove & Bushell (1978)
    # K is adjusted to the number of non-NaN values when na_rm=True
    return (1 / this_K) * np.nansum(X_h_p)


def computeEnright_Ap(X, p, adjust_for_NA=True):
    # Computes the Enright (1965) root-mean-square amplitude A_p for a
    # single hypothesised period p.
    # Implements Eq. 2 of Sokolove & Bushell (1978), pp. 137.
    #
    # Input column (raw CSV)  : SVM_sum  — signal vector magnitude sum per 60-s epoch
    # Column after io.py      : SVM      (SVM_sum renamed)
    # Column after aggregation: Activity — mean(SVM) per calendar hour
    # Matrix input            : X[d, h] = Activity for day d, hour h
    #                           shape (D, 24), NaN for hours with no recording
    #         p               — hypothesised period to test (integer hours, e.g. 14..34)
    #
    # A_p is a measure of how much the average activity profile varies across
    # the p phase bins.  A large A_p indicates a strong rhythm at period p.
    #
    # Formula:
    #
    #               ___________________________________________
    #              /    1    P
    #   A_p  =   \/   ───  * sum_h( (X_bar_h_p - X_bar_p)^2 )
    #                  P   h=1
    #
    # where:
    #   P          = number of hours per day (24) — the observation window
    #   X_bar_h_p  = mean activity for phase bin h at period p
    #                (from computeMean_X_h_p, handling NaN in partial days)
    #   X_bar_p    = (1/p) * sum_h(X_bar_h_p)
    #                the mean of phase-bin means at period p
    #
    # The period with the highest A_p is the dominant rhythm period.

    P = X.shape[1]   # hours per day (24)
    K = X.shape[0]   # number of days
    X_v = X.flatten()

    # compute phase-bin means for all h in [0, P-1]
    X_bar_h_p = np.array([
        computeMean_X_h_p(X_v, h, p, K, adjust_for_NA)
        for h in range(P)
    ])

    # mean of phase-bin means (Eq. 2 denominator normaliser)
    X_bar_p = (1 / p) * np.nansum(X_bar_h_p)

    # root mean square amplitude
    A_p = np.sqrt((1 / P) * np.nansum((X_bar_h_p - X_bar_p) ** 2))
    return A_p


def periodogram_Enright(X, periods=np.arange(14, 35), adjust_for_NA=True):
    # Computes a complete Enright periodogram over a range of candidate periods.
    # Returns a DataFrame with columns [Period, Value] where Value = A_p.
    #
    # Default period range 14–34 hours covers the full circadian-relevant window.
    # NaNs are returned for periods where the data contains too many missing values.

    pgram = pd.DataFrame({'Period': periods, 'Value': np.nan})
    for i, p in enumerate(periods):
        pgram.loc[i, 'Value'] = computeEnright_Ap(X, p=p,
                                                   adjust_for_NA=adjust_for_NA)
    return pgram


# ── Chi-Square Periodogram ─────────────────────────────────────────────────────
# References:
#   Tackenberg, M.C., & Hughey, J.J. (2021). The risks of using the
#     chi-square periodogram to estimate the period of biological rhythms.
#     PLoS Computational Biology, 17(1), e1008567.


def computeQ_P(X_v, N, P, K):
    # Computes the chi-square statistic Q_p for a single hypothesised period P.
    # Follows the conventions of Tackenberg & Hughey (2021).
    #
    # NOTE: variable names here (N, K, P) follow Tackenberg & Hughey, not
    # Sokolove & Bushell — they differ between the two papers.
    #
    # Input column (raw CSV)  : SVM_sum  — signal vector magnitude sum per 60-s epoch
    # Column after aggregation: Activity — mean(SVM) per calendar hour
    # Matrix origin           : X[d, h] = Activity, then flattened row-by-row to X_v
    #         N               — total length of X_v
    #         P               — hypothesised period in hours
    #         K               — floor(N/P)  — number of COMPLETE cycles that fit (conservative)
    #
    # The first K*P elements of X_v are arranged into a K-row x P-column matrix.
    # Any remaining elements (N mod P) are DROPPED (conservative rule).
    #
    # Formula:
    #
    #          K * N_incl * sum_h( (X_h - X_bar)^2 )
    #   Q_p = ──────────────────────────────────────────
    #                sum_i( (X_i - X_bar)^2 )
    #
    # where:
    #   K       = number of complete cycles  = floor(N/P)
    #   N_incl  = number of non-NaN values in the first K*P elements
    #   X_h     = column mean for phase bin h in the K x P matrix
    #   X_bar   = grand mean of the K*P included elements
    #   X_i     = individual values in the K*P block
    #
    # Q_p approximately follows a chi-square distribution with (P-1) degrees
    # of freedom under H0: no rhythm at period P.
    # A large Q_p (small p-value) indicates a significant rhythm at period P.

    X_included  = X_v[:int(K * P)]
    X_m         = X_included.reshape((int(K), P))
    X_bar       = np.nanmean(X_included)
    N_included  = len(X_included) - np.isnan(X_included).sum()

    X_h = np.nanmean(X_m, axis=0)   # column means

    numer = K * N_included * np.nansum((X_h - X_bar) ** 2)
    denom = np.nansum((X_included - X_bar) ** 2)

    return numer / denom if denom != 0 else np.nan


def computeQ_P_Greedy(X_v, N, P, K):
    # Greedy variant of computeQ_P.
    # Follows the GREEDY algorithm of Tackenberg & Hughey (2021).
    #
    # Input column (raw CSV)  : SVM_sum  — signal vector magnitude sum per 60-s epoch
    # Column after aggregation: Activity — mean(SVM) per calendar hour
    # Matrix origin           : X[d, h] = Activity, then flattened row-by-row to X_v
    #
    # Unlike the conservative rule (which drops the last N mod P data points),
    # the greedy algorithm pads the series with NaN to form a complete
    # ceil(K) x P matrix, using ALL available data.
    # This removes the discontinuity in Q_p that can occur at period lengths
    # that happen to evenly divide N.
    #
    # The formula is identical to computeQ_P except:
    #   K_incl  = N_incl / P  (fractional — accounts for the padded NaNs)
    #   The denominator sums over all cells of the padded K x P matrix.

    # pad to a complete rectangular matrix of ceil(K) rows
    pad_len = int(np.ceil(K) * P - N)
    if pad_len > 0:
        X_v = np.pad(X_v, (0, pad_len), constant_values=np.nan)

    X_m   = X_v.reshape((-1, P))
    X_bar = np.nanmean(X_m)
    X_h   = np.nanmean(X_m, axis=0)

    N_included = N - np.isnan(X_v[:N]).sum()
    K_included = N_included / P   # fractional K for the greedy weighting

    numer = K_included * N_included * np.nansum((X_h - X_bar) ** 2)
    denom = np.nansum((X_m - X_bar) ** 2)

    return numer / denom if denom != 0 else np.nan


def periodogram_ChiSquare(X, periods=np.arange(14, 35), greedy=False):
    # Computes a complete chi-square periodogram over a range of candidate periods.
    # Returns a DataFrame with columns [Period, Value] where Value = Q_p.
    #
    # Arguments:
    #   X       — numpy array [days x 24 hours] of mean hourly SVM activity
    #   periods — candidate periods to test (default 14–34 hours)
    #   greedy  — if True, use the greedy algorithm (no data truncation);
    #             if False (default), use the conservative floor(N/P) rule

    pg  = pd.DataFrame({'Period': periods, 'Value': np.nan})
    X_v = X.flatten()
    N   = len(X_v)

    for i, p in enumerate(periods):
        if greedy:
            this_K = N / p
            pg.loc[i, 'Value'] = computeQ_P_Greedy(X_v, N, p, this_K)
        else:
            this_K = int(np.floor(N / p))   # conservative: complete cycles only
            pg.loc[i, 'Value'] = computeQ_P(X_v, N, p, this_K)

    return pg


def add_pValue_ChiSquare(pg):
    # Appends a log p-value column to the chi-square periodogram.
    #
    # Under the null hypothesis of no rhythm, Q_p approximately follows a
    # chi-square distribution with (P-1) degrees of freedom.
    # The log p-value is computed as  log P( chi^2_{P-1} >= Q_p ).
    #
    # WARNING: with fewer than ~10 days of data the chi-square approximation
    # becomes unreliable — interpret p-values cautiously for short recordings.
    #
    # Uses scipy.stats.chi2.logsf (log survival function) for numerical
    # precision when p-values are very small.

    pg = pg.copy()
    pg['log_Pvalue'] = chi2.logsf(pg['Value'], df=pg['Period'] - 1)
    return pg
