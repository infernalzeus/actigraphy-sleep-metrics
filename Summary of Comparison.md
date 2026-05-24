Summary of Comparison
I cross-checked the Python files against the corresponding R files (periodogram-functions.R, non-parametric-measures.R, aggregating-functions.R, visualisation-functions.R, pre-process.R, etc.).
1. periodogram.py — Excellent Match

computeMean_X_h_p, computeEnright_Ap, periodogram_Enright: Direct, line-by-line translation of Enright periodogram logic from Sokolove & Bushell (1978). Handles NA removal and indexing identically.
computeQ_P, computeQ_P_Greedy, periodogram_ChiSquare, add_pValue_ChiSquare: Faithful to Tackenberg & Hughey (2021) implementations, including the "greedy" padding logic for rectangular matrices and chi-square p-value calculation via scipy.stats.chi2.logsf.
Minor differences are only in language idioms (e.g., np.nansum, vectorized operations, pandas DataFrames) — behavior is equivalent.

2. nonparametric.py — Very Good Match

interdailyStability: Matches R interdailyStability (Witting et al., 1990), including padding and NA handling.
intradailyVariability: Implements LOCF_NA (Last Observation Carried Forward) exactly as in R, then computes the IV statistic.
compute_M10 / compute_L5: Uses the bidirectional rolling mean filter (filterHour_ForwardBack) as described in the R code and Van Someren et al. (1999). Matches the R helper functions.
LOCF_NA: Python version correctly emulates the R cumulative sum / indexing approach.

3. preprocess.py — Strong Match

timeBin, aggregateHours: Exact logic for hourly binning of SVM (sum vector magnitude) activity.
dayByHourMatrix: Pivots to [days × hours] matrix as in R.
rollingWindowInd and nonwear_detect: Implements the 60-min window / 15-min step non-wear detection (van Hees et al. method) with identical thresholds and logic.

4. visualize.py — Good Match

activityHeatmap: Uses np.roll to center midnight (shift by 12 hours), then seaborn heatmap. This replicates the R activityHeatmap (which uses cbind + column subsetting for the same effect).

5. io.py — Functional Equivalent

Reading CSVs and downsampling (ds_accel_csv) mirrors the R preprocessing pipeline.

6. __init__.py — Clean re-export of the API.
Minor Notes / Improvements (not bugs):

Python is more robust with NaNs thanks to np.nan* functions.
Performance is likely better due to vectorization.
Edge cases (all-NaN, very short recordings) should behave the same.
No breaking differences found in the mathematical core.

sri.py (Sleep Regularity Index)
This is currently just a stub/wrapper:

GGIR_from_csv and SRI_from_GGIR raise warnings and pass.
This matches the comment in the file: it expects an R installation + run_ggir.R script (or rpy2).

The referenced GitHub repo (dpwindred/sleepreg) is a full R package that integrates GGIR for sleep-wake detection, then computes SRI (Phillips et al., 2017), handling naps, WASO, miscalculated nights, non-wear, etc.
Your sri.py is a reasonable placeholder, but full implementation would require:

Calling GGIR (via subprocess/rpy2), or
Reimplementing the SRI logic from sleepreg in Python (non-trivial, as it depends heavily on GGIR outputs).

Recent papers note that different SRI calculators (e.g., sleepreg vs. GGIR built-in) can produce meaningfully different scores, so consistency with the original package matters