# Step 2 — Technical Reference

## Overview

Step 2 reads an Actigraph CSV file of 60-second activity epochs and computes
validated sleep-wake regularity metrics from it.  It is a Python translation of
the R repository `sleep-wake-code-main` and produces identical numeric results.

The two metric groups are:

| Group | Metrics | Source |
|---|---|---|
| `nonparametric` | IS, IV, M10, L5 | Witting et al. 1990; Van Someren et al. 1999 |
| `periodogram` | Enright A_p, Chi-square Q_p | Sokolove & Bushell 1978; Tackenberg & Hughey 2021 |

---

## Program Architecture

```
step2/
  cli.py            Entry point.  Parses arguments, drives the pipeline,
                    writes outputs.  Run as: python -m cli read <file.csv>

  io.py             readParticipantCSV()
                    Reads the Actigraph CSV, fixes the timestamp format,
                    renames SVM_sum -> SVM.

  preprocess.py     timeBin()         epoch interval check
                    aggregateHours()  60-s epochs -> hourly means
                    dayByHourMatrix() hourly long-format -> [days x 24] matrix

  nonparametric.py  interdailyStability()   IS
                    intradailyVariability() IV
                    compute_M10()           M10 onset and mean activity
                    compute_L5()            L5 onset and mean activity

  periodogram.py    periodogram_Enright()    Enright A_p table
                    periodogram_ChiSquare()  Chi-square Q_p table
                    add_pValue_ChiSquare()   appends log p-values

  visualize.py      activityHeatmap()        seaborn heatmap figure
                    generate_pdf_report()    multi-page PDF

  outputs/          <stem>_metrics.csv
                    <stem>_periodogram.csv
                    <stem>_report.pdf
```

---

## Input Data

### File format

Actigraph ActiLife export, 60-second epoch, CSV.

### Columns used by Step 2

| Column | Type | Used for |
|---|---|---|
| `Time` | timestamp string `YYYY-MM-DD HH:MM:SS:mmm` | epoch alignment, hourly aggregation, day labelling |
| `SVM_sum` | float (counts/min) | all activity metrics — IS, IV, M10, L5, periodograms |

All other columns (`Ax_mean`, `Ay_mean`, `Az_mean`, `Lux_mean`, `Button_sum`,
`Temperature_mean`, `Ax_sd`, `Ay_sd`, `Az_sd`, `Lux_peak`) are read but
not used by the Step 2 metric calculations.

### SVM — Signal Vector Magnitude

SVM (Signal Vector Magnitude) is the Euclidean norm of the 3-axis acceleration
vector, summed over the 60-second epoch:

```
SVM_sum = sum_t( sqrt(Ax(t)^2 + Ay(t)^2 + Az(t)^2) )
```

It is a single non-directional measure of movement intensity per minute and is
the standard activity proxy used in non-parametric actigraphy analysis.

### Timestamp quirk

The Actigraph export uses a colon as the millisecond separator:
`2025-08-09 12:00:00:000` rather than the ISO standard `2025-08-09 12:00:00.000`.
`io.py` corrects this before parsing.

---

## Data Pipeline

```
Raw CSV  (27 373 rows x 12 cols, 60-s epochs)
    |
    | readParticipantCSV()
    |   fix timestamp format, rename SVM_sum -> SVM
    v
Epoch DataFrame  (rows = epochs, cols include Time, SVM)
    |
    | aggregateHours()
    |   floor Time to hour boundary
    |   group by hour, take mean(SVM)
    v
Hourly DataFrame  (~456 rows x 2 cols: Time, Activity)
    |
    | dayByHourMatrix()
    |   pivot: rows = calendar dates, cols = hours 0..23
    |   missing hours (partial days) -> NaN
    v
Activity Matrix  [20 days x 24 hours]  numpy float64
    |
    +-- interdailyStability()  -> IS  (scalar)
    +-- intradailyVariability() -> IV  (scalar)
    +-- compute_M10()           -> M10 hour, M10 mean activity
    +-- compute_L5()            -> L5 hour,  L5 mean activity
    +-- periodogram_Enright()   -> DataFrame [21 x 2]: Period, Enright_Value
    +-- periodogram_ChiSquare() -> DataFrame [21 x 3]: Period, ChiSq_Qp, log_Pvalue
    |
    v
outputs/<stem>_metrics.csv
outputs/<stem>_periodogram.csv
outputs/<stem>_report.pdf
```

---

## Mathematical Operations

### Step 1 — Hourly aggregation

Every row in the raw CSV is one 60-second epoch.  To produce the activity matrix
required by the metric functions, these are collapsed to one value per hour:

```
Activity(date d, hour h) = mean( SVM_sum[i] : floor(Time[i]) == (d, h) )
```

If a recording starts at 12:14, epoch 12:14–12:15 (and all others in that hour)
are averaged into a single `Activity(day_1, hour_12)` value.  Hours with no
data receive `NaN`.

---

### Step 2 — Day x Hour matrix

The hourly series is pivoted into a rectangular matrix:

```
M[d, h]  =  mean SVM for calendar day d, hour h
           (NaN where the device was not recording)
```

Shape: `(D, 24)` where D = number of distinct calendar dates in the recording.

---

### Metric 1 — Interdaily Stability (IS)

IS measures how consistent the 24-hour rest-activity pattern is across days.

**Inputs from matrix M:**
- All values of M, flattened row-by-row to vector X_v of length N
- Column means X_h = mean(M[:, h]) across all days, for h = 0..23

**Formula:**

```
         N * sum_{h=0}^{23}( (X_h - X_bar)^2 )
IS =  ────────────────────────────────────────────
         24 * sum_{i=0}^{N-1}( (X_i - X_bar)^2 )
```

- N      = number of non-NaN hourly values
- X_h    = mean activity at hour h averaged across all days  (column mean)
- X_bar  = grand mean of all valid hourly values
- 24     = period length P (hours)

**Interpretation:**
- Numerator measures between-hour variance of the mean daily profile
- Denominator measures total variance across all individual hourly observations
- IS = 0: no day-to-day pattern (pure noise)
- IS = 1: identical 24-hour pattern every day

---

### Metric 2 — Intradaily Variability (IV)

IV measures within-day fragmentation of the activity rhythm.

**Inputs from matrix M:**
- M flattened row-by-row to X_v
- Internal NaN gaps filled by LOCF (last observation carried forward)
  before computing the first difference
- Leading/trailing NaNs (partial first/last days) are removed

**LOCF rule:**
If `X_v[i]` is NaN and `X_v[i-1]` is valid, set `X_v[i] = X_v[i-1]`.
This assumes no change between adjacent hours during a data gap
(e.g. missing mid-day hour).

**Formula:**

```
         N * sum_{i=0}^{N-2}( (X_{i+1} - X_i)^2 )
IV =  ──────────────────────────────────────────────
         (N-1) * sum_{i=0}^{N-1}( (X_i - X_bar)^2 )
```

- N       = number of valid hourly values after LOCF and NaN removal
- X_{i+1} - X_i = first difference (numerator = sum of squared differences)
- X_bar   = mean of the cleaned series

**Interpretation:**
- IV ~ 0: smooth, sinusoidal rhythm
- IV ~ 2: maximally fragmented (pure random noise)

---

### Metric 3 — M10 (Most Active 10-Hour Window)

M10 identifies the 10-hour window of peak activity, averaged across all days.

**Inputs from matrix M:**
- Column means: X_h = mean(M[:, h])  for h = 0..23
  (mean activity at each hour of the day, collapsed across all recording days)

**Algorithm:**

1. Apply a **bidirectional circular rolling mean** of width 10 to X_h:

   ```
   forward  f2[h] = mean( X_h[h], X_h[h+1], ..., X_h[h+9] )   (circular)
   backward f1[h] = mean( X_h[h-9], ..., X_h[h-1], X_h[h] )   (circular)
   combined  f[h] = 0.5 * (f1[h] + f2[h])
   ```

   Bidirectionality removes the phase bias that a one-sided filter would
   introduce — the identified window is centred on the reported hour.
   Circular indexing means the window wraps across midnight (23:00 -> 00:00).

2. M10 onset = the hour h with the **highest** combined rolling mean f[h]
3. M10 mean  = that highest value

**Outputs:**
- `M10_hour` — clock hour (0–23) at which the most-active window begins
- `M10_mean_activity` — mean SVM within that 10-hour window, across all days

---

### Metric 4 — L5 (Least Active 5-Hour Window)

L5 identifies the 5-hour window of minimum activity — the rest/sleep window.

**Inputs and algorithm:** identical to M10 but with window width 5 and taking
the **lowest** rather than highest rolling mean.

**Outputs:**
- `L5_hour` — clock hour at which the least-active (rest) window begins
- `L5_mean_activity` — mean SVM within that 5-hour window, across all days

**Relative amplitude (RA)** — not computed here but commonly derived as:
```
RA = (M10 - L5) / (M10 + L5)
```

---

### Metric 5 — Enright Periodogram

The Enright periodogram tests for the strength of rhythms at a range of
candidate periods (default 14–34 hours, covering the circadian window).

**Inputs from matrix M:**
- M[d, h] arranged row-by-row into vector X_v of length N = D * 24
- Candidate period p tested one at a time

**Per-period computation:**

For each candidate period p:

1. Compute phase-bin means.  For phase bin h (h = 0..23):

   ```
   X_bar_{h,p} = mean( X_v[h], X_v[h+p], X_v[h+2p], ... )
   ```

   This selects all observations at the same relative position within each
   hypothesised p-hour cycle.  NaN values (partial days) are excluded and
   the count is adjusted accordingly.

2. Compute the mean of phase-bin means:

   ```
   X_bar_p = (1/p) * sum_{h=0}^{23}( X_bar_{h,p} )
   ```

3. Compute the root-mean-square amplitude A_p (Eq. 2, Sokolove & Bushell 1978):

   ```
                _______________________________________________
               /    1    23
   A_p  =    \/   ───  * sum_{h=0}( (X_bar_{h,p} - X_bar_p)^2 )
                  24   h=0
   ```

**Output:** DataFrame with columns `Period` and `Enright_Value`.

**Interpretation:** the period with the highest A_p is the dominant rhythm
period.  For a healthy 24-hour rest-activity cycle, the peak should fall at
p = 24.

---

### Metric 6 — Chi-Square Periodogram

Also tested over periods 14–34 hours.  Provides a formal statistical test for
each candidate period.

**Per-period computation (conservative rule):**

For each candidate period P:

1. Take the first `K * P` elements of X_v where `K = floor(N / P)`.
   This uses only complete cycles (remaining elements are dropped).

2. Reshape into a K-row x P-column matrix.

3. Compute the Q_p statistic:

   ```
          K * N_incl * sum_{h=0}^{P-1}( (X_h - X_bar)^2 )
   Q_p = ─────────────────────────────────────────────────────
                   sum_{i}( (X_i - X_bar)^2 )
   ```

   - K        = number of complete cycles
   - N_incl   = number of non-NaN values in the K*P block
   - X_h      = column mean for phase bin h
   - X_bar    = grand mean of the K*P block
   - X_i      = individual values in the K*P block

4. Under H0 (no rhythm), Q_p ~ chi-squared with (P-1) degrees of freedom.
   The log p-value is: `log P( chi^2_{P-1} >= Q_p )`

**Greedy variant:** pads X_v with NaN to fill a complete ceil(K) x P matrix,
using all data with fractional K.  This eliminates the discontinuity that
occurs when N is exactly divisible by P.

**Output:** DataFrame with columns `Period`, `ChiSq_Qp`, `log_Pvalue`.

**Caution:** with fewer than ~10 complete days the chi-square approximation
becomes unreliable.

---

## Output Files

### `<stem>_metrics.csv`

One row per input file.  Columns:

| Column | Description |
|---|---|
| `fname` | source filename |
| `processing_timestamp` | ISO-8601 timestamp of when the run completed |
| `IS` | inter-daily stability (0–1) |
| `IV` | intra-daily variability (0–2) |
| `M10_hour` | hour of day (0–23) at which the most-active 10-h window starts |
| `M10_mean_activity` | mean SVM within that window |
| `L5_hour` | hour of day (0–23) at which the least-active 5-h window starts |
| `L5_mean_activity` | mean SVM within that window |

### `<stem>_periodogram.csv`

One row per tested period.  Columns:

| Column | Description |
|---|---|
| `Period` | candidate period in hours (14–34) |
| `Enright_Value` | Enright amplitude A_p |
| `ChiSq_Qp` | chi-square statistic Q_p |
| `log_Pvalue` | log p-value under chi-square null hypothesis |

### `<stem>_report.pdf`

Multi-page PDF:

| Page | Content |
|---|---|
| 1 | Activity heatmap — M[days x 24 h], midnight-centred |
| 2 | Non-parametric metrics table (IS, IV, M10, L5) |
| 3 | Enright periodogram line plot |
| 4 | Chi-square periodogram — Q_p and log p-value |
