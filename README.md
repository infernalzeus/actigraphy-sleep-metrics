# actigraphy-sleep-metrics — Step 2

Python pipeline that converts epoched actigraphy CSV output into
circadian rest-activity metrics.  Python translation of the R repository
`sleep-wake-code-main`.

---

## Table of contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Input format](#input-format)
- [Usage](#usage)
- [Measure groups](#measure-groups)
- [Outputs](#outputs)
- [Module map](#module-map)
- [Mathematical background](#mathematical-background)

---

## Requirements

- Python 3.10 or later
- pip packages listed in `requirements.txt`

---

## Installation

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda install numpy pandas matplotlib scipy seaborn
```

---

## Input format

A Step 1 actigraphy output at **60-second epochs** in CSV format. The file must
contain at minimum:

| Column | Description |
|---|---|
| `Time` | Timestamp — format `YYYY-MM-DD HH:MM:SS:mmm` |
| `SVM_sum` | Signal vector magnitude sum over the 60-s epoch (not zero-crossing counts) |

---

## Usage

Run from inside the `step2` directory:

```bash
python -m cli read <path/to/input_file.csv> [--measures GROUP] [--output-dir DIR] [--verbose]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `input_file` | yes | Path to the Actigraph CSV file |
| `--measures GROUP` | no | `nonparametric`, `periodogram`, `temp_metrics`, or `sri`. Omit to run all four. |
| `--output-dir DIR` | no | Destination for outputs; defaults to the repository `outputs/` directory. |
| `--timezone ZONE` | no | IANA timezone override; otherwise read paired Step 1 metadata. |
| `--m-window-hours` | no | M window in hours; standard default 10. |
| `--l-window-hours` | no | L window in hours; standard default 5. |
| `--mxlx-resolution-minutes` | no | Window start search resolution; standard default 1 minute. |
| `--verbose` | no | Print step-by-step progress and computed values |

### Examples

Run all four measure groups (default):

```bash
python -m cli read "CD000_left wrist_60s.csv"
```

Run only non-parametric aggregate metrics with verbose output:

```bash
python -m cli read "CD000_left wrist_60s.csv" --measures nonparametric --verbose
```

Run only periodograms:

```bash
python -m cli read "CD000_left wrist_60s.csv" --measures periodogram
```

Run only per-day M10 / L5:

```bash
python -m cli read "CD000_left wrist_60s.csv" --measures temp_metrics --verbose
```

Run only the Sleep Regularity Index:

```bash
python -m cli read "CD000_left wrist_60s.csv" --measures sri --verbose
```

---

## Measure groups

### `nonparametric`

IS and IV are computed from the hourly series. M10/L5 use the standard
minute-level average 24-hour profile and exact circular 600/300-minute windows.
Missing observations are not filled. Partial outer days are allowed, but an
internal gap makes IV unavailable rather than joining observations across it.

| Metric | Range | Interpretation |
|---|---|---|
| IS — Interdaily Stability | 0 – 1 | Consistency of the 24-h pattern across days. 0 = no pattern, 1 = identical every day |
| IV — Intradaily Variability | 0 – 2 | Within-day fragmentation. 0 = smooth sinusoidal rhythm, ~2 = random noise |
| M10 onset time | HH:MM | Start of the most-active 10-h window (averaged across all days) |
| M10 mean activity | input activity units | Mean SVM within that most-active window |
| L5 onset time | HH:MM | Start of the least-active 5-h rest window (averaged across all days) |
| L5 mean activity | input activity units | Mean SVM within that rest window |

### `temp_metrics`

Per-day M10 and L5, computed independently for each calendar day using a
**noon-to-noon window** (each "day" runs from 12:00 on date *d* to 11:00 on
date *d+1*, so the sleep period falls in the centre of the window).

| Metric | Description |
|---|---|
| M10 onset | Clock hour with the highest 10-h rolling mean activity on that day |
| M10 mean activity | Filtered rolling mean at the M10 onset hour |
| M10 actual activity | Raw hourly mean at the M10 onset hour |
| L5 onset | Clock hour with the lowest 5-h rolling mean activity on that day |
| L5 mean activity | Filtered rolling mean at the L5 onset hour |
| L5 actual activity | Raw hourly mean at the L5 onset hour |

Edge days (the first and last day of the recording) may have partial data.
Missing hours are filled using Last Observation Carried Forward (LOCF) before
filtering so that the rolling mean is always defined.  Days with no recorded
data at all are reported as `—` / `None`.

### `sri`

Sleep Regularity Index — the probability that two epochs 24 h apart share the
same sleep/wake state, rescaled to a percentage. Computed from optional **GGIR zero-crossing counts**, or the original
`SVM_sum` input when counts are absent (no R dependency): each 60-s epoch is
scored sleep/wake with the Cole-Kripke algorithm, aligned into a
`days × epochs-per-day` matrix by clock time, and each day is compared to the
next.

| Metric | Range | Interpretation |
|---|---|---|
| SRI | −100 – 100 | +100 = identical sleep/wake pattern every day; 0 = chance; −100 = state flips exactly 24 h later |
| pct_epochs_sleep | 0 – 100 | Share of scored epochs classified as sleep — a sanity check on the classifier |

SRI is **independent of M10 / L5** (those describe the average activity day;
SRI describes day-to-day reproducibility of the binary sleep/wake state). It is
the close cousin of Interdaily Stability (IS), but compares each day to the
*next* day rather than to the grand-average profile.

> **Input-scale note.** Legacy `SVM_sum/60` is retained for backwards compatibility,
> but is not the zero-crossing input used by GGIR CK. Sleep labels and SRI are
> sensitive to input scale. The optional counts path below changes the input
> definition without fitting weights or thresholds to the diary.

### `periodogram`

Rhythm-period analysis over candidate periods 14 – 34 hours.

| Output | Description |
|---|---|
| Enright A_p | Root-mean-square amplitude at each candidate period. Peak = dominant rhythm period |
| Chi-square Q_p | Chi-square statistic at each candidate period |
| log p-value | Log probability under H₀ of no rhythm at that period |

---

## Outputs

By default outputs are written to `step2/outputs/`; `--output-dir` can direct a
run elsewhere without changing the old command. Files are named after the input
stem.

| File | Produced by | Contents |
|---|---|---|
| `<stem>_nonparametric.csv` | `nonparametric` | IS, IV, aggregate M10 / L5 — one row per file |
| `<stem>_daily.csv` | `temp_metrics` | Per-day M10 / L5 — one row per recording day |
| `<stem>_periodogram.csv` | `periodogram` | Enright A_p and chi-square Q_p for periods 14 – 34 h |
| `<stem>_sri.csv` | `sri` | Sleep Regularity Index + scoring summary — one row per file |
| `<stem>_report.pdf` | all groups | Multi-page PDF — see below |

### PDF report page order

| Page | Content | Requires |
|---|---|---|
| 1 | Activity heatmap (noon left, midnight centre) | always |
| 2 | Scalar metrics summary table | `nonparametric` |
| next | SRI summary table + sleep/wake raster | `sri` |
| next+ | Per-day M10 / L5 table (paginated, 30 rows/page) | `temp_metrics` |
| next | Enright periodogram plot | `periodogram` |
| last | Chi-square periodogram + log p-value plot | `periodogram` |

Pages for measure groups that were not run are omitted automatically.

---

## Module map

```
step2/
├── cli.py              Entry point — argument parsing, pipeline orchestration
├── io.py               CSV reader — timestamp normalisation, SVM_sum -> SVM rename
├── preprocess.py       Hourly aggregation, day-by-hour matrix construction
├── nonparametric.py    IS, IV, canonical minute-level aggregate M10/L5
├── temp_metrics.py     Per-day oneDay_M10, oneDay_L5, compute_daily_metrics
├── periodogram.py      Enright and chi-square periodograms
├── visualize.py        Activity heatmap, sleep/wake raster, PDF report generator
├── sri.py              Sleep Regularity Index — Cole-Kripke scoring, compute_SRI
├── requirements.txt    pip dependencies
└── outputs/            All generated CSV and PDF files (created on first run)
```

### Column pipeline

```
Raw CSV          io.py           preprocess.py            matrix
SVM_sum   -->   SVM       -->   Activity (mean/hour)  -->  mat[day, hour]
(60-s epoch)  (renamed)        (mean SVM per         (shape D×24,
                                calendar hour)         NaN = no recording)
```

---

## Mathematical background

See `Summary of Metrics.md` for the complete mathematical derivation of every
formula, which input columns they use, and what each output column means.

---

## References

- Witting et al. (1990). *Biological Psychiatry*, 27(6), 563-572 — IS and IV
- Van Someren et al. (1999). *Chronobiology International*, 16(4), 505-518 — M10, L5
- Sokolove & Bushell (1978). *Journal of Theoretical Biology*, 72(1), 131-160 — Enright periodogram
- Tackenberg & Hughey (2021). *PLoS Computational Biology*, 17(1), e1008567 — chi-square periodogram
- Cole et al. (1992). *Sleep*, 15(5), 461-469 — Cole-Kripke sleep/wake scoring
- Phillips et al. (2017). *Scientific Reports*, 7, 3216 — Sleep Regularity Index (SRI)
- Fischer et al. (2021). *PMC8503839* — SRI / IS methods review

## GGIR Cole-Kripke path

Updated Step 1 writes GGIR-style ZC by default. Step 2 reads the paired metadata,
verifies the declared sidecar and uses it for Cole-Kripke/SRI. Old epoch CSVs
without a declared count stream retain the historical SVM fallback, explicitly
labelled as non-validated. Output state encoding remains `0=sleep`, `1=wake`.

`readParticipantCSV(path, fname)` recognises `preprocessing.ck_counts` in the paired
`<stem>.metadata.json`, verifies the declared files' checksums, and adds the
`CK_activity` column to the returned DataFrame. SVM is unchanged. There
is no import or installation dependency on Step 1. Missing/corrupted **declared**
counts raise an error rather than silently falling back. Undeclared sidecars are
ignored. Keep the main CSV, metadata and `ck_counts/` file together.

`compute_SRI(d)` selects GGIR when `CK_activity` or `CK_count` is present. Here
`CK_activity` means **minute ZC sum / 6**, not a 2-second average and not SVM.
`normalize_per_second` applies only to the legacy SVM path. The dictionary schema
and downstream SRI aggregation remain unchanged.

For plots, `cole_kripke(d)` accepts a minute DataFrame and returns the same float
array format aligned with the supplied rows. `score_sleep_wake(d)` returns a
complete timestamped minute series and also accepts 2-second `CK_count` input.
Existing plain-array calls retain their original behaviour: arrays contain no
information about timestamps or measurement provenance. Callers that explicitly
discard the optional counts column must retain it to opt into the new algorithm.

Neishabouri counts are a separate optional Step 1 output. They are not
automatically fed into Cole-Kripke: GGIR's maintainers note that their scale
differs from the historical counts on which common sleep classifiers were based.
Until the team selects and validates an explicit profile, Step 2 refuses to
silently reinterpret a Neishabouri-only file as SVM CK input.

### Exact GGIR correspondence

Reference: **GGIR 3.3-9**, pinned commit
[`1a849d174ca3b3649267d9d7487fa75ab8c00c40`](https://github.com/wadpac/GGIR/tree/1a849d174ca3b3649267d9d7487fa75ab8c00c40),
checked 18 September 2026. The implementation follows the executable source,
including its boundary/alignment behaviour, rather than only a textbook formula.

| Stage | Pinned GGIR operation | Local implementation |
|---|---|---|
| Axis/scale | `Sadeh_axis="Y"`, `zc.scale=1` | Metadata-declared axis, unit scale |
| Missing counts | `g.sib.det`'s `fix_NA_invector`: NA -> 0 | `counts_to_minutes` fills missing 2-second counts with zero |
| One-minute reduction | `sumPerWindow(..., summingwindow=60)`, then `/6` | Sum 30 consecutive 2-second counts, divide by 6 |
| Window | `create_rollfun_mat(..., Ncol=7)`, endpoint replication, newest column first | Same reversed seven-column windows |
| Weighted score | `0.001 * matrix %*% c(67,74,230,76,58,54,106)` | Same coefficients, order and scale |
| Prefix | `PS = c(rep(2,4), PS)` | Prepend four scores of 2 |
| Classification | `PS < 1` gives sleep=1 | Same comparison, invert to preserve this API's wake=1 |
| Reformat | Repeat each minute state to the input epoch, resize to original length | Return one state per original minute; equivalent to sampling GGIR's repeated states at minute starts |
| Rescoring | No Webster rescoring | None |

**Alignment detail:** the R helper already centres/pads the seven-column matrix,
and HASIB then prepends four scores. The resulting timestamp alignment differs
from simply applying weights at `t-4,...,t+2`. This port reproduces the actual R
sequence. With zero-based indices, after the first four output minutes, weights
`[106,54,58,76,230,74,67]` address activity indices `t-7,...,t-1`, with endpoints
clamped. The first four scores are 2; the tail is truncated to the output timeline.
This behaviour is covered by impulse and boundary fixtures and is not silently
"corrected" to another CK variant.

**Missing-data detail:** count NaNs become zero before aggregation, following
`g.sib.det`; the former proposed "any missing neighbour -> NaN" rule is not used.
GGIR's earlier non-wear detection and cross-day imputation (`g.impute`) are separate
stages and are not implemented here. Structural gaps in the local timestamp grid
are inserted before the count NA rule; inconsistent/off-grid timestamps and
conflicting duplicates raise errors. Identical duplicates collapse. Incomplete
trailing count intervals are aligned with the existing minute CSV; prefix/resize
behaviour is verified against R, including a partial final minute.

### Scope and scientific interpretation

The numerical match covers the metadata-declared Step 1 ZC counts through the
GGIR CK classification core, with the stated local time/IO adapters. It does not
claim full-pipeline GGIR equivalence with non-wear imputation, sleep-period
detection or daily reporting. `nonparametric.py` uses the documented canonical
minute-level M10/L5 implementation and gap-aware IS/IV; periodogram and the
experimental daily metrics are preserved. `compute_SRI` continues to aggregate
states using the existing 24-hour-pair formula and 60-second output grid. GGIR's
own SRI function has additional daily reporting, validity masks and optional
smoothing, which are outside this change.

GGIR ZC remains an approximation of historical proprietary hardware. Neither the
legacy SVM path nor agreement with the R implementation establishes sleep validity
at the wrist, upper arm or ankle. A diary sleep-attempt interval is not minute-by-
minute sleep ground truth. No thresholds were fitted to these diary results.

Primary sources:
[GGIR ZC extraction](https://github.com/wadpac/GGIR/blob/1a849d174ca3b3649267d9d7487fa75ab8c00c40/R/g.applymetrics.R),
[defaults](https://github.com/wadpac/GGIR/blob/1a849d174ca3b3649267d9d7487fa75ab8c00c40/R/load_params.R),
[input preparation](https://github.com/wadpac/GGIR/blob/1a849d174ca3b3649267d9d7487fa75ab8c00c40/R/g.sib.det.R),
[HASIB and its helpers](https://github.com/wadpac/GGIR/blob/1a849d174ca3b3649267d9d7487fa75ab8c00c40/R/HASIB.R),
[non-wear imputation boundary](https://github.com/wadpac/GGIR/blob/1a849d174ca3b3649267d9d7487fa75ab8c00c40/R/g.impute.R),
[GGIR SRI boundary](https://github.com/wadpac/GGIR/blob/1a849d174ca3b3649267d9d7487fa75ab8c00c40/R/CalcSleepRegularityIndex.R),
and [Cole et al. 1992](https://doi.org/10.1093/sleep/15.5.461).

### Validation

Run `python -m unittest discover -s tests -v`. The tests include nine synthetic
cases with counts, weighted scores and states generated by the unmodified pinned
R function, including impulses, boundaries, missing counts and a partial minute.
`tests/generate_ggir_fixture.R` makes the oracle reproducible; R is not required
for normal Python execution. Three real 50-minute segments also matched the
original R functions from raw acceleration through CK classification. Eighteen
full-recording legacy regression comparisons (ordinary, missing and duplicate
input, with and without normalization) retained the original outputs and both
public CK/SRI function signatures.

GGIR reference-derived components are attributed above; the upstream Apache-2.0
license is included in `LICENSE-GGIR`. Existing repository licensing is unchanged.

The completed local rerun also matched original R HASIB for all 31,109
full-recording minute labels and 31,063 cropped common-period labels.
