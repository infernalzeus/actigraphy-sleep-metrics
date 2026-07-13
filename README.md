# actigraphy-sleep-metrics — Step 2

Python pipeline that converts epoched Actigraph CSV output into validated
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

An Actigraph ActiLife export at **60-second epochs** in CSV format.  The file
must contain at minimum:

| Column | Description |
|---|---|
| `Time` | Timestamp — format `YYYY-MM-DD HH:MM:SS:mmm` |
| `SVM_sum` | Signal vector magnitude sum (counts per 60-s epoch) |

---

## Usage

Run from inside the `step2` directory:

```bash
python -m cli read <path/to/input_file.csv> [--measures GROUP] [--verbose]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `input_file` | yes | Path to the Actigraph CSV file |
| `--measures GROUP` | no | `nonparametric`, `periodogram`, `temp_metrics`, or `sri`. Omit to run all four. |
| `--verbose` | no | Print step-by-step progress and computed values |

### Examples

Run all three measure groups (default):

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

Scalar rest-activity rhythm metrics computed from the day-by-hour activity
matrix averaged across the full recording.

| Metric | Range | Interpretation |
|---|---|---|
| IS — Interdaily Stability | 0 – 1 | Consistency of the 24-h pattern across days. 0 = no pattern, 1 = identical every day |
| IV — Intradaily Variability | 0 – 2 | Within-day fragmentation. 0 = smooth sinusoidal rhythm, ~2 = random noise |
| M10 hour | 0 – 23 | Midpoint hour of the most-active 10-h window (averaged across all days) |
| M10 mean activity | counts/h | Mean SVM within that most-active window |
| L5 hour | 0 – 23 | Midpoint hour of the least-active 5-h rest window (averaged across all days) |
| L5 mean activity | counts/h | Mean SVM within that rest window |

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
same sleep/wake state, rescaled to a percentage. Computed **directly from the
epoch-level `SVM_sum` series** (no GGIR / R dependency): each 60-s epoch is
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

> **Calibration note.** The Cole-Kripke constants were validated on activity
> counts from older devices, not on Actigraph `SVM_sum`. The *shape* of the
> sleep/wake series (and thus SRI) is robust to the exact scale, but if the
> reported `pct_epochs_sleep` is implausible for your device, tune the `scale`
> (or `threshold`) argument of `compute_SRI` against a night of known sleep
> timing.

### `periodogram`

Rhythm-period analysis over candidate periods 14 – 34 hours.

| Output | Description |
|---|---|
| Enright A_p | Root-mean-square amplitude at each candidate period. Peak = dominant rhythm period |
| Chi-square Q_p | Chi-square statistic at each candidate period |
| log p-value | Log probability under H₀ of no rhythm at that period |

---

## Outputs

All output files are written to `step2/outputs/` and named after the input
file stem.

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
├── nonparametric.py    IS, IV, aggregate compute_M10, compute_L5
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
