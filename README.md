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
| `--measures GROUP` | no | `nonparametric`, `periodogram`, or `temp_metrics`. Omit to run all three. |
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
| `<stem>_report.pdf` | all groups | Multi-page PDF — see below |

### PDF report page order

| Page | Content | Requires |
|---|---|---|
| 1 | Activity heatmap (noon left, midnight centre) | always |
| 2 | Scalar metrics summary table | `nonparametric` |
| 3+ | Per-day M10 / L5 table (paginated, 30 rows/page) | `temp_metrics` |
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
├── visualize.py        Activity heatmap, PDF report generator
├── sri.py              Deprecated — re-exports from temp_metrics for compatibility
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
