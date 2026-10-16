"""Optional CK count input contract, independent of the Step 1 Python package."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def timed_series(d, column):
    """Sorted, deduplicated timestamps; conflicting duplicates are invalid input."""
    frame = d[["Time", column]].copy()
    frame["Time"] = pd.to_datetime(frame["Time"].astype(str).str.replace(r":(\d{3})$", r".\1", regex=True))
    frame = frame.dropna(subset=["Time"]).sort_values("Time", kind="stable")
    if frame.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]), name=column)
    if frame.groupby("Time")[column].nunique(dropna=False).gt(1).any():
        raise ValueError(f"Conflicting duplicate timestamps in {column}.")
    series = frame.drop_duplicates("Time").set_index("Time")[column].astype(float)
    if (series.dropna() < 0).any() or np.isinf(series).any():
        raise ValueError(f"{column} must be nonnegative and finite, or NaN for missing data.")
    return series


def regular_grid(series, seconds):
    if series.empty:
        return series
    offsets = (series.index - series.index[0]).to_numpy(dtype="timedelta64[ns]").astype(np.int64)
    if np.any(offsets % (seconds * 1_000_000_000)):
        raise ValueError(f"CK timestamps must lie on one regular {seconds}-second grid.")
    return series.reindex(pd.date_range(series.index[0], series.index[-1], freq=f"{seconds}s"))


def counts_to_minutes(d):
    """GGIR g.sib.det + HASIB: NA counts -> 0; complete minute sum / 6.

    Keep the recording-start phase. Incomplete trailing minutes are represented
    by NaN here only for alignment with the unchanged main CSV; the scorer's
    prefix/resize semantics determine their returned state.
    """
    counts = regular_grid(timed_series(d, "CK_count"), 2)
    if counts.empty:
        return counts.rename("CK_activity")
    if not np.equal(counts.dropna(), np.floor(counts.dropna())).all():
        raise ValueError("CK_count must contain integer zero-crossing counts.")
    minutes = counts.fillna(0).resample("60s", origin=counts.index[0]).sum(min_count=30) / 6.0
    return minutes.rename("CK_activity")


def attach_ck_input(d, file_path):
    """Load only a metadata-declared, hash-verified sidecar. Old files stay old.

    Keep the sidecar and paired metadata when copying data. Never silently fall
    back to SVM if metadata explicitly declares CK input but that input is broken.
    """
    path = Path(file_path)
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        return d
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    info = metadata.get("preprocessing", {}).get("ck_counts")
    if info is None:
        return d
    if info.get("schema_version") != 1 or info.get("epoch_sec") != 2 or info.get("axis") not in ("x", "y", "z"):
        raise ValueError("Unsupported CK sidecar schema, epoch or axis.")
    sidecar = (path.parent / info["file"]).resolve()
    if not sidecar.is_relative_to(path.parent.resolve()):
        raise ValueError("CK sidecar must be inside the input directory.")
    for file, key in ((path, "main_csv_sha256"), (sidecar, "counts_csv_sha256")):
        if hashlib.sha256(file.read_bytes()).hexdigest() != info.get(key):
            raise ValueError(f"CK input checksum mismatch: {file.name}; regenerate the paired files.")
    counts = pd.read_csv(sidecar)
    minute = counts_to_minutes(counts)
    times = pd.DatetimeIndex(d["Time"])
    if len(times) and len(minute):
        if np.any((times - minute.index[0]).to_numpy(dtype="timedelta64[ns]").astype(np.int64) % 60_000_000_000):
            raise ValueError("CK sidecar requires main CSV timestamps on the same 60-second grid.")
        if times.min() < minute.index.min() or times.max() > minute.index.max() + pd.Timedelta(seconds=60):
            raise ValueError("CK sidecar does not cover the main CSV recording.")
    d = d.copy()
    d["CK_activity"] = minute.reindex(times).to_numpy()
    return d
