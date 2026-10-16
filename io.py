import pandas as pd
import os
import numpy as np
import json
from pathlib import Path
from ck_input import attach_ck_input


_WINDOWS_TIMEZONES = {
    "GMT Standard Time": "Europe/London",
    "GMT Daylight Time": "Europe/London",
}


def _read_metadata(file_path):
    metadata_path = Path(file_path).with_suffix('.metadata.json')
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding='utf-8'))


def _metadata_timezone(metadata, override=None):
    if override:
        return override
    timezone = (
        metadata.get('preprocessing', {}).get('timezone')
        or metadata.get('recording', {}).get('time_zone')
    )
    return _WINDOWS_TIMEZONES.get(timezone, timezone)


def _localize_time(d, timezone):
    """Attach a real timezone without changing device wall-clock timestamps."""
    d.attrs['timezone'] = timezone
    d.attrs['dst_excluded_dates'] = []
    if not timezone:
        return d
    try:
        localized = d['Time'].dt.tz_localize(
            timezone, ambiguous='raise', nonexistent='raise'
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot localize device timestamps to {timezone}: {exc}. "
            "Ambiguous or nonexistent naive times must be resolved upstream."
        ) from exc
    d['Time'] = localized
    offsets = localized.map(lambda value: value.utcoffset())
    offset_counts = pd.DataFrame({
        'date': localized.dt.date,
        'offset': offsets,
    }).groupby('date')['offset'].nunique()
    d.attrs['dst_excluded_dates'] = [
        date for date, count in offset_counts.items() if count > 1
    ]
    return d

def readParticipantCSV(path, fname, timezone=None):
    """
    Read a single participant CSV and normalise columns.
    Matches R: readParticipantCSV() in read-participant.R
    """
    file_path = os.path.join(path, fname) if path else fname
    d = pd.read_csv(file_path)

    # Actigraph exports timestamps as "YYYY-MM-DD HH:MM:SS:mmm"
    # (milliseconds delimited by ':' instead of '.') — normalise before parsing
    d['Time'] = pd.to_datetime(
        d['Time'].str.replace(r':(\d{3})$', r'.\1', regex=True)
    )

    # Map SVM_sum → SVM so downstream code matches the R column name convention
    if 'SVM_sum' in d.columns and 'SVM' not in d.columns:
        d = d.rename(columns={'SVM_sum': 'SVM'})

    metadata = _read_metadata(file_path)
    d = attach_ck_input(d, file_path)
    d = _localize_time(d, _metadata_timezone(metadata, timezone))
    d.attrs['count_method'] = metadata.get('preprocessing', {}).get('count_method')
    return d

def ds_accel_csv(acceldir, alloutdir=None, dsdir=None, col_timestamp=None, col_accel=None):
    """
    Down-samples .csv files to 1Hz, increasing speed and creating required input format for GGIR.
    """
    if not acceldir:
        raise ValueError("Specify directory containing accelerometer .csv files")
    if not alloutdir:
        alloutdir = f"{acceldir}_output"
    os.makedirs(alloutdir, exist_ok=True)
    
    if not dsdir:
        dsdir = os.path.join(alloutdir, "ds_output")
    os.makedirs(dsdir, exist_ok=True)
    
    if col_timestamp is None:
        raise ValueError("Specify column index (0-based) for timestamp")
    if col_accel is None:
        raise ValueError("Specify column indices (0-based) for x,y,z data")

    fl = [f for f in os.listdir(acceldir) if f.endswith('.csv')]
    
    for fname in fl:
        new_fname = fname.replace("_", "-").replace(" ", "-")
        wrdir = os.path.join(dsdir, new_fname)
        
        if not os.path.exists(wrdir):
            file_path = os.path.join(acceldir, fname)
            # Read specific columns for memory efficiency
            cols_to_use = [col_timestamp] + col_accel
            try:
                apptr = pd.read_csv(file_path, usecols=cols_to_use)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
                continue
                
            ts_col = apptr.columns[0] # The timestamp column
            accel_cols = apptr.columns[1:] # The xyz columns
            
            ts = apptr[ts_col].values
            
            # Simple downsampling assuming roughly uniform high frequency
            secl = np.floor((ts[-1] - ts[0]))
            if secl > 0:
                int_step = len(ts) / secl
                rind = np.round(np.arange(0, len(ts), int_step)).astype(int)
                rind = rind[rind < len(ts)]
                
                apptw = pd.DataFrame()
                apptw['t'] = np.linspace(np.round(ts[0]), np.round(ts[0]) + len(rind) - 1, len(rind))
                for i, c in enumerate(accel_cols):
                    apptw[['x','y','z'][i]] = apptr[c].iloc[rind].values
                    
                apptw.to_csv(wrdir, index=False)
                print(f"Down-sampled file extracted: {fname}")
