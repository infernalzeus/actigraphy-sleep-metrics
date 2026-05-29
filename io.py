import pandas as pd
import os
import numpy as np
from pathlib import Path

def readParticipantCSV(path, fname):
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
