import numpy as np
import pandas as pd

def interdailyStability(mat_df, P=24):
    X_v = mat_df.flatten()
    N = len(X_v)
    K = N / P
    
    # Pad out with nan
    pad_len = int(np.ceil(K) * P - N)
    if pad_len > 0:
        X_v = np.pad(X_v, (0, pad_len), constant_values=np.nan)
        
    X_m = X_v.reshape((-1, P))
    
    X_bar = np.nanmean(X_m)
    X_h = np.nanmean(X_m, axis=0)
    
    N_included = N - np.isnan(X_v[:N]).sum()
    
    numer = N_included * np.nansum((X_h - X_bar)**2)
    denom = P * np.nansum((X_v - X_bar)**2)
    
    return numer / denom

def LOCF_NA(x):
    mask = np.isnan(x)
    if not np.any(mask):
        return x
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx].copy()
    # Set leading nans back to nan if necessary
    if mask[0]:
        first_valid = np.argmax(~mask)
        if first_valid == 0 and mask[0]: # all nans
            return out
        out[:first_valid] = np.nan
    return out

def intradailyVariability(mat_df, LOCF=True):
    X_v = mat_df.flatten()
    
    if LOCF:
        X_v = LOCF_NA(X_v)
        
    # Remove remaining nans
    X_v = X_v[~np.isnan(X_v)]
    
    if len(X_v) == 0:
        return np.nan
        
    X_bar = np.mean(X_v)
    N = len(X_v)
    
    dX_v = np.diff(X_v)
    
    numer = N * np.sum(dX_v**2)
    denom = (N - 1) * np.sum((X_bar - X_v)**2)
    
    return numer / denom

def filterHour(x, n):
    weights = np.ones(n) / n
    return np.convolve(np.pad(x, (0, n-1), mode='wrap'), weights, mode='valid')

def filterHour_ForwardBack(x, n):
    f_x_1 = filterHour(x[::-1], n)[::-1]
    f_x_2 = filterHour(x, n)
    f_x = 0.5 * (f_x_1 + f_x_2)
    
    df = pd.DataFrame({
        'Hour': np.arange(24),
        'HourMean': x,
        'filterHourMean': f_x
    })
    return df

def compute_M10(X_m):
    X_h = np.nanmean(X_m, axis=0)
    df = filterHour_ForwardBack(X_h, 10)
    
    df_sorted = df.sort_values(by='filterHourMean', ascending=False)
    
    return {
        'M10_hour': df_sorted.iloc[0]['Hour'],
        'M10_mean_activity': df_sorted.iloc[0]['filterHourMean']
    }

def compute_L5(X_m):
    X_h = np.nanmean(X_m, axis=0)
    df = filterHour_ForwardBack(X_h, 5)
    
    df_sorted = df.sort_values(by='filterHourMean', ascending=True)
    
    return {
        'L5_hour': df_sorted.iloc[0]['Hour'],
        'L5_mean_activity': df_sorted.iloc[0]['filterHourMean']
    }
