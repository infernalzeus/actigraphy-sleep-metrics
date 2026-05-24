import pandas as pd
import numpy as np
import os

def timeBin(d):
    """
    Takes a dataframe of participant data and returns the 'delta' (in minutes) used for binning
    """
    if len(d) > 1:
        delta = (d['Time'].iloc[1] - d['Time'].iloc[0]).total_seconds() / 60.0
        return delta
    return 0

def aggregateHours(d):
    """
    Aggregates 'SVM' column by hour
    """
    delta = timeBin(d)
    if delta > 60:
        raise ValueError("aggregateHours: data is too coarse (greater than 60 minutes bins)")
        
    d_copy = d.copy()
    d_copy['hour_label'] = d_copy['Time'].dt.floor('H')
    
    # group by hour label and calculate mean
    agg_df = d_copy.groupby('hour_label')['SVM'].mean().reset_index()
    agg_df = agg_df.rename(columns={'hour_label': 'Time', 'SVM': 'Activity'})
    
    return agg_df

def dayByHourMatrix(df):
    """
    df: output of aggregateHours
    Returns matrix with rows=Days, cols=Hours, elements=hourly activity summaries
    """
    df_copy = df.copy()
    df_copy['Hour'] = df_copy['Time'].dt.hour
    df_copy['Date'] = df_copy['Time'].dt.date
    
    # Pivot to wide format
    m_df = df_copy.pivot(index='Date', columns='Hour', values='Activity')
    
    return m_df.values, m_df.index.values

def rollingWindowInd(t, window, step):
    """
    Fits rolling windows to time series and outputs start/end indices.
    """
    t_arr = np.array(t)
    tLen = t_arr[-1] - t_arr[0]
    
    if window >= tLen:
        raise ValueError("Window length must not be longer than time series length")
    if step > window:
        raise ValueError("Step length must not be longer than window length")
        
    tAbs = t_arr - t_arr[0]
    nWin = int(np.ceil((tLen - window) / step + 1))
    
    stInd = np.zeros(nWin, dtype=int)
    enInd = np.zeros(nWin, dtype=int)
    
    for jj in range(nWin):
        tSt = step * jj
        tEn = tSt + window
        
        stInd[jj] = np.argmin(np.abs(tAbs - tSt))
        if jj == nWin - 1:
            enInd[jj] = len(t_arr) - 1
        else:
            enInd[jj] = np.argmin(np.abs(tAbs - tEn))
            
    if stInd[-1] == enInd[-1]:
        stInd = stInd[:-1]
        enInd = enInd[:-1]
        
    # Check for cases where windows are too long / short
    duration = t_arr[enInd] - t_arr[stInd]
    naBl = (duration < 0.8 * window) | (duration > 1.2 * window)
    
    stInd = np.where(naBl, -1, stInd)
    enInd = np.where(naBl, -1, enInd)
    
    return stInd, enInd

def nonwear_detect(dsdir, nwdir=None, rmc_col_time=0, rmc_col_acc=[1,2,3], sdThres=0.12753, rngThres=0.4905):
    """
    Evaluates epoch-by-epoch non-wear status of accelerometer devices
    """
    if not dsdir:
        raise ValueError("Specify directory of accelerometer data")
    if not nwdir:
        nwdir = os.path.join(os.path.dirname(dsdir), "nw_output")
    os.makedirs(nwdir, exist_ok=True)
    
    dsLiNames = [f for f in os.listdir(dsdir) if f.endswith('.csv')]
    
    for fname in dsLiNames:
        wrdir = os.path.join(nwdir, fname)
        if not os.path.exists(wrdir):
            try:
                appt = pd.read_csv(os.path.join(dsdir, fname))
                ts = appt.iloc[:, rmc_col_time].values
                
                stInd, enInd = rollingWindowInd(ts, 3600, 900)
                valid_mask = stInd != -1
                stInd = stInd[valid_mask]
                enInd = enInd[valid_mask]
                
                mid_ts = (ts[enInd] - ts[stInd]) / 2 + ts[stInd] - 450
                st15NW = pd.DataFrame({'ts': mid_ts, 'nonwear': False})
                
                for j in range(len(st15NW)):
                    countAx = 0
                    for k in rmc_col_acc:
                        if countAx < 2:
                            winDat = appt.iloc[stInd[j]:enInd[j]+1, k].values
                            if len(winDat) > 1:
                                sdev = np.std(winDat, ddof=1)
                                rng = np.ptp(winDat)
                                if sdev < sdThres and rng < rngThres:
                                    countAx += 1
                    if countAx >= 2:
                        st15NW.loc[j, 'nonwear'] = True
                        
                st15NW.to_csv(wrdir, index=False)
                print(f"Non-wear data extracted: {fname}")
            except Exception as e:
                print(f"Error processing {fname}: {e}")
