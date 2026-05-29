import pandas as pd
import numpy as np
import os

def timeBin(d):
    # Returns the epoch interval in minutes — the gap between the first two rows.
    # Used to verify the data is at 60-second (or finer) resolution before
    # aggregating to hourly means.
    # Equivalent to R: timeBin <- function(d) { as.numeric(d$Time[2] - d$Time[1]) }
    if len(d) > 1:
        return (d['Time'].iloc[1] - d['Time'].iloc[0]).total_seconds() / 60.0
    return 0


def aggregateHours(d):
    # Aggregates the raw epoch-level SVM series to hourly mean activity values.
    #
    # Input column : SVM  (signal vector magnitude sum, 60-s epochs from the Actigraph)
    #                Epoch interval must be <= 60 min (checked via timeBin).
    #
    # Method:
    #   1. Floor each timestamp to its hour boundary
    #      (e.g. 12:34 -> 12:00,  23:59 -> 23:00)
    #   2. Group all epochs that share the same floored hour
    #   3. Take the mean SVM across those epochs -> one value per hour
    #
    # Output: DataFrame with columns [Time (hour boundary), Activity (mean SVM)]
    #
    # Equivalent to R:
    #   group_by(floor_date(Time, "hour")) %>% summarise(SVM = mean(SVM, na.rm=TRUE))

    delta = timeBin(d)
    if delta > 60:
        raise ValueError("aggregateHours: data is too coarse (greater than 60 minutes bins)")

    d_copy = d.copy()
    d_copy['hour_label'] = d_copy['Time'].dt.floor('h')

    agg_df = d_copy.groupby('hour_label')['SVM'].mean().reset_index()
    agg_df = agg_df.rename(columns={'hour_label': 'Time', 'SVM': 'Activity'})

    return agg_df


def dayByHourMatrix(df):
    # Reshapes the hourly long-format series into a [days x 24 hours] matrix.
    #
    # Input:  output of aggregateHours — one row per hour
    # Output: numpy array shape (D, 24) where D = number of calendar days spanned,
    #         plus the array of corresponding date labels.
    #         Hours with no data (partial first/last days) are filled with NaN.
    #
    # The resulting matrix is the standard input format for all metric functions:
    #   rows    = days  (day index, 0-based)
    #   columns = hours (0 = midnight, 1 = 01:00, ..., 23 = 23:00)
    #   values  = mean SVM activity for that hour on that day
    #
    # Equivalent to R:
    #   reshape2::dcast(df, Day ~ Hour, value.var = "Activity")
    # with an explicit reindex to ensure all 24 hour columns are present
    # even when the first or last day has no data at certain hours.

    df_copy = df.copy()
    df_copy['Hour'] = df_copy['Time'].dt.hour
    df_copy['Date'] = df_copy['Time'].dt.date

    m_df = df_copy.pivot(index='Date', columns='Hour', values='Activity')
    # guarantee all 24 columns; missing hours -> NaN
    m_df = m_df.reindex(columns=range(24))
    m_df.columns.name = None

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
