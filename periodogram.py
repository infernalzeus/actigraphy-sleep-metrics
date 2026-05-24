import numpy as np
import pandas as pd
from scipy.stats import chi2

def computeMean_X_h_p(X, h, p, K, na_rm=True):
    j = np.arange(K)
    indices = h + j * p
    indices = indices[indices < len(X)]
    X_h_p = X[indices]
    
    if na_rm:
        X_h_p = X_h_p[~np.isnan(X_h_p)]
        
    this_K = len(X_h_p)
    if this_K == 0:
        return np.nan
        
    return (1 / this_K) * np.nansum(X_h_p)

def computeEnright_Ap(X, p, adjust_for_NA=True):
    P = X.shape[1]
    K = X.shape[0]
    X_v = X.flatten()
    
    X_bar_h_p = np.zeros(P)
    for h in range(P):
        X_bar_h_p[h] = computeMean_X_h_p(X_v, h, p, K, adjust_for_NA)
        
    X_bar_p = (1 / p) * np.nansum(X_bar_h_p)
    
    A_p = np.sqrt((1 / P) * np.nansum((X_bar_h_p - X_bar_p)**2))
    return A_p

def periodogram_Enright(X, periods=np.arange(14, 35), adjust_for_NA=True):
    pgram = pd.DataFrame({
        'Period': periods,
        'Value': np.nan
    })
    
    for i, p in enumerate(periods):
        pgram.loc[i, 'Value'] = computeEnright_Ap(X, p, adjust_for_NA)
        
    return pgram

def computeQ_P(X_v, N, P, K):
    X_included = X_v[:int(K*P)]
    X_m = X_included.reshape((int(K), P))
    
    X_bar = np.nanmean(X_included)
    N_included = len(X_included) - np.isnan(X_included).sum()
    
    X_h = np.nanmean(X_m, axis=0)
    
    numer = K * N_included * np.nansum((X_h - X_bar)**2)
    denom = np.nansum((X_included - X_bar)**2)
    
    return numer / denom if denom != 0 else np.nan

def computeQ_P_Greedy(X_v, N, P, K):
    pad_len = int(np.ceil(K) * P - N)
    if pad_len > 0:
        X_v = np.pad(X_v, (0, pad_len), constant_values=np.nan)
        
    X_m = X_v.reshape((-1, P))
    X_bar = np.nanmean(X_m)
    
    X_h = np.nanmean(X_m, axis=0)
    
    N_included = N - np.isnan(X_v[:N]).sum()
    K_included = N_included / P
    
    numer = K_included * N_included * np.nansum((X_h - X_bar)**2)
    denom = np.nansum((X_m - X_bar)**2)
    
    return numer / denom if denom != 0 else np.nan

def periodogram_ChiSquare(X, periods=np.arange(14, 35), greedy=False):
    pg = pd.DataFrame({
        'Period': periods,
        'Value': np.nan
    })
    
    X_v = X.flatten()
    N = len(X_v)
    
    for i, p in enumerate(periods):
        if greedy:
            this_K = N / p
            pg.loc[i, 'Value'] = computeQ_P_Greedy(X_v, N, p, this_K)
        else:
            this_K = int(np.floor(N / p))
            pg.loc[i, 'Value'] = computeQ_P(X_v, N, p, this_K)
            
    return pg

def add_pValue_ChiSquare(pg):
    pg = pg.copy()
    pg['log_Pvalue'] = chi2.logsf(pg['Value'], df=pg['Period'] - 1)
    return pg
