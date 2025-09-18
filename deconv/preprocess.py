import numpy as np

def rolling_baseline(y, window_pts=501, quantile=0.1):
    if window_pts <= 1:
        return np.zeros_like(y)
    pad = window_pts//2
    ypad = np.pad(y, pad, mode='edge')
    out = np.zeros_like(y, dtype=float)
    for i in range(len(y)):
        w = ypad[i:i+window_pts]
        out[i] = np.quantile(w, quantile)
    return out
