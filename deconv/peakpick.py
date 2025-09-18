from dataclasses import dataclass
import numpy as np

@dataclass
class Peak:
    mz: float
    intensity: float
    index: int
    left_index: int
    right_index: int
    width_pts: int
    snr: float

def smooth_gaussian(y: np.ndarray, sigma_pts: float = 2.0) -> np.ndarray:
    if sigma_pts <= 0:
        return y.astype(float, copy=True)
    radius = int(3*sigma_pts)
    x = np.arange(-radius, radius+1)
    k = np.exp(-(x**2)/(2*sigma_pts**2))
    k /= k.sum()
    return np.convolve(y, k, mode="same")

def estimate_noise(y: np.ndarray) -> float:
    mad = np.median(np.abs(y - np.median(y))) + 1e-12
    return mad / 0.6745

def _local_maxima(y: np.ndarray):
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1

def _peak_width_bounds(y: np.ndarray, i: int, frac: float = 0.5):
    thresh = y[i] * frac
    L = i
    while L > 0 and y[L] > thresh and y[L-1] <= y[L]:
        L -= 1
    R = i
    n = len(y)
    while R < n-1 and y[R] > thresh and y[R+1] <= y[R]:
        R += 1
    return L, R

def pick_peaks(mz: np.ndarray,
               intensity: np.ndarray,
               min_snr: float = 3.0,
               smooth_sigma: float = 2.0,
               rel_height_for_width: float = 0.5,
               max_peaks: int | None = None):
    y = smooth_gaussian(intensity, smooth_sigma)
    noise = estimate_noise(y)
    thresh = max(min_snr * noise, 0.0)
    candidate_idx = _local_maxima(y)
    peaks = []
    for i in candidate_idx:
        if y[i] <= thresh:
            continue
        L, R = _peak_width_bounds(y, i, frac=rel_height_for_width)
        width = max(1, R - L + 1)
        snr = (y[i] - 0.5*(y[L] + y[R])) / (noise + 1e-12)
        peaks.append(Peak(float(mz[i]), float(y[i]), int(i), int(L), int(R), int(width), float(snr)))
    peaks.sort(key=lambda p: p.intensity, reverse=True)
    if max_peaks is not None and len(peaks) > max_peaks:
        peaks = peaks[:max_peaks]
    return peaks
