# peakfinder.py
from dataclasses import dataclass
import numpy as np
import csv

@dataclass
class Peak:
    mz: float
    intensity: float
    width: float
    snr: float

def load_txt(path: str):
    """Load m/z and intensity columns from a txt file."""
    data = np.loadtxt(path, skiprows=1)  # assumes header
    return data[:,0], data[:,1]

def smooth_gaussian(y, sigma_pts=2):
    if sigma_pts <= 0:
        return y.copy()
    radius = int(3*sigma_pts)
    x = np.arange(-radius, radius+1)
    k = np.exp(-(x**2)/(2*sigma_pts**2))
    k /= k.sum()
    return np.convolve(y, k, mode="same")

def estimate_noise(y):
    mad = np.median(np.abs(y - np.median(y))) + 1e-12
    return mad / 0.6745  # ~sigma if Gaussian

def find_peaks(mz, inten, min_snr=3.0, smooth_sigma=2.0):
    y = smooth_gaussian(inten, smooth_sigma)
    noise = estimate_noise(y)
    thresh = min_snr * noise

    peaks = []
    for i in range(1, len(y)-1):
        if y[i] > y[i-1] and y[i] > y[i+1] and y[i] > thresh:
            # half-height width
            half = y[i] / 2
            L = i
            while L > 0 and y[L] > half: L -= 1
            R = i
            while R < len(y)-1 and y[R] > half: R += 1
            width = mz[R] - mz[L]
            snr = y[i] / (noise+1e-12)
            peaks.append(Peak(mz[i], y[i], width, snr))
    return peaks
    
def save_peaks_csv(peaks, path: str):
    """Save list of Peak objects to CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mz", "intensity", "width", "snr"])
        for p in peaks:
            writer.writerow([f"{p.mz:.6f}", f"{p.intensity:.2f}", f"{p.width:.6f}", f"{p.snr:.2f}"])
