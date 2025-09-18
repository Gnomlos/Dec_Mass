import numpy as np
from deconv.peakpick import pick_peaks, estimate_noise

def synth(peaks, n=20000, x0=0.0, x1=2000.0, fwhm=0.02, noise=20, seed=1):
    x = np.linspace(x0, x1, n)
    y = np.zeros_like(x)
    sigma = fwhm/2.35482
    def add(center, height):
        y[:] += height*np.exp(-(x-center)**2/(2*sigma**2))
    for c, h in peaks:
        add(c, h)
    rng = np.random.default_rng(seed)
    y += rng.poisson(noise, size=n)
    return x, y

def test_detects_main_peaks():
    mz, I = synth([(600.0, 2e4), (900.0, 3e4), (1200.0, 1.5e4)], fwhm=0.02, noise=30)
    peaks = pick_peaks(mz, I, min_snr=5.0, smooth_sigma=2.0)
    assert any(abs(p.mz-600.0)<0.05 for p in peaks)
    assert any(abs(p.mz-900.0)<0.05 for p in peaks)
    assert any(abs(p.mz-1200.0)<0.05 for p in peaks)

def test_noise_estimator_positive():
    _, I = synth([(700.0, 1e4)], noise=50)
    sig = estimate_noise(I)
    assert sig > 0
