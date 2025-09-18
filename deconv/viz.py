# viz.py
import matplotlib.pyplot as plt

def plot_spectrum_with_peaks(mz, inten, peaks, title="Spectrum with peaks"):
    plt.figure(figsize=(12, 4))
    plt.plot(mz, inten, color="black", linewidth=1, label="spectrum")
    for p in peaks:
        plt.scatter(p.mz, p.intensity, color="red", zorder=5)
        plt.text(p.mz, p.intensity*1.05, f"{p.mz:.2f}", 
                 ha="center", va="bottom", fontsize=8, rotation=90)
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.tight_layout()
    plt.show()
