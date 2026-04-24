"""Epileptiform spike detection.

Detects high-amplitude transient events (sharp waves / spikes) in an EEG
channel using a robust z-score threshold and minimum-spacing constraint.
This is a lightweight, classical detector — useful as a clinically
recognizable signal alongside the ML classifier.
"""
import numpy as np
from scipy.signal import find_peaks


def detect_spikes(signal, sfreq, z_thresh=3.5, min_spacing_ms=50):
    """Detect spike events in a 1D signal.

    Parameters
    ----------
    signal : array-like
        Single-channel EEG samples.
    sfreq : float
        Sampling frequency the samples were captured at (Hz).
    z_thresh : float
        How many robust standard deviations above the median absolute
        deviation an absolute peak must rise to be counted as a spike.
    min_spacing_ms : float
        Minimum spacing between two consecutive spikes, in milliseconds.

    Returns
    -------
    dict with keys:
        count        : int, number of spikes detected
        times_sec    : list[float], spike onset times in seconds
        rate_per_min : float, spikes per minute over the analyzed window
        threshold    : float, amplitude threshold used
    """
    x = np.asarray(signal, dtype=float)
    if x.size == 0 or sfreq <= 0:
        return {"count": 0, "times_sec": [], "rate_per_min": 0.0, "threshold": 0.0}

    # Robust center / scale (median + MAD) — resistant to seizure bursts
    # dominating the statistics.
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median))) or 1e-9
    threshold = z_thresh * 1.4826 * mad  # 1.4826 → MAD≈std for normal data

    abs_dev = np.abs(x - median)
    distance = max(1, int((min_spacing_ms / 1000.0) * sfreq))
    peaks, _ = find_peaks(abs_dev, height=threshold, distance=distance)

    times = (peaks / sfreq).tolist()
    duration_sec = x.size / sfreq
    rate = (len(peaks) / duration_sec) * 60.0 if duration_sec > 0 else 0.0

    return {
        "count": int(len(peaks)),
        "times_sec": [round(t, 3) for t in times],
        "rate_per_min": round(rate, 2),
        "threshold": round(threshold, 4),
    }
