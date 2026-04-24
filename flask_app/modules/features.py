"""Feature extraction for EEG segments."""
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy

# NumPy 2.x renamed `trapz` to `trapezoid`. Pick whichever exists.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

# Standard EEG frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}

FEATURE_NAMES = [
    "mean", "variance", "energy", "shannon_entropy", "psd_total",
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
]


def _band_power(freqs, psd, band):
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(_trapz(psd[mask], freqs[mask]))


def extract_segment_features(segment, sfreq):
    """Extract features for a single segment averaged across channels.

    segment: (n_channels, n_samples) array.
    Returns 1D feature vector of length len(FEATURE_NAMES).
    """
    # Average across channels for compact, robust features.
    sig = segment.mean(axis=0)

    mean = float(np.mean(sig))
    variance = float(np.var(sig))
    energy = float(np.sum(sig ** 2))

    # Shannon entropy on histogram of amplitudes.
    hist, _ = np.histogram(sig, bins=32, density=True)
    hist = hist + 1e-12
    sh_entropy = float(scipy_entropy(hist))

    # PSD via Welch.
    nperseg = min(256, len(sig))
    freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg)
    psd_total = float(_trapz(psd, freqs))

    band_powers = [_band_power(freqs, psd, BANDS[b])
                   for b in ("delta", "theta", "alpha", "beta", "gamma")]

    return np.array([mean, variance, energy, sh_entropy, psd_total, *band_powers])


def extract_features(segments, sfreq):
    """Extract feature matrix for a stack of segments.

    segments: (n_windows, n_channels, n_samples).
    Returns (n_windows, n_features) numpy array.
    """
    if len(segments) == 0:
        return np.empty((0, len(FEATURE_NAMES)))
    feats = [extract_segment_features(seg, sfreq) for seg in segments]
    return np.vstack(feats)
