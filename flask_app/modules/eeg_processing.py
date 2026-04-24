"""EEG loading and preprocessing utilities."""
import numpy as np
from scipy.signal import butter, filtfilt
import mne
import warnings

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


def load_edf(file_path):
    """Load an .edf file and return raw signal array (channels x samples) and sfreq."""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    data = raw.get_data()  # (n_channels, n_samples) in volts
    sfreq = float(raw.info["sfreq"])
    ch_names = raw.ch_names
    return data, sfreq, ch_names


def bandpass_filter(data, sfreq, low=0.5, high=40.0, order=4):
    """Apply a Butterworth bandpass filter to each channel."""
    nyq = 0.5 * sfreq
    low_n = low / nyq
    high_n = min(high / nyq, 0.99)
    b, a = butter(order, [low_n, high_n], btype="band")
    filtered = filtfilt(b, a, data, axis=-1)
    return filtered


def normalize(data):
    """Z-score normalize each channel."""
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True) + 1e-8
    return (data - mean) / std


def segment(data, sfreq, window_sec=5.0):
    """Segment signal into fixed-size windows.

    Returns array shape (n_windows, n_channels, win_samples).
    """
    win = int(window_sec * sfreq)
    n_channels, n_samples = data.shape
    n_windows = n_samples // win
    if n_windows == 0:
        return np.empty((0, n_channels, win))
    trimmed = data[:, : n_windows * win]
    segs = trimmed.reshape(n_channels, n_windows, win).transpose(1, 0, 2)
    return segs


def preprocess_file(file_path, window_sec=5.0, max_windows=240):
    """Full preprocessing pipeline for an uploaded EDF file.

    Strategy: instead of analyzing only the first N seconds (which would
    miss seizures that occur later), we sample ``max_windows`` 5-second
    windows distributed *uniformly across the entire recording*. This
    keeps memory bounded while still covering seizure events anywhere
    in a long recording.

    Returns dict with segments, sfreq, channel names, a preview waveform,
    and the actual time offset of each sampled window.
    """
    data, sfreq, ch_names = load_edf(file_path)
    n_channels, n_samples = data.shape
    duration_sec = n_samples / sfreq
    win = int(window_sec * sfreq)

    # Bandpass on the full signal (cheap relative to memory of segments).
    filtered = bandpass_filter(data, sfreq)
    normed = normalize(filtered)

    # Choose window start indices spread across the file.
    n_full = max(1, n_samples // win)
    n_take = min(max_windows, n_full)
    if n_take <= 1:
        starts = np.array([0])
    else:
        starts = np.linspace(0, n_samples - win, n_take, dtype=int)

    segs = np.empty((n_take, n_channels, win), dtype=np.float32)
    for i, s in enumerate(starts):
        segs[i] = normed[:, s:s + win]

    window_times = (starts / sfreq).tolist()

    # Preview: a short representative slice (first 60 s) of the first channel.
    preview_seconds = min(60.0, duration_sec)
    preview_samples = int(preview_seconds * sfreq)
    preview_channel = filtered[0, :preview_samples]
    target_points = 1000
    step = max(1, len(preview_channel) // target_points)
    preview = preview_channel[::step][:target_points].tolist()

    return {
        "segments": segs,
        "sfreq": sfreq,
        "ch_names": ch_names,
        "preview": preview,
        "preview_duration_sec": float(preview_seconds),
        "duration_sec": float(duration_sec),
        "n_channels": int(n_channels),
        "window_times": window_times,
        "window_sec": float(window_sec),
    }
