"""Model training/loading for seizure detection.

Trains a small Random Forest on synthetic EEG-like data at startup so the
app is fully self-contained and runs with one click. The synthetic generator
produces two classes that mimic broad statistical properties of normal vs
ictal EEG (higher variance, energy, and low-frequency power during seizures).
"""
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .features import FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
RNG = np.random.default_rng(42)


def _synthetic_normal(n=400):
    """Synthetic features for normal EEG: lower variance/energy, balanced bands."""
    feats = np.zeros((n, len(FEATURE_NAMES)))
    feats[:, 0] = RNG.normal(0, 0.05, n)         # mean
    feats[:, 1] = RNG.normal(1.0, 0.15, n)       # variance
    feats[:, 2] = RNG.normal(800, 120, n)        # energy
    feats[:, 3] = RNG.normal(3.2, 0.2, n)        # entropy (high)
    feats[:, 4] = RNG.normal(1.0, 0.15, n)       # psd_total
    feats[:, 5] = RNG.normal(0.20, 0.05, n)      # delta
    feats[:, 6] = RNG.normal(0.18, 0.04, n)      # theta
    feats[:, 7] = RNG.normal(0.22, 0.05, n)      # alpha
    feats[:, 8] = RNG.normal(0.25, 0.05, n)      # beta
    feats[:, 9] = RNG.normal(0.15, 0.04, n)      # gamma
    return feats


def _synthetic_seizure(n=400):
    """Synthetic features for seizure EEG: higher energy, low-freq dominance, lower entropy."""
    feats = np.zeros((n, len(FEATURE_NAMES)))
    feats[:, 0] = RNG.normal(0, 0.15, n)
    feats[:, 1] = RNG.normal(2.6, 0.4, n)        # higher variance
    feats[:, 2] = RNG.normal(2200, 350, n)       # much higher energy
    feats[:, 3] = RNG.normal(2.4, 0.25, n)       # lower entropy (more rhythmic)
    feats[:, 4] = RNG.normal(2.5, 0.4, n)        # higher PSD total
    feats[:, 5] = RNG.normal(0.55, 0.08, n)      # delta surge
    feats[:, 6] = RNG.normal(0.40, 0.07, n)      # theta surge
    feats[:, 7] = RNG.normal(0.18, 0.05, n)
    feats[:, 8] = RNG.normal(0.15, 0.04, n)
    feats[:, 9] = RNG.normal(0.10, 0.03, n)
    return feats


def train_synthetic_model():
    """Train and persist a RandomForest pipeline on synthetic features."""
    X_norm = _synthetic_normal(500)
    X_seiz = _synthetic_seizure(500)
    X = np.vstack([X_norm, X_seiz])
    y = np.concatenate([np.zeros(len(X_norm)), np.ones(len(X_seiz))])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=120, max_depth=8, random_state=42, n_jobs=1)),
    ])
    pipe.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    return pipe


def load_or_train_model():
    """Load model.pkl if present, otherwise train a fresh synthetic model."""
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return train_synthetic_model()


def predict(model, features):
    """Predict per-segment seizure probability and aggregate.

    Returns dict with per-segment probs, segment labels, overall label,
    overall confidence, and feature importances.
    """
    if len(features) == 0:
        return {
            "segment_probs": [],
            "segment_labels": [],
            "overall_label": "Unknown",
            "overall_confidence": 0.0,
            "seizure_ratio": 0.0,
            "feature_importance": {},
        }

    probs = model.predict_proba(features)[:, 1]  # P(seizure)
    seg_labels = (probs >= 0.5).astype(int).tolist()
    seizure_ratio = float(np.mean(seg_labels))

    # Overall: seizure if any window crosses a strong threshold or >=20% segments flagged.
    strong_hit = bool(np.any(probs >= 0.7))
    is_seizure = strong_hit or seizure_ratio >= 0.2
    overall_label = "Seizure Detected" if is_seizure else "Normal"
    overall_conf = float(np.max(probs)) if is_seizure else float(1 - np.mean(probs))

    rf = model.named_steps["rf"]
    importances = rf.feature_importances_.tolist()

    return {
        "segment_probs": [float(p) for p in probs],
        "segment_labels": seg_labels,
        "overall_label": overall_label,
        "overall_confidence": overall_conf,
        "seizure_ratio": seizure_ratio,
        "feature_importance": importances,
    }
