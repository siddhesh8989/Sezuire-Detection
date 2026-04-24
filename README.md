# NeuroScan — AI EEG Seizure Detection

A Flask web application that detects seizure activity in EEG recordings
from the **CHB-MIT Scalp EEG Database**. Upload a `.edf` file and the app
preprocesses the signal, extracts statistical and spectral features,
runs a Random Forest classifier, detects epileptiform spikes, and
visualizes everything in a clean medical-style dashboard.

![NeuroScan dashboard](https://via.placeholder.com/900x400?text=NeuroScan+Dashboard)

---

## Features

- **File upload** for CHB-MIT `.edf` recordings (drag & drop)
- **Signal preprocessing**: bandpass filter (0.5–40 Hz), z-score
  normalization, 5-second windowing
- **Feature extraction**: mean, variance, energy, entropy, plus delta /
  theta / alpha / beta / gamma band powers
- **Random Forest classifier** trained at startup on synthetic
  CHB-like windows
- **Seizure probability score** with animated progress bar
- **Per-window analysis** — flags abnormal time ranges
  (e.g. *"Abnormal activity detected between 30s–45s"*)
- **Frequency band visualization** (Chart.js bar chart)
- **Epileptiform spike detection** on the raw waveform with rate-per-minute
- **Interpretation text** explaining what drove the prediction
- **Animated alert box** — green for normal, pulsing red for seizure
- **Login system** with session-based auth (demo accounts included)
- **Multi-page UI**: Home, About, ML Insights — fully responsive

---

## Tech stack

- **Backend**: Python 3.11, Flask
- **Signal processing**: NumPy, SciPy, MNE, pyEDFlib
- **ML**: scikit-learn (Random Forest)
- **Frontend**: HTML, CSS, vanilla JavaScript, Chart.js
- **Storage**: in-memory only — no external database

---

## Project layout

```
flask_app/
├── app.py                    # Flask routes, auth, prediction endpoint
├── modules/
│   ├── eeg_processing.py     # Loading, filtering, segmentation
│   ├── features.py           # Feature extraction (stats + band powers)
│   ├── model.py              # Random Forest training / prediction
│   └── spike_detection.py    # MAD-based spike detector
├── templates/
│   ├── _base.html            # Shared layout (navbar, footer)
│   ├── login.html            # Login page
│   ├── index.html            # Home — upload + results
│   ├── about.html            # Project description
│   └── ml_insights.html      # Model details + metrics
├── static/
│   ├── style.css             # Full stylesheet
│   └── script.js             # Frontend logic + Chart.js charts
└── uploads/                  # Temporary upload directory (auto-cleaned)
```

---

## Run locally

### 1. Prerequisites

- Python **3.11** (3.10+ should also work)
- `pip` and `venv`

### 2. Clone and enter the project

```bash
git clone <your-repo-url> neuroscan
cd neuroscan
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows PowerShell
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install flask numpy scipy scikit-learn mne pyEDFlib
```

### 5. Run the Flask app

```bash
cd flask_app
PORT=5000 python app.py
```

The app will:

1. Train the Random Forest on synthetic data at startup (a few seconds)
2. Start serving on `http://localhost:5000`

### 6. Open the app

Visit **<http://localhost:5000>** in your browser.

You'll be redirected to the login page. Use one of the demo accounts:

| Username | Password   |
|----------|-----------|
| `admin`  | `admin`   |
| `demo`   | `demo123` |

### 7. Upload an EEG file

Download a sample `.edf` file from the
[CHB-MIT database on PhysioNet](https://physionet.org/content/chbmit/1.0.0/)
(for example `chb01_03.edf`) and drag it into the upload area.

---

## Configuration

Optional environment variables:

| Variable         | Default        | Description                          |
|------------------|---------------|--------------------------------------|
| `PORT`           | `5000`        | Port the Flask server binds to       |
| `SESSION_SECRET` | `dev-secret…` | Secret key for Flask session cookies |

For production, **always** set `SESSION_SECRET` to a long random string.

---

## How it works

1. **Upload** — `.edf` file is saved temporarily to `flask_app/uploads/`.
2. **Preprocess** — MNE loads the file, a 4th-order Butterworth bandpass
   (0.5–40 Hz) is applied, channels are z-score normalized, and the
   signal is split into 5-second windows.
3. **Feature extraction** — each window is summarized with statistical
   (mean, variance, energy, entropy, RMS) and spectral (delta, theta,
   alpha, beta, gamma band powers) features.
4. **Classification** — a Random Forest predicts the seizure probability
   for every window. Per-window probabilities are aggregated to an
   overall verdict.
5. **Spike detection** — a robust MAD threshold is applied to the
   first-channel waveform to flag epileptiform sharp transients.
6. **Render** — results, probability progress bar, abnormal time
   ranges, frequency band chart, spike count, waveform with spike
   markers, and feature importance are returned as JSON and drawn with
   Chart.js.

The uploaded file is deleted from disk immediately after processing —
nothing is persisted between requests.

---

## Disclaimer

NeuroScan is a **demonstration / educational project** built around a
classical ML baseline. It is **not** a medical device, has not been
clinically validated, and must not be used for diagnosis or treatment
decisions.
