// NeuroScan frontend logic.
const fileInput = document.getElementById("fileInput");
const dropzone = document.querySelector(".dropzone");
const fileName = document.getElementById("fileName");
const submitBtn = document.getElementById("submitBtn");
const uploadForm = document.getElementById("uploadForm");

const loadingCard = document.getElementById("loadingCard");
const resultsSection = document.getElementById("resultsSection");
const errorCard = document.getElementById("errorCard");
const errorText = document.getElementById("errorText");

const alertBox = document.getElementById("alertBox");
const alertIcon = document.getElementById("alertIcon");
const alertTitle = document.getElementById("alertTitle");
const alertSubtitle = document.getElementById("alertSubtitle");

const resultFile = document.getElementById("resultFile");
const probabilityVal = document.getElementById("probabilityVal");
const probabilityBar = document.getElementById("probabilityBar");
const confidenceVal = document.getElementById("confidenceVal");
const ratioVal = document.getElementById("ratioVal");
const durationVal = document.getElementById("durationVal");
const channelsVal = document.getElementById("channelsVal");
const explanationText = document.getElementById("explanationText");
const segmentsList = document.getElementById("segmentsList");
const modelStatus = document.getElementById("modelStatus");
const spikeCount = document.getElementById("spikeCount");
const spikeRate = document.getElementById("spikeRate");
const spikeNote = document.getElementById("spikeNote");

let waveChart, probChart, featChart, bandChart;

// --- File picker / drag & drop -------------------------------------------------
dropzone.addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") return;
  fileInput.click();
});

["dragover", "dragenter"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    updateFileLabel();
  }
});

fileInput.addEventListener("change", updateFileLabel);

function updateFileLabel() {
  const f = fileInput.files[0];
  if (!f) {
    fileName.textContent = "No file selected";
    submitBtn.disabled = true;
    return;
  }
  const lower = f.name.toLowerCase();
  if (lower.endsWith(".seizures") || lower.endsWith(".edf.seizures")) {
    fileName.textContent =
      `${f.name} — this is the CHB-MIT annotation file. Pick the matching .edf instead.`;
    submitBtn.disabled = true;
    return;
  }
  if (!lower.endsWith(".edf")) {
    fileName.textContent = `${f.name} — only .edf files are supported.`;
    submitBtn.disabled = true;
    return;
  }
  fileName.textContent = `${f.name} (${(f.size / (1024 * 1024)).toFixed(2)} MB)`;
  submitBtn.disabled = false;
}

// --- Submit --------------------------------------------------------------------
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files[0]) return;

  resultsSection.classList.add("hidden");
  errorCard.classList.add("hidden");
  loadingCard.classList.remove("hidden");
  submitBtn.disabled = true;

  const fd = new FormData();
  fd.append("file", fileInput.files[0]);

  try {
    const res = await fetch("/predict", { method: "POST", body: fd });
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); }
    catch (_) {
      throw new Error(
        res.status === 401
          ? "Your session expired — please sign in again."
          : `Server returned status ${res.status}. ${text.slice(0, 200)}`
      );
    }
    if (!res.ok) throw new Error(data.error || `Server error (${res.status})`);
    renderResults(data);
  } catch (err) {
    errorText.textContent = err.message;
    errorCard.classList.remove("hidden");
  } finally {
    loadingCard.classList.add("hidden");
    submitBtn.disabled = false;
  }
});

// --- Results rendering ---------------------------------------------------------
function renderResults(data) {
  resultsSection.classList.remove("hidden");

  const isSeizure = data.prediction === "Seizure Detected";

  // Alert box
  alertBox.className = "alert-box " + (isSeizure ? "seizure" : "normal");
  alertIcon.textContent = isSeizure ? "!" : "✓";
  alertTitle.textContent = isSeizure ? "Seizure Detected" : "Normal";
  alertSubtitle.textContent = isSeizure
    ? "The model flagged abnormal EEG activity. Review highlighted segments below."
    : "EEG patterns are within typical resting ranges.";

  // Status pill in header
  modelStatus.innerHTML = isSeizure
    ? '<span class="dot dot-red"></span> Abnormal pattern detected'
    : '<span class="dot dot-green"></span> Model ready';

  // Probability + progress bar
  const prob = data.seizure_probability;
  probabilityVal.textContent = `${prob.toFixed(1)}%`;
  // Animate from 0 to target.
  probabilityBar.style.width = "0%";
  requestAnimationFrame(() => {
    probabilityBar.style.width = `${prob}%`;
  });
  probabilityBar.classList.toggle("high", prob >= 50);

  // Metrics
  resultFile.textContent = `${data.filename} · ${data.n_segments} windows analyzed`;
  confidenceVal.textContent = `${data.confidence}%`;
  ratioVal.textContent = `${data.seizure_ratio}%`;
  durationVal.textContent = `${data.duration_sec}s`;
  channelsVal.textContent = data.n_channels;
  explanationText.textContent = data.explanation;

  // Abnormal segments list
  renderSegments(data.abnormal_ranges);

  // Spike detection card
  renderSpikes(data.spikes, data.duration_sec);

  // Charts
  drawWaveform(data.preview, data.preview_duration_sec ?? data.duration_sec, data.spikes);
  drawProbabilities(data.segment_probs, data.window_times);
  drawBandPowers(data.band_powers);
  drawFeatureImportance(data.feature_importance);

  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSegments(ranges) {
  segmentsList.innerHTML = "";
  if (!ranges || ranges.length === 0) {
    segmentsList.classList.add("empty");
    const li = document.createElement("li");
    li.textContent = "No abnormal segments detected.";
    segmentsList.appendChild(li);
    return;
  }
  segmentsList.classList.remove("empty");
  ranges.forEach((r) => {
    const li = document.createElement("li");
    li.textContent = `Abnormal activity detected between ${r.start.toFixed(0)}s – ${r.end.toFixed(0)}s`;
    segmentsList.appendChild(li);
  });
}

function destroy(chart) { if (chart) chart.destroy(); }

function renderSpikes(spikes, duration) {
  if (!spikes) return;
  spikeCount.textContent = spikes.count;
  spikeRate.textContent = spikes.rate_per_min.toFixed(1);
  let severity = "Low — within normal background activity.";
  if (spikes.rate_per_min >= 30) severity = "High — frequent epileptiform discharges.";
  else if (spikes.rate_per_min >= 10) severity = "Moderate — elevated transient activity.";
  spikeNote.textContent =
    `Detected ${spikes.count} sharp transient${spikes.count === 1 ? "" : "s"} ` +
    `over ${duration.toFixed(1)}s of EEG (${spikes.rate_per_min.toFixed(1)} / min). ${severity}`;
}

function drawWaveform(preview, duration, spikes) {
  destroy(waveChart);
  const ctx = document.getElementById("waveChart").getContext("2d");
  const n = preview.length;
  const dt = duration / Math.max(n, 1);
  const xs = preview.map((_, i) => +(i * dt).toFixed(3));

  const datasets = [{
    label: "Channel 1 (μV)",
    data: preview.map((y, i) => ({ x: xs[i], y })),
    borderColor: "#1f6feb",
    backgroundColor: "rgba(31,111,235,0.08)",
    borderWidth: 1.2,
    pointRadius: 0,
    tension: 0.2,
    fill: true,
    showLine: true,
  }];

  if (spikes && spikes.times_sec && spikes.times_sec.length) {
    const spikePoints = spikes.times_sec.map((t) => {
      const idx = Math.min(n - 1, Math.max(0, Math.round(t / dt)));
      return { x: t, y: preview[idx] };
    });
    datasets.push({
      label: "Spike",
      type: "scatter",
      data: spikePoints,
      borderColor: "#d83a52",
      backgroundColor: "#d83a52",
      pointRadius: 4,
      pointHoverRadius: 6,
      showLine: false,
    });
  }

  waveChart = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: "bottom", labels: { color: "#6b7a90" } },
        tooltip: { callbacks: { title: (items) => `t = ${items[0].parsed.x.toFixed(2)}s` } },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Time (s)", color: "#6b7a90" },
          ticks: { color: "#6b7a90" }, grid: { color: "#eef2f8" },
        },
        y: { ticks: { color: "#6b7a90" }, grid: { color: "#eef2f8" } },
      },
    },
  });
}

function drawProbabilities(probs, times) {
  destroy(probChart);
  const ctx = document.getElementById("probChart").getContext("2d");
  const labels = times && times.length === probs.length
    ? times.map((t) => `${Math.round(t)}s`)
    : probs.map((_, i) => `W${i + 1}`);
  probChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "P(seizure)",
        data: probs,
        backgroundColor: probs.map((p) =>
          p >= 0.5 ? "rgba(216,58,82,0.85)" : "rgba(31,111,235,0.7)"
        ),
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: {
          title: (items) => `Window @ ${labels[items[0].dataIndex]}`,
          label: (item) => `P(seizure) = ${item.parsed.y.toFixed(3)}`,
        } },
      },
      scales: {
        y: { min: 0, max: 1, ticks: { color: "#6b7a90" }, grid: { color: "#eef2f8" } },
        x: {
          ticks: { color: "#6b7a90", maxRotation: 0, autoSkip: true, maxTicksLimit: 12 },
          grid: { display: false },
          title: { display: true, text: "Time in recording", color: "#6b7a90" },
        },
      },
    },
  });
}

function drawBandPowers(bands) {
  destroy(bandChart);
  const ctx = document.getElementById("bandChart").getContext("2d");
  const order = ["delta", "theta", "alpha", "beta", "gamma"];
  const labels = order.map((b) => b.charAt(0).toUpperCase() + b.slice(1));
  const values = order.map((b) => bands[b] ?? 0);
  bandChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Avg Power",
        data: values,
        backgroundColor: ["#1f6feb", "#4d9bff", "#1aa971", "#e89726", "#d83a52"],
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { ticks: { color: "#6b7a90" }, grid: { color: "#eef2f8" } },
        x: { ticks: { color: "#6b7a90" }, grid: { display: false } },
      },
    },
  });
}

function drawFeatureImportance(importance) {
  destroy(featChart);
  const entries = Object.entries(importance).sort((a, b) => b[1] - a[1]);
  const ctx = document.getElementById("featChart").getContext("2d");
  featChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: entries.map(([k]) => k),
      datasets: [{
        data: entries.map(([, v]) => v),
        backgroundColor: "rgba(26,169,113,0.75)",
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#6b7a90" }, grid: { color: "#eef2f8" } },
        y: { ticks: { color: "#6b7a90" }, grid: { display: false } },
      },
    },
  });
}
