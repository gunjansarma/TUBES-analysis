"""
Sensor Stream Monitor
----------------------
Streaming GUI for continuously-logged sensor data (tab-delimited .txt file,
same layout as the original acquisition script: 2 header rows, then
Time / SPEED / POWER / FREQUENCY / VOLTAGE / CURRENT1 / CURRENT2 / Column8 / Column9).

Design notes (read before changing behaviour):
- Data is read by BYTE OFFSET, not by re-reading the whole file, so this stays
  fast no matter how large the log file grows.
- Only COMPLETE lines are parsed. If the acquisition process is mid-write on the
  last line, that partial line is left untouched and picked up on the next poll.
  This avoids the classic "half a row = NaN row" bug you get from naive tailing.
- Windows are cut on the Time column (0-300, 300-600, ...), not on row count,
  so it doesn't matter if the true sample rate drifts slightly from the
  nominal 1000 Hz.
- No Excel round-trip. Data lives in memory (pandas) between polls. If you
  want to persist a window, use the "Download this window as CSV" button.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import gaussian_kde, kurtosis, skew

COLUMNS = [
    "Time",
    "SPEED",
    "POWER",
    "FREQUENCY",
    "VOLTAGE",
    "CURRENT1",
    "CURRENT2",
    "Column8",
    "Column9",
]
ANALYZABLE_CHANNELS = ["SPEED", "POWER", "FREQUENCY", "VOLTAGE", "CURRENT1", "CURRENT2"]

MAX_TIMESERIES_POINTS = 5000  # downsample for plotting only, not for stats/FFT
MAX_DIST_POINTS = 20000  # KDE gets slow well before 300k points


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def read_new_rows(path: str, offset: int) -> tuple[pd.DataFrame, int]:
    """Read only newly-appended, COMPLETE lines from `path` starting at byte `offset`.
    Returns (new_rows_df, new_offset). Leaves a trailing partial line unread."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS), offset

    size = os.path.getsize(path)
    if size <= offset:
        return pd.DataFrame(columns=COLUMNS), offset

    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read()

    last_newline = raw.rfind(b"\n")
    if last_newline == -1:
        return pd.DataFrame(columns=COLUMNS), offset  # no complete line yet

    complete_raw = raw[: last_newline + 1]
    new_offset = offset + len(complete_raw)
    text = complete_raw.decode(errors="replace")
    lines = text.splitlines()

    if offset == 0:
        lines = lines[2:]  # the two header rows, only present at the very start

    if not lines:
        return pd.DataFrame(columns=COLUMNS), new_offset

    try:
        new_df = pd.read_csv(
            StringIO("\n".join(lines)), sep="\t", header=None, engine="python"
        )
    except Exception:
        return pd.DataFrame(columns=COLUMNS), new_offset

    new_df.columns = COLUMNS[: new_df.shape[1]]
    new_df = new_df.apply(pd.to_numeric, errors="coerce")
    return new_df, new_offset


# ---------------------------------------------------------------------------
# Metrics / plots (same statistics as the original script)
# ---------------------------------------------------------------------------
def compute_metrics(data: pd.Series) -> dict:
    data = data.dropna()
    data = data[np.isfinite(data)]
    mean = data.mean()
    return {
        "N": len(data),
        "Mean": mean,
        "Median": data.median(),
        "Std Dev": data.std(),
        "Variance": data.var(),
        "Skewness": skew(data) if len(data) > 2 else np.nan,
        "Kurtosis": kurtosis(data) if len(data) > 2 else np.nan,
        "CV (%)": (data.std() / mean * 100) if mean else np.nan,
    }


def plot_timeseries(df: pd.DataFrame, col: str) -> go.Figure:
    d = df[["Time", col]].dropna()
    if len(d) > MAX_TIMESERIES_POINTS:
        step = len(d) // MAX_TIMESERIES_POINTS
        d = d.iloc[::step]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["Time"],
            y=d[col],
            mode="lines",
            line=dict(color="#3366cc", width=1.5),
            name=col,
        )
    )
    fig.update_layout(
        title=f"{col} — Time Series",
        xaxis_title="Time (s)",
        yaxis_title=col,
        height=340,
        margin=dict(t=40, b=30),
    )
    return fig


def compute_fft(values: np.ndarray, sampling_rate: float):
    values = values - np.mean(values)
    n = len(values)
    yf = fft(values)
    xf = fftfreq(n, 1 / sampling_rate)
    pos = xf[: n // 2]
    mag = np.abs(yf[: n // 2])
    return pos, mag


def get_fft_band(
    df: pd.DataFrame, col: str, sampling_rate: float, low: float, high: float
):
    """FFT of the channel, masked to [low, high] Hz — the exact data the FFT plot shows,
    reused for the diagnostics below it so the FFT isn't computed twice."""
    values = df[col].dropna().values
    freqs, mags = compute_fft(values, sampling_rate)
    mask = (freqs >= low) & (freqs <= high)
    return freqs[mask], mags[mask]


def compute_fft_for_channel(df: pd.DataFrame, col: str, sampling_rate: float):
    """Raw (unmasked) FFT of the channel, so multiple bands (main plot band, shape-analysis
    band) can be sliced from it via mask_band() without recomputing the FFT each time.
    """
    values = df[col].dropna().values
    return compute_fft(values, sampling_rate)


def mask_band(freqs: np.ndarray, mags: np.ndarray, low: float, high: float):
    """Re-mask an already-computed FFT to a different band, without recomputing the FFT.
    Used so the shape-classification band (e.g. 100-120 Hz) can differ from the main
    FFT-plot band (e.g. 100-150 Hz) without doubling the FFT cost per render."""
    mask = (freqs >= low) & (freqs <= high)
    return freqs[mask], mags[mask]


def plot_fft(
    freqs: np.ndarray, mags: np.ndarray, col: str, low: float, high: float
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=freqs, y=mags, mode="lines", line=dict(color="#cc3366", width=1.5))
    )
    fig.update_layout(
        title=f"FFT of {col} ({low:.0f}–{high:.0f} Hz)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        height=340,
        margin=dict(t=40, b=30),
    )
    return fig


def compute_fft_diagnostics(
    freqs: np.ndarray, mags: np.ndarray, peak_prominence_frac: float
) -> dict:
    """Spectral entropy, peak-to-noise ratio, envelope smoothness, and harmonic spacing
    variance — all computed on the same masked band the FFT plot shows."""
    result = {
        "Spectral Entropy": np.nan,
        "Peak-to-Noise Ratio (dB)": np.nan,
        "Envelope Smoothness": np.nan,
        "Harmonic Spacing Var. (Hz²)": np.nan,
    }
    if len(mags) < 4 or not np.any(mags > 0):
        return result

    # Spectral entropy: normalized Shannon entropy of the power distribution.
    power = mags**2
    total = power.sum()
    if total > 0:
        p = power / total
        p_nonzero = p[p > 0]
        entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
        result["Spectral Entropy"] = float(entropy / np.log2(len(mags)))

    # Peak-to-noise ratio: strongest bin vs. median of the rest of the band.
    peak = mags.max()
    noise_floor = np.median(mags)
    if noise_floor > 0:
        result["Peak-to-Noise Ratio (dB)"] = float(20 * np.log10(peak / noise_floor))

    # Envelope smoothness: fraction of the spectral shape explained by a smooth trend.
    n = len(mags)
    window = min(21, n if n % 2 == 1 else n - 1)
    if window >= 5 and mags.std() > 0:
        trend = savgol_filter(mags, window_length=window, polyorder=3)
        residual = mags - trend
        result["Envelope Smoothness"] = float(
            np.clip(1 - residual.std() / mags.std(), 0, 1)
        )

    # Harmonic spacing variance: variance of the frequency gaps between detected peaks.
    if peak > 0:
        peak_idx, _ = find_peaks(mags, prominence=peak_prominence_frac * peak)
        if len(peak_idx) >= 3:
            peak_freqs = np.sort(freqs[peak_idx])
            result["Harmonic Spacing Var. (Hz²)"] = float(np.var(np.diff(peak_freqs)))

    return result


def bin_peaks_by_frequency(freqs: np.ndarray, mags: np.ndarray, bin_hz: float):
    """Downsample the full-window FFT (already computed over the whole 300s, unchanged)
    into bin_hz-wide frequency bins, taking the peak magnitude within each bin. Since
    bins are ordered strictly by frequency, the resulting (freq, mag) points connect
    into a well-posed function of frequency — unlike time-binning, there's no risk of
    the x-axis doubling back on itself."""
    if len(freqs) == 0:
        return np.array([]), np.array([])
    low, high = freqs.min(), freqs.max()
    n_bins = max(1, int(np.ceil((high - low) / bin_hz)))
    bin_freqs, bin_mags = [], []
    for i in range(n_bins):
        b_lo = low + i * bin_hz
        b_hi = b_lo + bin_hz
        mask = (freqs >= b_lo) & (freqs < b_hi)
        if not np.any(mask):
            continue
        local_mags = mags[mask]
        local_freqs = freqs[mask]
        peak_i = np.argmax(local_mags)
        bin_freqs.append(local_freqs[peak_i])
        bin_mags.append(local_mags[peak_i])
    return np.array(bin_freqs), np.array(bin_mags)


def plot_peak_envelope_on_fft(
    base_freqs, base_mags, peak_freqs, peak_mags, col, low, high, bin_hz
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=base_freqs,
            y=base_mags,
            mode="lines",
            line=dict(color="rgba(204,51,102,0.25)", width=1),
            name="Full-window FFT (300s, unbinned)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=peak_freqs,
            y=peak_mags,
            mode="lines+markers",
            line=dict(color="#cc3366", width=2),
            marker=dict(size=6),
            name=f"Peak per {bin_hz:g} Hz bin, connected",
        )
    )
    fig.update_layout(
        title=f"FFT of {col} ({low:.0f}–{high:.0f} Hz) — {bin_hz:g} Hz peak envelope",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        height=380,
        margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def classify_curve_shape(x: np.ndarray, y: np.ndarray, r2_threshold: float) -> dict:
    """Fits the connected peak-envelope points (magnitude vs. frequency) with a quadratic
    and reports whether it looks parabolic vs. shows no clear shape, gated on R²-quadratic
    alone. This is a heuristic shape score you specified, not a validated weld-quality
    classifier — see UI caption."""
    result = {"r2_quadratic": np.nan, "concavity": None, "label": "Not enough data"}
    if len(x) < 5:
        return result

    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0:
        result["label"] = "Flat (no variation)"
        return result

    quad_coeffs = np.polyfit(x, y, 2)
    quad_pred = np.polyval(quad_coeffs, x)

    r2_quad = 1 - np.sum((y - quad_pred) ** 2) / ss_tot
    result["r2_quadratic"] = float(r2_quad)
    result["concavity"] = (
        "downward (rise then fall)" if quad_coeffs[0] < 0 else "upward (dip then rise)"
    )

    result["label"] = "Parabolic" if r2_quad >= r2_threshold else "No clear shape"
    return result


def plot_distribution(df: pd.DataFrame, col: str) -> go.Figure:
    data = df[col].dropna()
    data = data[np.isfinite(data)]
    if len(data) > MAX_DIST_POINTS:
        data = data.sample(MAX_DIST_POINTS, random_state=0)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=data,
            histnorm="probability density",
            marker_color="#2ca02c",
            opacity=0.5,
            name="Histogram",
        )
    )

    if len(data) > 2 and data.std() > 0:
        kde = gaussian_kde(data)
        x_grid = np.linspace(data.min(), data.max(), 200)
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=kde(x_grid),
                mode="lines",
                line=dict(color="#1b7a1b", width=2),
                name="KDE",
            )
        )

    mean, median = data.mean(), data.median()
    fig.add_vline(
        x=mean,
        line_color="black",
        annotation_text=f"Mean={mean:.3f}",
        annotation_position="top",
    )
    fig.add_vline(
        x=median,
        line_dash="dash",
        line_color="purple",
        annotation_text=f"Median={median:.3f}",
        annotation_position="bottom",
    )

    fig.update_layout(
        title=f"Distribution of {col}",
        height=340,
        margin=dict(t=40, b=30),
        showlegend=False,
        xaxis_title=col,
        yaxis_title="Density",
    )
    return fig


def export_plot_png(
    fig: go.Figure,
    folder: str,
    channel: str,
    window_start: float,
    window_end: float,
    plot_name: str,
) -> tuple[bool, str]:
    """Write one Plotly figure to <folder> as a PNG. Filename = real-world export time
    (what you asked for) + the window's elapsed-second range (so exports from different
    windows don't collide or look identical — the Time column itself has no real calendar
    timestamp, only elapsed seconds, so this is the closest stand-in).
    Returns (success, full_path_or_error_message)."""
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as e:
        return False, f"Could not create/access folder '{folder}': {e}"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = (
        f"{timestamp}_{channel}_{window_start:.0f}-{window_end:.0f}s_{plot_name}.png"
    )
    full_path = os.path.join(folder, filename)
    try:
        fig.write_image(full_path, format="png", width=1200, height=700, scale=2)
        return True, full_path
    except Exception as e:
        return False, (
            f"PNG export failed ({e}). This usually means the 'kaleido' package "
            f"isn't installed — run: pip install kaleido"
        )


def segment_into_windows(df: pd.DataFrame, window_seconds: float):
    """Split a full DataFrame into sequential [start, end) windows by Time.
    Used for manually-supplied (already-complete) data, as opposed to the
    live stream which advances one window at a time as data arrives."""
    if df.empty:
        return []
    max_t = df["Time"].max()
    n_windows = int(max_t // window_seconds) + 1
    windows = []
    for i in range(n_windows):
        start = i * window_seconds
        end = start + window_seconds
        wdf = df[(df["Time"] >= start) & (df["Time"] < end)]
        if not wdf.empty:
            windows.append((start, end, wdf))
    return windows


def parse_tabular_text(text: str, has_header_rows: bool) -> pd.DataFrame:
    """Parse pasted, uploaded, or link-fetched tab-delimited text into the standard schema."""
    lines = text.splitlines()
    if has_header_rows:
        lines = lines[2:]
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(StringIO("\n".join(lines)), sep="\t", header=None, engine="python")
    df.columns = COLUMNS[: df.shape[1]]
    return df.apply(pd.to_numeric, errors="coerce")


def fetch_url_text(url: str, timeout: float = 15.0) -> tuple[str, str]:
    """Plain HTTP(S) GET, no auth, no schema assumptions beyond 'returns text'.
    Returns (text, error) — exactly one of which is non-empty/non-None.
    This is deliberately generic: it knows nothing about iba, gRPC, or any
    specific protocol — if your link needs a proprietary client, this won't
    speak to it, it'll just fail with a clear error."""
    if not url or not url.strip():
        return "", "No URL provided."
    if not (url.startswith("http://") or url.startswith("https://")):
        return "", "URL must start with http:// or https://"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sensor-gui/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            encoding = resp.headers.get_content_charset() or "utf-8"
        return raw.decode(encoding, errors="replace"), ""
    except urllib.error.HTTPError as e:
        return "", f"Server returned HTTP {e.code} ({e.reason})."
    except urllib.error.URLError as e:
        return "", f"Could not reach that URL: {e.reason}"
    except TimeoutError:
        return "", f"Request timed out after {timeout:g}s."
    except Exception as e:
        return "", f"Fetch failed: {e}"


def read_new_rows_from_url(
    url: str, has_header_rows: bool, last_max_time: float
) -> tuple[pd.DataFrame, float, str]:
    """Poll `url` once, parse the response, and return only rows with Time > last_max_time.
    Works whether the endpoint returns the full growing dataset on every call or only the
    newest chunk — either way nothing gets double-counted, since the watermark is the data's
    own Time column rather than a byte offset (which HTTP has no equivalent of).
    Returns (new_rows_df, updated_max_time, error_message)."""
    text, err = fetch_url_text(url)
    if err:
        return pd.DataFrame(columns=COLUMNS), last_max_time, err

    parsed = parse_tabular_text(text, has_header_rows)
    if parsed.empty or "Time" not in parsed.columns:
        return pd.DataFrame(columns=COLUMNS), last_max_time, ""

    parsed = parsed.dropna(subset=["Time"])
    new_rows = parsed[parsed["Time"] > last_max_time]
    if new_rows.empty:
        return pd.DataFrame(columns=COLUMNS), last_max_time, ""

    updated_max_time = max(last_max_time, new_rows["Time"].max())
    return new_rows, updated_max_time, ""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sensor Stream Monitor", layout="wide")

st.sidebar.header("Configuration")
live_source_mode = st.sidebar.radio(
    "Live data source", ["Local file", "URL (polled)"], horizontal=True, index=1
)
if live_source_mode == "Local file":
    file_path = st.sidebar.text_input(
        "Log file path",
        value="/content/drive/MyDrive/Tube-making_analysis/09_57_37.txt",
    )
    live_url = None
    live_url_has_header = False
else:
    file_path = None
    live_url = st.sidebar.text_input(
        "Live data URL", placeholder="https://example.com/latest-data"
    )
    live_url_has_header = st.sidebar.checkbox(
        "Response starts with the 2 header rows", value=False
    )
    st.sidebar.caption(
        "Polled every 'Poll interval' below. Only rows with Time greater than what's "
        "already been ingested are kept — safe whether the endpoint returns the full "
        "dataset or just the newest chunk each time."
    )
sampling_rate = st.sidebar.number_input("Sampling rate (Hz)", value=1000, min_value=1)
window_seconds = st.sidebar.number_input("Window length (s)", value=300, min_value=1)
fft_low = st.sidebar.number_input("FFT band low (Hz)", value=100.0)
fft_high = st.sidebar.number_input("FFT band high (Hz)", value=150.0)
peak_bin_hz = st.sidebar.number_input(
    "Peak-envelope bin width (Hz)", value=1.0, min_value=0.1, step=0.5
)
shape_r2_threshold = st.sidebar.slider(
    "Parabola shape fit threshold (R²)", 0.30, 0.95, 0.70, step=0.05
)
st.sidebar.caption("Shape classification band (independent of the FFT band above):")
shape_band_low = st.sidebar.number_input("Shape band low (Hz)", value=100.0)
shape_band_high = st.sidebar.number_input("Shape band high (Hz)", value=120.0)
require_inverted = st.sidebar.checkbox(
    "Only count downward/inverted parabola as 'good weld'", value=True
)
peak_prominence_pct = st.sidebar.slider(
    "Harmonic peak prominence (% of peak)", 1, 50, 10
)
poll_seconds = st.sidebar.number_input("Poll interval (s)", value=3, min_value=1)
auto_refresh = st.sidebar.checkbox("Auto-poll for new data", value=True)
channel = st.sidebar.selectbox(
    "Channel to analyze", ANALYZABLE_CHANNELS, index=1
)  # default POWER

st.sidebar.divider()
st.sidebar.subheader("Plot Export")
export_enabled = st.sidebar.checkbox(
    "Auto-export plots when a new live window completes", value=False
)
export_folder = st.sidebar.text_input(
    "Export folder path",
    placeholder=r"C:\Exports\SensorPlots or /home/user/exports",
    disabled=not export_enabled,
)
st.sidebar.caption(
    "A local folder on the machine running this app — not a cloud link. Saves 4 PNGs "
    "(time series, FFT, distribution, peak-envelope/shape) each time a live-stream window "
    "finishes, filenamed by real export time + the window's elapsed-second range. "
    "Manual Data Entry is not auto-exported."
)
if export_enabled and not export_folder.strip():
    st.sidebar.warning("Enter a folder path above, or exports will be skipped.")

reset_label = (
    "Reset stream (re-read file from byte 0)"
    if live_source_mode == "Local file"
    else "Reset stream (re-fetch URL from scratch)"
)
if st.sidebar.button(reset_label):
    st.session_state.byte_offset = 0
    st.session_state.url_last_max_time = -1.0
    st.session_state.buffer = pd.DataFrame(columns=COLUMNS)
    st.session_state.window_start = 0.0
    st.session_state.current_window = None

for key, default in [
    ("byte_offset", 0),
    ("url_last_max_time", -1.0),
    ("buffer", pd.DataFrame(columns=COLUMNS)),
    ("window_start", 0.0),
    ("current_window", None),
    ("manual_df", pd.DataFrame(columns=COLUMNS)),
    ("last_upload_sig", None),
    ("last_export_results", None),
    ("last_export_time", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- ingest whatever is new since last poll ---
live_fetch_error = ""
if live_source_mode == "Local file":
    new_df, st.session_state.byte_offset = read_new_rows(
        file_path, st.session_state.byte_offset
    )
else:
    new_df, st.session_state.url_last_max_time, live_fetch_error = (
        read_new_rows_from_url(
            live_url, live_url_has_header, st.session_state.url_last_max_time
        )
    )
if not new_df.empty:
    st.session_state.buffer = pd.concat(
        [st.session_state.buffer, new_df], ignore_index=True
    )

# --- advance the window if a full window_seconds span is now available ---
window_end = st.session_state.window_start + window_seconds
buf = st.session_state.buffer
window_ready = (not buf.empty) and (buf["Time"].max() >= window_end)

if window_ready:
    mask = (buf["Time"] >= st.session_state.window_start) & (buf["Time"] < window_end)
    st.session_state.current_window = buf.loc[mask].copy()
    st.session_state.buffer = buf.loc[buf["Time"] >= window_end].copy()

    # --- auto-export, exactly once per completed window (this block only runs on a
    # window transition, not on every poll) — Manual Data Entry is untouched by design ---
    if export_enabled and export_folder.strip():
        w_export = st.session_state.current_window
        w_start, w_end = w_export["Time"].min(), w_export["Time"].max()
        export_results = []

        ts_fig = plot_timeseries(w_export, channel)
        export_results.append(
            (
                "Time series",
                *export_plot_png(
                    ts_fig, export_folder, channel, w_start, w_end, "timeseries"
                ),
            )
        )

        raw_f, raw_m = compute_fft_for_channel(w_export, channel, sampling_rate)
        fft_f, fft_m = mask_band(raw_f, raw_m, fft_low, fft_high)
        fft_fig = plot_fft(fft_f, fft_m, channel, fft_low, fft_high)
        export_results.append(
            (
                "FFT",
                *export_plot_png(
                    fft_fig, export_folder, channel, w_start, w_end, "fft"
                ),
            )
        )

        dist_fig = plot_distribution(w_export, channel)
        export_results.append(
            (
                "Distribution",
                *export_plot_png(
                    dist_fig, export_folder, channel, w_start, w_end, "distribution"
                ),
            )
        )

        shape_f, shape_m = mask_band(raw_f, raw_m, shape_band_low, shape_band_high)
        peak_f, peak_m = bin_peaks_by_frequency(shape_f, shape_m, peak_bin_hz)
        if len(peak_f) >= 5:
            shape_fig = plot_peak_envelope_on_fft(
                shape_f,
                shape_m,
                peak_f,
                peak_m,
                channel,
                shape_band_low,
                shape_band_high,
                peak_bin_hz,
            )
            export_results.append(
                (
                    "Peak-envelope/shape",
                    *export_plot_png(
                        shape_fig, export_folder, channel, w_start, w_end, "shape"
                    ),
                )
            )
        else:
            export_results.append(
                (
                    "Peak-envelope/shape",
                    False,
                    "Skipped — not enough bins in the shape band for this window",
                )
            )

        st.session_state.last_export_results = export_results
        st.session_state.last_export_time = datetime.now()

    st.session_state.window_start = window_end

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
st.title("Sensor Stream Monitor")
st.caption(
    f"Channel: **{channel}** · Window: **{window_seconds}s** · Nominal sampling: **{sampling_rate} Hz**"
)

if st.session_state.last_export_results is not None:
    n_ok = sum(1 for _, ok, _ in st.session_state.last_export_results if ok)
    n_total = len(st.session_state.last_export_results)
    label = f"Last auto-export — {st.session_state.last_export_time.strftime('%Y-%m-%d %H:%M:%S')} ({n_ok}/{n_total} succeeded)"
    with st.expander(label, expanded=(n_ok < n_total)):
        for name, ok, msg in st.session_state.last_export_results:
            st.caption(f"{'✅' if ok else '❌'} {name} → {msg}")

w = st.session_state.current_window
if w is not None and not w.empty:
    st.subheader(
        f"Window {w['Time'].min():.1f}s – {w['Time'].max():.1f}s  ({len(w):,} rows)"
    )

    st.plotly_chart(plot_timeseries(w, channel), use_container_width=True)

    raw_freqs, raw_mags = compute_fft_for_channel(w, channel, sampling_rate)
    fft_freqs, fft_mags = mask_band(raw_freqs, raw_mags, fft_low, fft_high)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_fft(fft_freqs, fft_mags, channel, fft_low, fft_high),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(plot_distribution(w, channel), use_container_width=True)

    st.subheader("FFT Diagnostics")
    fft_diag = compute_fft_diagnostics(fft_freqs, fft_mags, peak_prominence_pct / 100)
    fcols = st.columns(4)
    for i, (k, v) in enumerate(fft_diag.items()):
        display = "N/A" if (v is None or np.isnan(v)) else f"{v:,.4f}"
        fcols[i % 4].metric(k, display)

    st.subheader("Peak Envelope & Shape Classification")
    st.caption(
        f"Scoped to {shape_band_low:g}–{shape_band_high:g} Hz — independent of the {fft_low:g}–{fft_high:g} Hz FFT band above."
    )
    shape_freqs, shape_mags = mask_band(
        raw_freqs, raw_mags, shape_band_low, shape_band_high
    )
    peak_freqs, peak_mags = bin_peaks_by_frequency(shape_freqs, shape_mags, peak_bin_hz)
    if len(peak_freqs) < 5:
        st.info(
            "Not enough bins in the shape band for analysis — widen the shape band or shrink the bin width."
        )
    else:
        st.plotly_chart(
            plot_peak_envelope_on_fft(
                shape_freqs,
                shape_mags,
                peak_freqs,
                peak_mags,
                channel,
                shape_band_low,
                shape_band_high,
                peak_bin_hz,
            ),
            use_container_width=True,
        )
        shape = classify_curve_shape(peak_freqs, peak_mags, shape_r2_threshold)
        is_good = shape["label"] == "Parabolic" and (
            not require_inverted or (shape["concavity"] or "").startswith("downward")
        )
        scols = st.columns(3)
        scols[0].metric(
            "R² (quadratic fit)",
            (
                f"{shape['r2_quadratic']:.3f}"
                if not np.isnan(shape["r2_quadratic"])
                else "N/A"
            ),
        )
        scols[1].metric("Concavity", shape["concavity"] or "N/A")
        scols[2].metric("Shape", "Inverted parabola" if is_good else shape["label"])
        if is_good:
            st.success(
                f"Shape: inverted parabola ({shape['concavity']}) → heuristic reads as **possible good weld**."
            )
        elif shape["label"] == "No clear shape":
            st.warning(
                "Shape: no clear fit → heuristic reads as **possible bad weld**."
            )
        elif shape["label"] == "Parabolic":
            st.info(
                f"Shape: parabolic, but {shape['concavity']} — not inverted, so it doesn't satisfy the good-weld rule as specified."
            )
        else:
            st.info(f"Shape: {shape['label']} — doesn't match the good-weld rule.")
        st.caption(
            "⚠️ Inverted-parabola-equals-good-weld is a heuristic rule you specified, not something "
            "validated against known weld outcomes here. Check it against a batch of welds with "
            "**confirmed** good/bad results before using this as an actual accept/reject gate."
        )

    st.subheader("Time-Series Metrics")
    metrics = compute_metrics(w[channel])
    cols = st.columns(4)
    for i, (k, v) in enumerate(metrics.items()):
        display = f"{v:,.4f}" if isinstance(v, (float, np.floating)) else f"{v}"
        cols[i % 4].metric(k, display)

    st.download_button(
        "Download this window as CSV",
        data=w.to_csv(index=False).encode(),
        file_name=f"window_{w['Time'].min():.0f}_{w['Time'].max():.0f}s.csv",
        mime="text/csv",
    )
else:
    buffered = len(st.session_state.buffer)
    latest_t = st.session_state.buffer["Time"].max() if buffered else 0.0
    st.info(
        f"Waiting for the first complete {window_seconds}s window — "
        f"{buffered:,} rows buffered so far, latest Time = {latest_t:.1f}s "
        f"({window_end - latest_t:.1f}s of data still needed)."
    )

if live_fetch_error:
    st.error(
        f"Last poll of the URL failed: {live_fetch_error} (will retry next poll in {poll_seconds}s)"
    )

if live_source_mode == "Local file":
    st.caption(
        f"File offset read so far: {st.session_state.byte_offset:,} bytes · polling every {poll_seconds}s"
    )
else:
    watermark = st.session_state.url_last_max_time
    watermark_display = "none yet" if watermark < 0 else f"{watermark:.3f}s"
    st.caption(
        f"Latest Time ingested from URL: {watermark_display} · polling every {poll_seconds}s"
    )

st.divider()

# ---------------------------------------------------------------------------
# Manual data entry
# ---------------------------------------------------------------------------
st.subheader("Manual Data Entry")
st.caption(
    "Upload, paste, or fetch-from-link data outside the live stream — same columns, "
    "tab-separated. It's processed through the identical pipeline (windowing, FFT, "
    "distribution, metrics, peak-envelope shape classification), but browsed by window "
    "instead of advancing automatically, since the whole batch is available at once. "
    "If you're actively typing in a text box below, turn off 'Auto-poll for new data' "
    "in the sidebar first — a background refresh mid-edit will reset the box."
)

mc1, mc2, mc3 = st.columns(3)

with mc1:
    uploaded = st.file_uploader(
        "Upload a file (.txt/.csv, same format as the log file)", type=["txt", "csv"]
    )
    has_header = st.checkbox(
        "File starts with the 2 header rows (same as the log format)", value=True
    )
    if uploaded is not None:
        sig = (uploaded.name, uploaded.size)
        if sig != st.session_state.last_upload_sig:
            text = uploaded.getvalue().decode(errors="replace")
            parsed = parse_tabular_text(text, has_header)
            if not parsed.empty:
                st.session_state.manual_df = pd.concat(
                    [st.session_state.manual_df, parsed], ignore_index=True
                )
                st.session_state.last_upload_sig = sig
                st.success(f"Added {len(parsed):,} rows from {uploaded.name}.")

with mc2:
    pasted = st.text_area(
        "...or paste rows directly (tab-separated, no header — just data rows)",
        height=120,
        placeholder="0.000\t1500.2\t60.1\t50.0\t230.5\t10.1\t10.0\t0\t0",
    )
    if st.button("Add pasted rows"):
        if pasted.strip():
            parsed = parse_tabular_text(pasted, has_header_rows=False)
            if not parsed.empty:
                st.session_state.manual_df = pd.concat(
                    [st.session_state.manual_df, parsed], ignore_index=True
                )
                st.success(f"Added {len(parsed):,} rows from pasted text.")
        else:
            st.warning("Nothing to add — the box is empty.")

with mc3:
    link_url = st.text_input(
        "...or fetch from a link (HTTP/HTTPS, returns the same tab-separated data)",
        placeholder="https://example.com/latest-data",
    )
    link_has_header = st.checkbox(
        "Link response starts with the 2 header rows", value=False
    )
    if st.button("Fetch from link"):
        text, err = fetch_url_text(link_url)
        if err:
            st.error(f"Fetch failed: {err}")
        else:
            parsed = parse_tabular_text(text, link_has_header)
            if parsed.empty:
                st.warning(
                    "Fetched the link, but got 0 parseable rows — check the response format "
                    "matches tab-separated Time/SPEED/POWER/... and the header-rows checkbox above."
                )
            else:
                st.session_state.manual_df = pd.concat(
                    [st.session_state.manual_df, parsed], ignore_index=True
                )
                st.success(f"Added {len(parsed):,} rows fetched from the link.")
    st.caption(
        "Plain GET, no authentication. If your link needs an API key or token, tell me and I'll add a field for it."
    )

if st.button("Clear all manual data"):
    st.session_state.manual_df = pd.DataFrame(columns=COLUMNS)
    st.session_state.last_upload_sig = None

manual_df = st.session_state.manual_df
if not manual_df.empty:
    manual_windows = segment_into_windows(manual_df, window_seconds)
    st.caption(
        f"{len(manual_df):,} manual rows loaded, spanning {manual_df['Time'].max():.1f}s "
        f"→ segmented into {len(manual_windows)} window(s) of {window_seconds}s."
    )

    labels = [
        f"Window {i+1}: {s:.1f}s – {e:.1f}s ({len(d):,} rows)"
        for i, (s, e, d) in enumerate(manual_windows)
    ]
    choice = st.selectbox(
        "Select a window to analyze",
        options=range(len(labels)),
        format_func=lambda i: labels[i],
    )
    _, _, mw = manual_windows[choice]

    st.plotly_chart(plot_timeseries(mw, channel), use_container_width=True)

    manual_raw_freqs, manual_raw_mags = compute_fft_for_channel(
        mw, channel, sampling_rate
    )
    manual_fft_freqs, manual_fft_mags = mask_band(
        manual_raw_freqs, manual_raw_mags, fft_low, fft_high
    )
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.plotly_chart(
            plot_fft(manual_fft_freqs, manual_fft_mags, channel, fft_low, fft_high),
            use_container_width=True,
        )
    with mcol2:
        st.plotly_chart(plot_distribution(mw, channel), use_container_width=True)

    st.subheader("FFT Diagnostics (manual window)")
    manual_fft_diag = compute_fft_diagnostics(
        manual_fft_freqs, manual_fft_mags, peak_prominence_pct / 100
    )
    mfcols = st.columns(4)
    for i, (k, v) in enumerate(manual_fft_diag.items()):
        display = "N/A" if (v is None or np.isnan(v)) else f"{v:,.4f}"
        mfcols[i % 4].metric(k, display)

    st.subheader("Peak Envelope & Shape Classification (manual window)")
    st.caption(
        f"Scoped to {shape_band_low:g}–{shape_band_high:g} Hz — independent of the {fft_low:g}–{fft_high:g} Hz FFT band above."
    )
    manual_shape_freqs, manual_shape_mags = mask_band(
        manual_raw_freqs, manual_raw_mags, shape_band_low, shape_band_high
    )
    manual_peak_freqs, manual_peak_mags = bin_peaks_by_frequency(
        manual_shape_freqs, manual_shape_mags, peak_bin_hz
    )
    if len(manual_peak_freqs) < 5:
        st.info(
            "Not enough bins in the shape band for analysis — widen the shape band or shrink the bin width."
        )
    else:
        st.plotly_chart(
            plot_peak_envelope_on_fft(
                manual_shape_freqs,
                manual_shape_mags,
                manual_peak_freqs,
                manual_peak_mags,
                channel,
                shape_band_low,
                shape_band_high,
                peak_bin_hz,
            ),
            use_container_width=True,
        )
        manual_shape = classify_curve_shape(
            manual_peak_freqs, manual_peak_mags, shape_r2_threshold
        )
        manual_is_good = manual_shape["label"] == "Parabolic" and (
            not require_inverted
            or (manual_shape["concavity"] or "").startswith("downward")
        )
        mscols = st.columns(3)
        mscols[0].metric(
            "R² (quadratic fit)",
            (
                f"{manual_shape['r2_quadratic']:.3f}"
                if not np.isnan(manual_shape["r2_quadratic"])
                else "N/A"
            ),
        )
        mscols[1].metric("Concavity", manual_shape["concavity"] or "N/A")
        mscols[2].metric(
            "Shape", "Inverted parabola" if manual_is_good else manual_shape["label"]
        )
        if manual_is_good:
            st.success(
                f"Shape: inverted parabola ({manual_shape['concavity']}) → heuristic reads as **possible good weld**."
            )
        elif manual_shape["label"] == "No clear shape":
            st.warning(
                "Shape: no clear fit → heuristic reads as **possible bad weld**."
            )
        elif manual_shape["label"] == "Parabolic":
            st.info(
                f"Shape: parabolic, but {manual_shape['concavity']} — not inverted, so it doesn't satisfy the good-weld rule as specified."
            )
        else:
            st.info(
                f"Shape: {manual_shape['label']} — doesn't match the good-weld rule."
            )
        st.caption(
            "⚠️ Inverted-parabola-equals-good-weld is a heuristic rule you specified, not something "
            "validated against known weld outcomes here. Check it against a batch of welds with "
            "**confirmed** good/bad results before using this as an actual accept/reject gate."
        )

    st.subheader("Time-Series Metrics (manual window)")
    manual_metrics = compute_metrics(mw[channel])
    mcols = st.columns(4)
    for i, (k, v) in enumerate(manual_metrics.items()):
        display = f"{v:,.4f}" if isinstance(v, (float, np.floating)) else f"{v}"
        mcols[i % 4].metric(k, display)

    st.download_button(
        "Download this manual window as CSV",
        data=mw.to_csv(index=False).encode(),
        file_name=f"manual_window_{choice+1}.csv",
        mime="text/csv",
    )
else:
    st.caption("No manual data loaded yet.")

if auto_refresh:
    time.sleep(poll_seconds)
    st.rerun()