"""
Feature extraction — 82-dimensional feature vector per hit.

Breakdown:
  [0:50]   Welch PSD — 50 log-spaced frequency bins (relative, sum=1)
  [50:63]  MFCC mean — 13 coefficients (mean-subtracted per coefficient)
  [63:76]  MFCC std  — 13 coefficients
  [76]     Decay time constant τ (exponential fit)
  [77]     Energy ratio (late/early)
  [78]     Crest factor (peak/RMS)
  [79]     Log peak amplitude
  [80:82]  Spectral centroid mean + std
"""

import warnings
import numpy as np
import librosa
from scipy.signal import welch
from scipy.optimize import curve_fit

from config import SR, N_MFCC, N_FFT, HIT_WINDOW_LEN, N_MELS, HOP_LENGTH_MEL, SPEC_TIME_FRAMES

# ─── PSD ──────────────────────────────────────────────────────────────────────

def extract_psd(window: np.ndarray, n_bins: int = 50) -> np.ndarray:
    """
    Welch PSD normalised to sum=1 (relative PSD), log-spaced frequency bins.
    Normalisation makes features robust to recording-level differences.
    """
    f, pxx = welch(window, fs=SR, nperseg=512, noverlap=256)
    # Log-spaced bin edges from 50 Hz to Nyquist
    edges = np.logspace(np.log10(50), np.log10(SR / 2), n_bins + 1)
    binned = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = (f >= edges[i]) & (f < edges[i + 1])
        if mask.any():
            binned[i] = float(pxx[mask].mean())
    total = binned.sum()
    if total > 0:
        binned /= total
    return binned


# ─── MFCC ─────────────────────────────────────────────────────────────────────

def extract_mfcc(window: np.ndarray, n_mfcc: int = N_MFCC) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (mfcc_mean, mfcc_std) — both shape (n_mfcc,).
    Mean subtraction (per coefficient across the window) is applied first —
    this is the single most impactful normalisation for cross-session robustness.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc = librosa.feature.mfcc(y=window, sr=SR, n_mfcc=n_mfcc, n_fft=N_FFT)
    # Per-coefficient mean subtraction
    mfcc -= mfcc.mean(axis=1, keepdims=True)
    return mfcc.mean(axis=1).astype(np.float32), mfcc.std(axis=1).astype(np.float32)


# ─── Decay ────────────────────────────────────────────────────────────────────

def _exp_model(t, A, tau):
    return A * np.exp(-t / tau)


def extract_decay(window: np.ndarray) -> float:
    """
    Fit A·exp(-t/τ) to the RMS energy envelope.
    Returns τ (seconds). Larger τ = slower decay = tighter flange.
    Falls back to 0.05 on fit failure.
    """
    # RMS in 5ms frames
    frame_len = int(0.005 * SR)
    n_frames  = len(window) // frame_len
    rms_env   = np.array([
        np.sqrt(np.mean(window[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ], dtype=np.float32)
    t = np.arange(n_frames) * 0.005
    # Trim to non-zero region
    thresh = 0.02 * rms_env.max()
    valid  = rms_env > thresh
    if valid.sum() < 5:
        return 0.05
    try:
        p0 = (rms_env[valid].max(), 0.15)
        popt, _ = curve_fit(_exp_model, t[valid], rms_env[valid], p0=p0,
                             bounds=([0, 0.001], [np.inf, 2.0]), maxfev=2000)
        tau = float(np.clip(popt[1], 0.001, 2.0))
    except Exception:
        tau = 0.05
    return tau


# ─── Energy ratio ─────────────────────────────────────────────────────────────

def extract_energy_ratio(window: np.ndarray, split_ms: float = 50.0) -> float:
    """
    E_late / E_early where split is at split_ms ms after the hit onset.
    High ratio → flange still ringing → tight.
    """
    split_n = int(split_ms / 1000 * SR)
    early = float(np.sum(window[:split_n] ** 2)) + 1e-12
    late  = float(np.sum(window[split_n:] ** 2))
    return min(late / early, 100.0)  # cap for numerical safety


# ─── Spectral centroid ────────────────────────────────────────────────────────

def extract_spectral_centroid(window: np.ndarray) -> tuple[float, float]:
    cent = librosa.feature.spectral_centroid(y=window, sr=SR, n_fft=N_FFT)[0]
    return float(cent.mean()), float(cent.std())


# ─── Full 82-dim vector ───────────────────────────────────────────────────────

def extract_features(window: np.ndarray) -> np.ndarray:
    """Return 82-dim feature vector for a single hit window."""
    psd          = extract_psd(window)                              # 50
    mfcc_m, mfcc_s = extract_mfcc(window)                         # 13 + 13
    tau          = extract_decay(window)                            # 1
    energy_ratio = extract_energy_ratio(window)                    # 1
    peak_amp     = float(np.abs(window).max())
    rms          = float(np.sqrt(np.mean(window ** 2))) + 1e-12
    crest        = float(peak_amp / rms)                            # 1
    log_peak     = float(np.log1p(peak_amp))                       # 1
    sc_mean, sc_std = extract_spectral_centroid(window)            # 2

    feat = np.concatenate([
        psd,
        mfcc_m,
        mfcc_s,
        [tau, energy_ratio, crest, log_peak, sc_mean, sc_std],
    ]).astype(np.float32)
    return feat


FEATURE_NAMES: list[str] = (
    [f"psd_{i}"      for i in range(50)] +
    [f"mfcc_mean_{i}" for i in range(13)] +
    [f"mfcc_std_{i}"  for i in range(13)] +
    ["tau", "energy_ratio", "crest_factor", "log_peak_amp", "sc_mean", "sc_std"]
)


# ─── Mel spectrogram (CNN input) ─────────────────────────────────────────────

def extract_mel_spectrogram(window: np.ndarray) -> np.ndarray:
    """
    Returns mel spectrogram of shape (N_MELS, SPEC_TIME_FRAMES) in dB.
    Used as CNN input (add channel dim in model).
    """
    mel = librosa.feature.melspectrogram(
        y=window, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH_MEL
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Pad / trim to fixed width
    if mel_db.shape[1] < SPEC_TIME_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, SPEC_TIME_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :SPEC_TIME_FRAMES]
    return mel_db.astype(np.float32)
