"""
Feature extraction — 82-dimensional physics-informed vector.
EXACTLY matches Cell 4 of final_project_saurav_silwal.ipynb.

Group 1: Relative PSD in 20 log-spaced bins (50 Hz – 8 kHz)    → 20 dims
Group 2: MFCC mean/std + delta MFCC mean/std (13 coeffs each)  → 52 dims
Group 3: Physics features (centroid, bandwidth, rolloff, ZCR,
         peak freq, decay τ, energy ratio, RMS, Q-factor)       → 10 dims
Total                                                            → 82 dims
"""

import warnings
import numpy as np
import librosa
from scipy.signal import welch

from config import SR, N_MELS, N_FFT, HOP_LENGTH_MEL, SPEC_TIME_FRAMES

# ── Constants matching notebook ────────────────────────────────────────────
N_PSD_BINS     = 20
PSD_FMIN       = 50.0
PSD_FMAX       = 8000.0
WELCH_NPERSEG  = 2048
WELCH_NOVERLAP = 1024

N_MFCC         = 13
MFCC_NFFT      = 2048
MFCC_HOP       = 512

DECAY_FIT_MS       = 200
EARLY_LATE_FRAC    = 0.20
PEAK_SAMPLE_IN_WIN = int(0.020 * SR)   # 960 samples = 20 ms pre-peak

FMIN_MEL = 0
FMAX_MEL = SR // 2   # Nyquist = 24 000 Hz


# ── Group 1: Relative PSD ─────────────────────────────────────────────────

def relative_psd_log_bins(y, sr=SR, n_bins=N_PSD_BINS,
                          f_min=PSD_FMIN, f_max=PSD_FMAX):
    """Welch PSD → 20 log-spaced bins → normalized so sum=1."""
    f, pxx = welch(y, fs=sr,
                   nperseg=min(WELCH_NPERSEG, len(y)),
                   noverlap=min(WELCH_NOVERLAP, len(y) // 2))
    edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)
    bins  = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask    = (f >= edges[i]) & (f < edges[i + 1])
        bins[i] = pxx[mask].sum()
    total = bins.sum()
    if total > 1e-20:
        bins /= total
    return bins, f, pxx


# ── Group 2: MFCC + delta statistics ──────────────────────────────────────

def mfcc_stats(y, sr=SR):
    """13 MFCCs → mean+std (26) + delta mean+std (26) = 52 dims."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                      n_fft=MFCC_NFFT, hop_length=MFCC_HOP)
        delta = librosa.feature.delta(mfcc)
    return np.concatenate([
        mfcc.mean(axis=1),  mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
    ]).astype(np.float32)


# ── Group 3: Physics features ──────────────────────────────────────────────

def peak_frequency(f, pxx):
    if pxx.max() <= 0:
        return 0.0
    return float(f[np.argmax(pxx)])


def q_factor(f, pxx):
    """Q = f_peak / -3 dB bandwidth. High Q = tight (rings cleanly)."""
    if pxx.max() <= 0:
        return 0.0
    pdb      = 10 * np.log10(pxx + 1e-20)
    peak_idx = int(np.argmax(pdb))
    threshold = pdb[peak_idx] - 3.0
    L = peak_idx
    while L > 0 and pdb[L] >= threshold:
        L -= 1
    R = peak_idx
    while R < len(pdb) - 1 and pdb[R] >= threshold:
        R += 1
    bw = max(f[R] - f[L], 1.0)
    return float(f[peak_idx] / bw)


def decay_tau(y, peak_sample=PEAK_SAMPLE_IN_WIN, sr=SR, fit_ms=DECAY_FIT_MS):
    """Decay time constant τ. Loose → small τ. Tight → large τ."""
    n_fit  = int(fit_ms * sr / 1000)
    seg    = y[peak_sample:min(peak_sample + n_fit, len(y))]
    if len(seg) < 100:
        return np.nan
    env_w = max(1, int(0.005 * sr))
    env   = np.convolve(np.abs(seg), np.ones(env_w) / env_w, mode='same')
    if env.max() < 1e-8:
        return np.nan
    active = np.where(env > 0.05 * env.max())[0]
    if len(active) < 50:
        return np.nan
    n_active = active[-1] + 1
    eps      = env.max() * 1e-4
    log_env  = np.log(env[:n_active] + eps)
    t        = np.arange(n_active) / sr
    slope, _ = np.polyfit(t, log_env, 1)
    if slope >= 0:
        return np.nan
    tau = -1.0 / slope
    return float(tau) if 0.001 <= tau <= 10.0 else np.nan


def energy_ratio(y, frac=EARLY_LATE_FRAC):
    """E_late / E_early. Tight flanges still ringing → high ratio."""
    n_chunk = int(frac * len(y))
    e_early = np.sqrt(np.mean(y[:n_chunk] ** 2))
    e_late  = np.sqrt(np.mean(y[-n_chunk:] ** 2))
    return float(e_late / (e_early + 1e-12))


# ── Master 82-dim extractor ───────────────────────────────────────────────

def extract_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Return 82-dim feature vector for one hit window."""
    psd_bins, f_psd, pxx = relative_psd_log_bins(y, sr)    # 20

    cepstral = mfcc_stats(y, sr)                            # 52

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc   = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        sb   = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        sr85 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        zcr  = librosa.feature.zero_crossing_rate(y)[0]

    physics = np.array([
        sc.mean(),  sc.std(),           # 2: centroid mean/std
        sb.mean(),                      # 1: bandwidth mean
        sr85.mean(),                    # 1: rolloff 85%
        zcr.mean(),                     # 1: zero-crossing rate
        peak_frequency(f_psd, pxx),     # 1: dominant freq
        decay_tau(y),                   # 1: τ (NaN → imputed after)
        energy_ratio(y),                # 1: E_late / E_early
        float(np.sqrt(np.mean(y ** 2))),# 1: RMS energy
        q_factor(f_psd, pxx),           # 1: Q-factor
    ], dtype=np.float32)               # 10 total

    return np.concatenate([psd_bins, cepstral, physics])


def impute_nans(X: np.ndarray, y_labels: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Per-class median imputation for NaN columns (tau can be NaN)."""
    X = X.copy()
    nan_cols = np.where(np.isnan(X).any(axis=0))[0]
    for c in nan_cols:
        for cls in range(n_classes):
            cls_mask   = (y_labels == cls)
            median_val = float(np.nanmedian(X[cls_mask, c]))
            if np.isnan(median_val):
                median_val = float(np.nanmedian(X[:, c]))
            fill_mask = cls_mask & np.isnan(X[:, c])
            X[fill_mask, c] = median_val
    return X


# ── Feature name list ─────────────────────────────────────────────────────

def _build_feature_names() -> list[str]:
    names = []
    edges = np.logspace(np.log10(PSD_FMIN), np.log10(PSD_FMAX), N_PSD_BINS + 1)
    for i in range(N_PSD_BINS):
        names.append(f'psd_{edges[i]:.0f}_{edges[i+1]:.0f}Hz')
    names += [f'mfcc{i:02d}_mean'  for i in range(N_MFCC)]
    names += [f'mfcc{i:02d}_std'   for i in range(N_MFCC)]
    names += [f'dmfcc{i:02d}_mean' for i in range(N_MFCC)]
    names += [f'dmfcc{i:02d}_std'  for i in range(N_MFCC)]
    names += ['spec_centroid_mean', 'spec_centroid_std',
              'spec_bandwidth_mean', 'spec_rolloff85_mean',
              'zero_cross_rate_mean', 'peak_frequency',
              'decay_tau', 'energy_ratio', 'rms_energy', 'q_factor']
    return names

FEATURE_NAMES: list[str] = _build_feature_names()
assert len(FEATURE_NAMES) == 82


# ── Mel spectrogram (CNN / BiLSTM input) — matches notebook Cell 5 ────────

def extract_mel_spectrogram(y: np.ndarray, sr: int = SR,
                             n_mels: int = N_MELS,
                             n_fft: int = N_FFT,
                             hop_length: int = HOP_LENGTH_MEL,
                             target_frames: int = SPEC_TIME_FRAMES) -> np.ndarray:
    """One hit → standardized log-mel spectrogram of shape (n_mels, target_frames)."""
    mel    = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
        fmin=FMIN_MEL, fmax=FMAX_MEL, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    n_frames = mel_db.shape[1]
    if n_frames < target_frames:
        pad_val = float(mel_db.min())
        mel_db  = np.pad(mel_db, ((0, 0), (0, target_frames - n_frames)),
                         mode='constant', constant_values=pad_val)
    elif n_frames > target_frames:
        start  = (n_frames - target_frames) // 2
        mel_db = mel_db[:, start:start + target_frames]

    # per-sample standardize
    mu, sigma = mel_db.mean(), mel_db.std()
    return (mel_db - mu) / (sigma + 1e-6)   # shape (64, 128)
