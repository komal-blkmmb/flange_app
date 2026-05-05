"""
Audio loading and hit extraction utilities.
All parameters come from config.py — identical to the notebook pipeline.
"""

import warnings
import numpy as np
import librosa
from scipy.signal import find_peaks

from config import (
    SR, RMS_FRAME_LENGTH, RMS_HOP_LENGTH, PEAK_REL_THRESH,
    PEAK_MIN_DIST_S, PRE_PEAK_S, POST_PEAK_S, HIT_WINDOW_LEN,
    MIN_PEAK_AMP, MIN_CREST_FACTOR, MIN_ATTACK_RATIO, ATTACK_WIN_S,
)

PRE_SAMPLES  = int(PRE_PEAK_S  * SR)   # 960
POST_SAMPLES = int(POST_PEAK_S * SR)   # 24 000
FADE_SAMPLES = int(0.10 * HIT_WINDOW_LEN)


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(filepath: str, sr: int = SR) -> np.ndarray:
    """Load any audio file at target SR. Returns float32 mono array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(filepath, sr=sr, mono=True)
    return y.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Hit detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_peaks(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Find hit-peak sample indices via RMS envelope."""
    rms = librosa.feature.rms(
        y=y, frame_length=RMS_FRAME_LENGTH, hop_length=RMS_HOP_LENGTH
    )[0]
    if rms.max() <= 0:
        return np.array([], dtype=int)
    height_thresh  = PEAK_REL_THRESH * rms.max()
    min_dist_frames = int(PEAK_MIN_DIST_S * sr / RMS_HOP_LENGTH)
    peak_frames, _ = find_peaks(rms, height=height_thresh, distance=min_dist_frames)
    return (peak_frames * RMS_HOP_LENGTH).astype(int)


def extract_window(y: np.ndarray, peak_sample: int) -> np.ndarray:
    """Extract 520 ms window around peak. Zero-pads at file edges."""
    window = np.zeros(PRE_SAMPLES + POST_SAMPLES, dtype=np.float32)
    src_start = max(0, peak_sample - PRE_SAMPLES)
    src_end   = min(len(y), peak_sample + POST_SAMPLES)
    dst_start = src_start - (peak_sample - PRE_SAMPLES)
    dst_end   = dst_start + (src_end - src_start)
    window[dst_start:dst_end] = y[src_start:src_end]
    return window


def apply_hann_fadeout(window: np.ndarray) -> np.ndarray:
    """Cosine fade-out on the last 10% of the window."""
    fade = np.hanning(2 * FADE_SAMPLES)[FADE_SAMPLES:]
    out = window.copy()
    out[-FADE_SAMPLES:] *= fade
    return out


def compute_quality(window: np.ndarray) -> tuple[float, float, float]:
    """Returns (peak_amp, crest_factor, attack_ratio)."""
    abs_w    = np.abs(window)
    peak_amp = float(abs_w.max())
    rms_val  = float(np.sqrt(np.mean(window ** 2)))
    crest    = peak_amp / (rms_val + 1e-12)
    atk_n    = int(ATTACK_WIN_S * SR)
    attack   = float(abs_w[:atk_n].max()) / (peak_amp + 1e-12)
    return peak_amp, crest, attack


def extract_hits_from_file(
    filepath: str,
    class_idx: int,
    flange_id: int,
    area_id: int,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Full pipeline for one audio file.
    Returns (kept_windows, quality_log_entries).
    """
    y = load_audio(filepath)
    peaks = detect_peaks(y)
    kept_windows: list[np.ndarray] = []
    quality_log: list[dict] = []

    for i, p in enumerate(peaks):
        win = extract_window(y, p)
        peak_amp, crest, attack = compute_quality(win)
        passed = (
            peak_amp >= MIN_PEAK_AMP
            and crest  >= MIN_CREST_FACTOR
            and attack >= MIN_ATTACK_RATIO
        )
        quality_log.append({
            "hit_idx": i,
            "peak_sample": int(p),
            "peak_time_s": round(float(p / SR), 4),
            "peak_amp": round(peak_amp, 4),
            "crest_factor": round(crest, 3),
            "attack_ratio": round(attack, 3),
            "kept": passed,
        })
        if passed:
            win = apply_hann_fadeout(win)
            kept_windows.append(win)

    return kept_windows, quality_log


def get_rms_envelope(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (rms_values, time_axis_seconds) for frontend waveform display."""
    rms = librosa.feature.rms(
        y=y, frame_length=RMS_FRAME_LENGTH, hop_length=RMS_HOP_LENGTH
    )[0]
    times = librosa.frames_to_time(
        np.arange(len(rms)), sr=SR, hop_length=RMS_HOP_LENGTH
    )
    return rms, times
