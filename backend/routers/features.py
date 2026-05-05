"""
Router: /api/features
Extracts 82-dimensional feature vectors from all hits and computes
mel spectrograms. Returns feature data + 2D PCA for visualisation.
"""

import numpy as np
from fastapi import APIRouter, Header, HTTPException

from session import session_manager
from ml.feature_extraction import (
    extract_features, extract_mel_spectrogram,
    extract_psd, extract_mfcc, extract_decay,
    extract_energy_ratio, FEATURE_NAMES
)
from config import IDX_TO_CLASS, CLASS_NAMES

router = APIRouter(prefix="/api", tags=["features"])


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Simple PCA to 2 components (no sklearn needed for basic viz)."""
    X_c = X - X.mean(axis=0)
    cov = np.cov(X_c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    components = eigvecs[:, idx[:2]]
    X_pca = X_c @ components
    var_ratio = eigvals[idx[:2]] / eigvals.sum()
    return X_pca, components, var_ratio.tolist()


@router.post("/features")
async def extract_all_features(session_id: str = Header(..., alias="X-Session-Id")):
    """
    Run feature extraction on all hits stored in session.
    Stores X_feat, feature_names, and PCA coords in session.
    Returns summary + PCA scatter data for the frontend.
    """
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    hits = session.hits
    if not hits or hits["n_hits"] == 0:
        raise HTTPException(status_code=400, detail="No hits found. Run /api/process first.")

    waveforms     = hits["waveforms"]
    labels        = np.array(hits["labels"])
    flange_groups = np.array(hits["flange_groups"])

    # Extract features for every hit
    X_feat = np.stack([
        extract_features(np.array(w, dtype=np.float32))
        for w in waveforms
    ], axis=0)   # (N, 82)

    # PCA for scatter plot
    X_pca, components, var_ratio = pca_2d(X_feat)

    # Per-class mean feature profile (for bar chart comparison)
    class_profiles = {}
    for idx in [0, 1, 2]:
        mask = labels == idx
        if mask.any():
            class_profiles[str(IDX_TO_CLASS[idx])] = X_feat[mask].mean(axis=0).tolist()

    # Store in session
    session.features = {
        "X_feat":         X_feat.tolist(),
        "feature_names":  FEATURE_NAMES,
        "labels":         labels.tolist(),
        "flange_groups":  flange_groups.tolist(),
        "X_pca":          X_pca.tolist(),
        "pca_var_ratio":  var_ratio,
        "class_profiles": class_profiles,
        "n_features":     X_feat.shape[1],
        "n_hits":         X_feat.shape[0],
    }
    session.touch()

    # Build scatter data: list of {x, y, label_idx, label_name, flange}
    scatter = [
        {
            "x":          round(float(X_pca[i, 0]), 4),
            "y":          round(float(X_pca[i, 1]), 4),
            "label_idx":  int(labels[i]),
            "label_name": CLASS_NAMES[int(labels[i])],
            "flange":     int(flange_groups[i]),
        }
        for i in range(len(labels))
    ]

    return {
        "status":        "done",
        "n_hits":        int(X_feat.shape[0]),
        "n_features":    int(X_feat.shape[1]),
        "feature_names": FEATURE_NAMES,
        "scatter":       scatter,
        "pca_var_ratio": var_ratio,
        "class_profiles": class_profiles,
        "feature_stats": {
            name: {
                "mean": round(float(X_feat[:, i].mean()), 4),
                "std":  round(float(X_feat[:, i].std()),  4),
                "min":  round(float(X_feat[:, i].min()),  4),
                "max":  round(float(X_feat[:, i].max()),  4),
            }
            for i, name in enumerate(FEATURE_NAMES)
        },
    }


@router.get("/features/hit/{hit_idx}")
async def get_hit_features(
    hit_idx: int,
    session_id: str = Header(..., alias="X-Session-Id"),
):
    """
    Return all feature visualisation data for a single hit:
    waveform, mel spectrogram, PSD, MFCC, decay curve, energy ratio.
    Used by the Feature Extraction screen's hit picker.
    """
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    hits = session.hits
    if not hits or hit_idx >= hits["n_hits"]:
        raise HTTPException(status_code=404, detail=f"Hit {hit_idx} not found")

    win = np.array(hits["waveforms"][hit_idx], dtype=np.float32)

    # Individual feature components for visualisation
    psd          = extract_psd(win)
    mfcc_m, mfcc_s = extract_mfcc(win)
    tau          = extract_decay(win)
    energy_ratio = extract_energy_ratio(win)
    mel          = extract_mel_spectrogram(win)

    # Decay envelope for chart
    frame_len = int(0.005 * 48000)
    n_frames  = len(win) // frame_len
    rms_env   = [
        float(np.sqrt(np.mean(win[i * frame_len:(i + 1) * frame_len] ** 2)))
        for i in range(n_frames)
    ]
    decay_t = [round(i * 0.005, 4) for i in range(n_frames)]

    # Full feature vector
    feat = extract_features(win)

    return {
        "hit_idx":     hit_idx,
        "label_idx":   hits["labels"][hit_idx],
        "label_name":  CLASS_NAMES[hits["labels"][hit_idx]],
        "flange_id":   hits["flange_groups"][hit_idx],
        "psd":         psd.tolist(),
        "mfcc_mean":   mfcc_m.tolist(),
        "mfcc_std":    mfcc_s.tolist(),
        "tau":         round(float(tau), 4),
        "energy_ratio": round(float(energy_ratio), 4),
        "mel_spectrogram": mel.tolist(),   # (64, 128) — Plotly heatmap
        "decay_rms":   rms_env,
        "decay_t":     decay_t,
        "feature_vector": feat.tolist(),
        "feature_names":  FEATURE_NAMES,
    }
