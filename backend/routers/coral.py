"""
Router: POST /api/coral
CORAL (Correlation Alignment) domain adaptation.

Aligns the covariance structure of lab test features to match
the training feature distribution, then runs ensemble classification.

Reference: Sun & Saenko, "Return of Frustratingly Easy Domain Adaptation", AAAI 2016.
"""

import numpy as np
from fastapi import APIRouter, Header, HTTPException
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

from session import session_manager
from ml.feature_extraction import extract_features, FEATURE_NAMES
from utils.audio import extract_hits_from_file
from config import SEED, IDX_TO_CLASS, CLASS_NAMES

router = APIRouter(prefix="/api", tags=["coral"])


# ─── CORAL math ────────────────────────────────────────────────────────────────

def coral_align(X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
    """
    Align X_target covariance to match X_source covariance.
    Returns whitened + re-coloured target features.

    Steps:
      1. Whiten target: X_t_white = X_t @ Ct^{-1/2}
      2. Re-colour with source: X_t_aligned = X_t_white @ Cs^{1/2}
    """
    d = X_source.shape[1]

    def cov(X):
        return np.cov(X.T) + 1e-5 * np.eye(d)

    Cs = cov(X_source)
    Ct = cov(X_target)

    # Matrix square root via eigendecomposition
    def mat_sqrt(M):
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.clip(eigvals, 1e-10, None)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def mat_sqrt_inv(M):
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.clip(eigvals, 1e-10, None)
        return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    Ct_inv_sqrt = mat_sqrt_inv(Ct)
    Cs_sqrt     = mat_sqrt(Cs)

    X_aligned = (X_target @ Ct_inv_sqrt) @ Cs_sqrt
    return X_aligned.astype(np.float32)


def cov_distance(X_a: np.ndarray, X_b: np.ndarray) -> float:
    """Frobenius norm of covariance difference — measures domain gap."""
    Ca = np.cov(X_a.T)
    Cb = np.cov(X_b.T)
    return float(np.linalg.norm(Ca - Cb, ord="fro"))


# ─── Route ────────────────────────────────────────────────────────────────────

@router.post("/coral")
async def run_coral(session_id: str = Header(..., alias="X-Session-Id")):
    """
    1. Extract features from lab test files (if not already done)
    2. Apply CORAL alignment: target → source distribution
    3. Run all trained shallow models on aligned features
    4. Compute per-flange consensus vote
    """
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.features:
        raise HTTPException(status_code=400, detail="Extract training features first")
    if not session.training_results:
        raise HTTPException(status_code=400, detail="Train models first")
    if not session.lab_files:
        raise HTTPException(status_code=400, detail="No lab files uploaded")

    # ── Extract lab features ────────────────────────────────────────────────
    lab_waveforms:   list[np.ndarray] = []
    lab_meta:        list[dict]       = []

    for finfo in session.lab_files:
        windows, _ = extract_hits_from_file(
            filepath=finfo["filepath"],
            class_idx=0,          # unknown — placeholder
            flange_id=finfo["flange_id"],
            area_id=finfo["area_id"],
        )
        for win in windows:
            lab_waveforms.append(win)
            lab_meta.append({
                "flange_id": finfo["flange_id"],
                "area_id":   finfo["area_id"],
                "filename":  finfo["filename"],
            })

    if not lab_waveforms:
        raise HTTPException(status_code=400, detail="No usable hits found in lab files")

    X_lab = np.stack([extract_features(w) for w in lab_waveforms], axis=0)

    # ── Source features ──────────────────────────────────────────────────────
    X_src = np.array(session.features["X_feat"], dtype=np.float32)
    y_src = np.array(session.features["labels"],  dtype=np.int64)
    g_src = np.array(session.features["flange_groups"], dtype=np.int64)

    # Z-score both using source statistics
    scaler = StandardScaler().fit(X_src)
    X_src_s = scaler.transform(X_src)
    X_lab_s = scaler.transform(X_lab)

    # ── Covariance distance before ──────────────────────────────────────────
    dist_before = cov_distance(X_src_s, X_lab_s)

    # ── CORAL alignment ─────────────────────────────────────────────────────
    X_lab_aligned = coral_align(X_src_s, X_lab_s)
    dist_after    = cov_distance(X_src_s, X_lab_aligned)

    # ── PCA for visualisation (reuse training PCA components) ───────────────
    feat_data = session.features
    # Simple 2D projection using stored PCA (recompute from X_src for safety)
    X_src_c = X_src_s - X_src_s.mean(axis=0)
    _, eigvecs = np.linalg.eigh(np.cov(X_src_c.T))
    components = eigvecs[:, -2:]   # top 2 eigenvectors

    X_lab_pca_before  = (X_lab_s        - X_src_s.mean(axis=0)) @ components
    X_lab_pca_after   = (X_lab_aligned  - X_src_s.mean(axis=0)) @ components
    X_src_pca         = X_src_c @ components

    # ── Classify with each trained shallow model ─────────────────────────────
    MODELS = {
        "SVM": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                   class_weight="balanced", random_state=SEED),
        "LR":  LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                   multi_class="multinomial", solver="lbfgs", random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "RF":  RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=SEED, n_jobs=-1),
    }

    model_preds: dict[str, dict] = {}
    for name, clf in MODELS.items():
        clf.fit(X_src_s, y_src)
        proba_before = clf.predict_proba(X_lab_s)        if hasattr(clf, "predict_proba") else None
        proba_after  = clf.predict_proba(X_lab_aligned)  if hasattr(clf, "predict_proba") else None

        pred_before = clf.predict(X_lab_s).tolist()
        clf.fit(X_src_s, y_src)                        # refit (same data, ensure alignment)
        pred_after  = clf.predict(X_lab_aligned).tolist()

        model_preds[name] = {
            "pred_before": pred_before,
            "pred_after":  pred_after,
            "proba_before": proba_before.tolist() if proba_before is not None else None,
            "proba_after":  proba_after.tolist()  if proba_after  is not None else None,
        }

    # ── Per-flange consensus ────────────────────────────────────────────────
    flanges = sorted(set(m["flange_id"] for m in lab_meta))
    consensus_rows: list[dict] = []

    for fl in flanges:
        fl_mask = [i for i, m in enumerate(lab_meta) if m["flange_id"] == fl]
        votes_before, votes_after = [], []
        p0_before, p0_after = [], []

        for name, preds in model_preds.items():
            # Hit-level majority vote per flange per model
            hit_preds_before = [preds["pred_before"][i] for i in fl_mask]
            hit_preds_after  = [preds["pred_after"][i]  for i in fl_mask]

            model_vote_before = Counter(hit_preds_before).most_common(1)[0][0]
            model_vote_after  = Counter(hit_preds_after).most_common(1)[0][0]

            votes_before.append(IDX_TO_CLASS[model_vote_before])
            votes_after.append(IDX_TO_CLASS[model_vote_after])

            if preds["proba_after"] is not None:
                p0_after.append(
                    float(np.mean([preds["proba_after"][i][0] for i in fl_mask]))
                )
            if preds["proba_before"] is not None:
                p0_before.append(
                    float(np.mean([preds["proba_before"][i][0] for i in fl_mask]))
                )

        winner_before = Counter(votes_before).most_common(1)[0][0]
        winner_after  = Counter(votes_after).most_common(1)[0][0]

        consensus_rows.append({
            "flange":          fl,
            "consensus_before": winner_before,
            "consensus_after":  winner_after,
            "votes_before":    votes_before,
            "votes_after":     votes_after,
            "avg_p0_before":   round(float(np.mean(p0_before)), 3) if p0_before else None,
            "avg_p0_after":    round(float(np.mean(p0_after)),  3) if p0_after  else None,
            "n_hits":          len(fl_mask),
        })

    result = {
        "cov_distance_before": round(dist_before, 4),
        "cov_distance_after":  round(dist_after,  4),
        "improvement_pct":     round((dist_before - dist_after) / (dist_before + 1e-9) * 100, 1),
        "n_lab_hits":          len(lab_waveforms),
        "model_predictions":   model_preds,
        "consensus":           consensus_rows,
        "pca": {
            "source":       X_src_pca.tolist(),
            "source_labels": y_src.tolist(),
            "lab_before":   X_lab_pca_before.tolist(),
            "lab_after":    X_lab_pca_after.tolist(),
            "lab_meta":     lab_meta,
        },
    }

    session.coral_result = result
    session.touch()

    return result
