"""
Router: POST /api/classify
Upload unlabeled WAV recordings, extract hits, and classify with trained models.

Phase 1 — Raw:
  All 6 models, weighted ensemble by Task-2 LOIO accuracy.

Phase 2 — CORAL:
  • Shallow (SVM, LR, KNN) + MLP: CORAL alignment in scaled 82-dim feature space.
  • CNN, LSTM: Deep CORAL — embed via penultimate layer, align in embedding space,
    refit lightweight logistic-regression head on training embeddings.

Filename convention: 'Area 1 Flange 2.wav' (case/space insensitive, underscores OK).
"""

import os
import re
import tempfile
import traceback

import numpy as np
from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from sklearn.linear_model import LogisticRegression

from config import SEED
from ml.feature_extraction import extract_features, extract_mel_spectrogram
from session import session_manager
from utils.audio import extract_hits_from_file

router = APIRouter(prefix="/api", tags=["classify"])

CLASS_VALUES = [0, 25, 50]   # class index → ft-lbs


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_filename(name: str) -> tuple[int, int]:
    """Return (flange_id, area_id) from any 'Area X Flange Y' variant."""
    stem = os.path.splitext(name)[0]
    area_m   = re.search(r'area\s*[-_]?\s*(\d+)', stem, re.IGNORECASE)
    flange_m = re.search(r'flange\s*[-_]?\s*(\d+)', stem, re.IGNORECASE)
    return (
        int(flange_m.group(1)) if flange_m else 0,
        int(area_m.group(1))   if area_m   else 0,
    )


def _impute_with_train(X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Replace NaN in test features with training column medians."""
    X = X_test.copy()
    medians = np.nanmedian(X_train, axis=0)
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = medians[cols]
    return X


def _coral(X_src: np.ndarray, X_tgt: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    CORAL with mean shift — matches the notebook's coral_align_target_to_source.
    Whiten X_tgt covariance, re-colour with X_src covariance, shift to source mean.
    """
    X_src = X_src.astype(np.float64)
    X_tgt = X_tgt.astype(np.float64)
    d = X_src.shape[1]
    mu_s = X_src.mean(axis=0, keepdims=True)
    mu_t = X_tgt.mean(axis=0, keepdims=True)
    Cs = np.cov((X_src - mu_s).T) + eps * np.eye(d)
    Ct = np.cov((X_tgt - mu_t).T) + eps * np.eye(d)

    def _sqrt(M):
        v, Q = np.linalg.eigh(M)
        return Q @ np.diag(np.sqrt(np.maximum(v, eps))) @ Q.T

    def _inv_sqrt(M):
        v, Q = np.linalg.eigh(M)
        return Q @ np.diag(1.0 / np.sqrt(np.maximum(v, eps))) @ Q.T

    aligned = (X_tgt - mu_t) @ _inv_sqrt(Ct) @ _sqrt(Cs) + mu_s
    return aligned.astype(np.float32)


def _cov_dist(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(np.cov(A.T) - np.cov(B.T), ord="fro"))


def _agg_per_flange(proba: np.ndarray, meta: list[dict]) -> list[dict]:
    """Average hit probabilities per flange and return per-flange verdict dicts."""
    flanges = sorted({m["flange_id"] for m in meta})
    rows = []
    for fl in flanges:
        idx = [i for i, m in enumerate(meta) if m["flange_id"] == fl]
        avg = proba[idx].mean(axis=0)
        pi  = int(avg.argmax())
        s   = sorted(avg)
        rows.append({
            "flange_id":  fl,
            "n_hits":     len(idx),
            "prediction": CLASS_VALUES[pi],
            "proba":      {"p_0": round(float(avg[0]), 4),
                           "p_25": round(float(avg[1]), 4),
                           "p_50": round(float(avg[2]), 4)},
            "confidence": round(float(avg.max()), 4),
            "margin":     round(float(s[-1] - s[-2]), 4),
        })
    return rows


def _weighted_ens(proba_dict: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights.get(k, 0) for k in proba_dict) or 1.0
    result = None
    for k, p in proba_dict.items():
        w = weights.get(k, 0) / total
        result = w * p if result is None else result + w * p
    return result


def _build_embedder(model, input_shape):
    """Sub-model ending at the penultimate non-output layer of a Keras Sequential."""
    try:
        from tensorflow import keras
        layers = [l for l in model.layers if not isinstance(l, keras.layers.InputLayer)]
        embed_layer = None
        for l in reversed(layers[:-1]):
            if isinstance(l, (keras.layers.Dense, keras.layers.Dropout,
                              keras.layers.GlobalAveragePooling2D,
                              keras.layers.Bidirectional)):
                embed_layer = l
                break
        if embed_layer is None:
            return None
        x = inp = keras.Input(shape=input_shape)
        for l in layers:
            x = l(x)
            if l is embed_layer:
                break
        return keras.Model(inputs=inp, outputs=x)
    except Exception:
        return None


def _lr_head(emb_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs",
        class_weight="balanced", max_iter=2000, C=1.0, random_state=SEED,
    ).fit(emb_train, y_train)


# ─── Route ───────────────────────────────────────────────────────────────────

@router.post("/classify")
async def classify_recordings(
    session_id: str = Header(..., alias="X-Session-Id"),
    files: list[UploadFile] = File(...),
):
    """
    Upload WAV files → extract hits → classify (raw + CORAL) → return per-flange verdicts.
    """
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if not session.features:
        raise HTTPException(400, "Extract features first (Step 4)")
    if not session.training_results:
        raise HTTPException(400, "Train models first (Step 5)")
    if not session.models:
        raise HTTPException(400, "No saved models found — please retrain.")

    # ── 1. Parse filenames + extract hits ────────────────────────────────────
    all_waveforms: list[np.ndarray] = []
    all_meta:      list[dict]       = []
    recordings:    list[dict]       = []

    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            flange_id, area_id = _parse_filename(f.filename)
            path = os.path.join(tmp, f.filename)
            with open(path, "wb") as fp:
                fp.write(await f.read())

            windows, _ = extract_hits_from_file(
                filepath=path, class_idx=0, flange_id=flange_id, area_id=area_id
            )
            n_before = len(all_waveforms)
            for w in windows:
                all_waveforms.append(w)
                all_meta.append({"flange_id": flange_id, "area_id": area_id,
                                  "filename": f.filename})
            recordings.append({
                "filename":  f.filename,
                "flange_id": flange_id,
                "area_id":   area_id,
                "n_hits":    len(all_waveforms) - n_before,
            })

    if not all_waveforms:
        raise HTTPException(400, "No usable hits detected in the uploaded files")

    N = len(all_waveforms)

    # ── 2. Feature extraction ─────────────────────────────────────────────────
    X_test = np.stack([extract_features(w) for w in all_waveforms]).astype(np.float32)
    X_src  = np.array(session.features["X_feat"], dtype=np.float32)
    y_src  = np.array(session.features["labels"],  dtype=np.int64)
    X_test = _impute_with_train(X_test, X_src)

    # Mel spectrograms (lazy — only if CNN or LSTM saved)
    need_deep = any(k in session.models for k in ("CNN", "LSTM", "BiLSTM"))
    X_spec = X_seq = None
    if need_deep:
        X_spec = np.stack([
            extract_mel_spectrogram(w)[..., np.newaxis] for w in all_waveforms
        ])  # (N, 64, 128, 1)
        X_seq = np.stack([
            extract_mel_spectrogram(w).T for w in all_waveforms
        ])  # (N, 128, 64)

    # ── 3. Ensemble weights (Task-2 LOIO accuracy) ────────────────────────────
    weights: dict[str, float] = {
        m: float(session.training_results[m]["task2_mean"])
        for m in session.models
        if m in session.training_results
    }

    # ── 4. Phase 1 — Raw classification ──────────────────────────────────────
    raw_probas: dict[str, np.ndarray] = {}

    for mname in ("SVM", "LR", "KNN"):
        if mname not in session.models:
            continue
        try:
            clf    = session.models[mname]["clf"]
            scaler = session.models[mname]["scaler"]
            raw_probas[mname] = clf.predict_proba(scaler.transform(X_test))
        except Exception:
            traceback.print_exc()

    if "MLP" in session.models:
        try:
            import tensorflow as tf
            mlp    = tf.keras.models.load_model(session.models["MLP"]["model_path"])
            scaler = session.models["MLP"]["scaler"]
            raw_probas["MLP"] = mlp.predict(scaler.transform(X_test), verbose=0)
        except Exception:
            traceback.print_exc()

    if "CNN" in session.models and X_spec is not None:
        try:
            import tensorflow as tf
            cnn = tf.keras.models.load_model(session.models["CNN"]["model_path"])
            raw_probas["CNN"] = cnn.predict(X_spec, verbose=0)
        except Exception:
            traceback.print_exc()

    for lstm_key in ("LSTM", "BiLSTM"):
        if lstm_key in session.models and X_seq is not None:
            try:
                import tensorflow as tf
                lstm = tf.keras.models.load_model(session.models[lstm_key]["model_path"])
                raw_probas[lstm_key] = lstm.predict(X_seq, verbose=0)
            except Exception:
                traceback.print_exc()
            break

    if raw_probas:
        raw_probas["Ensemble"] = _weighted_ens(raw_probas, weights)

    raw_ensemble = raw_probas.get("Ensemble")
    raw_per_flange = _agg_per_flange(raw_ensemble, all_meta) if raw_ensemble is not None else []

    # ── 5. Phase 2 — CORAL classification ────────────────────────────────────
    coral_probas: dict[str, np.ndarray] = {}
    dist_before = dist_after = None
    Xt_coral = None

    src_scaler = next(
        (session.models[m]["scaler"] for m in ("SVM", "LR", "KNN") if m in session.models),
        None,
    )

    if src_scaler is not None:
        Xs_s = src_scaler.transform(X_src).astype(np.float64)
        Xt_s = src_scaler.transform(X_test).astype(np.float64)
        dist_before = _cov_dist(Xs_s, Xt_s)
        Xt_coral    = _coral(Xs_s, Xt_s)
        dist_after  = _cov_dist(Xs_s, Xt_coral)

        # Feature-CORAL: shallow models
        for mname in ("SVM", "LR", "KNN"):
            if mname not in session.models:
                continue
            try:
                clf = session.models[mname]["clf"]
                coral_probas[mname] = clf.predict_proba(Xt_coral)
            except Exception:
                traceback.print_exc()

        # Feature-CORAL: MLP (uses its own scaler, same fit data)
        if "MLP" in session.models:
            try:
                import tensorflow as tf
                mlp      = tf.keras.models.load_model(session.models["MLP"]["model_path"])
                sc_mlp   = session.models["MLP"]["scaler"]
                Xs_mlp   = sc_mlp.transform(X_src).astype(np.float64)
                Xt_mlp   = sc_mlp.transform(X_test).astype(np.float64)
                coral_probas["MLP"] = mlp.predict(_coral(Xs_mlp, Xt_mlp), verbose=0)
            except Exception:
                traceback.print_exc()

    # Deep CORAL: CNN
    if "CNN" in session.models and X_spec is not None:
        try:
            import tensorflow as tf
            cnn = tf.keras.models.load_model(session.models["CNN"]["model_path"])
            _ = cnn(X_spec[:1])  # warm-up

            wvs = session.hits.get("waveforms", [])
            Xsp_train = np.stack([
                extract_mel_spectrogram(np.array(w, dtype=np.float32))[..., np.newaxis]
                for w in wvs
            ])

            embedder = _build_embedder(cnn, (64, 128, 1))
            if embedder is not None:
                emb_tr    = embedder.predict(Xsp_train, verbose=0)
                emb_te    = embedder.predict(X_spec,    verbose=0)
                emb_coral = _coral(emb_tr, emb_te)
                coral_probas["CNN"] = _lr_head(emb_tr, y_src).predict_proba(emb_coral)
        except Exception:
            traceback.print_exc()

    # Deep CORAL: LSTM / BiLSTM
    for lstm_key in ("LSTM", "BiLSTM"):
        if lstm_key not in session.models or X_seq is None:
            continue
        try:
            import tensorflow as tf
            lstm = tf.keras.models.load_model(session.models[lstm_key]["model_path"])
            _ = lstm(X_seq[:1])

            wvs = session.hits.get("waveforms", [])
            Xseq_train = np.stack([
                extract_mel_spectrogram(np.array(w, dtype=np.float32)).T
                for w in wvs
            ])

            embedder = _build_embedder(lstm, (128, 64))
            if embedder is not None:
                emb_tr    = embedder.predict(Xseq_train, verbose=0)
                emb_te    = embedder.predict(X_seq,      verbose=0)
                emb_coral = _coral(emb_tr, emb_te)
                coral_probas[lstm_key] = _lr_head(emb_tr, y_src).predict_proba(emb_coral)
        except Exception:
            traceback.print_exc()
        break

    if coral_probas:
        coral_probas["Ensemble"] = _weighted_ens(coral_probas, weights)

    coral_ensemble = coral_probas.get("Ensemble")
    coral_per_flange = _agg_per_flange(coral_ensemble, all_meta) if coral_ensemble is not None else []

    # ── 6. PCA scatter (scaled-feature space) ────────────────────────────────
    Xs_pca = src_scaler.transform(X_src) if src_scaler else X_src
    mean_s = Xs_pca.mean(axis=0)
    Xs_c   = Xs_pca - mean_s
    _, evecs = np.linalg.eigh(np.cov(Xs_c.T))
    pcs = evecs[:, -2:]   # top 2 eigenvectors

    train_2d = Xs_c @ pcs

    Xt_pca = src_scaler.transform(X_test) if src_scaler else X_test
    test_2d_raw   = (Xt_pca   - mean_s) @ pcs
    test_2d_coral = (Xt_coral - mean_s) @ pcs if Xt_coral is not None else test_2d_raw

    ens_raw = raw_ensemble
    ens_coral = coral_ensemble if coral_ensemble is not None else ens_raw
    raw_preds   = np.argmax(ens_raw,   axis=1) if ens_raw   is not None else np.zeros(N, int)
    coral_preds = np.argmax(ens_coral, axis=1) if ens_coral is not None else raw_preds

    pca_train = [
        {"x": round(float(train_2d[i, 0]), 4), "y": round(float(train_2d[i, 1]), 4),
         "label": int(y_src[i])}
        for i in range(len(y_src))
    ]
    pca_test_raw = [
        {"x": round(float(test_2d_raw[i, 0]), 4), "y": round(float(test_2d_raw[i, 1]), 4),
         "pred_label": int(raw_preds[i]), "flange_id": all_meta[i]["flange_id"]}
        for i in range(N)
    ]
    pca_test_coral = [
        {"x": round(float(test_2d_coral[i, 0]), 4), "y": round(float(test_2d_coral[i, 1]), 4),
         "pred_label": int(coral_preds[i]), "flange_id": all_meta[i]["flange_id"]}
        for i in range(N)
    ]

    raw_final   = raw_per_flange
    coral_final = coral_per_flange if coral_per_flange else raw_final

    return {
        "status":     "done",
        "n_hits":     N,
        "recordings": recordings,
        "raw": {
            "ensemble_weights": {k: round(float(v), 4) for k, v in weights.items()},
            "final_prediction": raw_final,
        },
        "coral": {
            "cov_distance_before": round(float(dist_before), 4) if dist_before is not None else None,
            "cov_distance_after":  round(float(dist_after),  4) if dist_after  is not None else None,
            "improvement_pct":     round(
                (dist_before - dist_after) / (dist_before + 1e-9) * 100, 1
            ) if dist_before is not None and dist_after is not None else None,
            "final_prediction": coral_final,
        },
        "pca": {
            "train":      pca_train,
            "test_raw":   pca_test_raw,
            "test_coral": pca_test_coral,
        },
    }
