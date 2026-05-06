"""
Router: POST /api/train   — launch training as a background task
        GET  /api/results  — fetch completed results
        WS   /ws/train/{task_id} — stream live epoch/fold metrics

Models (matching final_project_saurav_silwal.ipynb exactly):
  Shallow (82-dim tabular features): SVM, LR, KNN
  Deep (82-dim tabular):             MLP  (Keras, 3 hidden layers + dropout)
  Deep (mel spectrogram):            CNN, BiLSTM (Keras)

All models: Task 1 (70/30 dependent split) + Task 2 (LOIO cross-validation)
"""

import asyncio
import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from fastapi import APIRouter, Header, HTTPException, WebSocket, WebSocketDisconnect

from session import session_manager
from ws.training_ws import ws_manager, emit
from ml.feature_extraction import (
    extract_mel_spectrogram, impute_nans, FEATURE_NAMES
)
from config import SEED, TEST_SIZE, IDX_TO_CLASS, CLASS_NAMES, N_CLASSES

router    = APIRouter(tags=["training"])
_executor = ThreadPoolExecutor(max_workers=2)


# ─────────────────────────────────────────────────────────────────────────────
# Shallow model factories (sklearn)
# ─────────────────────────────────────────────────────────────────────────────

SHALLOW_FACTORIES = {
    "SVM": lambda: SVC(
        kernel="rbf", C=10.0, gamma="scale",
        probability=True, class_weight="balanced", random_state=SEED,
    ),
    "LR": lambda: LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced",
        multi_class="multinomial", solver="lbfgs", random_state=SEED,
    ),
    "KNN": lambda: KNeighborsClassifier(
        n_neighbors=5, metric="euclidean", weights="uniform",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Keras model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp(input_dim: int, n_classes: int = 3):
    """3-layer MLP with BatchNorm + Dropout. Matches notebook Cell 9."""
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(128),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(64),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(n_mels: int = 64, n_frames: int = 128, n_classes: int = 3):
    """CNN on log-mel spectrogram (64×128×1). Matches notebook Cell 10."""
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(n_mels, n_frames, 1)),

        keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_bilstm(n_frames: int = 128, n_mels: int = 64, n_classes: int = 3):
    """Bidirectional LSTM on mel sequences (128 time steps × 64 mel features).
    Matches notebook Cell 11."""
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(n_frames, n_mels)),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Dropout(0.3),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket epoch callback for Keras
# ─────────────────────────────────────────────────────────────────────────────

class WSCallback:
    """Keras callback that emits epoch metrics over WebSocket."""

    def __init__(self, loop, queue, model_name, total_epochs):
        self.loop        = loop
        self.queue       = queue
        self.model_name  = model_name
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        emit(self.loop, self.queue, {
            "type":       "epoch",
            "model":      self.model_name,
            "epoch":      epoch + 1,
            "total":      self.total_epochs,
            "train_acc":  round(float(logs.get("accuracy", 0)), 4),
            "val_acc":    round(float(logs.get("val_accuracy", 0)), 4),
            "train_loss": round(float(logs.get("loss", 0)), 4),
            "val_loss":   round(float(logs.get("val_loss", 0)), 4),
        })


def _make_keras_callback(loop, queue, model_name, total_epochs):
    """Return a tf.keras.callbacks.Callback subclass instance."""
    import tensorflow as tf

    cb = WSCallback(loop, queue, model_name, total_epochs)

    class _CB(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            cb.on_epoch_end(epoch, logs)

    return _CB()


# ─────────────────────────────────────────────────────────────────────────────
# Shallow model training (SVM / LR / KNN)
# ─────────────────────────────────────────────────────────────────────────────

def _train_shallow(model_name, X, y, groups, loop, queue, session):
    try:
        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)

        # Task 1
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_s, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
        )
        clf = SHALLOW_FACTORIES[model_name]()
        clf.fit(X_tr, y_tr)
        y_p1   = clf.predict(X_te)
        acc_t1 = float(accuracy_score(y_te, y_p1))
        f1_t1  = float(f1_score(y_te, y_p1, average="macro"))
        cm_t1  = confusion_matrix(y_te, y_p1, labels=[0, 1, 2]).tolist()

        emit(loop, queue, {"type": "task1_done", "model": model_name,
                           "acc": round(acc_t1, 4), "f1": round(f1_t1, 4)})

        # Task 2 — LOIO
        logo         = LeaveOneGroupOut()
        fold_accs    = []
        fold_records = []
        all_yt, all_yp = [], []

        for fold_i, (tr_idx, te_idx) in enumerate(logo.split(X_s, y, groups)):
            flange_out = int(groups[te_idx[0]])
            clf2       = SHALLOW_FACTORIES[model_name]()
            clf2.fit(X_s[tr_idx], y[tr_idx])
            yp   = clf2.predict(X_s[te_idx])
            acc_f = float(accuracy_score(y[te_idx], yp))
            fold_accs.append(acc_f)
            fold_records.append({"fold": fold_i + 1, "flange_out": flange_out,
                                  "acc": round(acc_f, 4), "n_test": len(te_idx)})
            all_yt.extend(y[te_idx].tolist())
            all_yp.extend(yp.tolist())
            emit(loop, queue, {"type": "fold_done", "model": model_name,
                                "fold": fold_i + 1, "flange_out": flange_out,
                                "acc": round(acc_f, 4)})

        cm_t2  = confusion_matrix(all_yt, all_yp, labels=[0, 1, 2]).tolist()
        f1_t2  = float(f1_score(all_yt, all_yp, average="macro"))

        # Final model on all data
        clf_final = SHALLOW_FACTORIES[model_name]()
        clf_final.fit(X_s, y)
        train_acc = float(accuracy_score(y, clf_final.predict(X_s)))

        result = {
            "model":        model_name,
            "task1_acc":    round(acc_t1, 4),
            "task1_f1":     round(f1_t1, 4),
            "task1_cm":     cm_t1,
            "task2_mean":   round(float(np.mean(fold_accs)), 4),
            "task2_std":    round(float(np.std(fold_accs)), 4),
            "task2_f1":     round(f1_t2, 4),
            "task2_cm":     cm_t2,
            "folds":        fold_records,
            "train_acc":    round(train_acc, 4),
        }
        session.training_results[model_name] = result
        session.touch()
        emit(loop, queue, {"type": "model_done", "model": model_name, **result})

    except Exception as e:
        emit(loop, queue, {"type": "error", "model": model_name, "message": str(e)})
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# MLP training (Keras, tabular features)
# ─────────────────────────────────────────────────────────────────────────────

def _train_mlp(X, y, groups, loop, queue, session, epochs=60):
    model_name = "MLP"
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)

        # Task 1
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_s, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
        )
        model  = build_mlp(X_s.shape[1])
        cb     = _make_keras_callback(loop, queue, model_name, epochs)
        es     = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=32,
                  validation_split=0.15,
                  callbacks=[cb, es], verbose=0)

        y_p1   = np.argmax(model.predict(X_te, verbose=0), axis=1)
        acc_t1 = float(accuracy_score(y_te, y_p1))
        f1_t1  = float(f1_score(y_te, y_p1, average="macro"))
        cm_t1  = confusion_matrix(y_te, y_p1, labels=[0, 1, 2]).tolist()

        emit(loop, queue, {"type": "task1_done", "model": model_name,
                           "acc": round(acc_t1, 4), "f1": round(f1_t1, 4)})

        # Task 2 — LOIO
        logo         = LeaveOneGroupOut()
        fold_accs    = []
        fold_records = []
        all_yt, all_yp = [], []

        for fold_i, (tr_idx, te_idx) in enumerate(logo.split(X_s, y, groups)):
            flange_out = int(groups[te_idx[0]])
            m2 = build_mlp(X_s.shape[1])
            es2 = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
            m2.fit(X_s[tr_idx], y[tr_idx], epochs=epochs, batch_size=32,
                   validation_split=0.15, callbacks=[es2], verbose=0)
            yp    = np.argmax(m2.predict(X_s[te_idx], verbose=0), axis=1)
            acc_f = float(accuracy_score(y[te_idx], yp))
            fold_accs.append(acc_f)
            fold_records.append({"fold": fold_i + 1, "flange_out": flange_out,
                                  "acc": round(acc_f, 4), "n_test": len(te_idx)})
            all_yt.extend(y[te_idx].tolist())
            all_yp.extend(yp.tolist())
            emit(loop, queue, {"type": "fold_done", "model": model_name,
                                "fold": fold_i + 1, "flange_out": flange_out,
                                "acc": round(acc_f, 4)})

        cm_t2 = confusion_matrix(all_yt, all_yp, labels=[0, 1, 2]).tolist()
        f1_t2 = float(f1_score(all_yt, all_yp, average="macro"))
        train_acc = float(accuracy_score(y, np.argmax(model.predict(X_s, verbose=0), axis=1)))

        result = {
            "model": model_name, "task1_acc": round(acc_t1, 4),
            "task1_f1": round(f1_t1, 4), "task1_cm": cm_t1,
            "task2_mean": round(float(np.mean(fold_accs)), 4),
            "task2_std":  round(float(np.std(fold_accs)), 4),
            "task2_f1":   round(f1_t2, 4), "task2_cm": cm_t2,
            "folds": fold_records, "train_acc": round(train_acc, 4),
        }
        session.training_results[model_name] = result
        session.touch()
        emit(loop, queue, {"type": "model_done", "model": model_name, **result})

    except Exception as e:
        emit(loop, queue, {"type": "error", "model": model_name, "message": str(e)})
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# CNN training (Keras, mel spectrograms)
# ─────────────────────────────────────────────────────────────────────────────

def _train_cnn(waveforms, y, groups, loop, queue, session, epochs=50):
    model_name = "CNN"
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)

        # Build spectrogram tensor (N, 64, 128, 1)
        emit(loop, queue, {"type": "task1_done", "model": model_name,
                           "acc": 0.0, "f1": 0.0, "message": "Extracting spectrograms..."})
        X_spec = np.stack([
            extract_mel_spectrogram(np.array(w, dtype=np.float32))
            for w in waveforms
        ], axis=0)[..., np.newaxis]   # (N, 64, 128, 1)

        # Task 1
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_spec, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
        )
        model = build_cnn()
        cb    = _make_keras_callback(loop, queue, model_name, epochs)
        es    = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=32,
                  validation_split=0.15, callbacks=[cb, es], verbose=0)

        y_p1   = np.argmax(model.predict(X_te, verbose=0), axis=1)
        acc_t1 = float(accuracy_score(y_te, y_p1))
        f1_t1  = float(f1_score(y_te, y_p1, average="macro"))
        cm_t1  = confusion_matrix(y_te, y_p1, labels=[0, 1, 2]).tolist()

        # Task 2 — LOIO
        logo = LeaveOneGroupOut()
        fold_accs, fold_records, all_yt, all_yp = [], [], [], []

        for fold_i, (tr_idx, te_idx) in enumerate(logo.split(X_spec, y, groups)):
            flange_out = int(groups[te_idx[0]])
            m2 = build_cnn()
            es2 = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
            m2.fit(X_spec[tr_idx], y[tr_idx], epochs=epochs, batch_size=32,
                   validation_split=0.15, callbacks=[es2], verbose=0)
            yp    = np.argmax(m2.predict(X_spec[te_idx], verbose=0), axis=1)
            acc_f = float(accuracy_score(y[te_idx], yp))
            fold_accs.append(acc_f)
            fold_records.append({"fold": fold_i + 1, "flange_out": flange_out,
                                  "acc": round(acc_f, 4), "n_test": len(te_idx)})
            all_yt.extend(y[te_idx].tolist())
            all_yp.extend(yp.tolist())
            emit(loop, queue, {"type": "fold_done", "model": model_name,
                                "fold": fold_i + 1, "flange_out": flange_out,
                                "acc": round(acc_f, 4)})

        cm_t2     = confusion_matrix(all_yt, all_yp, labels=[0, 1, 2]).tolist()
        f1_t2     = float(f1_score(all_yt, all_yp, average="macro"))
        train_acc = float(accuracy_score(y, np.argmax(model.predict(X_spec, verbose=0), axis=1)))

        result = {
            "model": model_name, "task1_acc": round(acc_t1, 4),
            "task1_f1": round(f1_t1, 4), "task1_cm": cm_t1,
            "task2_mean": round(float(np.mean(fold_accs)), 4),
            "task2_std":  round(float(np.std(fold_accs)), 4),
            "task2_f1":   round(f1_t2, 4), "task2_cm": cm_t2,
            "folds": fold_records, "train_acc": round(train_acc, 4),
        }
        session.training_results[model_name] = result
        session.touch()
        emit(loop, queue, {"type": "model_done", "model": model_name, **result})

    except Exception as e:
        emit(loop, queue, {"type": "error", "model": model_name, "message": str(e)})
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM training (Keras, mel sequences)
# ─────────────────────────────────────────────────────────────────────────────

def _train_bilstm(waveforms, y, groups, loop, queue, session, epochs=50, model_name="LSTM"):
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)

        # Reshape to (N, 128, 64) — time steps × mel features
        X_seq = np.stack([
            extract_mel_spectrogram(np.array(w, dtype=np.float32)).T   # (128, 64)
            for w in waveforms
        ], axis=0)

        # Task 1
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_seq, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
        )
        model = build_bilstm()
        cb    = _make_keras_callback(loop, queue, model_name, epochs)
        es    = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=32,
                  validation_split=0.15, callbacks=[cb, es], verbose=0)

        y_p1   = np.argmax(model.predict(X_te, verbose=0), axis=1)
        acc_t1 = float(accuracy_score(y_te, y_p1))
        f1_t1  = float(f1_score(y_te, y_p1, average="macro"))
        cm_t1  = confusion_matrix(y_te, y_p1, labels=[0, 1, 2]).tolist()

        # Task 2 — LOIO
        logo = LeaveOneGroupOut()
        fold_accs, fold_records, all_yt, all_yp = [], [], [], []

        for fold_i, (tr_idx, te_idx) in enumerate(logo.split(X_seq, y, groups)):
            flange_out = int(groups[te_idx[0]])
            m2 = build_bilstm()
            es2 = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
            m2.fit(X_seq[tr_idx], y[tr_idx], epochs=epochs, batch_size=32,
                   validation_split=0.15, callbacks=[es2], verbose=0)
            yp    = np.argmax(m2.predict(X_seq[te_idx], verbose=0), axis=1)
            acc_f = float(accuracy_score(y[te_idx], yp))
            fold_accs.append(acc_f)
            fold_records.append({"fold": fold_i + 1, "flange_out": flange_out,
                                  "acc": round(acc_f, 4), "n_test": len(te_idx)})
            all_yt.extend(y[te_idx].tolist())
            all_yp.extend(yp.tolist())
            emit(loop, queue, {"type": "fold_done", "model": model_name,
                                "fold": fold_i + 1, "flange_out": flange_out,
                                "acc": round(acc_f, 4)})

        cm_t2     = confusion_matrix(all_yt, all_yp, labels=[0, 1, 2]).tolist()
        f1_t2     = float(f1_score(all_yt, all_yp, average="macro"))
        train_acc = float(accuracy_score(y, np.argmax(model.predict(X_seq, verbose=0), axis=1)))

        result = {
            "model": model_name, "task1_acc": round(acc_t1, 4),
            "task1_f1": round(f1_t1, 4), "task1_cm": cm_t1,
            "task2_mean": round(float(np.mean(fold_accs)), 4),
            "task2_std":  round(float(np.std(fold_accs)), 4),
            "task2_f1":   round(f1_t2, 4), "task2_cm": cm_t2,
            "folds": fold_records, "train_acc": round(train_acc, 4),
        }
        session.training_results[model_name] = result
        session.touch()
        emit(loop, queue, {"type": "model_done", "model": model_name, **result})

    except Exception as e:
        emit(loop, queue, {"type": "error", "model": model_name, "message": str(e)})
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Master training thread
# ─────────────────────────────────────────────────────────────────────────────

ALL_MODELS = ["SVM", "LR", "KNN", "MLP", "CNN", "BiLSTM"]

def _train_all(task_id: str, session_id: str, models: list[str], main_loop: asyncio.AbstractEventLoop):

    session = session_manager.get(session_id)
    if session is None:
        return
    queue = ws_manager.get_queue(task_id)
    if queue is None:
        return

    loop = main_loop

    feats = session.features
    if not feats:
        emit(loop, queue, {"type": "error", "message": "Features not extracted yet"})
        return

    X      = np.array(feats["X_feat"],       dtype=np.float32)
    y      = np.array(feats["labels"],        dtype=np.int64)
    groups = np.array(feats["flange_groups"], dtype=np.int64)

    # NaN imputation (tau column can have NaNs)
    X = impute_nans(X, y)

    waveforms = session.hits.get("waveforms", [])

    for m in models:
        if m in SHALLOW_FACTORIES:
            _train_shallow(m, X, y, groups, loop, queue, session)
        elif m == "MLP":
            _train_mlp(X, y, groups, loop, queue, session)
        elif m == "CNN":
            _train_cnn(waveforms, y, groups, loop, queue, session)
        elif m in ("BiLSTM", "LSTM"):
            _train_bilstm(waveforms, y, groups, loop, queue, session, model_name=m)

    emit(loop, queue, {"type": "all_done", "task_id": task_id})


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/api/train")
async def start_training(
    session_id: str = Header(..., alias="X-Session-Id"),
    body: dict = None,
):
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.features:
        raise HTTPException(status_code=400, detail="Extract features first: POST /api/features")

    models  = (body or {}).get("models", ALL_MODELS)
    task_id = str(uuid.uuid4())
    ws_manager.create_queue(task_id)
    session.training_tasks[task_id] = models
    session.touch()

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _train_all, task_id, session_id, models, loop)

    return {"task_id": task_id, "models": models}


@router.get("/api/results")
async def get_results(session_id: str = Header(..., alias="X-Session-Id")):
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "models_trained": list(session.training_results.keys()),
        "results":        session.training_results,
    }


@router.websocket("/ws/train/{task_id}")
async def training_websocket(websocket: WebSocket, task_id: str):
    await ws_manager.connect(task_id, websocket)
    try:
        await ws_manager.stream(task_id, websocket)
    except WebSocketDisconnect:
        ws_manager.disconnect(task_id)
