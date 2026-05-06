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
import os
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


def _model_dir() -> str:
    """Persistent directory for Keras model files, one per session."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_cache")
    os.makedirs(d, exist_ok=True)
    return d


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
        solver="lbfgs", random_state=SEED,
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
        keras.layers.Dropout(0.35),

        keras.layers.Dense(64),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dropout(0.25),

        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(n_mels: int = 64, n_frames: int = 128, n_classes: int = 3):
    """CNN architecture matching notebook Cell 10."""
    from tensorflow import keras

    inp = keras.layers.Input(shape=(n_mels, n_frames, 1))
    x = keras.layers.Conv2D(32, 3, padding="same")(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.50)(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_bilstm(n_frames: int = 128, n_mels: int = 64, n_classes: int = 3):
    """Bidirectional LSTM architecture matching notebook Cell 11."""
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(n_frames, n_mels)),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.40),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.30),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.20),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _make_spec_dataset(X, y, batch_size=32, augment=False, shuffle=True):
    import tensorflow as tf

    def _augment(spec, label):
        if tf.random.uniform([]) < 0.5:
            spec = spec + tf.random.normal(tf.shape(spec), mean=0.0, stddev=0.10)
        if tf.random.uniform([]) < 0.5:
            f = tf.random.uniform([], 1, 5, dtype=tf.int32)
            f0 = tf.random.uniform([], 0, 64 - f, dtype=tf.int32)
            mask = tf.concat([
                tf.ones([f0, 128, 1]),
                tf.zeros([f, 128, 1]),
                tf.ones([64 - f0 - f, 128, 1]),
            ], axis=0)
            spec = spec * mask
        if tf.random.uniform([]) < 0.5:
            t = tf.random.uniform([], 1, 9, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, 128 - t, dtype=tf.int32)
            mask = tf.concat([
                tf.ones([64, t0, 1]),
                tf.zeros([64, t, 1]),
                tf.ones([64, 128 - t0 - t, 1]),
            ], axis=1)
            spec = spec * mask
        return spec, label

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _make_seq_dataset(X, y, batch_size=32, augment=False, shuffle=True):
    import tensorflow as tf

    def _augment(seq, label):
        if tf.random.uniform([]) < 0.5:
            seq = seq + tf.random.normal(tf.shape(seq), 0.0, 0.10)
        if tf.random.uniform([]) < 0.5:
            f = tf.random.uniform([], 1, 5, dtype=tf.int32)
            f0 = tf.random.uniform([], 0, 64 - f, dtype=tf.int32)
            mask = tf.concat([
                tf.ones([128, f0]),
                tf.zeros([128, f]),
                tf.ones([128, 64 - f0 - f]),
            ], axis=1)
            seq = seq * mask
        if tf.random.uniform([]) < 0.5:
            t = tf.random.uniform([], 1, 9, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, 128 - t, dtype=tf.int32)
            mask = tf.concat([
                tf.ones([t0, 64]),
                tf.zeros([t, 64]),
                tf.ones([128 - t0 - t, 64]),
            ], axis=0)
            seq = seq * mask
        return seq, label

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class WSCallback:
    """Keras callback helper that emits epoch metrics over websocket queue."""

    def __init__(self, loop, queue, model_name, total_epochs):
        self.loop = loop
        self.queue = queue
        self.model_name = model_name
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        emit(self.loop, self.queue, {
            "type": "epoch",
            "model": self.model_name,
            "epoch": epoch + 1,
            "total": self.total_epochs,
            "train_acc": round(float(logs.get("accuracy", 0)), 4),
            "val_acc": round(float(logs.get("val_accuracy", 0)), 4),
            "train_loss": round(float(logs.get("loss", 0)), 4),
            "val_loss": round(float(logs.get("val_loss", 0)), 4),
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

        # Final model on all data (also saved for Step 9 classification)
        clf_final = SHALLOW_FACTORIES[model_name]()
        clf_final.fit(X_s, y)
        train_acc = float(accuracy_score(y, clf_final.predict(X_s)))
        session.models[model_name] = {"clf": clf_final, "scaler": scaler}

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

def _train_mlp(X, y, groups, loop, queue, session, epochs=200):
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
        es     = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
        rlrop  = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
        )
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=32,
                  validation_split=0.10,
                  callbacks=[cb, es, rlrop], verbose=0)

        y_p1   = np.argmax(model(X_te, training=False).numpy(), axis=1)
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
            es2 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
            rlrop2 = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
            )
            m2.fit(X_s[tr_idx], y[tr_idx], epochs=epochs, batch_size=32,
                   validation_split=0.10, callbacks=[es2, rlrop2], verbose=0)
            yp    = np.argmax(m2(X_s[te_idx], training=False).numpy(), axis=1)
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
        Xtr_f, Xval_f, ytr_f, yval_f = train_test_split(
            X_s, y, test_size=0.10, stratify=y, random_state=SEED
        )
        mlp_final = build_mlp(X_s.shape[1])
        es_f = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
        rlrop_f = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
        )
        mlp_final.fit(
            Xtr_f, ytr_f, validation_data=(Xval_f, yval_f),
            epochs=epochs, batch_size=32, verbose=0, callbacks=[es_f, rlrop_f]
        )
        train_acc = float(accuracy_score(y, np.argmax(mlp_final(X_s, training=False).numpy(), axis=1)))

        # Save model + scaler for Step 9 classification
        try:
            mp = os.path.join(_model_dir(), f"{session.session_id}_MLP.keras")
            mlp_final.save(mp)
            session.models["MLP"] = {"scaler": scaler, "model_path": mp}
        except Exception:
            traceback.print_exc()

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

def _train_cnn(waveforms, y, groups, loop, queue, session, epochs=100):
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
        Xtr_in, Xval, ytr_in, yval = train_test_split(
            X_tr, y_tr, test_size=0.10, stratify=y_tr, random_state=SEED
        )
        train_ds = _make_spec_dataset(Xtr_in, ytr_in, batch_size=32, augment=True, shuffle=True)
        val_ds = _make_spec_dataset(Xval, yval, batch_size=32, augment=False, shuffle=False)
        model = build_cnn()
        cb = _make_keras_callback(loop, queue, model_name, epochs)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        rlrop = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6
        )
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[cb, es, rlrop], verbose=0)

        y_p1   = np.argmax(model(X_te, training=False).numpy(), axis=1)
        acc_t1 = float(accuracy_score(y_te, y_p1))
        f1_t1  = float(f1_score(y_te, y_p1, average="macro"))
        cm_t1  = confusion_matrix(y_te, y_p1, labels=[0, 1, 2]).tolist()

        # Task 2 — LOIO
        logo = LeaveOneGroupOut()
        fold_accs, fold_records, all_yt, all_yp = [], [], [], []

        for fold_i, (tr_idx, te_idx) in enumerate(logo.split(X_spec, y, groups)):
            flange_out = int(groups[te_idx[0]])
            Xtr_f, Xval_f, ytr_f, yval_f = train_test_split(
                X_spec[tr_idx], y[tr_idx], test_size=0.10, stratify=y[tr_idx], random_state=SEED
            )
            train_ds_f = _make_spec_dataset(Xtr_f, ytr_f, batch_size=32, augment=True, shuffle=True)
            val_ds_f = _make_spec_dataset(Xval_f, yval_f, batch_size=32, augment=False, shuffle=False)
            m2 = build_cnn()
            es2 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
            rlrop2 = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6
            )
            cb2 = _make_keras_callback(loop, queue, model_name, epochs)
            m2.fit(train_ds_f, validation_data=val_ds_f, epochs=epochs, callbacks=[cb2, es2, rlrop2], verbose=0)
            yp    = np.argmax(m2(X_spec[te_idx], training=False).numpy(), axis=1)
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
        train_acc = float(accuracy_score(y, np.argmax(model(X_spec, training=False).numpy(), axis=1)))

        # Save Task-1 model for classify/Deep CORAL
        try:
            mp = os.path.join(_model_dir(), f"{session.session_id}_CNN.keras")
            model.save(mp)
            session.models["CNN"] = {"model_path": mp}
        except Exception:
            traceback.print_exc()

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

def _train_bilstm(waveforms, y, groups, loop, queue, session, epochs=80, model_name="LSTM"):
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
        Xtr_in, Xval, ytr_in, yval = train_test_split(
            X_tr, y_tr, test_size=0.10, stratify=y_tr, random_state=SEED
        )
        train_ds = _make_seq_dataset(Xtr_in, ytr_in, batch_size=32, augment=True, shuffle=True)
        val_ds = _make_seq_dataset(Xval, yval, batch_size=32, augment=False, shuffle=False)
        model = build_bilstm()
        cb = _make_keras_callback(loop, queue, model_name, epochs)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        rlrop = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6
        )
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[cb, es, rlrop], verbose=0)

        y_p1   = np.argmax(model(X_te, training=False).numpy(), axis=1)
        acc_t1 = float(accuracy_score(y_te, y_p1))
        f1_t1  = float(f1_score(y_te, y_p1, average="macro"))
        cm_t1  = confusion_matrix(y_te, y_p1, labels=[0, 1, 2]).tolist()

        # Task 2 — LOIO
        logo = LeaveOneGroupOut()
        fold_accs, fold_records, all_yt, all_yp = [], [], [], []

        for fold_i, (tr_idx, te_idx) in enumerate(logo.split(X_seq, y, groups)):
            flange_out = int(groups[te_idx[0]])
            Xtr_f, Xval_f, ytr_f, yval_f = train_test_split(
                X_seq[tr_idx], y[tr_idx], test_size=0.10, stratify=y[tr_idx], random_state=SEED
            )
            train_ds_f = _make_seq_dataset(Xtr_f, ytr_f, batch_size=32, augment=True, shuffle=True)
            val_ds_f = _make_seq_dataset(Xval_f, yval_f, batch_size=32, augment=False, shuffle=False)
            m2 = build_bilstm()
            es2 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
            rlrop2 = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6
            )
            cb2 = _make_keras_callback(loop, queue, model_name, epochs)
            m2.fit(train_ds_f, validation_data=val_ds_f, epochs=epochs, callbacks=[cb2, es2, rlrop2], verbose=0)
            yp    = np.argmax(m2(X_seq[te_idx], training=False).numpy(), axis=1)
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

        # Save Task-1 model for classify/Deep CORAL
        try:
            mp = os.path.join(_model_dir(), f"{session.session_id}_{model_name}.keras")
            model.save(mp)
            session.models[model_name] = {"model_path": mp}
        except Exception:
            traceback.print_exc()

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

ALL_MODELS = ["SVM", "LR", "KNN", "MLP", "CNN", "LSTM"]

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
