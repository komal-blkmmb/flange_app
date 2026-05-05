"""
In-memory session store. Each browser session gets a UUID that acts as the
key for all uploaded files, extracted features, and training results.

No database needed — data is ephemeral and cleared after SESSION_TTL_SECS.
"""

import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from config import SESSION_TTL_SECS


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # Step 2 — uploaded files
    uploaded_files: list[dict] = field(default_factory=list)   # [{filename, filepath, class_label, flange_id, area_id}]
    lab_files: list[dict] = field(default_factory=list)        # test files for CORAL

    # Step 3 — extracted hits (stored as list of float lists to avoid numpy pickling issues)
    hits: dict = field(default_factory=dict)          # {waveforms, labels, flange_groups, area_groups}
    processing_stats: dict = field(default_factory=dict)

    # Step 4 — features
    features: dict = field(default_factory=dict)      # {X_feat, feature_names, X_pca, pca_components}
    lab_features: dict = field(default_factory=dict)  # features for lab test files

    # Step 5 — training results keyed by model name
    training_results: dict = field(default_factory=dict)   # {"SVM": ModelResult, ...}
    training_tasks: dict = field(default_factory=dict)     # task_id → model_name (for WebSocket routing)

    # Step 6 — ensemble
    ensemble_result: dict = field(default_factory=dict)

    # Step 7 / 8 — CORAL
    coral_result: dict = field(default_factory=dict)

    # Arbitrary scratch storage for any additional data
    extra: dict = field(default_factory=dict)

    def touch(self):
        self.last_active = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > SESSION_TTL_SECS


class SessionManager:
    """Thread-safe in-memory session store with automatic TTL cleanup."""

    def __init__(self):
        self._store: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._start_cleanup_thread()

    def create(self) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id)
        with self._lock:
            self._store[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            s = self._store.get(session_id)
            if s and not s.is_expired():
                s.touch()
                return s
            if s and s.is_expired():
                del self._store[session_id]
            return None

    def require(self, session_id: str) -> Session:
        """Get session or raise ValueError (used in routes)."""
        s = self.get(session_id)
        if s is None:
            raise ValueError(f"Session '{session_id}' not found or expired.")
        return s

    def delete(self, session_id: str):
        with self._lock:
            self._store.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._store)

    def _cleanup(self):
        while True:
            time.sleep(300)  # check every 5 minutes
            with self._lock:
                expired = [sid for sid, s in self._store.items() if s.is_expired()]
                for sid in expired:
                    del self._store[sid]

    def _start_cleanup_thread(self):
        t = threading.Thread(target=self._cleanup, daemon=True)
        t.start()


# Singleton — imported everywhere
session_manager = SessionManager()
