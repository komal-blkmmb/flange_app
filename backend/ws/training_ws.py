"""
WebSocket manager for streaming live training progress.

Architecture:
  - Each training task gets a unique task_id (UUID)
  - The training background thread puts JSON events into an asyncio.Queue
  - The WebSocket endpoint drains that queue and sends to the browser
  - Events are typed: 'epoch', 'fold_done', 'model_done', 'error', 'all_done'
"""

import asyncio
import json
import uuid
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class TrainingWSManager:
    """Manages queues and WebSocket connections for live training streams."""

    def __init__(self):
        # task_id → asyncio.Queue of JSON-serialisable dicts
        self._queues: dict[str, asyncio.Queue] = {}
        # task_id → WebSocket (at most one listener per task)
        self._sockets: dict[str, WebSocket] = {}

    # ── Queue management (called from sync training threads via run_coroutine_threadsafe) ──

    def create_queue(self, task_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues[task_id] = q
        return q

    def get_queue(self, task_id: str) -> asyncio.Queue | None:
        return self._queues.get(task_id)

    def remove_queue(self, task_id: str):
        self._queues.pop(task_id, None)

    # ── WebSocket lifecycle ──

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        self._sockets[task_id] = websocket

    def disconnect(self, task_id: str):
        self._sockets.pop(task_id, None)

    # ── Streaming loop (run inside the WebSocket endpoint) ──

    async def stream(self, task_id: str, websocket: WebSocket):
        """Drain the queue and forward every event to the WebSocket client."""
        queue = self._queues.get(task_id)
        if queue is None:
            await websocket.send_json({"type": "error", "message": "Task not found"})
            return

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping"})
                    continue

                await websocket.send_json(event)
                queue.task_done()

                if event.get("type") in ("all_done", "error"):
                    break
        except Exception:
            pass
        finally:
            self.disconnect(task_id)
            self.remove_queue(task_id)


# Singleton
ws_manager = TrainingWSManager()


# ── Helper used by training threads (synchronous) ──────────────────────────────

def emit(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, event: dict):
    """
    Thread-safe emit from a synchronous training thread into the async queue.
    Call this from inside background training functions.
    """
    asyncio.run_coroutine_threadsafe(queue.put(event), loop)
