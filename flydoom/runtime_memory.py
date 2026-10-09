"""Load the Vision observer on demand and release its process after idle time."""

import gc
from threading import Event, Lock, Thread
from time import monotonic

from flydoom.vision import VisionClient, verified_manifest


class IdleVision:
    def __init__(self, idle_seconds=60):
        manifest = verified_manifest("models/laya-vision", "models/laya-vision-source")
        self.metadata = {k: manifest[k] for k in ("repo", "revision", "source_commit")}
        self.metadata.update(device="unloaded", weights_trained_in_session=False)
        self.client, self.last_used = None, monotonic()
        self.lock, self.closed = Lock(), Event()
        self.idle_seconds = idle_seconds
        Thread(target=self._monitor, daemon=True).start()

    def _monitor(self):
        while not self.closed.wait(5):
            with self.lock:
                if self.client and monotonic() - self.last_used > self.idle_seconds:
                    self.client.close()
                    self.client = None

    def predict(self, frame):
        with self.lock:
            if self.closed.is_set():
                raise ValueError("Vision observer has been closed")
            if self.client is None:
                self.client = VisionClient()
                self.metadata.update(self.client.metadata)
            try:
                return self.client.predict(frame)
            finally:
                self.last_used = monotonic()

    def release(self):
        with self.lock:
            if self.client:
                self.client.close()
                self.client = None
        gc.collect()
        return {"vision_loaded": False, "note": "Vision reloads on the next decision. Active model, graph and files are retained."}

    def close(self):
        self.closed.set()
        self.release()
