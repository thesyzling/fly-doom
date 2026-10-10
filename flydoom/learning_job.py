"""Local subprocess supervision for a single bounded learning cycle."""

import json
import os
from pathlib import Path
import subprocess
import sys

from flydoom.learning_cycle import ROOT, champion, rollback


class LearningJob:
    def __init__(self): self.process = None

    def state(self):
        path = ROOT / "latest.json"
        status = json.loads(path.read_text()) if path.exists() else {"status": "not_started", "phase": "idle"}
        active = self.process is not None and self.process.poll() is None
        registry = ROOT / "registry.json"
        can_rollback = bool(json.loads(registry.read_text()).get("history")) if registry.exists() else False
        if self.process is not None and not active and self.process.returncode and status.get("status") == "running":
            status.update(status="failed", error=f"Worker exited with code {self.process.returncode}; see worker.log")
        return {**status, "worker_active": active, "locked": (ROOT / "cycle.lock").exists(), "can_rollback": can_rollback,
                "selected_checkpoint": champion(), "log": str(ROOT / "worker.log")}

    def start(self):
        if self.state()["worker_active"] or (ROOT / "cycle.lock").exists(): raise ValueError("A learning cycle is already active")
        ROOT.mkdir(parents=True, exist_ok=True)
        with (ROOT / "worker.log").open("a", encoding="utf-8") as log:
            self.process = subprocess.Popen([sys.executable, "-u", "-m", "flydoom.synaptic_consensus"],
                stdout=log, stderr=log, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return {"started": True, "pid": self.process.pid}

    def cancel(self):
        status = self.state()
        if status.get("status") != "running" or not status.get("output"): raise ValueError("No running cycle to cancel")
        folder = Path(status["output"]).resolve()
        if folder.parent != ROOT.resolve(): raise ValueError("Invalid learning output")
        (folder / "cancel.request").write_text("Cancel after the current bounded operation.\n", encoding="ascii")
        return {"cancellation_requested": True}

    def rollback(self): return rollback()
