"""Pinned Laya Vision labels in a separate process, with six explicit options."""

import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
from threading import Lock, Thread

from flydoom.movement_core import ACTIONS
from flydoom.vision import VisionClient, verified_manifest


class MovementTeacher(VisionClient):
    def __init__(self, log):
        self.log = open(log, "a", encoding="utf-8")
        self.lock, self.messages = Lock(), Queue()
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONIOENCODING": "utf-8",
               "KMP_BLOCKTIME": "0", "OMP_WAIT_POLICY": "PASSIVE"}
        self.process = subprocess.Popen([sys.executable, "-u", "-m", "flydoom.movement_teacher"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8", env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        def reader():
            for line in self.process.stdout:
                self.messages.put(line)
            self.messages.put(None)
        Thread(target=reader, daemon=True).start()
        try:
            self.metadata = self.receive(180)["ready"]
        except BaseException:
            self.close()
            raise


def worker():
    import numpy as np
    model, source = Path("models/laya-vision"), Path("models/laya-vision-source")
    manifest = verified_manifest(model, source)
    sys.path.insert(0, str(source.resolve()))
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import laya
        from PIL import Image
        torch.set_num_threads(4)
        agent = laya.load_vlm(str(model.resolve()), device="cuda" if torch.cuda.is_available() else "cpu", dtype="fp32")
        agent.model.eval()
    question = {"action": {"type": "choice", "instructions":
        "Play Doom target shooting. Align with the monster and shoot. Use forward or backward movement only when repositioning helps; do not move needlessly.",
        "criteria": dict(zip(ACTIONS, ["Wait", "Strafe left", "Strafe right", "Shoot", "Move forward", "Move backward"]))}}
    print(json.dumps({"ready": {"revision": manifest["revision"], "source_commit": manifest["source_commit"],
                                "question": question, "actions": ACTIONS, "weights_trained": False}}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            frame = np.frombuffer(base64.b64decode(request["pixels"], validate=True), np.uint8).reshape(request["shape"])
            if frame.shape != (240, 320, 3):
                raise ValueError("Require native RGB observations")
            raw = {}
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                result = agent.predict({"image": Image.fromarray(frame)}, question, strict=True, n_permutations=1, _raw_logits=raw)
            logits = np.asarray(raw["action"], dtype=float)
            if logits.shape != (6,) or not np.isfinite(logits).all():
                raise ValueError("Invalid six-action logits")
            temperature = max(float(result["provenance"]["temperatures"]["action"]), 1e-3)
            z = logits / temperature
            probabilities = np.exp(z - z.max()); probabilities /= probabilities.sum()
            print(json.dumps({"result": {"probabilities": probabilities.tolist(), "logits": logits.tolist(),
                  "temperature": temperature, "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest()}}), flush=True)
        except Exception as error:
            print(json.dumps({"error": str(error)}), flush=True)


if __name__ == "__main__":
    worker()
