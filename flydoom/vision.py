"""Pinned, offline Laya Vision inference in an isolated Python import process."""

import argparse
import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
from threading import Lock, Thread
from time import perf_counter


def verified_manifest(model, source):
    model, source = Path(model), Path(source)
    manifest = json.loads((model / "download.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "laya_vision_download_v1":
        raise ValueError("Run flydoom.vision_setup before loading Laya Vision")
    for root, key in ((model, "sha256"), (source, "source_sha256")):
        for name, expected in manifest[key].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Artifact path leaves its pinned directory")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Laya Vision artifact changed: {name}")
    return manifest


def run_worker(model, source, device):
    import numpy as np
    manifest = verified_manifest(model, source)
    sys.path.insert(0, str(Path(source).resolve()))
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import laya
        from laya.games import doom_question
        from PIL import Image
        torch.set_num_threads(4)
        selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        agent = laya.load_vlm(str(Path(model).resolve()), device=selected_device, dtype="fp32")
        agent.model.eval()
        buttons = ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]
        question = doom_question("basic", buttons)
        captured = {}

        def capture(module, inputs, output):
            captured["activation"] = inputs[0].detach().float().cpu().numpy()[0]
            captured["logits"] = output.detach().float().cpu().numpy()[0, :, 0]

        hook = agent.model.scorer[-1].register_forward_hook(capture)
        weight = agent.model.scorer[-1].weight.detach().float().cpu().numpy()[0]
        bias = float(agent.model.scorer[-1].bias.detach().float().cpu()[0])
    metadata = {"repo": manifest["repo"], "revision": manifest["revision"],
                "source_commit": manifest["source_commit"], "device": str(agent.device),
                "dtype": "fp32", "parameters": sum(p.numel() for p in agent.model.parameters()),
                "question": question, "action_order": buttons, "weights_trained_in_session": False,
                "scorer_features": len(weight), "scorer_bias": bias, "scorer_weight": weight.tolist(),
                "manifest_sha256": hashlib.sha256((Path(model) / "download.json").read_bytes()).hexdigest()}
    print(json.dumps({"ready": metadata}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                break
            shape = request["shape"]
            if shape != [240, 320, 3]:
                raise ValueError("Require the native 320 by 240 RGB observation")
            frame = np.frombuffer(base64.b64decode(request["pixels"], validate=True), dtype=np.uint8).reshape(shape)
            started = perf_counter()
            raw = {}
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                result = agent.predict({"image": Image.fromarray(frame)}, question,
                                       strict=True, n_permutations=1, _raw_logits=raw)
            logits = np.asarray(raw["action"])
            np.testing.assert_allclose(logits, captured["logits"], atol=1e-5, rtol=1e-5)
            temperature = float(result["provenance"]["temperatures"]["action"])
            z = logits / max(temperature, 1e-3)
            probabilities = np.exp(z - z.max())
            probabilities /= probabilities.sum()
            terms = captured["activation"] * weight
            np.testing.assert_allclose(terms.sum(axis=1) + bias, logits, atol=1e-4, rtol=1e-4)
            details = []
            for i, name in enumerate(buttons):
                chosen = np.argsort(-np.abs(terms[i]), kind="stable")[:16]
                details.append({"action": name, "raw_logit": float(logits[i]), "bias": bias,
                    "sum_all_contributions": float(terms[i].sum()),
                    "other_contributions": float(terms[i].sum() - terms[i, chosen].sum()),
                    "terms": [{"feature": int(j), "weight": float(weight[j]),
                               "activation": float(captured["activation"][i, j]),
                               "contribution": float(terms[i, j])} for j in chosen]})
            answer = {"probabilities": [0.0, *probabilities.tolist()],
                      "action": buttons[int(probabilities.argmax())], "logits": logits.tolist(),
                      "temperature": temperature, "head": details, "seconds": perf_counter() - started,
                      "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                      "input_provenance": result["provenance"]}
            print(json.dumps({"result": answer}, allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)
    hook.remove()


class VisionClient:
    """Serialize inference calls and bound worker hangs; never import fork Laya here."""

    def __init__(self, *, model="models/laya-vision", source="models/laya-vision-source", device="auto",
                 log="runs/vision-worker.log"):
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log, "a", encoding="utf-8")
        self.lock, self.messages = Lock(), Queue()
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false"}
        self.process = subprocess.Popen([sys.executable, "-u", "-m", "flydoom.vision", "--worker",
            "--model", str(model), "--source", str(source), "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8",
            env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

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

    def receive(self, timeout):
        try:
            line = self.messages.get(timeout=timeout)
        except Empty:
            self.close()
            raise TimeoutError("Laya Vision worker timed out; inspect its log") from None
        if line is None:
            raise RuntimeError(f"Laya Vision worker exited; inspect {self.log.name}")
        message = json.loads(line)
        if "error" in message:
            raise RuntimeError(message["error"])
        return message

    def predict(self, frame):
        import numpy as np
        if frame.dtype != np.uint8 or frame.shape != (240, 320, 3):
            raise ValueError("Expected uint8 RGB frame of shape (240, 320, 3)")
        with self.lock:
            request = {"shape": list(frame.shape), "pixels": base64.b64encode(frame.tobytes()).decode("ascii")}
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            result = self.receive(120)["result"]
            if result["frame_sha256"] != hashlib.sha256(frame.tobytes()).hexdigest():
                raise ValueError("Vision worker returned a different observation")
            return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()
        self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run_worker(args.model, args.source, args.device)


if __name__ == "__main__":
    main()
