"""Live six-action pilot with bounded RAM, no Vision worker and explicit motor checks."""

import argparse
from collections import deque
from datetime import datetime
import gc
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from threading import Condition, RLock, Thread
import webbrowser

import numpy as np
from safetensors.torch import load_file
import torch

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.experiment import experiment_handler, reserved_seeds
from flydoom.live_activity import ConnectionCatalog, ReadoutTelemetry, frame_png
from flydoom.live_brain import LiveSession, load_checkpoint
from flydoom.live_map import build_map
from flydoom.movement_core import ACTIONS, MEMORY, SCHEMA, Memory, MovementReadout, apply_action, make_game, vector
from flydoom.movement_study import PARENT, CALIBRATION, DATA
from flydoom.vision_distill import read_json


class MovementCatalog(ConnectionCatalog):
    def feature(self, index):
        count = len(self.outputs)
        if index < 2 * count: return str(self.ids[self.outputs[index % count]]), "voltage" if index < count else "spike rate"
        offset = index - 2 * count
        return ("memory:" + MEMORY[offset], "causal action memory") if offset < len(MEMORY) else ("previous:" + ACTIONS[offset - len(MEMORY)], "previous action")

    def inspect(self, key, sample):
        if key in self.lookup: return super().inspect(key, sample)
        edges = []
        def edge(source, target, weight, **extra):
            edges.append({"source": {"id": source}, "target": {"id": target}, "weight": float(weight), **extra})
        result = {"id": key, "label": key, "sequence": sample["sequence"], "edges": edges, "explanation": "Six-action engineered readout; eight internal steps per decision."}
        if key.startswith("student:"):
            i = int(key.split(":")[1])
            if not 0 <= i < self.hidden: raise ValueError("Invalid student cell")
            features = sample["features"]
            normalized = np.zeros(self.input_weights.shape[1]) if features is None else (features - self.model.mean.numpy()) / self.model.scale.numpy()
            products = normalized * self.input_weights[i]
            for j in self.strongest(products, 8):
                edge(self.feature(j)[0], key, self.input_weights[i, j], normalized_value=float(normalized[j]), input_drive_contribution=float(products[j]))
            for k, name in enumerate(ACTIONS): edge(key, "action:" + name, self.action_weights[k, i])
            result.update(kind="student", drive=float(sample["drive"][i]), spikes=int(round(sample["activity"][i] * 8)),
                          reconstructed_input_drive=float(products.sum() + self.model.input.bias.detach()[i]))
        elif key.startswith("action:"):
            index = ACTIONS.index(key.split(":")[1]); products = self.action_weights[index] * sample["activity"]
            for i in self.strongest(products, 8): edge(f"student:{i}", key, self.action_weights[index, i])
            result.update(kind="action", explanation=f"Probability {sample['probabilities'][index]:.5f}; logit {products.sum() + self.action_bias[index]:.5f}")
        else:
            result.update(kind="memory", explanation="Causal six-action history; resets for every episode.")
        return result


class Session(LiveSession):
    def __init__(self, catalog, output, identity, seed, limit):
        self.catalog, self.controller, self.model = catalog, catalog.controller, catalog.model
        self.output = output; output.mkdir(parents=True, exist_ok=False)
        self.identity, self.seed, self.max_decisions = identity, seed, limit
        self.samples, self.condition = deque(maxlen=8), Condition()
        self.paused, self.stopped, self.busy, self.steps, self.teaching = True, False, False, 0, False
        self.phase, self.error, self.sequence, self.manual = "loading", None, 0, None
        self.map_data = None
        self.report = {"schema": "six_action_live_v1", "status": "running", "checkpoint_sha256": identity["student_sha256"],
                       "seed": seed, "actions": ACTIONS, "training_during_run": False, "laya_used_during_play": False, "episodes": []}

    def publish(self, frame, number, probabilities, telemetry, features, action, effect=None, manual=False, after=None):
        self.sequence += 1
        sample = {"sequence": self.sequence, "episode": 1, "decision": number, "frame": frame_png(frame),
                  "after_frame": frame_png(after) if after is not None else None,
                  "probabilities": probabilities.tolist(), "applied_probabilities": np.eye(6)[action].tolist() if manual else probabilities.tolist(),
                  "action": ACTIONS[action] if number else "Not decided", "student_action": ACTIONS[int(probabilities.argmax())],
                  "alpha": 0., "manual_action": manual, "return": self.total_return, "effect": effect,
                  "activity": telemetry.activity.copy(), "drive": telemetry.drive.copy(), "memory": self.memory.encode().tolist(),
                  "features": features, "voltage": self.controller.network.voltage.copy(),
                  "counts": self.controller.last_counts.astype(np.uint16).copy() if number else np.zeros(len(self.catalog.ids), np.uint16),
                  "brain_spikes": int(self.controller.last_counts.sum()) if number else 0}
        with self.condition:
            self.samples.append(sample); self.busy = False; self.phase = "ready"

    def snapshot(self, sequence=None):
        with self.condition:
            state = {"phase": self.phase, "paused": self.paused, "busy": self.busy, "error": self.error, "finished_episodes": self.report["episodes"]}
            row = (self.samples[-1] if self.samples else None) if sequence is None else next((r for r in self.samples if r["sequence"] == sequence), None)
            if row is None: return state
            public = {k: v for k, v in row.items() if k not in {"features", "activity", "drive", "voltage", "counts"}}
            return {**public, **state, "student_spikes": np.rint(row["activity"] * 8).astype(int).tolist()}

    def run(self):
        telemetry, game = ReadoutTelemetry(self.model), None
        try:
            game = make_game(self.seed); game.new_episode(); self.controller.reset(); self.memory = Memory(); self.total_return = 0.
            self.publish(game.get_state().screen_buffer, 0, np.ones(6)/6, telemetry, None, 0)
            counts = dict.fromkeys(ACTIONS, 0); number = 0
            with (self.output / "decisions.jsonl").open("w", encoding="utf-8") as trace:
                while number < self.max_decisions and not game.is_episode_finished():
                    if not self.permit(): break
                    frame = game.get_state().screen_buffer.copy(); decision = self.controller.decide(frame)
                    if decision["voltage_min_mv"] < -90: raise ValueError("Engineering voltage bound exceeded")
                    features = vector(self.controller, self.memory)
                    with torch.inference_mode(): probabilities = self.model(torch.tensor(features)[None]).softmax(1)[0].numpy()
                    with self.condition:
                        manual = self.manual is not None; action = self.manual if manual else int(probabilities.argmax()); self.manual = None
                    effect = apply_action(game, action); number += 1; counts[ACTIONS[action]] += 1; self.total_return = game.get_total_reward()
                    after = game.get_state().screen_buffer.copy() if game.get_state() is not None else None
                    self.publish(frame, number, probabilities, telemetry, features, action, effect, manual, after)
                    np.savez_compressed(self.output / f"decision-{number:04d}.npz", features=features, probabilities=probabilities,
                                        activity=telemetry.activity, frame=frame, applied_action=action)
                    trace.write(json.dumps(self.snapshot()) + "\n"); trace.flush(); self.memory.advance(action)
            self.report["episodes"] = [{"seed": self.seed, "decisions": number, "kills": int(game.get_game_variable(__import__('vizdoom').GameVariable.KILLCOUNT)), "return": self.total_return, "action_counts": counts}]
            self.report["status"] = "stopped" if self.stopped else "completed"
        except Exception as error:
            self.error = str(error); self.report.update(status="error", error=self.error)
        finally:
            telemetry.close()
            if game: game.close()
            write_json(self.output / "report.json", self.report)
            with self.condition: self.phase = self.report["status"]; self.busy = False


class Workbench:
    def __init__(self, catalog, identity, output):
        self.catalog, self.identity, self.output = catalog, identity, output
        output.mkdir(parents=True, exist_ok=False); self.lock = RLock(); self.worker = None; self.map_data = None
        reservation = Path("runs/vision-recovery-reservation-v2/plan.json")
        self.protected = reserved_seeds(reservation) if reservation.exists() else set()
        self.replays = type("EmptyReplays", (), {"listing": lambda _: []})()

    def new_run(self, seed=74000, episodes=1, max_decisions=75, alpha=0):
        if type(seed) is not int or not 0 <= seed < 2**32 or episodes != 1 or not 1 <= max_decisions <= 75 or alpha != 0:
            raise ValueError("Six-action mode requires one episode, 1..75 decisions, a uint32 seed and alpha zero")
        if seed in self.protected: raise ValueError("Reserved future benchmark seed")
        with self.lock:
            if self.worker and self.worker.is_alive(): raise ValueError("End the previous run first")
            self.session = Session(self.catalog, self.output / datetime.now().strftime("run-%H%M%S-%f"), self.identity, seed, max_decisions)
            self.worker = Thread(target=self.session.run, daemon=True); self.worker.start()
            return {"output": str(self.session.output)}

    def metadata(self):
        return {"neurons": len(self.catalog.ids), "connections": self.catalog.incoming.nnz, "hidden": self.catalog.hidden,
                "actions": ACTIONS, "memory_features": MEMORY, "checkpoint_sha256": self.identity["student_sha256"],
                "seed": self.session.seed, "episodes": 1, "max_decisions": self.session.max_decisions, "alpha": 0.,
                "output": str(self.session.output), "vision": {"revision": self.identity["teacher"]["revision"]},
                "student_distilled_from_vision": True, "movement": True, "anatomy_available": True,
                "distillation_metrics": {k: self.identity[k] for k in ("initial", "final", "selected_epoch")}}

    def brain_map(self):
        if self.map_data is None:
            self.map_data = build_map(self.catalog)
            for k in (4, 5):
                for i in self.catalog.strongest(self.catalog.action_weights[k], 4):
                    self.map_data["edges"].append({"source": f"student:{i}", "target": "action:" + ACTIONS[k], "weight": float(self.catalog.action_weights[k, i])})
        return self.map_data

    def weights(self, full=False):
        return {"checkpoint_sha256": self.identity["student_sha256"], "arrays": {k: v.detach().tolist() for k, v in self.catalog.model.state_dict().items() if full or k in ("readout.weight", "readout.bias")}}
    def state(self): return self.session.snapshot()
    def models(self): return {"models": [], "scope": "Six-action pilot; see checkpoint report for its separate benchmark."}
    def choose(self, **kwargs): raise ValueError("Use the six-action checkpoint selected at startup")
    def control(self, command): return self.session.control(command)
    def inspect(self, key, sequence=None): return self.session.inspect(key, sequence)
    def map_activity(self, sequence): return self.session.map_activity(sequence)
    def release_memory(self):
        with self.session.condition:
            if self.session.busy or not self.session.paused: raise ValueError("Pause before releasing memory")
            self.session.samples = deque(list(self.session.samples)[-1:], maxlen=8)
        gc.collect(); return {"note": "History released. This mode loads no GPU teacher; the active CPU graph is retained."}
    def manual(self, action):
        with self.session.condition:
            if action not in ACTIONS or not self.session.paused or self.session.busy or self.session.phase != "ready": raise ValueError("Manual motor checks require a paused, idle run")
            self.session.manual = ACTIONS.index(action); self.session.control("step")
        return {"manual": action}
    def close(self):
        if self.worker and self.worker.is_alive(): self.session.control("stop"); self.worker.join(timeout=30)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--checkpoint", type=Path, default=Path("runs/movement-pilot-v1")); parser.add_argument("--port", type=int, default=8772); parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv); identity = read_json(args.checkpoint / "report.json")
    if (identity.get("schema") != SCHEMA or identity.get("status") != "completed" or not identity.get("student_trained")
            or tuple(identity["actions"]) != ACTIONS or identity["student_sha256"] != digest(args.checkpoint / "student.safetensors", "sha256")
            or identity["core_sha256"] != digest(Path(__file__).with_name("movement_core.py"), "sha256")
            or identity["calibration_sha256"] != digest(CALIBRATION, "sha256")):
        raise ValueError("Six-action checkpoint is incomplete or changed")
    ids, controller, parent, _ = load_checkpoint(PARENT, CALIBRATION, DATA)
    catalog = MovementCatalog.from_directory(ids, controller, parent, DATA)
    model = MovementReadout(identity["input_size"], identity["hidden"]); model.load_state_dict(load_file(str(args.checkpoint / "student.safetensors"))); model.eval()
    catalog.model, catalog.memory_names = model, MEMORY
    catalog.input_weights = model.input.weight.detach().numpy().copy(); catalog.action_weights = model.readout.weight.detach().numpy().copy(); catalog.action_bias = model.readout.bias.detach().numpy().copy(); catalog.recurrent = model.recurrent.weight.detach().numpy().copy()
    work = Workbench(catalog, identity, Path("runs") / datetime.now().strftime("movement-live-%Y%m%d-%H%M%S"))
    base = experiment_handler(work)
    class Handler(base):
        def do_POST(self):
            if self.path != "/api/manual": return super().do_POST()
            if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{self.server.server_port}"): return self.send({"error": "Use the local origin"}, status=403)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Content-Type") != "application/json" or not 0 < size <= 128: raise ValueError("Invalid manual request")
                body = json.loads(self.rfile.read(size)); self.send(work.manual(body["action"]))
            except (ValueError, KeyError, TypeError) as error: self.send({"error": str(error)}, status=400)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        work.new_run(); print(f"Six-action live workbench: http://127.0.0.1:{args.port}", flush=True)
        if not args.no_browser: webbrowser.open(f"http://127.0.0.1:{args.port}")
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt: pass
    finally: work.close(); server.server_close()


if __name__ == "__main__": main()
