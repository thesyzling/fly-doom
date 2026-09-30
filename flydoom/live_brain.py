"""Watch and inspect a trained fly-network student in a local interactive browser."""

import argparse
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Condition, Thread
import traceback
from urllib.parse import parse_qs, urlsplit
import webbrowser

import numpy as np
from safetensors.torch import load_file
import torch

from flydoom import student, action_memory
from flydoom.calibration import load_calibrated, output_features, write_json
from flydoom.data import digest
from flydoom.learning_data import ACTIONS, make_game
from flydoom.live_activity import ConnectionCatalog, ReadoutTelemetry, frame_png


STATIC = Path(__file__).parent / "web" / "live"


def load_checkpoint(checkpoint, calibration, data_dir):
    """Keep the existing training implementation and every checkpoint unchanged."""
    checkpoint = Path(checkpoint)
    report = json.loads((checkpoint / "report.json").read_text(encoding="utf-8"))
    if (report.get("schema") not in {"spiking_student_v1", action_memory.SCHEMA} or report.get("status") != "completed"
            or not report.get("student_trained")):
        raise ValueError("Student checkpoint is not complete")
    if (digest(checkpoint / "student.safetensors", "sha256") != report["student_sha256"]
            or digest(calibration, "sha256") != report["calibration_sha256"]
            or digest(student.__file__, "sha256") != report["implementation_sha256"]):
        raise ValueError("Student artifact, calibration, or implementation changed")
    memory = report["schema"] == action_memory.SCHEMA
    if memory and (report.get("memory_implementation_sha256") != digest(action_memory.__file__, "sha256")
                   or report.get("memory_feature_names") != list(action_memory.NAMES)
                   or report["input_size"] != 2 * len(report["output_root_ids"]) + 4 + len(action_memory.NAMES)):
        raise ValueError("Action memory implementation or feature mapping changed")
    ids, controller, _ = load_calibrated(data_dir, calibration)
    indices = np.sort(np.concatenate(controller.mapping.output_groups))
    if [str(ids[i]) for i in indices] != report["output_root_ids"]:
        raise ValueError("Student neuron mapping mismatch")
    torch.set_num_threads(4)
    model = student.SpikingReadout(report["input_size"], report["hidden"])
    model.load_state_dict(load_file(str(checkpoint / "student.safetensors")))
    model.eval()
    model.action_memory = memory
    return ids, controller, model, report


class LiveSession:
    def __init__(self, catalog, output, report, *, episodes=3, seed=51000, max_decisions=75, autoplay=False, seeds=None):
        if seeds is not None:
            if not 1 <= len(seeds) <= 20 or len(set(seeds)) != len(seeds) or any(not 0 <= s < 2**32 for s in seeds):
                raise ValueError("Require 1..20 distinct uint32 seeds")
            episodes, seed = len(seeds), seeds[0]
        if not 1 <= episodes <= 20 or not 1 <= max_decisions <= 75 or not 0 <= seed <= 2**32 - episodes:
            raise ValueError("Invalid episode limits or seed")
        self.catalog, self.controller, self.model = catalog, catalog.controller, catalog.model
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.episodes, self.seed, self.max_decisions = episodes, seed, max_decisions
        self.seeds = list(seeds) if seeds is not None else list(range(seed, seed + episodes))
        self.memory = action_memory.ActionMemory()
        self.condition = Condition()
        self.paused, self.stopped, self.steps, self.busy = not autoplay, False, 0, False
        self.phase, self.error = "loading", None
        self.samples = deque(maxlen=24)
        self.sequence = 0
        self.report = {"schema": "live_student_observation_v1", "status": "running", "episodes": [],
                       "laya_used_during_play": False, "connectome_weights_trained": False,
                       "checkpoint_sha256": report["student_sha256"],
                       "action_memory": getattr(self.model, "action_memory", False),
                       "disconnected": self.controller.control == "disconnected" if hasattr(self.controller, "control") else False,
                       "planned_seeds": self.seeds,
                       "teacher_accepted": report.get("teacher_accepted"),
                       "observer_sha256": digest(__file__, "sha256"),
                       "telemetry_sha256": digest(Path(__file__).with_name("live_activity.py"), "sha256")}

    def control(self, command):
        if command not in {"run", "pause", "step", "stop"}:
            raise ValueError("Unknown control")
        with self.condition:
            if self.phase in {"completed", "stopped", "error"}:
                raise ValueError("This run has ended; start a new command for another run")
            if command == "stop":
                self.stopped = True
            elif command == "run":
                self.paused, self.steps = False, 0
            else:
                self.paused = True
                self.steps = min(self.steps + 1, 5) if command == "step" else 0
            self.condition.notify_all()

    def permit(self):
        with self.condition:
            while self.paused and not self.steps and not self.stopped:
                self.condition.wait()
            if self.stopped:
                return False
            self.steps = max(0, self.steps - 1)
            self.busy = True
            return True

    def publish(self, frame, episode, number, previous, probabilities, telemetry, decision, reward):
        net = self.controller.network
        self.sequence += 1
        sample = {"sequence": self.sequence, "episode": episode, "decision": number,
                  "previous": previous, "probabilities": np.asarray(probabilities).tolist(),
                  "activity": telemetry.activity.copy(), "drive": telemetry.drive.copy(),
                  "voltage": net.voltage.copy(), "counts": self.controller.last_counts.copy() if number else np.zeros(len(net.voltage), dtype=np.int32),
                  "frame": frame_png(frame), "return": reward,
                  "action": ACTIONS[int(np.argmax(probabilities))] if number else "Not decided",
                  "brain_spikes": decision.get("spikes", 0),
                  "brain_compute_seconds": decision.get("compute_seconds", 0),
                  "brain_ms": self.controller.steps * net.params.dt_ms,
                  "voltage_min_mv": decision.get("voltage_min_mv", net.params.rest_mv)}
        if getattr(self.model, "action_memory", False):
            sample["memory"] = action_memory.encode(self.memory.observe()).tolist()
        with self.condition:
            self.samples.append(sample)
            self.busy = False
            self.phase = "ready"

    def metadata(self):
        return {"neurons": len(self.catalog.ids), "connections": self.catalog.incoming.nnz,
                "inputs": len(self.catalog.input_lookup), "outputs": len(self.catalog.outputs),
                "hidden": self.catalog.hidden, "actions": ACTIONS, "episodes": self.episodes,
                "seed": self.seed, "max_decisions": self.max_decisions,
                "seeds": self.seeds, "memory_features": list(self.catalog.memory_names),
                "teacher_accepted": self.report["teacher_accepted"], "output": str(self.output)}

    def state(self):
        with self.condition:
            status = {"phase": self.phase, "paused": self.paused, "busy": self.busy,
                      "error": self.error, "finished_episodes": [dict(e) for e in self.report["episodes"]]}
            sample = self.samples[-1] if self.samples else None
        if sample is None:
            return status
        public = {key: value for key, value in sample.items() if key not in {"voltage", "counts", "drive", "activity"}}
        public["student_spikes"] = np.rint(sample["activity"] * 8).astype(int).tolist()
        counts = sample["counts"]
        ranked = self.catalog.strongest(counts, 10)
        outputs = self.catalog.outputs
        descending = outputs[self.catalog.strongest(counts[outputs], 10)]
        public["active"] = [{**self.catalog.label(str(self.catalog.ids[i])), "spikes": int(counts[i]),
                              "voltage_mv": float(sample["voltage"][i])} for i in ranked]
        public["descending"] = [{**self.catalog.label(str(self.catalog.ids[i])), "spikes": int(counts[i]),
                                  "voltage_mv": float(sample["voltage"][i])} for i in descending]
        return {**status, **public}

    def inspect(self, key, sequence=None):
        with self.condition:
            samples = list(self.samples)
        if not samples:
            raise ValueError("Waiting for the first observation")
        sample = samples[-1] if sequence is None else next((s for s in samples if s["sequence"] == sequence), None)
        if sample is None:
            raise ValueError("Observation expired; inspect the latest decision")
        result = self.catalog.inspect(key, sample)
        history = []
        for s in samples:
            if s["sequence"] > sample["sequence"] or s["episode"] != sample["episode"]:
                continue
            if key in self.catalog.lookup:
                i = self.catalog.lookup[key]
                history.append({"decision": s["decision"], "spikes": int(s["counts"][i]), "voltage_mv": float(s["voltage"][i])})
            elif key.startswith("student:"):
                i = int(key.split(":")[1])
                history.append({"decision": s["decision"], "spikes": int(round(float(s["activity"][i]) * 8))})
        return {**result, "history": history}

    def run(self, on_finished=None):
        import vizdoom as vzd
        telemetry = ReadoutTelemetry(self.model)
        game = None
        try:
            game, buttons = make_game(self.seed, False)
            with (self.output / "decisions.jsonl").open("w", encoding="utf-8") as trace:
                for episode in range(self.episodes):
                    if self.stopped:
                        break
                    game.set_seed(self.seeds[episode])
                    game.new_episode()
                    self.controller.reset()
                    self.memory = action_memory.ActionMemory()
                    previous, number = 0, 0
                    telemetry.activity.fill(0)
                    telemetry.drive.fill(0)
                    self.publish(game.get_state().screen_buffer, episode + 1, 0, previous,
                                 [0.25] * 4, telemetry, {}, 0)
                    counts = dict.fromkeys(ACTIONS, 0)
                    while not game.is_episode_finished() and number < self.max_decisions:
                        if not self.permit():
                            break
                        frame = game.get_state().screen_buffer.copy()
                        decision = self.controller.decide(frame)
                        if decision["voltage_min_mv"] < -90:
                            raise ValueError("Live game failed the engineering voltage gate")
                        vector = np.concatenate((output_features(self.controller), np.eye(4, dtype=np.float32)[previous]))
                        if getattr(self.model, "action_memory", False):
                            vector = action_memory.augment(vector[None, :], [self.memory.observe()])[0]
                        with torch.no_grad():
                            probabilities = self.model(torch.tensor(vector).unsqueeze(0)).softmax(-1)[0].numpy()
                        action = int(probabilities.argmax())
                        for _ in range(4):
                            if game.is_episode_finished():
                                break
                            game.make_action([int(b == ACTIONS[action]) for b in buttons], 1)
                        number += 1
                        counts[ACTIONS[action]] += 1
                        self.publish(frame, episode + 1, number, previous, probabilities, telemetry,
                                     decision, game.get_total_reward())
                        trace.write(json.dumps({"episode": episode + 1, "decision": number,
                            "action": ACTIONS[action], "probabilities": probabilities.tolist(),
                            "return": game.get_total_reward(), "brain_spikes": decision["spikes"],
                            "brain_compute_seconds": decision["compute_seconds"],
                            "neural_feature_norm": float(np.linalg.norm(vector[:2 * len(self.catalog.outputs)])),
                            "memory": action_memory.encode(self.memory.observe()).tolist() if getattr(self.model, "action_memory", False) else None,
                            "student_spikes": np.rint(telemetry.activity * 8).astype(int).tolist()}) + "\n")
                        trace.flush()
                        self.memory.advance(action)
                        previous = action
                    summary = {"seed": self.seeds[episode], "decisions": number, "action_counts": counts,
                               "return": game.get_total_reward(),
                               "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
                               "end_reason": "stopped" if self.stopped else "game_finished" if game.is_episode_finished() else "decision_limit"}
                    with self.condition:
                        self.report["episodes"].append(summary)
                    print(f"Episode {episode + 1}: {summary}", flush=True)
            self.report["status"] = "stopped" if self.stopped else "completed"
        except Exception as error:
            self.error = str(error)
            self.report.update(status="error", error=self.error)
            traceback.print_exc()
        finally:
            telemetry.close()
            if game:
                game.close()
            with self.condition:
                self.phase = self.report["status"]
                self.busy = False
            try:
                write_json(self.output / "report.json", self.report)
            finally:
                if on_finished:
                    on_finished()


def make_handler(session):
    class Handler(BaseHTTPRequestHandler):
        def send(self, body, mime="application/json", status=200):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                if parsed.path in {"/", "/app.js", "/style.css"}:
                    name, mime = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                                  "/style.css": ("style.css", "text/css")}[parsed.path]
                    self.send((STATIC / name).read_bytes(), mime + "; charset=utf-8")
                elif parsed.path == "/api/meta":
                    self.send(session.metadata())
                elif parsed.path == "/api/state":
                    self.send(session.state())
                elif parsed.path == "/api/neuron":
                    query = parse_qs(parsed.query)
                    self.send(session.inspect(query["id"][0], int(query["sequence"][0]) if "sequence" in query else None))
                else:
                    self.send({"error": "Unknown route"}, status=404)
            except (ValueError, KeyError, IndexError) as error:
                self.send({"error": str(error)}, status=400)

        def do_POST(self):
            expected = f"http://127.0.0.1:{self.server.server_port}"
            origin = self.headers.get("Origin")
            if origin and origin != expected:
                self.send({"error": "Use the local dashboard origin"}, status=403)
                return
            if self.path != "/api/control" or self.headers.get("Content-Type") != "application/json":
                self.send({"error": "Unknown route or content type"}, status=400)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 128:
                    raise ValueError("Invalid request size")
                session.control(json.loads(self.rfile.read(size))["command"])
                self.send({"ok": True})
            except (ValueError, KeyError, TypeError) as error:
                self.send({"error": str(error)}, status=400)

        def log_message(self, format, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/fly-student-experimental-v2"))
    parser.add_argument("--calibration", type=Path, default=Path("runs/calibration-training-v1/report.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/fafb783"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=51000)
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit distinct starts; overrides --seed and --episodes")
    parser.add_argument("--disconnected", action="store_true", help="Disable graph transmission for an evaluation control")
    parser.add_argument("--max-decisions", type=int, default=75)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--exit-on-complete", action="store_true", help="Close the server after a bounded automated check")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    print("Verifying the saved student and loading the full fly graph...", flush=True)
    try:
        ids, controller, model, report = load_checkpoint(args.checkpoint, args.calibration, args.data_dir)
        if args.disconnected:
            controller.control = "disconnected"
        catalog = ConnectionCatalog.from_directory(ids, controller, model, args.data_dir)
        output = args.output or Path("runs") / datetime.now().strftime("live-brain-%Y%m%d-%H%M%S-%f")
        session = LiveSession(catalog, output, report, episodes=args.episodes, seed=args.seed,
                              max_decisions=args.max_decisions, autoplay=args.autoplay, seeds=args.seeds)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    worker = Thread(target=session.run, kwargs={"on_finished": server.shutdown if args.exit_on_complete else None}, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Dashboard: {url}\nClick Run or Step once. Ctrl+C stops the server.\nReports: {output}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        with session.condition:
            session.stopped = True
            session.condition.notify_all()
        worker.join()
        server.server_close()
    if session.error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
