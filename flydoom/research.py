"""Research workbench: Laya Vision, frozen connectome, and explicit decision fusion."""

import argparse
import base64
from collections import deque
from datetime import datetime
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock, Thread
import traceback
from urllib.parse import parse_qs, urlsplit
import webbrowser

import numpy as np
import torch

from flydoom import action_memory
from flydoom.calibration import output_features, write_json
from flydoom.data import digest
from flydoom.learning_data import ACTIONS, make_game
from flydoom.live_activity import ConnectionCatalog, ReadoutTelemetry
from flydoom.live_brain import LiveSession, load_checkpoint, make_handler
from flydoom.vision import VisionClient
from flydoom.live_activity import frame_png
from flydoom.replay import ReplayLibrary, finish_live_replay


STATIC = Path(__file__).parent / "web" / "research"


def fuse(vision, student, alpha):
    """Convex mixture of action probabilities, not a learned synaptic connection."""
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Vision coefficient must be between zero and one")
    arrays = [np.asarray(p, dtype=np.float64) for p in (vision, student)]
    if any(p.shape != (4,) or not np.isfinite(p).all() or np.any(p < 0)
           or not np.isclose(p.sum(), 1, atol=1e-5) for p in arrays):
        raise ValueError("Require two normalized distributions in canonical action order")
    return alpha * arrays[0] + (1 - alpha) * arrays[1]


class ResearchSession(LiveSession):
    def __init__(self, catalog, output, report, vision, *, alpha=0.8, **kwargs):
        fuse([0, 1, 0, 0], [1, 0, 0, 0], alpha)
        super().__init__(catalog, output, report, **kwargs)
        self.vision, self.alpha = vision, float(alpha)
        self.samples = deque(maxlen=96)
        self.timeline = []
        self.report.update(schema="vision_fly_research_v1", laya_used_during_play=True,
                           vision=vision.metadata, alpha=self.alpha,
                           fusion="alpha * vision probabilities + (1-alpha) * student probabilities",
                           training_during_run=False, student_distilled_from_vision=bool(report.get("vision_distillation")),
                           scenario="ViZDoom basic / bundled Freedoom2 / RGB24 / 320x240 / HUD off",
                           tics_per_decision=4,
                           frame_timing="RGB observation before action; reward after up to 4 game tics",
                           source_sha256={name: digest(Path(__file__).with_name(name), "sha256")
                                          for name in ("research.py", "vision.py")})
        (self.output / "observations").mkdir()
        write_json(self.output / "report.json", self.report)

    def control(self, command):
        with self.condition:
            if command == "step" and self.busy:
                self.paused, self.steps = True, 0
                return
            super().control(command)

    def publish_pair(self, frame, episode, number, previous, probabilities, telemetry,
                     decision, reward, *, features=None, vision=None, applied=None, effect=None):
        with self.condition:
            super().publish(frame, episode, number, previous, probabilities, telemetry, decision, reward, features)
            sample = self.samples[-1]
            sample["counts"] = sample["counts"].astype(np.uint16)
            sample.update(vision=vision, alpha=self.alpha, effect=effect,
                          applied_probabilities=applied.tolist() if applied is not None else None,
                          action=ACTIONS[int(applied.argmax())] if applied is not None else "Not decided",
                          student_action=ACTIONS[int(np.argmax(probabilities))] if number else "Not decided",
                          agreement=bool(np.argmax(probabilities) == np.argmax(vision["probabilities"])) if vision else None)
            self.timeline.append({k: sample[k] for k in
                                  ("sequence", "episode", "decision", "action", "student_action", "agreement", "return")})
            return sample

    def snapshot(self, sequence=None):
        with self.condition:
            sample = self.samples[-1] if sequence is None and self.samples else next(
                (s for s in self.samples if s["sequence"] == sequence), None)
            if sample is None:
                if sequence is not None:
                    raise ValueError("Snapshot expired; select one of the retained observations")
                return {"phase": self.phase, "busy": self.busy, "paused": self.paused, "error": self.error}
            public = {k: v for k, v in sample.items() if k not in
                      {"voltage", "counts", "drive", "activity", "features"}}
            public["student_spikes"] = np.rint(sample["activity"] * 8).astype(int).tolist()
            return {**public, "phase": self.phase, "busy": self.busy, "paused": self.paused,
                    "error": self.error, "finished_episodes": list(self.report["episodes"]),
                    "latest_sequence": self.sequence,
                    "timeline": [row for row in self.timeline if row["sequence"] >= self.samples[0]["sequence"]]}

    def state(self):
        return self.snapshot()

    def inspect(self, key, sequence=None):
        result = super().inspect(key, sequence)
        if result["kind"] != "student":
            return result
        with self.condition:
            sample = next((s for s in self.samples if s["sequence"] == result["sequence"]), None)
        if sample is None or sample["features"] is None:
            return result
        cell = int(key.split(":")[1])
        raw = sample["features"]
        mean = self.model.mean.detach().numpy()
        scale = self.model.scale.detach().numpy()
        normalized = (raw - mean) / scale
        contributions = normalized * self.catalog.input_weights[cell]
        result["input_bias"] = float(self.model.input.bias.detach()[cell])
        result["sum_all_input_contributions"] = float(contributions.sum())
        result["reconstructed_input_drive"] = result["sum_all_input_contributions"] + result["input_bias"]
        mapping = {self.catalog.feature(j): j for j in range(len(raw))}
        for edge in result["edges"]:
            if edge["target"]["id"] != key or not edge["channel"].startswith("normalized "):
                continue
            feature = mapping[(edge["source"]["id"], edge["channel"].removeprefix("normalized "))]
            edge.update(feature_index=feature, raw_value=float(raw[feature]),
                        training_mean=float(mean[feature]), training_scale=float(scale[feature]),
                        normalized_value=float(normalized[feature]), input_drive_contribution=float(contributions[feature]))
        return result

    def save_feedback(self, sequence, action):
        raise ValueError("Use the recorded paired observations for Vision integration; legacy feedback is separate")

    def run(self, on_finished=None):
        import vizdoom as vzd
        telemetry, game = ReadoutTelemetry(self.model), None
        try:
            game, buttons = make_game(self.seed, False)
            with (self.output / "decisions.jsonl").open("w", encoding="utf-8") as trace:
                for episode, seed in enumerate(self.seeds):
                    if self.stopped:
                        break
                    game.set_seed(seed)
                    game.new_episode()
                    self.controller.reset()
                    self.memory = action_memory.ActionMemory()
                    previous, number = 0, 0
                    telemetry.activity.fill(0)
                    telemetry.drive.fill(0)
                    self.publish_pair(game.get_state().screen_buffer, episode + 1, 0, previous,
                                      [.25] * 4, telemetry, {}, 0)
                    counts = dict.fromkeys(ACTIONS, 0)
                    while not game.is_episode_finished() and number < self.max_decisions:
                        if not self.permit():
                            break
                        frame = game.get_state().screen_buffer.copy()
                        vision = self.vision.predict(frame)
                        if self.stopped:
                            break
                        decision = self.controller.decide(frame)
                        if decision["voltage_min_mv"] < -90:
                            raise ValueError("Fly simulation exceeded its engineering voltage bound")
                        vector = np.concatenate((output_features(self.controller), np.eye(4, dtype=np.float32)[previous]))
                        if getattr(self.model, "action_memory", False):
                            vector = action_memory.augment(vector[None], [self.memory.observe()])[0]
                        with torch.inference_mode():
                            probabilities = self.model(torch.tensor(vector)[None]).softmax(-1)[0].numpy()
                        applied = fuse(vision["probabilities"], probabilities, self.alpha)
                        if self.stopped:
                            break
                        action = int(applied.argmax())
                        before_return = game.get_total_reward()
                        before_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
                        after_frames, tics = [], 0
                        for _ in range(4):
                            if game.is_episode_finished():
                                break
                            game.make_action([int(b == ACTIONS[action]) for b in buttons], 1)
                            tics += 1
                            state = game.get_state()
                            if state is not None:
                                after_frames.append(state.screen_buffer.copy())
                        kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
                        effect = {"reward_delta": game.get_total_reward() - before_return,
                                  "kill_delta": kills - before_kills, "kills": kills,
                                  "episode_finished": game.is_episode_finished(), "game_tics": tics,
                                  "terminal_image_available": not game.is_episode_finished()}
                        number += 1
                        counts[ACTIONS[action]] += 1
                        sample = self.publish_pair(frame, episode + 1, number, previous, probabilities,
                            telemetry, decision, game.get_total_reward(), features=vector, vision=vision, applied=applied, effect=effect)
                        stem = f"{sample['sequence']:05d}"
                        (self.output / "observations" / f"{stem}.png").write_bytes(
                            base64.b64decode(sample["frame"].split(",", 1)[1]))
                        playback_frames = [f"observations/{stem}.png"]
                        for tic, image in enumerate(after_frames, 1):
                            name = f"observations/{stem}-tic{tic}.png"
                            (self.output / name).write_bytes(base64.b64decode(frame_png(image).split(",", 1)[1]))
                            playback_frames.append(name)
                        np.savez_compressed(self.output / "observations" / f"{stem}.npz",
                            features=vector, teacher_probabilities=np.asarray(vision["probabilities"], dtype=np.float32),
                            student_probabilities=probabilities, applied_probabilities=applied,
                            student_activity=telemetry.activity, student_drive=telemetry.drive)
                        row = {k: sample[k] for k in ("sequence", "episode", "decision", "previous", "action",
                               "student_action", "probabilities", "applied_probabilities", "alpha", "agreement",
                               "return", "brain_spikes", "brain_compute_seconds", "vision")}
                        row.update(seed=seed, effect=effect, playback_frames=playback_frames, observation=f"observations/{stem}",
                                   observation_sha256=digest(self.output / "observations" / f"{stem}.npz", "sha256"))
                        trace.write(json.dumps(row, allow_nan=False) + "\n")
                        trace.flush()
                        self.memory.advance(action)
                        previous = action
                    summary = {"seed": seed, "decisions": number, "action_counts": counts,
                               "return": game.get_total_reward(),
                               "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
                               "end_reason": "stopped" if self.stopped else "game_finished" if game.is_episode_finished() else "decision_limit"}
                    with self.condition:
                        self.report["episodes"].append(summary)
                    write_json(self.output / "report.json", self.report)
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
            write_json(self.output / "report.json", self.report)
            try:
                finish_live_replay(self.output, self.report)
            except Exception as error:
                self.report["replay_error"] = str(error)
                write_json(self.output / "report.json", self.report)
                traceback.print_exc()
            with self.condition:
                self.phase, self.busy = self.report["status"], False
            if on_finished:
                on_finished()


class Workbench:
    def __init__(self, catalog, checkpoint_report, vision, output):
        self.catalog, self.checkpoint_report, self.vision = catalog, checkpoint_report, vision
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.session, self.worker = None, None
        self.lock = Lock()
        self.map_data = None
        self.replays = ReplayLibrary()

    def new_run(self, *, seed=72000, episodes=1, max_decisions=75, alpha=.8, autoplay=False):
        if any(type(v) is not int for v in (seed, episodes, max_decisions)) or not 1 <= episodes <= 6:
            raise ValueError("Require an integer seed, decision bound, and 1..6 episodes")
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                raise ValueError("End the current run and wait for computation to finish before creating another")
            path = self.output / datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")
            self.session = ResearchSession(self.catalog, path, self.checkpoint_report, self.vision,
                seed=seed, episodes=episodes, max_decisions=max_decisions, alpha=alpha, autoplay=autoplay)
            self.session.map_data = self.map_data
            self.worker = Thread(target=self.session.run, daemon=True)
            self.worker.start()
            return {"output": str(path)}

    def metadata(self):
        return {**self.session.metadata(), "vision": self.vision.metadata,
                "checkpoint_sha256": self.checkpoint_report["student_sha256"],
                "student_training": self.checkpoint_report.get("training_method", "See checkpoint report"),
                "student_distilled_from_vision": bool(self.checkpoint_report.get("vision_distillation")),
                "distillation_metrics": {k: self.checkpoint_report.get(k) for k in
                    ("initial", "final", "selected_epoch", "parameter_changes", "validation_duplicates_removed")}
                    if self.checkpoint_report.get("vision_distillation") else None,
                "alpha": self.session.alpha, "research": True}

    def state(self):
        return self.session.state()

    def control(self, command):
        self.session.control(command)

    def save_feedback(self, sequence, action):
        return self.session.save_feedback(sequence, action)

    def brain_map(self):
        if self.map_data is None:
            self.map_data = self.session.brain_map()
        return self.map_data

    def map_activity(self, sequence):
        return self.session.map_activity(sequence)

    def inspect(self, key, sequence=None):
        return self.session.inspect(key, sequence)

    def weights(self, full=False):
        model = self.catalog.model
        return {"checkpoint_sha256": self.checkpoint_report["student_sha256"],
                "normalization": "(features - mean) / scale",
                "arrays": {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()
                           if full or k in ("readout.weight", "readout.bias", "recurrent.weight")},
                "vision_to_fly_synapses": None,
                "vision_scorer": {"weight": self.vision.metadata.get("scorer_weight"),
                                  "bias": self.vision.metadata.get("scorer_bias"),
                                  "scope": "Final scalar scorer shared across action options, not the full VLM"},
                "decision_fusion": {"vision": self.session.alpha, "student": 1 - self.session.alpha}}

    def close(self):
        if self.session:
            with self.session.condition:
                self.session.stopped = True
                self.session.condition.notify_all()
        if self.worker:
            self.worker.join(timeout=130)
        self.vision.close()


def research_handler(workbench):
    base = make_handler(workbench, static=STATIC)

    class Handler(base):
        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/api/snapshot":
                    self.send(workbench.session.snapshot(int(query["sequence"][0])))
                elif parsed.path == "/api/weights":
                    self.send(workbench.weights(query.get("full") == ["1"]))
                elif parsed.path == "/api/report":
                    self.send(workbench.session.report)
                elif parsed.path == "/api/replays":
                    self.send(workbench.replays.listing())
                elif parsed.path == "/api/replay":
                    self.send(workbench.replays.data(query["id"][0]))
                elif parsed.path == "/api/replay-signals":
                    from flydoom.replay_signals import inspect_archive
                    self.send(inspect_archive(workbench.replays, query["id"][0]))
                elif parsed.path == "/api/replay-frame":
                    self.send(workbench.replays.frame(query["id"][0], int(query["frame"][0])), mime="image/png")
                else:
                    super().do_GET()
            except (ValueError, KeyError, IndexError, OSError) as error:
                self.send({"error": str(error)}, status=400)

        def do_POST(self):
            if self.path != "/api/new-run":
                return super().do_POST()
            if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{self.server.server_port}"):
                return self.send({"error": "Use the local dashboard origin"}, status=403)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Content-Type") != "application/json" or not 0 < size <= 512:
                    raise ValueError("Require a bounded JSON run configuration")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict) or set(body) - {"seed", "episodes", "max_decisions", "alpha"}:
                    raise ValueError("Unknown run settings")
                self.send(workbench.new_run(**body))
            except (TypeError, ValueError, KeyError) as error:
                self.send({"error": str(error)}, status=400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/fly-student-memory-v1"))
    parser.add_argument("--calibration", type=Path, default=Path("runs/calibration-training-v1/report.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/fafb783"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vision-model", default="models/laya-vision")
    parser.add_argument("--vision-source", default="models/laya-vision-source")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--seed", type=int, default=72000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-decisions", type=int, default=75)
    parser.add_argument("--alpha", type=float, default=.8)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--autoplay", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    output = args.output or Path("runs") / datetime.now().strftime("research-%Y%m%d-%H%M%S-%f")
    print("Loading pinned Laya Vision and the verified fly/student graph...", flush=True)
    vision = VisionClient(model=args.vision_model, source=args.vision_source, device=args.device)
    workbench, server = None, None
    try:
        ids, controller, model, report = load_checkpoint(args.checkpoint, args.calibration, args.data_dir)
        catalog = ConnectionCatalog.from_directory(ids, controller, model, args.data_dir)
        workbench = Workbench(catalog, report, vision, output)
        workbench.new_run(seed=args.seed, episodes=args.episodes, max_decisions=args.max_decisions,
                          alpha=args.alpha, autoplay=args.autoplay)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), research_handler(workbench))
        print(f"Research workbench: http://127.0.0.1:{args.port}\nArtifacts: {output}", flush=True)
        if not args.no_browser:
            webbrowser.open(f"http://127.0.0.1:{args.port}")
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        if workbench:
            workbench.close()
        else:
            vision.close()
        if server:
            server.server_close()


if __name__ == "__main__":
    main()
