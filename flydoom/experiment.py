"""Connect completed benchmark checkpoints to explicit, paused live experiments."""

import argparse
from collections import deque
from datetime import datetime
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit, parse_qs
import webbrowser

from safetensors.torch import load_file

from flydoom import student
from flydoom.data import digest
from flydoom.live_activity import ConnectionCatalog
from flydoom.live_brain import load_checkpoint
from flydoom.research import Workbench, research_handler
from flydoom.runtime_memory import IdleVision
from flydoom.vision_distill import read_json, verified_parent
from flydoom.vision_recovery_eval import score, verify_plan
from flydoom.vision_validation import paired_summary


def benchmark_catalog(report_path, plan_path):
    """Bind displayed scores to the locked checkpoints, not directory names."""
    plan = verify_plan(plan_path)
    report = read_json(report_path)
    if (report.get("status") != "completed" or report.get("schema") != "vision_recovery_gameplay_v1"
            or report.get("plan_sha256") != digest(plan_path, "sha256") or report.get("plan") != plan
            or report.get("training_during_run") is not False
            or report.get("laya_used_during_play") is not False):
        raise ValueError("Require the completed, matching student-only benchmark")
    expected_seeds = [r["seed"] for r in plan["openings"]]
    expected_images = [r["rgb_sha256"] for r in plan["openings"]]
    entries = []
    for condition in plan["conditions"]:
        path = Path(condition["checkpoint"])
        identity = verified_parent(path)
        result = report["results"][condition["name"]]
        if (result.get("checkpoint_sha256") != identity["student_sha256"]
                or [r["seed"] for r in result["episodes"]] != expected_seeds
                or result["opening_hashes"] != expected_images
                or score(result["episodes"]) != result["scores"]):
            raise ValueError("Benchmark episodes or scores differ from the protocol")
        if condition["name"] != "parent" and report["paired"].get(condition["name"]) != paired_summary(
                report["results"]["parent"]["episodes"], result["episodes"], expected_images):
            raise ValueError("Benchmark uncertainty differs from the recorded episodes")
        entries.append({"id": condition["name"], "checkpoint": str(path),
                        "sha256": identity["student_sha256"], "training_complete": True,
                        "selected_epoch": identity.get("selected_epoch"),
                        "scores": result["scores"], "paired": report["paired"].get(condition["name"]),
                        "report": identity})
    return entries


def compatible_model(path, reference):
    """Stage a verified model before replacing any live state."""
    identity = verified_parent(path)
    for key in ("input_size", "hidden", "output_root_ids", "calibration_sha256", "memory_feature_names"):
        if identity[key] != reference[key]:
            raise ValueError(f"Live checkpoint has an incompatible {key}")
    model = student.SpikingReadout(identity["input_size"], identity["hidden"])
    model.load_state_dict(load_file(str(Path(path) / "student.safetensors")))
    model.eval()
    model.action_memory = True
    return model, identity


def reserved_seeds(path):
    """Keep a locally locked future test set out of interactive demonstrations."""
    path = Path(path)
    if digest(path, "sha256") != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Future test reservation changed")
    plan = read_json(path)
    if plan.get("schema") != "recovery_reservation_v2" or plan.get("status") != "reserved_not_collected":
        raise ValueError("Unsupported future test reservation")
    return {row["seed"] for row in plan["splits"]["evaluation"]}


class ExperimentWorkbench(Workbench):
    def __init__(self, catalog, report, vision, output, entries, protected_seeds=()):
        super().__init__(catalog, report, vision, output)
        self.lock = RLock()
        self.entries = {entry["id"]: entry for entry in entries}
        self.protected_seeds = set(protected_seeds)

    def check_live_seeds(self, seed, episodes):
        if type(seed) is not int or type(episodes) is not int or not 1 <= episodes <= 6:
            raise ValueError("Require an integer seed and 1..6 live episodes")
        if any(value in self.protected_seeds for value in range(seed, seed + episodes)):
            raise ValueError("This seed is reserved for the future benchmark; choose another live seed")

    def new_run(self, **kwargs):
        self.check_live_seeds(kwargs.get("seed", 72000), kwargs.get("episodes", 1))
        result = super().new_run(**kwargs)
        with self.session.condition:
            self.session.samples = deque(self.session.samples, maxlen=8)
        return result

    def release_memory(self):
        with self.lock, self.session.condition:
            if self.session.busy or (not self.session.paused and self.worker.is_alive()):
                raise ValueError("Pause and wait for the current decision before releasing memory")
            self.session.samples = deque(list(self.session.samples)[-1:], maxlen=8)
            return self.vision.release()

    def control(self, command):
        with self.lock:
            return super().control(command)

    def models(self):
        return {"active_sha256": self.checkpoint_report["student_sha256"],
                "models": [{k: v for k, v in entry.items() if k != "report"} for entry in self.entries.values()],
                "scope": "20 shared basic-scenario openings; three continuations of one parent. All paired intervals include zero. No automatic winner selection.",
                "live_rule": "A live run is a demonstration, not a new held-out benchmark. Weights stay fixed; at alpha zero Vision observes and the student acts."}

    def metadata(self):
        return {**super().metadata(), "model_selection": True, "anatomy_available": True,
                "snapshot_limit": 8, "vision_idle_unload_seconds": 60}

    def choose(self, model_id, seed=74000):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Require a uint32 seed")
        self.check_live_seeds(seed, 1)
        with self.lock:
            if model_id not in self.entries:
                raise ValueError("Unknown benchmark model")
            if self.worker and self.worker.is_alive() and (self.session.busy or not self.session.paused):
                raise ValueError("Pause the live run and wait for the current decision before loading a model")
            entry = self.entries[model_id]
            model, identity = compatible_model(entry["checkpoint"], self.checkpoint_report)
            if identity["student_sha256"] != entry["sha256"] or identity != entry["report"]:
                raise ValueError("The benchmark checkpoint changed since startup")
            old = self.catalog
            catalog = ConnectionCatalog(old.ids, old.controller, model, old.rows)
            if self.worker and self.worker.is_alive():
                self.session.control("stop")
                self.worker.join(timeout=10)
                if self.worker.is_alive():
                    raise ValueError("The previous run is still stopping; try again after it finishes")
            self.catalog, self.checkpoint_report, self.map_data = catalog, identity, None
            return {**self.new_run(seed=seed, episodes=1, max_decisions=75, alpha=0., autoplay=False),
                    "checkpoint_sha256": identity["student_sha256"], "model_id": model_id}


def experiment_handler(workbench):
    base = research_handler(workbench)

    class Handler(base):
        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/app.js":
                root = Path(__file__).parent / "web/research"
                return self.send((root / "anatomy.js").read_bytes() + b"\n" + (root / "app.js").read_bytes(), "text/javascript; charset=utf-8")
            if parsed.path == "/anatomy.js":
                return self.send((Path(__file__).parent / "web/research/anatomy.js").read_bytes(), "text/javascript; charset=utf-8")
            if parsed.path in {"/api/anatomy", "/api/skeleton"}:
                from flydoom import anatomy
                try:
                    data = workbench.brain_map()
                    value = anatomy.surfaces(data) if parsed.path == "/api/anatomy" else anatomy.skeleton(parse_qs(parsed.query)["id"][0], data)
                    return self.send(value)
                except (OSError, ValueError, KeyError) as error:
                    return self.send({"error": str(error)}, status=400)
            if urlsplit(self.path).path == "/api/models":
                return self.send(workbench.models())
            return super().do_GET()

        def do_POST(self):
            if self.path not in {"/api/model", "/api/release-memory"}:
                return super().do_POST()
            if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{self.server.server_port}"):
                return self.send({"error": "Use the local dashboard origin"}, status=403)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Content-Type") != "application/json" or not 0 < size <= 512:
                    raise ValueError("Require a bounded JSON model selection")
                body = json.loads(self.rfile.read(size))
                if self.path == "/api/release-memory":
                    if body != {}:
                        raise ValueError("Memory release takes an empty object")
                    return self.send(workbench.release_memory())
                if not isinstance(body, dict) or set(body) - {"model_id", "seed"}:
                    raise ValueError("Unknown model selection fields")
                self.send(workbench.choose(**body))
            except (TypeError, ValueError, KeyError, OSError) as error:
                self.send({"error": str(error)}, status=400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("runs/vision-recovery-gameplay-v1/report.json"))
    parser.add_argument("--plan", type=Path, default=Path("runs/vision-recovery-gameplan-v1/plan.json"))
    parser.add_argument("--model", default="parent")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--seed", type=int, default=74000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--movement", action="store_true", help="Open the separately trained six-action movement pilot")
    parser.add_argument("--reservation", type=Path, default=Path("runs/vision-recovery-reservation-v2/plan.json"))
    parser.add_argument("--check", action="store_true", help="Verify completed training and benchmark; do not start a game")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 0 <= args.seed < 2**32:
        parser.error("Invalid port or seed")
    if args.movement:
        from flydoom.movement_live import main as movement_main
        return movement_main(["--port", str(args.port)] + (["--no-browser"] if args.no_browser else []))
    entries = benchmark_catalog(args.benchmark, args.plan)
    selected = next((entry for entry in entries if entry["id"] == args.model), None)
    if selected is None:
        parser.error("Model must be one of: " + ", ".join(e["id"] for e in entries))
    if args.check:
        print(json.dumps({"status": "verified", "models": [e["id"] for e in entries],
                          "selected_sha256": selected["sha256"]}, indent=2))
        return
    plan = read_json(args.plan)
    output = args.output or Path("runs") / datetime.now().strftime("experiment-%Y%m%d-%H%M%S-%f")
    vision, workbench, server = None, None, None
    try:
        print("Loading the verified student, connectome and Vision observer...", flush=True)
        ids, controller, model, identity = load_checkpoint(selected["checkpoint"], plan["calibration"], plan["data_dir"])
        catalog = ConnectionCatalog.from_directory(ids, controller, model, plan["data_dir"])
        vision = IdleVision()
        protected = reserved_seeds(args.reservation) if args.reservation.exists() else set()
        workbench = ExperimentWorkbench(catalog, identity, vision, output, entries, protected)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), experiment_handler(workbench))
        workbench.new_run(seed=args.seed, episodes=1, max_decisions=75, alpha=0., autoplay=False)
        print(f"Experiment workbench: http://127.0.0.1:{args.port}\nModel: {args.model}\nArtifacts: {output}", flush=True)
        if not args.no_browser:
            webbrowser.open(f"http://127.0.0.1:{args.port}")
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        if workbench:
            workbench.close()
        elif vision:
            vision.close()
        if server:
            server.server_close()


if __name__ == "__main__":
    main()
