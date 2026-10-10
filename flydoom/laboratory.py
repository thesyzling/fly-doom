"""Research desk for the trained biological-edge pilot and six-action Doom."""

import argparse
from collections import deque
from datetime import datetime
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlsplit
import webbrowser

import numpy as np

from flydoom import anatomy, named_anatomy
from flydoom.data import digest
from flydoom.experiment import experiment_handler
from flydoom.movement_core import ACTIONS, MEMORY
from flydoom.movement_live import MovementCatalog, Session, Workbench
from flydoom.synaptic import load_system
from flydoom.synaptic_eligibility import restore
from flydoom.neural_archive import NeuralArchive, ARRAYS
from flydoom.live_map import activity_snapshot
from flydoom.learning_features import ContrastMotionController
from flydoom.learning_job import LearningJob
from flydoom.learning_cycle import champion
from flydoom.graph_paths import directed_path

STATIC = Path(__file__).parent / "web/laboratory"


class LabCatalog(MovementCatalog):
    def attach_patch(self, patch, report):
        self.patch, self.plastic_report = patch, report
        targets = np.searchsorted(self.incoming.indptr, patch.offsets, side="right") - 1
        sources = self.incoming.indices[patch.offsets]
        self.plastic_lookup = {(int(s), int(t)): j for j, (s, t) in enumerate(zip(sources, targets))}
        edge_gains = getattr(patch, "edge_gains", patch.gains[patch.groups])
        changed = np.abs(patch.base * (edge_gains - 1))
        ranking = np.argsort(-changed, kind="stable")[:128]
        self.changed_edges = [self.edge_details(str(self.ids[sources[j]]), str(self.ids[targets[j]])) for j in ranking]
        candidates = np.unique(targets)
        positions = np.array([float(self.rows[i]["pos_x"]) for i in candidates])
        ordered = candidates[np.argsort(positions)]
        self.examples = [str(self.ids[ordered[j]]) for j in np.linspace(0, len(ordered)-1, 6, dtype=int)]
        self.examples += [str(self.ids[self.incoming.indices[patch.offsets[j]]]) for j in ranking[:2]]
        self.examples = list(dict.fromkeys(self.examples))

    def label(self, key):
        if key.startswith(("action:", "previous:")) and key.split(":")[1] in ACTIONS:
            return {"id": key, "label": key.split(":")[1], "kind": key.split(":")[0]}
        return super().label(key)

    def edge_details(self, source, target):
        s, t = self.lookup[source], self.lookup[target]
        j = self.plastic_lookup.get((s, t))
        if j is not None:
            group = int(self.patch.groups[j]); base = float(self.patch.base[j])
            gain = float(self.patch.edge_gains[j] if hasattr(self.patch, "edge_gains") else self.patch.gains[group])
        else:
            start, end = self.incoming.indptr[t:t+2]
            matches = np.flatnonzero(self.incoming.indices[start:end] == s)
            if not len(matches): raise ValueError("No directed edge between these cells")
            base = float(self.incoming.data[start + matches[0]]); gain = 1.; group = None
        return {"source": source, "target": target, "base_weight": base, "weight": base * gain,
                "delta": base * (gain - 1), "gain": gain, "plastic": j is not None,
                "group": self.patch.labels[group] if group is not None else "fixed",
                "unit": "mV equivalent", "sign": "excitatory" if base > 0 else "inhibitory" if base < 0 else "silent"}

    def inspect(self, key, sample):
        result = super().inspect(key, sample)
        if key in self.lookup:
            i = self.lookup[key]
            result["annotation"] = self.rows[i]
            result["synaptic_current_mv"] = float(sample["current"][i])
            result["refractory_steps"] = int(sample["refractory"][i])
            for edge in result["edges"]:
                source, target = edge["source"]["id"], edge["target"]["id"]
                if source in self.lookup and target in self.lookup:
                    values = self.edge_details(source, target)
                    edge.update({k: v for k, v in values.items() if k not in {"source", "target", "weight"}})
        return result

    def search(self, query):
        query = query.strip().lower()
        if len(query) > 100: raise ValueError("Search is limited to 100 characters")
        matches = []
        if not query:
            candidates = [self.lookup[key] for key in self.examples]
        else:
            candidates = (i for i, r in enumerate(self.rows) if query in str(self.ids[i]) or any(query in r.get(k, '').lower() for k in ('cell_type', 'super_class', 'class', 'top_nt')))
        for i in candidates:
            matches.append({**self.label(str(self.ids[i])), "cell_type": self.rows[i].get("cell_type"), "super_class": self.rows[i].get("super_class")})
            if len(matches) >= 40: break
        return {"cells": matches, "limit": 40, "query": query}


class LabSession(Session):
    def __init__(self, *args):
        super().__init__(*args)
        self.visual_observer = getattr(self.catalog, 'visual_observer', None)
        if self.visual_observer is not None:
            self.visual_observer.reset()
            self.report['visual_observer'] = self.visual_observer.identity
        self.report.update(connectome_weights_trained=self.catalog.plastic_report["connectome_weights_trained"],
                           synaptic_checkpoint_sha256=self.catalog.plastic_report["synapses_sha256"],
                           trained_graph_sha256=self.catalog.plastic_report["trained_weights_sha256"])
        self.archive = NeuralArchive(self.output / "neural", {
            "seed": self.seed, "max_decisions": self.max_decisions,
            "student_sha256": self.identity["student_sha256"],
            "graph_sha256": self.report["trained_graph_sha256"],
            "encoder_mix": getattr(self.controller, "encoder_mix", 0.)})
        self.report["neural_archive"] = str(self.archive.folder)

    def run(self):
        try: super().run()
        finally:
            with self.condition: self.paused = True

    def control(self, command):
        with self.condition:
            if command in {"pause", "stop"} and self.phase in {"completed", "stopped", "error"}:
                self.paused = True
                return
        return super().control(command)

    def publish(self, *args, **kwargs):
        visual = None
        if self.visual_observer is not None:
            effect = kwargs.get('effect', args[6] if len(args) > 6 else None)
            visual = self.visual_observer.observe(args[0], int(effect['game_tics']) if effect else 0,
                                                  self.sequence+1, self.output/'visual')
        # Hold the same reentrant condition until all per-decision telemetry exists.
        with self.condition:
            super().publish(*args, **kwargs)
            self.samples[-1]["current"] = self.controller.network.current.copy()
            self.samples[-1]["refractory"] = self.controller.network.refractory.copy()
            self.samples[-1]['visual_observer'] = visual
            self.archive.save(self.samples[-1])

    def sample(self, sequence=None):
        with self.condition:
            if sequence is None: return self.samples[-1] if self.samples else None
            row = next((s for s in self.samples if s["sequence"] == sequence), None)
            return row if row is not None else self.archive.load(sequence)

    def snapshot(self, sequence=None):
        with self.condition:
            result = {"phase": self.phase, "paused": self.paused, "busy": self.busy,
                      "error": self.error, "finished_episodes": self.report["episodes"]}
            sample = self.sample(sequence)
            if sample is not None:
                result.update({k: v for k, v in sample.items() if k not in ARRAYS})
                result["student_spikes"] = np.rint(sample["activity"] * 8).astype(int).tolist()
            result["retained_sequences"] = [s["sequence"] for s in self.samples]
            result["archived_sequences"] = self.archive.sequences()
            return result

    def inspect(self, key, sequence=None):
        sample = self.sample(sequence)
        if sample is None: raise ValueError("Waiting for the first decision")
        return {**self.catalog.inspect(key, sample), "history": [], "recording": "Full per-decision state on disk"}

    def map_activity(self, sequence): return activity_snapshot(self.sample(sequence))


class Laboratory(Workbench):
    def __init__(self, *args):
        super().__init__(*args)
        self.learning = LearningJob()

    def release_memory(self):
        result = super().release_memory()
        self.session.archive.cache = None
        result["note"] = "RAM snapshots and disk-read cache released; complete neural recordings remain on disk. The active graph stays loaded."
        return result

    def archives(self):
        found = []
        paths = sorted(Path("runs").glob("synaptic-live-*/run-*/neural/manifest.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:100]
        for path in paths:
            value = json.loads(path.read_text())
            found.append({"id": path.parent.parent.as_posix(), "decisions": len(value["records"]),
                          **value["identity"], "compatible": value["identity"]["graph_sha256"] == self.catalog.plastic_report["trained_weights_sha256"]
                          and value["identity"].get("encoder_mix", 0.) == getattr(self.catalog.controller, "encoder_mix", 0.)})
        return {"runs": found}

    def open_archive(self, key):
        with self.lock:
            if self.worker and self.worker.is_alive(): raise ValueError("End the current run before opening an archive")
            row = next((r for r in self.archives()["runs"] if r["id"] == key), None)
            if row is None or not row["compatible"]: raise ValueError("Archive requires its exact graph and encoder checkpoint")
            archive = NeuralArchive(Path(key) / "neural")
            if archive.manifest["identity"]["student_sha256"] != self.identity["student_sha256"]: raise ValueError("Archive readout identity mismatch")
            from threading import Condition
            session = LabSession.__new__(LabSession)
            session.catalog, session.controller, session.model = self.catalog, self.catalog.controller, self.catalog.model
            session.output, session.identity, session.archive = Path(key), self.identity, archive
            session.condition, session.samples = Condition(), deque(maxlen=8)
            session.samples.append(archive.load(archive.sequences()[-1]))
            session.seed, session.max_decisions = row["seed"], row["max_decisions"]
            session.phase, session.paused, session.busy, session.error = "completed", True, False, None
            report = Path(key) / "report.json"
            session.report = json.loads(report.read_text()) if report.exists() else {"episodes": []}
            self.session = session
            return {"opened": key}

    def learning_command(self, command):
        with self.lock:
            if command == "cancel": return self.learning.cancel()
            if self.worker and self.worker.is_alive(): raise ValueError("End the live run before starting training or replacing a checkpoint")
            if command == "start": return self.learning.start()
            if command == "rollback": return self.learning.rollback()
            if command == "activate":
                if self.learning.state()["locked"]: raise ValueError("Wait for the learning cycle to finish")
                observer = getattr(self.catalog, 'visual_observer', None)
                self.catalog, self.identity = load_catalog(Path(champion()["path"]))
                self.catalog.visual_observer = observer
                self.map_data = None
                return self.new_run()
            raise ValueError("Unknown learning command")

    def new_run(self, seed=74000, episodes=1, max_decisions=75, alpha=0):
        if type(seed) is not int or not 0 <= seed < 2**32 or type(max_decisions) is not int or not 1 <= max_decisions <= 75 or episodes != 1 or alpha != 0:
            raise ValueError("Require one episode, 1..75 decisions, a uint32 seed and alpha zero")
        if seed in self.protected: raise ValueError("Reserved future benchmark seed")
        for plan in Path("runs/learning").glob("*/plan.json"):
            value = json.loads(plan.read_text())
            if any(seed == row["seed"] for split in ("gate", "test", "validation") for row in value["splits"][split]):
                raise ValueError("This seed belongs to a learning evaluation split")
        with self.lock:
            if self.worker and self.worker.is_alive(): raise ValueError("End the previous run first")
            self.session = LabSession(self.catalog, self.output / datetime.now().strftime("run-%H%M%S-%f"), self.identity, seed, max_decisions)
            self.worker = Thread(target=self.session.run, daemon=True); self.worker.start()
            return {"output": str(self.session.output)}

    def metadata(self):
        result = super().metadata()
        result.update(laboratory=True, synaptic_checkpoint=self.catalog.plastic_report,
                      morphology_examples=self.catalog.examples,
                      snapshot_limit=8, anatomy_source="78 named fafbseg neuropils in FlyWire space / v783 skeletons",
                      encoder_mix=getattr(self.catalog.controller, "encoder_mix", 0.))
        observer = getattr(self.catalog, 'visual_observer', None)
        result['visual_observer'] = observer.identity if observer else {'enabled': False}
        return result

    def signals(self, sequence):
        with self.session.condition:
            sample = self.session.sample(sequence)
            indices = np.argsort(-sample["counts"].astype(np.int64), kind="stable")[:32]
            active = [{**self.catalog.label(str(self.catalog.ids[i])), "spikes": int(sample["counts"][i]),
                       "voltage_mv": float(sample["voltage"][i])} for i in indices if sample["counts"][i]]
            return {"sequence": sequence, "active_cells": active, "active_cell_count": int(np.count_nonzero(sample["counts"])),
                    "voltage_min_mv": float(sample["voltage"].min()), "voltage_max_mv": float(sample["voltage"].max()),
                    "simulated_ms": sample["decision"] * self.catalog.controller.steps * self.catalog.controller.network.params.dt_ms}


def handler(work):
    base = experiment_handler(work)
    class Handler(base):
        def do_GET(self):
            parsed = urlsplit(self.path); query = parse_qs(parsed.query)
            try:
                files = {"/": ("index.html", "text/html"), "/style.css": ("style.css", "text/css"),
                         "/app.js": ("app.js", "text/javascript"), "/lab-map.js": ("map.js", "text/javascript")}
                if parsed.path in files:
                    name, mime = files[parsed.path]
                    return self.send((STATIC / name).read_bytes(), mime + "; charset=utf-8")
                if parsed.path == "/api/cells": return self.send(work.catalog.search(query.get("q", [""])[0]))
                if parsed.path == "/api/learning": return self.send(work.learning.state())
                if parsed.path == "/api/anatomy": return self.send(named_anatomy.surfaces(work.brain_map()))
                if parsed.path == "/api/learning/report":
                    state = work.learning.state()
                    folder = Path(state.get("output", ""))
                    if folder.resolve().parent != Path("runs/learning").resolve(): raise ValueError("No learning report")
                    return self.send(json.loads((folder / "report.json").read_text()))
                if parsed.path == "/api/learning/benchmarks":
                    from flydoom.learning_audit import summarize
                    folder = Path(work.learning.state().get("output", ""))
                    if folder.resolve().parent != Path("runs/learning").resolve(): raise ValueError("No learning benchmark")
                    path = folder / "benchmarks.json"
                    return self.send({"rows": summarize(json.loads(path.read_text())) if path.exists() else []})
                if parsed.path == "/api/archives": return self.send(work.archives())
                if parsed.path == "/api/path":
                    return self.send(directed_path(work.catalog, query["source"][0], query["target"][0], int(query.get("hops", [4])[0])))
                if parsed.path == "/api/plastic-edges": return self.send({"edges": work.catalog.changed_edges, "total_plastic_edges": len(work.catalog.patch.offsets)})
                if parsed.path == "/api/signals": return self.send(work.signals(int(query["sequence"][0])))
                if parsed.path == "/api/snapshot":
                    sequence = int(query["sequence"][0]); work.signals(sequence)
                    return self.send(work.session.snapshot(sequence))
                if parsed.path == "/api/design":
                    return self.send(Path("docs/LAB_DESIGN_RESEARCH.md").read_bytes(), "text/plain; charset=utf-8")
                return super().do_GET()
            except (ValueError, KeyError, OSError) as error:
                return self.send({"error": str(error)}, status=400)

        def do_POST(self):
            if self.path in {"/api/learning", "/api/archive/open"}:
                if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{self.server.server_port}"):
                    return self.send({"error": "Use the local origin"}, status=403)
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if self.headers.get("Content-Type") != "application/json" or not 0 < size <= 1024: raise ValueError("Invalid request")
                    body = json.loads(self.rfile.read(size))
                    result = work.learning_command(body["command"]) if self.path == "/api/learning" else work.open_archive(body["id"])
                    return self.send(result)
                except (ValueError, TypeError, KeyError, OSError) as error:
                    return self.send({"error": str(error)}, status=400)
            if self.path != "/api/manual": return super().do_POST()
            if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{self.server.server_port}"):
                return self.send({"error": "Use the local origin"}, status=403)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Content-Type") != "application/json" or not 0 < size <= 128: raise ValueError("Invalid manual request")
                body = json.loads(self.rfile.read(size)); return self.send(work.manual(body["action"]))
            except (ValueError, TypeError, KeyError) as error:
                return self.send({"error": str(error)}, status=400)
    return Handler


def load_catalog(folder):
    ids, controller, parent, model, rows, identity = load_system()
    patch, report = restore(controller, ids, folder, identity)
    controller = ContrastMotionController(controller, report.get("encoder_mix", 0.))
    catalog = LabCatalog(ids, controller, parent, rows)
    catalog.model, catalog.memory_names = model, MEMORY
    catalog.input_weights = model.input.weight.detach().numpy().copy()
    catalog.action_weights = model.readout.weight.detach().numpy().copy()
    catalog.action_bias = model.readout.bias.detach().numpy().copy()
    catalog.recurrent = model.recurrent.weight.detach().numpy().copy()
    catalog.attach_patch(patch, report)
    return catalog, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synapses", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument('--visual-checkpoint', type=Path, default=Path('runs/retinal-feedback-v1'))
    parser.add_argument('--no-visual-observer', action='store_true', help='Run only the existing motor policy')
    parser.add_argument('--retinal-control', action='store_true', help='Use the trained T4/T5 decoder with live reward feedback')
    parser.add_argument('--legacy-lif', action='store_true', help='Open the retained whole-brain LIF pilot')
    args = parser.parse_args()
    if args.retinal_control or (not args.legacy_lif and args.synapses is None and not args.no_visual_observer):
        from flydoom.integrated_desk import main as retinal_main
        return retinal_main((['--port',str(args.port)] if args.port is not None else [])+(['--no-browser'] if args.no_browser else []))
    if args.port is None:args.port=8770
    catalog, identity = load_catalog(args.synapses or Path(champion()["path"]))
    if not args.no_visual_observer and (args.visual_checkpoint/'report.json').exists():
        from flydoom.visual_observer import VisualObserver
        catalog.visual_observer = VisualObserver(args.visual_checkpoint, catalog.ids, catalog.rows)
    work = Laboratory(catalog, identity, Path("runs") / datetime.now().strftime("synaptic-live-%Y%m%d-%H%M%S-%f"))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(work))
    try:
        work.new_run(); print(f"Synaptic research desk: http://127.0.0.1:{args.port}", flush=True)
        if not args.no_browser: webbrowser.open(f"http://127.0.0.1:{args.port}")
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        work.close(); server.server_close()


if __name__ == "__main__": main()
