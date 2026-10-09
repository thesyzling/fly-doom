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

from flydoom import anatomy
from flydoom.data import digest
from flydoom.experiment import experiment_handler
from flydoom.movement_core import ACTIONS, MEMORY
from flydoom.movement_live import MovementCatalog, Session, Workbench
from flydoom.synaptic import load_system
from flydoom.synaptic_eligibility import restore

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
        self.report.update(connectome_weights_trained=self.catalog.plastic_report["connectome_weights_trained"],
                           synaptic_checkpoint_sha256=self.catalog.plastic_report["synapses_sha256"],
                           trained_graph_sha256=self.catalog.plastic_report["trained_weights_sha256"])

    def publish(self, *args, **kwargs):
        # Hold the same reentrant condition until all per-decision telemetry exists.
        with self.condition:
            super().publish(*args, **kwargs)
            self.samples[-1]["current"] = self.controller.network.current.copy()
            self.samples[-1]["refractory"] = self.controller.network.refractory.copy()

    def snapshot(self, sequence=None):
        with self.condition:
            result = super().snapshot(sequence)
            result.pop("current", None); result.pop("refractory", None)
            result["retained_sequences"] = [s["sequence"] for s in self.samples]
            return result


class Laboratory(Workbench):
    def new_run(self, seed=74000, episodes=1, max_decisions=75, alpha=0):
        if type(seed) is not int or not 0 <= seed < 2**32 or type(max_decisions) is not int or not 1 <= max_decisions <= 75 or episodes != 1 or alpha != 0:
            raise ValueError("Require one episode, 1..75 decisions, a uint32 seed and alpha zero")
        if seed in self.protected: raise ValueError("Reserved future benchmark seed")
        with self.lock:
            if self.worker and self.worker.is_alive(): raise ValueError("End the previous run first")
            self.session = LabSession(self.catalog, self.output / datetime.now().strftime("run-%H%M%S-%f"), self.identity, seed, max_decisions)
            self.worker = Thread(target=self.session.run, daemon=True); self.worker.start()
            return {"output": str(self.session.output)}

    def metadata(self):
        result = super().metadata()
        result.update(laboratory=True, synaptic_checkpoint=self.catalog.plastic_report,
                      morphology_examples=self.catalog.examples,
                      snapshot_limit=8, anatomy_source="FlyWire FAFB14.1 / v783 skeletons")
        return result

    def signals(self, sequence):
        with self.session.condition:
            sample = next((s for s in self.session.samples if s["sequence"] == sequence), None)
            if sample is None: raise ValueError("This neural snapshot has expired; select a retained decision")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synapses", type=Path, default=Path("runs/synaptic-eligibility-v1"))
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    ids, controller, parent, model, rows, identity = load_system()
    patch, report = restore(controller, ids, args.synapses, identity)
    catalog = LabCatalog(ids, controller, parent, rows)
    catalog.model, catalog.memory_names = model, MEMORY
    catalog.input_weights = model.input.weight.detach().numpy().copy()
    catalog.action_weights = model.readout.weight.detach().numpy().copy()
    catalog.action_bias = model.readout.bias.detach().numpy().copy()
    catalog.recurrent = model.recurrent.weight.detach().numpy().copy()
    catalog.attach_patch(patch, report)
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
