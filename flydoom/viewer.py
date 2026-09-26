"""Serve a local 3D viewer for completed, recorded brain-probe experiments."""

import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from flydoom.data import digest

STATIC = Path(__file__).parent / "web"


class RecordedExperiment:
    def __init__(self, run_dir, data_dir):
        self.run_dir = Path(run_dir)
        data_dir = Path(data_dir)
        report_path = self.run_dir / "report.json"
        if not report_path.is_file():
            raise ValueError("No completed report.json found. Finish brain_probe first or choose a completed --run-dir.")
        self.report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = self.report["provenance"]["prepared_sha256"]
        for name in ("root_ids.npy", "neuron_annotations.tsv"):
            if digest(data_dir / name, "sha256") != expected[name]:
                raise ValueError(f"Experiment/data checksum mismatch: {name}")
        self.ids = np.load(data_dir / "root_ids.npy", allow_pickle=False)
        with (data_dir / "neuron_annotations.tsv").open(encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f, delimiter="\t"))
        if len(self.rows) != len(self.ids) or any(int(r["root_id"]) != int(i) for r, i in zip(self.rows, self.ids)):
            raise ValueError("Annotations do not match the recorded neuron order")
        # The authors define anchor positions in 4 x 4 x 40 nm voxel space.
        self.positions_um = np.array([[float(r[f"pos_{c}"]) for c in "xyz"] for r in self.rows])
        self.positions_um *= [0.004, 0.004, 0.040]
        if not np.isfinite(self.positions_um).all():
            raise ValueError("Missing or non-finite anchor coordinates")
        center = (self.positions_um.max(axis=0) + self.positions_um.min(axis=0)) / 2
        extent = float(np.ptp(self.positions_um, axis=0).max())
        if extent <= 0:
            raise ValueError("Anchor positions have no spatial extent")
        self.positions = ((self.positions_um - center) * (1.7 / extent)).astype(np.float32)
        self.groups = sorted({r["super_class"] or "unknown" for r in self.rows})
        group_map = {name: i for i, name in enumerate(self.groups)}
        self.group_ids = np.array([group_map[r["super_class"] or "unknown"] for r in self.rows])
        self.conditions = {}
        self.buffers = {}
        for name, summary in self.report["conditions"].items():
            # Condition names are identifiers, never arbitrary filesystem paths.
            if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
                raise ValueError("Invalid condition identifier")
            path = self.run_dir / f"{name}.npz"
            if digest(path, "sha256") != self.report["output_sha256"][path.name]:
                raise ValueError(f"Recorded result checksum mismatch: {name}")
            with np.load(path, allow_pickle=False) as archive:
                ids, counts = archive["root_ids"], archive["spike_counts"]
                selected = archive["stimulated_root_ids"]
                temporal_keys = {"sample_times_ms", "spike_bins", "voltage_mv"}
                present = temporal_keys.intersection(archive.files)
                if present and present != temporal_keys:
                    raise ValueError("Incomplete temporal recording")
                temporal = {key: archive[key] for key in temporal_keys} if present else None
            if not np.array_equal(ids, self.ids) or counts.shape != self.ids.shape:
                raise ValueError("Recorded neuron IDs or counts have the wrong shape/order")
            if counts.dtype.kind not in "iu" or np.any(counts < 0) or int(counts.sum()) != summary["spikes"]:
                raise ValueError("Recorded spike counts do not match the report")
            if not np.isin(selected, self.ids).all():
                raise ValueError("Unknown stimulated neuron ID")
            if temporal is not None:
                times, bins, voltages = (temporal[k] for k in ("sample_times_ms", "spike_bins", "voltage_mv"))
                if (times.ndim != 1 or len(times) < 2 or times[0] != 0 or
                        not np.isfinite(times).all() or not np.all(np.diff(times) > 0) or
                        not np.isclose(times[-1], summary["duration_ms"]) or
                        bins.shape != (len(times), len(self.ids)) or voltages.shape != bins.shape):
                    raise ValueError("Invalid temporal recording dimensions or timestamps")
                if (bins.dtype.kind not in "iu" or np.any(bins < 0) or bins[0].any() or
                        not np.array_equal(bins.sum(axis=0), counts) or not np.isfinite(voltages).all()):
                    raise ValueError("Invalid temporal counts or voltages")
            stimulated = np.isin(self.ids, selected)
            group_totals = {group: int(counts[self.group_ids == i].sum()) for i, group in enumerate(self.groups)}
            self.conditions[name] = {"summary": summary, "counts": counts, "stimulated": stimulated,
                                     "group_totals": group_totals, "temporal": temporal}
            # Interleaved x/y/z, spike count, stimulation flag, superclass code.
            vertices = np.column_stack((self.positions, counts, stimulated, self.group_ids)).astype("<f4")
            self.buffers[name] = vertices.tobytes()

    def metadata(self):
        return {"run_name": self.run_dir.name, "created_utc": self.report["created_utc"],
                "mode": "Recorded experiment", "neurons": len(self.ids),
                "rest_mv": self.report.get("parameters", {}).get("rest_mv", -52.0),
                "groups": self.groups, "descending_group": self.groups.index("descending") if "descending" in self.groups else -1,
                "conditions": {name: {"summary": data["summary"], "group_totals": data["group_totals"],
                                      "max_spikes": int(data["counts"].max()),
                                      "sample_times_ms": data["temporal"]["sample_times_ms"].tolist() if data["temporal"] else [],
                                      "max_bin_spikes": int(data["temporal"]["spike_bins"].max()) if data["temporal"] else 0}
                               for name, data in self.conditions.items()},
                "gates": self.report["gates"], "limitations": self.report["limitations"]}

    def replay(self, condition):
        temporal = self.conditions[condition]["temporal"]
        if temporal is None:
            raise KeyError("This experiment has no temporal recording")
        # Frame-major, neuron-major pairs: interval spike count, endpoint voltage.
        return np.stack((temporal["spike_bins"], temporal["voltage_mv"]), axis=-1).astype("<f4").tobytes()

    def neuron(self, condition, index):
        if condition not in self.conditions or not 0 <= index < len(self.ids):
            raise KeyError("Unknown neuron or condition")
        r = self.rows[index]
        data = self.conditions[condition]
        return {"root_id": str(self.ids[index]), "cell_type": r["cell_type"] or "Unspecified",
                "super_class": r["super_class"], "side_annotation": r["side"],
                "predicted_transmitter": r["top_nt"] or "Unspecified",
                "curated_transmitters": r["known_nt"] or "Unspecified",
                "anchor_um": self.positions_um[index].round(3).tolist(),
                "spikes": int(data["counts"][index]), "directly_stimulated": bool(data["stimulated"][index])}


def make_handler(experiment):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path in ("/", "/app.js", "/style.css"):
                    name = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[path]
                    types = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}
                    body, mime = (STATIC / name).read_bytes(), types[name] + "; charset=utf-8"
                elif path == "/api/meta":
                    body, mime = json.dumps(experiment.metadata()).encode(), "application/json"
                elif path.startswith("/api/points/"):
                    body, mime = experiment.buffers[path.removeprefix("/api/points/")], "application/octet-stream"
                elif path.startswith("/api/replay/"):
                    body, mime = experiment.replay(path.removeprefix("/api/replay/")), "application/octet-stream"
                elif path.startswith("/api/neuron/"):
                    condition, index = path.removeprefix("/api/neuron/").split("/")
                    body, mime = json.dumps(experiment.neuron(condition, int(index))).encode(), "application/json"
                else:
                    self.send_error(404)
                    return
            except (KeyError, ValueError):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/brain-probe"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/fafb783"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    print("Loading and verifying recorded experiment...", flush=True)
    try:
        experiment = RecordedExperiment(args.run_dir, args.data_dir)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(experiment))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Viewer could not start: {error}\n")
    print(f"Open http://127.0.0.1:{args.port} in your browser. Keep this terminal open.", flush=True)
    print(f"Recorded run: {args.run_dir}. This viewer does not run training or a live simulation.", flush=True)
    print("Press Ctrl+C to stop the viewer.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
