"""Bounded visual-path diagnostics on the unchanged calibrated LIF graph.

Uniform photoreceptor stimulation tests transmission without a retinal map.
Spatial Mi1 stimulation uses explicitly provisional morphology correspondences;
it bypasses upstream visual cells and cannot establish retinal physiology.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from flydoom.calibration import load_calibrated
from flydoom.data import digest
from flydoom.eye_mapping import DATA, OUTPUT, write_json

CALIBRATION = Path("runs/calibration-training-v1/report.json")
CONDITIONS = ("no_input", "photoreceptor_flash", "Mi1_flash", "Mi1_disconnected",
              "Mi1_x_positive", "Mi1_x_negative", "Mi1_z_positive", "Mi1_z_negative")


def stimulus(condition, time_ms, directions):
    """Return a unit drive profile; x/z are author optical axes, not game axes."""
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3 or not np.isfinite(directions).all():
        raise ValueError("Expected finite optical vectors")
    if condition not in CONDITIONS or not np.isfinite(time_ms):
        raise ValueError("Invalid diagnostic condition or time")
    n = len(directions)
    if condition == "no_input" or not 40 <= time_ms < 160:
        return np.zeros(n, dtype=np.float32)
    if condition in CONDITIONS[:4]:
        return np.ones(n, dtype=np.float32)
    axis = 0 if "_x_" in condition else 2
    coordinate = directions[:, axis]
    if "negative" in condition:
        coordinate = -coordinate
    progress = (time_ms - 40) / 120
    # A translating bright band in author direction-cosine space. Not a movie
    # with calibrated angular speed or a claim about a fly's preferred direction.
    center = -1.25 + 2.5 * progress
    return (np.abs(coordinate - center) <= .25).astype(np.float32)


def run_condition(network, selected, directions, populations, condition, *, amplitude_mv=20.0):
    network.reset()
    p = network.params
    counts = np.zeros(len(network.voltage), dtype=np.int64)
    window_counts = np.zeros(len(network.voltage), dtype=np.int64)
    traces = []
    minimum = maximum = p.rest_mv
    completed = True
    started = perf_counter()
    for tick in range(p.steps(200)):
        drive = np.zeros(len(network.voltage), dtype=np.float32)
        drive[selected] = amplitude_mv * stimulus(condition, tick * p.dt_ms, directions)
        spikes = network.step(drive, connected=condition != "Mi1_disconnected")
        counts += spikes
        window_counts += spikes
        minimum = min(minimum, float(network.voltage.min()))
        maximum = max(maximum, float(network.voltage.max()))
        if minimum < -250 or spikes.mean() > .2:
            completed = False
        if (tick + 1) % p.steps(10) == 0 or not completed:
            elapsed = ((tick + 1) * p.dt_ms) - (traces[-1]["time_ms"] if traces else 0)
            traces.append({"time_ms": (tick + 1) * p.dt_ms,
                           "populations": {name: {"mean_delta_mv": float((network.voltage[idx] - p.rest_mv).mean()),
                                                   "mean_rate_hz": float(window_counts[idx].mean() * 1000 / elapsed)}
                                           for name, idx in populations.items()}})
            window_counts.fill(0)
        if not completed:
            break
    return {"condition": condition, "completed": completed,
            "input_cells": len(selected), "amplitude_mv": amplitude_mv,
            "duration_ms": network.tick * p.dt_ms, "minimum_voltage_mv": minimum,
            "maximum_voltage_mv_after_reset": maximum, "wall_seconds": perf_counter() - started,
            "spikes_by_population": {name: int(counts[idx].sum()) for name, idx in populations.items()},
            "trace": traces}, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, default=Path("runs/visual-integration-v1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mapping_report = json.loads((args.mapping / "report.json").read_text())
    for name, expected in mapping_report["output_sha256"].items():
        if digest(args.mapping / name, "sha256") != expected:
            raise ValueError("Mapping audit checksum mismatch")
    columns = json.loads((args.mapping / "columns.json").read_text())
    chosen = [r for r in columns if r["geometry_supported"]]
    if not chosen:
        raise ValueError("No morphology-supported Mi1 candidates; spatial probe cannot run")
    ids, controller, provenance = load_calibrated(DATA, CALIBRATION)
    network = controller.network
    import csv
    with (DATA / "neuron_annotations.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    lookup = {str(root): i for i, root in enumerate(ids)}
    selected = np.array([lookup[r["candidate_root_id"]] for r in chosen], dtype=int)
    directions = np.array([r["optical_direction"] for r in chosen])
    sides = {r["side"] for r in chosen}
    if len(sides) != 1:
        raise ValueError("A single-eye diagnostic requires one consistent anatomical side")
    side = sides.pop()
    populations = {}
    for name in ["R1-6", "R7", "R8", "L1", "L2", "Mi1", "Tm3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]:
        idx = np.array([i for i, r in enumerate(rows) if r["cell_type"] == name and r["side"] == side], dtype=int)
        if len(idx):
            populations[name + "_" + side] = idx
    populations["descending"] = np.array([i for i, r in enumerate(rows) if r["super_class"] == "descending"], dtype=int)
    receptors = np.array([i for i, r in enumerate(rows) if r["cell_type"] in {"R1-6", "R7", "R8"}], dtype=int)
    results = []
    output_hashes = {}
    for condition in CONDITIONS:
        print(f"Running {condition}", flush=True)
        input_idx = receptors if condition == "photoreceptor_flash" else selected
        vectors = np.zeros((len(input_idx), 3)) if condition == "photoreceptor_flash" else directions
        result, counts = run_condition(network, input_idx, vectors, populations, condition)
        filename = condition + ".npz"
        np.savez_compressed(args.output / filename, root_ids=ids, spike_counts=counts, input_root_ids=ids[input_idx])
        output_hashes[filename] = digest(args.output / filename, "sha256")
        results.append(result)
        print(f"  completed={result['completed']}, descending spikes={result['spikes_by_population']['descending']}", flush=True)
    report = {"schema": "visual_path_probe_v1", "trained": False, "live_policy_changed": False,
              "mapping_report_sha256": digest(args.mapping / "report.json", "sha256"),
              "calibration_sha256": digest(CALIBRATION, "sha256"),
              "model_parameters": asdict(network.params), "model_source": "unchanged calibrated base graph; no trained synaptic gains",
              "source_sha256": {n: digest(Path(__file__).parent / n, "sha256") for n in
                                ("visual_probe.py", "eye_mapping.py", "simulation.py", "calibration.py")},
              "output_sha256": output_hashes, "population_sizes": {k: len(v) for k, v in populations.items()},
              "supported_Mi1_columns": len(chosen), "conditions": results,
              "anatomical_side": side,
              "gates": {"bounded_diagnostics_completed": all(r["completed"] for r in results),
                        "retinal_mapping_validated": False, "graded_dynamics_implemented": False,
                        "physiological_validation_passed": False, "ready_for_task_training": False},
              "limitations": ["Spatial stimulation bypasses the retina and targets provisional Mi1 identities",
                              "All photoreceptors receive the same pulse in the retinal transmission control",
                              "Source x/z direction-cosine sweeps are not calibrated Doom camera motion",
                              "Uniform LIF dynamics and absent tonic baseline remain unvalidated",
                              "Response differences do not establish physiological direction selectivity"]}
    write_json(args.output / "report.json", report)
    from flydoom.visual_report import render
    render(args.output, args.mapping)
    print(f"Report: {args.output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
