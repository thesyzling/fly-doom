"""Bounded engineering gain selection, not a biological calibration claim."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from flydoom.bridge import NeuralController, NeuralMapping
from flydoom.data import digest
from flydoom.simulation import LIFNetwork, LIFParameters, load_connectome


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def output_features(controller):
    """Descending voltage and spike rate, preserving exact mapped cell order."""
    indices = np.sort(np.concatenate(controller.mapping.output_groups))
    return np.concatenate(((controller.network.voltage[indices] - controller.network.params.rest_mv) / 20,
                           controller.last_counts[indices] / (controller.steps * controller.network.params.dt_ms) * 10)).astype(np.float32)


def load_calibrated(data_dir, calibration_path):
    report = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    if not report.get("engineering_gate_passed") or report.get("schema") != "gain_calibration_v1":
        raise ValueError("Calibration has not passed the engineering gates")
    if report["simulation_sha256"] != digest(Path(__file__).with_name("simulation.py"), "sha256"):
        raise ValueError("Simulation implementation changed; rerun calibration")
    params = LIFParameters(**report["selected_parameters"])
    ids, rows, weights, signs, provenance = load_connectome(data_dir, params)
    if provenance["prepared_sha256"] != report["provenance"]["prepared_sha256"]:
        raise ValueError("Calibration belongs to different graph data")
    mapping = NeuralMapping.from_annotations(ids, rows, signs)
    controller = NeuralController(LIFNetwork(weights, params), mapping,
                                  brain_ms=report["brain_ms"], input_gain_mv=report["input_gain_mv"])
    return ids, controller, report


def calibrate(data_dir, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    params = LIFParameters()
    print("Loading verified graph for a gain sweep...", flush=True)
    ids, rows, source_weights, signs, provenance = load_connectome(data_dir, params)
    mapping = NeuralMapping.from_annotations(ids, rows, signs)
    del rows, signs
    frames = []
    for position in (0, 3, 6):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[:, position:position + 2] = 255
        frames.append(frame)
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    white = np.full_like(black, 255)
    report = {"schema": "gain_calibration_v1", "biologically_validated": False,
              "engineering_gate_passed": False, "provenance": provenance,
              "simulation_sha256": digest(Path(__file__).with_name("simulation.py"), "sha256"),
              "brain_ms": 50.0, "input_gain_mv": 40.0,
              "criteria": {"minimum_voltage_mv": -90, "recovery_max_deviation_mv": 0.1,
                           "minimum_pattern_distance": 1e-4,
                           "white_drive_ms": 1000, "recovery_ms": 400}, "candidates": []}
    for gain in (0.03, 0.01, 0.003):
        print(f"Testing synaptic gain {gain} mV/contact", flush=True)
        weights = source_weights.copy()
        weights.data *= gain / params.mv_per_contact
        selected_params = LIFParameters(mv_per_contact=gain)
        controller = NeuralController(LIFNetwork(weights, selected_params), mapping)
        baseline = controller.decide(black)
        responses, extrema = [], []
        for frame in frames:
            controller.reset()
            first = controller.decide(frame)
            second = controller.decide(frame)
            extrema.extend([first["voltage_min_mv"], second["voltage_min_mv"]])
            responses.append(output_features(controller))
        distances = [float(np.linalg.norm(responses[i] - responses[j]))
                     for i in range(3) for j in range(i)]
        controller.reset()
        spike_history = []
        for window in range(20):
            result = controller.decide(white)
            extrema.append(result["voltage_min_mv"])
            spike_history.append(result["spikes"])
            if (window + 1) % 5 == 0:
                print(f"  Stress {50 * (window + 1)}/1000 ms", flush=True)
        for _ in range(8):
            controller.decide(black)
        recovery = float(np.max(np.abs(controller.network.voltage - params.rest_mv)))
        controller.reset()
        controller.control = "disconnected"
        controller.decide(white)
        disconnected = output_features(controller)
        gates = {"rests_without_input": baseline["spikes"] == 0,
                 "voltage_in_engineering_range": min(extrema) >= -90,
                 "recovers_after_stimulation": recovery <= 0.1,
                 "distinguishes_three_patterns": min(distances) > 1e-4,
                 "output_needs_graph": bool(np.all(disconnected == 0))}
        candidate = {"gain": gain, "gates": gates, "min_voltage_mv": min(extrema),
                     "recovery_max_deviation_mv": recovery, "pattern_distances": distances,
                     "stress_spikes_per_50ms": spike_history}
        report["candidates"].append(candidate)
        print(f"  Gates: {gates}", flush=True)
        if all(gates.values()) and not report["engineering_gate_passed"]:
            report["engineering_gate_passed"] = True
            report["selected_parameters"] = asdict(selected_params)
        write_json(output / "report.json", report)
    print(f"Engineering gate passed: {report['engineering_gate_passed']} | {output / 'report.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/fafb783"))
    parser.add_argument("--output", type=Path, default=Path("runs/calibration"))
    args = parser.parse_args()
    calibrate(args.data_dir, args.output)


if __name__ == "__main__":
    main()
