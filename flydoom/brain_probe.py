"""Run bounded stimulation experiments; no pixels, game control, or training."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy

from flydoom.data import digest
from flydoom.simulation import LIFNetwork, LIFParameters, load_connectome


def run_probe(network, stimulus_indices, descending_indices, *, duration_ms=200,
              pulse_start_ms=20, pulse_end_ms=80, amplitude_mv=20, connected=True,
              recording=None, progress=None):
    """Optionally capture per-neuron spike bins and voltage snapshots every 10 ms."""
    p = network.params
    if not (0 <= pulse_start_ms < pulse_end_ms <= duration_ms):
        raise ValueError("Pulse times must lie inside the run")
    if not np.isfinite(amplitude_mv) or amplitude_mv < 0:
        raise ValueError("Amplitude must be finite and nonnegative")
    network.reset()
    n = len(network.voltage)
    stimulated = np.zeros(n, dtype=bool)
    stimulated[stimulus_indices] = True
    descending = np.zeros(n, dtype=bool)
    descending[descending_indices] = True
    outside = ~stimulated
    drive = np.zeros(n, dtype=np.float32)
    spike_counts = np.zeros(n, dtype=np.int64)
    ever_responded = np.zeros(n, dtype=bool)
    traces = []
    first_outside_spike_ms = None
    first_descending_spike_ms = None
    max_outside_deviation_mv = 0.0
    voltage_min, voltage_max = p.rest_mv, p.rest_mv
    step_seconds = []
    steps = p.steps(duration_ms)
    start_step, end_step = p.steps(pulse_start_ms), p.steps(pulse_end_ms)
    if recording is not None:
        recording.clear()
        sample_times = [0.0]
        spike_bins = [np.zeros(n, dtype=np.uint32)]
        voltage_frames = [network.voltage.copy()]
        previous_counts = np.zeros(n, dtype=np.int64)
    for step in range(steps):
        drive.fill(0)
        if start_step <= step < end_step:
            drive[stimulated] = amplitude_mv
        started = perf_counter()
        spikes = network.step(drive, connected=connected)
        step_seconds.append(perf_counter() - started)
        time_ms = network.tick * p.dt_ms
        spike_counts += spikes
        deviation = np.abs(network.voltage - p.rest_mv)
        ever_responded |= deviation > 1e-4
        if outside.any():
            max_outside_deviation_mv = max(max_outside_deviation_mv, float(deviation[outside].max()))
        voltage_min = min(voltage_min, float(network.voltage.min()))
        voltage_max = max(voltage_max, float(network.voltage.max()))
        if first_outside_spike_ms is None and spikes[outside].any():
            first_outside_spike_ms = time_ms
        if first_descending_spike_ms is None and spikes[descending].any():
            first_descending_spike_ms = time_ms
        if (step + 1) % max(1, p.steps(10)) == 0 or step == steps - 1:
            traces.append({"time_ms": time_ms, "cumulative_spikes": int(spike_counts.sum()),
                           "cumulative_descending_spikes": int(spike_counts[descending].sum()),
                           "mean_voltage_mv": float(network.voltage.mean()),
                           "min_voltage_mv": float(network.voltage.min())})
            if recording is not None:
                sample_times.append(time_ms)
                spike_bins.append((spike_counts - previous_counts).astype(np.uint32))
                voltage_frames.append(network.voltage.copy())
                previous_counts[:] = spike_counts
        if progress is not None and ((step + 1) % max(1, steps // 4) == 0 or step == steps - 1):
            progress(time_ms, int(spike_counts.sum()))
    if recording is not None:
        recording.update(sample_times_ms=np.asarray(sample_times),
                         spike_bins=np.stack(spike_bins), voltage_mv=np.stack(voltage_frames))
    elapsed = sum(step_seconds)
    return {
        "stimulated_neurons": len(stimulus_indices), "connected": connected,
        "duration_ms": steps * p.dt_ms,
        "pulse_start_ms": start_step * p.dt_ms, "pulse_end_ms": end_step * p.dt_ms,
        "amplitude_mv": amplitude_mv, "steps": steps,
        "spikes": int(spike_counts.sum()), "spiking_neurons": int(np.count_nonzero(spike_counts)),
        "stimulated_spikes": int(spike_counts[stimulated].sum()),
        "outside_stimulus_spikes": int(spike_counts[outside].sum()),
        "outside_stimulus_responding_neurons": int(ever_responded[outside].sum()),
        "outside_stimulus_max_deviation_mv": max_outside_deviation_mv,
        "first_outside_spike_ms": first_outside_spike_ms,
        "descending_spikes": int(spike_counts[descending].sum()),
        "descending_spiking_neurons": int(np.count_nonzero(spike_counts[descending])),
        "first_descending_spike_ms": first_descending_spike_ms,
        "min_voltage_mv": voltage_min, "max_voltage_mv_after_reset": voltage_max,
        "final_max_deviation_mv": float(np.max(np.abs(network.voltage - p.rest_mv))),
        "finite": bool(np.isfinite(network.voltage).all() and np.isfinite(network.current).all()),
        "step_compute_seconds": elapsed,
        "mean_step_ms": 1000 * elapsed / steps,
        "p95_step_ms": 1000 * float(np.percentile(step_seconds, 95)),
        "simulated_time_over_step_compute_time": steps * p.dt_ms / (1000 * elapsed),
        "trace": traces,
    }, spike_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/fafb783"))
    parser.add_argument("--output", type=Path, default=Path("runs/brain-probe"))
    parser.add_argument("--dt-ms", type=float, default=0.5)
    args = parser.parse_args()
    params = LIFParameters(dt_ms=args.dt_ms)
    started = perf_counter()
    ids, rows, weights, signs, provenance = load_connectome(args.data_dir, params)
    network = LIFNetwork(weights, params)
    load_seconds = perf_counter() - started
    sensory = np.array([i for i, r in enumerate(rows) if r["cell_type"] == "R1-6"], dtype=np.int64)
    descending = np.array([i for i, r in enumerate(rows) if r["super_class"] == "descending"], dtype=np.int64)
    positive = np.array([i for i, r in enumerate(rows)
                         if r["super_class"] == "visual_projection" and signs[i] > 0], dtype=np.int64)
    if min(len(sensory), len(descending), len(positive)) == 0:
        raise ValueError("A required diagnostic population is empty")
    args.output.mkdir(parents=True, exist_ok=True)
    conditions = [
        ("no_input", np.array([], dtype=np.int64), True),
        ("photoreceptor_pulse", sensory, True),
        ("photoreceptor_disconnected", sensory, False),
        ("excitatory_projection_pulse", positive, True),
    ]
    results = {}
    output_hashes = {}
    for name, selected, connected in conditions:
        print(f"Running {name}: {len(selected)} stimulated neurons", flush=True)
        recording = {}
        result, counts = run_probe(
            network, selected, descending, connected=connected, recording=recording,
            progress=lambda time_ms, spikes: print(f"  {time_ms:.0f}/200 ms simulated | {spikes} spikes", flush=True))
        result["spikes_by_super_class"] = dict(Counter())
        for row, count in zip(rows, counts):
            group = row["super_class"]
            result["spikes_by_super_class"][group] = result["spikes_by_super_class"].get(group, 0) + int(count)
        results[name] = result
        path = args.output / f"{name}.npz"
        np.savez_compressed(path, root_ids=ids, spike_counts=counts, stimulated_root_ids=ids[selected], **recording)
        output_hashes[path.name] = digest(path, "sha256")
        print(f"  spikes={result['spikes']}, descending={result['descending_spikes']}, "
              f"mean step={result['mean_step_ms']:.2f} ms", flush=True)
    gates = {
        "bounded_runs_finite": all(r["finite"] for r in results.values()),
        "rest_without_input": results["no_input"]["spikes"] == 0 and results["no_input"]["final_max_deviation_mv"] == 0,
        "stimulated_cells_spike": results["photoreceptor_pulse"]["stimulated_spikes"] > 0,
        "connected_voltage_response": results["photoreceptor_pulse"]["outside_stimulus_responding_neurons"] > 0,
        "disconnected_no_spread": results["photoreceptor_disconnected"]["outside_stimulus_spikes"] == 0
            and results["photoreceptor_disconnected"]["outside_stimulus_max_deviation_mv"] == 0,
        "excitatory_control_spreads": results["excitatory_projection_pulse"]["outside_stimulus_spikes"] > 0,
        "photoreceptor_to_descending_spikes": results["photoreceptor_pulse"]["descending_spikes"] > 0,
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "experimental_uniform_current_LIF_v1", "trained": False,
        "recording": "Per-neuron spike counts in successive intervals and endpoint voltage snapshots; initial frame at rest",
        "pixels_used": False, "doom_connected": False,
        "parameters": asdict(params), "effective_delay_ms": network.delay_steps * params.dt_ms,
        "effective_refractory_ms": network.refractory_steps * params.dt_ms,
        "neurons": len(ids), "descending_neurons": len(descending), "provenance": provenance,
        "persistent_network_mib": network.persistent_bytes / 2**20,
        "memory_scope": "CSR weights, state, refractory counters, and spike delay buffer; excludes load peak and Python overhead",
        "load_and_verify_seconds": load_seconds,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "platform": platform.platform(), "device": "CPU"},
        "source_sha256": {name: digest(Path(__file__).parent / name, "sha256")
                          for name in ("simulation.py", "brain_probe.py", "data.py")},
        "output_sha256": output_hashes, "gates": gates, "conditions": results,
        "limitations": ["No natural retinal encoding or tonic baseline", "Uniform LIF approximation includes normally graded cells",
                        "Modulatory and unresolved outputs are zeroed and counted", "Sign assumptions omit receptor specificity",
                        "Positive control bypasses the retina", "Finite bounded runs do not establish biological validity or long-term stability"],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gates": gates, "persistent_network_mib": report["persistent_network_mib"],
                      "report": str(args.output / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
