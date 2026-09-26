"""Experimental pixel-to-connectome-to-action adapter; no learned mappings.

Input bypasses the retina. ID-based assignments are engineering conventions,
not retinotopy, anatomical laterality, or known motor functions.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np


ACTION_NAMES = ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK")
GRID_SIDE = 8


def encode_frame(frame):
    """Average RGB intensity in 8x8 spatial bins, row-major, scaled to [0, 1]."""
    frame = np.asarray(frame)
    if (frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8
            or min(frame.shape[:2]) < GRID_SIDE):
        raise ValueError("Expected an RGB uint8 frame at least 8x8 pixels")
    gray = frame.astype(np.float32).mean(axis=2) / 255.0
    return np.asarray([cell.mean() for band in np.array_split(gray, GRID_SIDE, axis=0)
                       for cell in np.array_split(band, GRID_SIDE, axis=1)], dtype=np.float32)


@dataclass(frozen=True)
class NeuralMapping:
    input_indices: np.ndarray
    feature_indices: np.ndarray
    output_groups: tuple

    @classmethod
    def from_annotations(cls, ids, rows, signs):
        if len(ids) != len(rows) or len(ids) != len(signs):
            raise ValueError("Mapping inputs must share the same neuron order")
        # Integer IDs establish reproducible ordering, not a spatial ordering.
        ordered = np.argsort(ids)
        inputs = np.asarray([i for i in ordered
                             if rows[i]["super_class"] == "visual_projection" and signs[i] > 0],
                            dtype=np.int64)
        outputs = np.asarray([i for i in ordered if rows[i]["super_class"] == "descending"],
                             dtype=np.int64)
        if len(inputs) < GRID_SIDE**2 or len(outputs) < len(ACTION_NAMES):
            raise ValueError("Not enough visual projection or descending neurons")
        return cls(inputs, np.arange(len(inputs)) % GRID_SIDE**2,
                   tuple(outputs[k::len(ACTION_NAMES)] for k in range(len(ACTION_NAMES))))

    def describe(self, ids):
        return {
            "input_rule": "Positive-sign visual_projection cells, sorted by exact root ID, cyclic assignment to 64 image bins",
            "output_rule": "Descending cells, sorted by exact root ID, cyclic assignment to MOVE_LEFT, MOVE_RIGHT, ATTACK",
            "biological_spatial_or_motor_mapping": False,
            "input_root_ids": [str(ids[i]) for i in self.input_indices],
            "input_feature_indices": self.feature_indices.tolist(),
            "output_root_ids": {name: [str(ids[i]) for i in group]
                                for name, group in zip(ACTION_NAMES, self.output_groups)},
        }


def decode_spikes(counts, groups, duration_ms):
    """Use mean firing rate per cell; silence and ties yield WAIT."""
    if not np.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("Readout duration must be finite and positive")
    counts = np.asarray(counts)
    if counts.ndim != 1 or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Spike counts must be a finite nonnegative vector")
    if len(groups) != len(ACTION_NAMES) or any(len(group) == 0 for group in groups):
        raise ValueError("Require three nonempty output groups")
    rates = np.asarray([counts[group].mean() * 1000 / duration_ms for group in groups])
    winner = int(np.argmax(rates))
    tied = np.count_nonzero(np.isclose(rates, rates[winner], rtol=1e-7, atol=1e-9)) > 1
    action = "WAIT" if rates[winner] == 0 or tied else ACTION_NAMES[winner]
    return action, dict(zip(ACTION_NAMES, rates.tolist()))


class NeuralController:
    """Keep neural state between frames; reset it explicitly between episodes."""

    def __init__(self, network, mapping, *, brain_ms=50.0, input_gain_mv=40.0, control="connected"):
        if not np.isfinite(brain_ms) or not 0 < brain_ms <= 200:
            raise ValueError("Brain duration must be in (0, 200] ms per decision")
        if not np.isfinite(input_gain_mv) or not 0 <= input_gain_mv <= 100:
            raise ValueError("Input gain must be in [0, 100] mV-equivalent")
        if control not in {"connected", "disconnected", "zero-input"}:
            raise ValueError("Unknown control condition")
        self.network = network
        self.mapping = mapping
        self.steps = network.params.steps(brain_ms)
        self.input_gain_mv = input_gain_mv
        self.control = control
        n = len(network.voltage)
        indices = [mapping.input_indices, *mapping.output_groups]
        for group in indices:
            if len(group) == 0 or np.any(group < 0) or np.any(group >= n):
                raise ValueError("Mapping indices must lie inside the network")
        joined = np.concatenate(indices)
        if len(np.unique(joined)) != len(joined) or len(mapping.output_groups) != 3:
            raise ValueError("Input and three output populations must be disjoint")
        if (mapping.feature_indices.shape != mapping.input_indices.shape
                or np.any(mapping.feature_indices < 0) or np.any(mapping.feature_indices >= GRID_SIDE**2)):
            raise ValueError("Each input neuron must have a valid feature index")

    def reset(self):
        self.network.reset()

    def decide(self, frame):
        features = encode_frame(frame)
        net, mapping = self.network, self.mapping
        drive = np.zeros_like(net.voltage)
        if self.control != "zero-input":
            drive[mapping.input_indices] = features[mapping.feature_indices] * self.input_gain_mv
        counts = np.zeros(len(drive), dtype=np.int64)
        voltage_min = float(net.voltage.min())
        voltage_max = float(net.voltage.max())
        started = perf_counter()
        for _ in range(self.steps):
            counts += net.step(drive, connected=self.control != "disconnected")
            voltage_min = min(voltage_min, float(net.voltage.min()))
            voltage_max = max(voltage_max, float(net.voltage.max()))
        compute_seconds = perf_counter() - started
        duration_ms = self.steps * net.params.dt_ms
        action, rates = decode_spikes(counts, mapping.output_groups, duration_ms)
        return {
            "action": action, "rates_hz_per_neuron": rates,
            "features": features.tolist(), "applied_drive_max_mv": float(drive.max()),
            "spikes": int(counts.sum()),
            "input_spikes": int(counts[mapping.input_indices].sum()),
            "descending_spikes": {name: int(counts[group].sum())
                                  for name, group in zip(ACTION_NAMES, mapping.output_groups)},
            "voltage_min_mv": voltage_min, "voltage_max_mv_after_reset": voltage_max,
            "brain_window_ms": duration_ms, "brain_time_ms": net.tick * net.params.dt_ms,
            "compute_seconds": compute_seconds,
        }
