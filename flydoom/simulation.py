"""Experimental current-based LIF dynamics on the prepared connectome.

This is an independently implemented diagnostic model, not a reproduction of
Shiu et al. and not a validated fly visual system. See docs/SIMULATION.md.
"""

from collections import Counter
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse

from flydoom.data import digest


@dataclass(frozen=True)
class LIFParameters:
    dt_ms: float = 0.5
    membrane_tau_ms: float = 20.0
    synapse_tau_ms: float = 5.0
    rest_mv: float = -52.0
    reset_mv: float = -52.0
    threshold_mv: float = -45.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    mv_per_contact: float = 0.275

    def __post_init__(self):
        if not all(math.isfinite(x) for x in asdict(self).values()):
            raise ValueError("All parameters must be finite")
        if min(self.dt_ms, self.membrane_tau_ms, self.synapse_tau_ms,
               self.delay_ms, self.mv_per_contact) <= 0 or self.refractory_ms < 0:
            raise ValueError("Time scales and gain must be positive; refractory must be nonnegative")
        if max(self.rest_mv, self.reset_mv) >= self.threshold_mv:
            raise ValueError("Rest and reset must be below threshold")

    def steps(self, milliseconds):
        return int(math.ceil(milliseconds / self.dt_ms - 1e-12))


def transmitter_signs(rows):
    """Assign explicit, provisional presynaptic signs, recording every rule.

    Photoreceptors use an inhibitory visual-channel approximation. Otherwise,
    unambiguous curated fast-transmitter signs precede predicted identities.
    Unknown, modulatory-only, or conflicting curated fast signs produce zero.
    """
    fast = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
    signs = np.zeros(len(rows), dtype=np.float32)
    rules = Counter()
    for i, row in enumerate(rows):
        if row["cell_type"] in {"R1-6", "R7", "R8"}:
            signs[i] = -1
            rules["photoreceptor_inhibitory_visual_channel_override"] += 1
            continue
        known = {x.strip().lower() for x in row["known_nt"].split(",") if x.strip()}
        known_signs = {fast[x] for x in known if x in fast}
        if len(known_signs) == 1:
            signs[i] = known_signs.pop()
            rules["curated_fast_transmitter"] += 1
        elif known:
            rules["curated_conflict_or_unsupported_silenced"] += 1
        elif row["top_nt"] in fast:
            signs[i] = fast[row["top_nt"]]
            rules["predicted_fast_transmitter"] += 1
        else:
            rules["unknown_or_modulatory_prediction_silenced"] += 1
    return signs, dict(rules)


def signed_weights(counts, signs, gain):
    if counts.shape[0] != counts.shape[1] or signs.shape != (counts.shape[0],):
        raise ValueError("Expected a square graph and one sign per neuron")
    if not np.isfinite(gain) or gain <= 0 or not np.isin(signs, [-1, 0, 1]).all():
        raise ValueError("Invalid gain or transmitter signs")
    weights = counts.astype(np.float32).tocsr(copy=True)
    if not np.isfinite(weights.data).all() or np.any(weights.data <= 0):
        raise ValueError("Source counts must be finite and positive")
    # CSR columns are presynaptic neurons. Keep explicit zero edges for audit.
    weights.data *= signs[weights.indices] * gain
    return weights


def load_connectome(directory, params):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    annotations = json.loads((directory / "annotations_manifest.json").read_text(encoding="utf-8"))
    expected = {**manifest["output_sha256"], "neuron_annotations.tsv": annotations["aligned_sha256"]}
    for name, checksum in expected.items():
        if digest(directory / name, "sha256") != checksum:
            raise ValueError(f"Prepared data checksum mismatch: {name}")
    if annotations["graph_root_ids_sha256"] != expected["root_ids.npy"]:
        raise ValueError("Annotations belong to a different graph ID list")
    ids = np.load(directory / "root_ids.npy", allow_pickle=False)
    counts = sparse.load_npz(directory / "synapse_counts.npz")
    with (directory / "neuron_annotations.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != len(ids) or any(int(r["root_id"]) != int(n) for r, n in zip(rows, ids)):
        raise ValueError("Annotation row order does not match neuron IDs")
    if counts.shape != (len(ids), len(ids)) or len(ids) != manifest["neurons"]:
        raise ValueError("Graph dimensions do not match the manifest")
    signs, rules = transmitter_signs(rows)
    weights = signed_weights(counts, signs, params.mv_per_contact)
    provenance = {
        "dataset": manifest["dataset"], "prepared_sha256": expected,
        "sign_rules": rules, "positive_neurons": int((signs > 0).sum()),
        "negative_neurons": int((signs < 0).sum()), "silent_output_neurons": int((signs == 0).sum()),
        "structural_edges": weights.nnz, "effective_nonzero_edges": int(np.count_nonzero(weights.data)),
    }
    return ids, rows, weights, signs, provenance


class LIFNetwork:
    """Uniform-delay LIF model with exact subthreshold integration per step.

    A step integrates [t, t+dt]; spikes are emitted at the right endpoint.
    A spike at t arrives at t+ceil(delay/dt)*dt. Voltage is held at reset
    during refractoriness; synaptic current still receives input and decays.
    """

    def __init__(self, weights, params=None):
        self.params = params or LIFParameters()
        self.weights = sparse.csr_matrix(weights, dtype=np.float32)
        n, m = self.weights.shape
        if n == 0 or n != m or not np.isfinite(self.weights.data).all():
            raise ValueError("Weights must be a nonempty finite square matrix")
        p = self.params
        self.delay_steps = p.steps(p.delay_ms)
        self.refractory_steps = p.steps(p.refractory_ms)
        self.membrane_decay = math.exp(-p.dt_ms / p.membrane_tau_ms)
        self.synapse_decay = math.exp(-p.dt_ms / p.synapse_tau_ms)
        if math.isclose(p.membrane_tau_ms, p.synapse_tau_ms, rel_tol=1e-10):
            self.synapse_integral = (p.dt_ms / p.membrane_tau_ms) * self.membrane_decay
        else:
            self.synapse_integral = p.synapse_tau_ms / (p.synapse_tau_ms - p.membrane_tau_ms) * (
                self.synapse_decay - self.membrane_decay)
        self.voltage = np.empty(n, dtype=np.float32)
        self.current = np.zeros(n, dtype=np.float32)
        self.refractory = np.zeros(n, dtype=np.int32)
        self.pending = np.zeros((self.delay_steps + 1, n), dtype=np.float32)
        self.reset()

    def reset(self):
        self.voltage.fill(self.params.rest_mv)
        self.current.fill(0)
        self.refractory.fill(0)
        self.pending.fill(0)
        self.tick = 0

    @property
    def persistent_bytes(self):
        arrays = (self.weights.data, self.weights.indices, self.weights.indptr,
                  self.voltage, self.current, self.refractory, self.pending)
        return sum(a.nbytes for a in arrays)

    def step(self, drive_mv, *, connected=True):
        """Apply a constant external drive during one step; return spike flags.

        drive_mv is a voltage-equivalent input, not amperes or a measured
        photoreceptor response. connected=False disables recurrent transmission.
        """
        drive = np.asarray(drive_mv, dtype=np.float32)
        if drive.shape != self.voltage.shape or not np.isfinite(drive).all():
            raise ValueError("Drive must be a finite vector with one value per neuron")
        slot = self.tick % len(self.pending)
        if connected:
            self.current += self.weights @ self.pending[slot]
        self.pending[slot].fill(0)
        active = self.refractory == 0
        p = self.params
        updated = (p.rest_mv + (self.voltage - p.rest_mv) * self.membrane_decay
                   + drive * (1 - self.membrane_decay) + self.current * self.synapse_integral)
        self.voltage[:] = np.where(active, updated, p.reset_mv)
        self.current *= self.synapse_decay
        self.refractory[:] = np.maximum(self.refractory - 1, 0)
        spikes = active & (self.voltage >= p.threshold_mv)
        self.voltage[spikes] = p.reset_mv
        self.refractory[spikes] = self.refractory_steps
        # The spike is emitted at tick+1, not the start of this interval.
        destination = (self.tick + 1 + self.delay_steps) % len(self.pending)
        self.pending[destination] = spikes
        self.tick += 1
        if not np.isfinite(self.voltage).all() or not np.isfinite(self.current).all():
            raise FloatingPointError("Non-finite neural state")
        return spikes
