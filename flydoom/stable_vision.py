"""Cancellation-free evoked release; the original graded equation is unchanged.

Legacy checkpoints retain their original arithmetic. New experiments must opt
in explicitly and recalibrate/evaluate their decoder on the corrected states.
"""
import numpy as np
import json
from pathlib import Path
from scipy import sparse

from flydoom.graded_vision import GradedVision

MODE = 'stable_evoked_release_v1'
SCHEMA = 'stable_sensorimotor_v1'


def evoked_release(baseline, delta):
    """Compute relu(baseline + delta) - relu(baseline) without losing tiny deltas."""
    baseline, delta = np.asarray(baseline), np.asarray(delta)
    if baseline.shape != delta.shape:
        raise ValueError('Baseline and evoked state must have identical shapes')
    return np.maximum(delta + np.minimum(baseline, 0), -np.maximum(baseline, 0))


class StableGradedVision(GradedVision):
    def step(self, drive, *, connected=True):
        drive = np.asarray(drive, dtype=np.float32)
        if drive.shape != self.delta.shape or not np.isfinite(drive).all():
            raise ValueError('Expected one finite drive per cell')
        release = evoked_release(self.baseline, self.delta)
        release[self.blocked_release] = 0
        coupling = self.weights @ release if connected else 0
        self.delta[:] = self.decay*self.delta + (1-self.decay)*(drive+coupling)
        if not np.isfinite(self.delta).all():
            raise FloatingPointError('Non-finite graded activity')
        return self.delta


def upgrade(encoder):
    """Replace arithmetic only, preserving the exact checkpoint parameters."""
    old = encoder.model
    model = StableGradedVision.__new__(StableGradedVision)
    model.__dict__.update(old.__dict__)
    encoder.model = model
    encoder.reset()
    return encoder


def load_any(folder):
    """Load verified artifacts and honor their explicitly versioned arithmetic."""
    from flydoom.adaptive_policy import load_any as legacy_load
    from flydoom.adaptive_policy import AdaptiveActor
    from flydoom.data import digest
    from flydoom.retinal_policy import WINDOW_MS
    from flydoom.sensorimotor import SensorimotorEncoder, plastic_groups
    folder = Path(folder)
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    if report.get('schema') != SCHEMA:
        if report.get('neural_numerics'):
            raise ValueError('Corrected arithmetic requires its versioned schema')
        return legacy_load(folder)
    if report.get('status') != 'completed' or report.get('neural_numerics') != MODE:
        raise ValueError('Require completed, supported neural arithmetic')
    if 'stable_vision.py' not in report['source_sha256']:
        raise ValueError('Corrected arithmetic must be source-pinned')
    for name, sha in report['source_sha256'].items():
        if digest(Path(__file__).with_name(name), 'sha256') != sha:
            raise ValueError('Neural implementation changed: '+name)
    for name, sha in report['output_sha256'].items():
        if digest(folder/name, 'sha256') != sha:
            raise ValueError('Neural artifact changed: '+name)
    encoder = SensorimotorEncoder.__new__(SensorimotorEncoder)
    with np.load(folder/'circuit.npz', allow_pickle=False) as d:
        for key in ('ids', 'types', 'inputs', 'outputs', 'uv', 'visible', 'directions'):
            setattr(encoder, key, d[key].copy())
        encoder.model = StableGradedVision(sparse.load_npz(folder/'weights.npz'), d['tau'], dt_ms=1000/120/10)
        encoder.model.baseline = d['baseline'].copy()
        encoder.steps = int(d['steps'])
        for key in ('adaptation_gain', 'adaptation_tau_ms', 'dark_gain'):
            setattr(encoder, key, float(d[key]))
    encoder.model.decay = np.exp(-(WINDOW_MS/encoder.steps)/encoder.model.tau)
    meta = json.loads((folder/'circuit.json').read_text(encoding='utf-8'))
    encoder.camera, encoder.output_kind, encoder.bridge = meta['camera'], meta['output_kind'], meta['bridge']
    encoder.base_weights = encoder.model.weights.data.copy()
    encoder.base_tau = encoder.model.tau.copy()
    encoder.group_names, encoder.edge_group = plastic_groups(encoder.model.weights, encoder.types, 128)
    encoder.gains = np.ones(len(encoder.group_names))
    encoder.reset()
    actor = AdaptiveActor.load(folder/'actor.npz')
    np.testing.assert_array_equal(encoder.ids[encoder.outputs[actor.indices]], actor.root_ids)
    return encoder, actor, report
