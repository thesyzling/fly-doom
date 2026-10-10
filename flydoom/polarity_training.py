"""Fit bounded existing type-pair gains to qualitative response-polarity priors.

This is a weakly supervised pilot, not a fit to measured physiological traces.
T4/T5 labels are withheld from the objective. Time constants are not trained.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from flydoom.data import digest
from flydoom.eye_mapping import write_json
from flydoom.graded_vision import GradedVision
from flydoom.physiology_sources import REFERENCE_ROOT, reference_targets
from flydoom.retinal_experiment import prepare
from flydoom.retinal_mapping import RESULT

OUTPUT = Path('runs/polarity-training-v1')


def fixed_point(weights, drive, *, tolerance=1e-10):
    """Solve the tonic driven state of a contractive graded network."""
    weights = sparse.csr_matrix(weights, dtype=np.float64)
    drive = np.asarray(drive, dtype=np.float64)
    if weights.shape != (drive.size, drive.size) or not np.isfinite(drive).all():
        raise ValueError('Invalid state dimensions or drive')
    if not np.isfinite(weights.data).all() or np.asarray(abs(weights).sum(axis=1)).max() >= 1:
        raise ValueError('Expected finite contractive weights')
    state = np.ones(drive.size)
    for _ in range(1000):
        new = 1 + drive + weights @ np.maximum(state, 0)
        if np.max(np.abs(new - state)) < tolerance:
            return new
        state = new
    raise ValueError('Fixed point did not converge')


def response(weights, drive):
    return fixed_point(weights, drive) - fixed_point(weights, np.zeros_like(drive))


class PairGains:
    """Scale existing inputs to Mi4 by presynaptic type, retaining signed zeros."""

    def __init__(self, weights, types, target='Mi4'):
        self.base = sparse.csr_matrix(weights, dtype=np.float64)
        self.types = np.asarray(types)
        if self.base.shape != (len(self.types), len(self.types)):
            raise ValueError('Type/graph size mismatch')
        rows = np.repeat(np.arange(len(types)), np.diff(self.base.indptr))
        selected = (self.types[rows] == target) & (self.base.data != 0)
        names = sorted(set(self.types[self.base.indices[selected]]))
        self.names = [f'{source}->{target}' for source in names]
        self.positions = [np.flatnonzero(selected & (self.types[self.base.indices] == source)) for source in names]

    def weights(self, log_gains):
        log_gains = np.asarray(log_gains, dtype=np.float64)
        if log_gains.shape != (len(self.names),) or not np.isfinite(log_gains).all():
            raise ValueError('Expected one finite log gain per existing type pair')
        if np.any(log_gains < np.log(.05) - 1e-8) or np.any(log_gains > np.log(1.5) + 1e-8):
            raise ValueError('Gain exceeds declared bounds')
        weights = self.base.copy()
        for positions, gain in zip(self.positions, np.exp(log_gains)):
            weights.data[positions] *= gain
        return weights


def prior_loss(values, signs, scales):
    """A dimensionless engineering margin, not an experimental amplitude."""
    return float(np.mean(np.maximum(0, .25 - signs * values / scales) ** 2))


def polarity_metrics(values, populations, targets, train):
    rows = {}
    for kind, indices in populations.items():
        if kind not in targets or targets[kind] == 0:
            continue
        mean = float(np.mean(values[indices], dtype=np.float64))
        rows[kind] = {'mean_delta': mean, 'expected_sign': targets[kind],
                      'matches': bool(targets[kind] * mean > 1e-8),
                      'split': 'train' if kind in train else 'held_out_cell_type'}
    return rows


def pulse(weights, system, amplitude, *, connected=True):
    net = GradedVision(weights, np.full(len(system['ids']), 20.), dt_ms=.5)
    traces, peak = [], 0.
    for tick in range(400):
        drive = np.zeros(len(system['ids']), dtype=np.float32)
        if 40 <= tick * .5 < 160:
            drive[system['selected']] = amplitude
        net.step(drive, connected=connected)
        peak = max(peak, float(np.abs(net.delta).max()))
        if peak > 2:
            raise FloatingPointError('Activity exceeds declared bound')
        if (tick + 1) % 20 == 0:
            traces.append([float(net.delta[idx].mean(dtype=np.float64)) for idx in system['populations'].values()])
    return np.array(traces), peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if (args.output / 'report.json').exists():
        raise ValueError('Choose a new output directory to preserve previous evidence')
    args.output.mkdir(parents=True, exist_ok=True)
    targets = reference_targets()
    train = ['L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9']
    heldout = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
    plan = {'schema': 'polarity_fit_plan_v1', 'train_types': train, 'held_out_types': heldout,
            'fit_condition': 'Steady +0.5 input at mapped R1-6 cells',
            'trainable': 'Existing nonzero source-type to Mi4 shared positive gains only',
            'selection_reason': 'Mi4 was the only polarity mismatch in the already inspected prototype',
            'exploratory': True, 'raw_recordings_used': False, 'tau_trained': False,
            'gain_bounds': [.05, 1.5], 'margin': .25, 'regularization': .0001,
            'optimizer': 'L-BFGS-B; zero log gains; maxiter=60; finite-difference eps=1e-4',
            'evaluation': 'Unfitted transient flashes at +/-0.25 and +/-0.5; no input; disconnected',
            'holdout_note': 'Withheld label groups in one author synthesis, not independent experimental validation',
            'reference_sha256': digest(REFERENCE_ROOT / 'groundtruth_utils.py', 'sha256')}
    write_json(args.output / 'plan.json', plan)
    system = prepare(RESULT)
    populations = system['populations']
    types = [row['cell_type'] for row in system['rows']]
    pairs = PairGains(system['model'].weights, types)
    drive = np.zeros(len(types))
    drive[system['selected']] = .5
    initial = response(pairs.base, drive)
    base_means = np.array([initial[populations[k]].mean() for k in train])
    scales = np.maximum(np.abs(base_means), 1e-5)
    signs = np.array([targets[k] for k in train])
    history = []

    def objective(parameters):
        values = response(pairs.weights(parameters), drive)
        means = np.array([values[populations[k]].mean() for k in train])
        return prior_loss(means, signs, scales) + .0001 * float(np.mean(parameters ** 2))

    def progress(parameters):
        history.append(objective(parameters))
        print(f'Iteration {len(history)} loss={history[-1]:.6g}', flush=True)

    initial_loss = objective(np.zeros(len(pairs.names)))
    fitted = minimize(objective, np.zeros(len(pairs.names)), method='L-BFGS-B',
                      bounds=[(np.log(.05), np.log(1.5))] * len(pairs.names), callback=progress,
                      options={'maxiter': 60, 'eps': 1e-4, 'ftol': 1e-10, 'gtol': 1e-7})
    weights = pairs.weights(fitted.x)
    final = response(weights, drive)
    initial_metrics = polarity_metrics(initial, populations, targets, train)
    final_metrics = polarity_metrics(final, populations, targets, train)
    conditions, arrays = {}, {}
    for label, w in [('initial', pairs.base), ('fitted', weights)]:
        for name, amplitude, connected in [('bright_025', .25, True), ('dark_025', -.25, True),
                                          ('bright_050', .5, True), ('dark_050', -.5, True),
                                          ('no_input', 0., True), ('disconnected', .5, False)]:
            trace, peak = pulse(w, system, amplitude, connected=connected)
            key = f'{label}_{name}'
            arrays[key] = trace
            # Average plateau while the flash is still active (110 through 160 ms).
            mean = trace[10:16].mean(axis=0)
            conditions[key] = {'peak_abs_cell_delta': peak,
                               'population_mean_110_160_ms': dict(zip(populations, mean.tolist()))}
            print(f'Evaluated {key}', flush=True)
    sparse.save_npz(args.output / 'weights.npz', weights)
    np.savez_compressed(args.output / 'traces.npz', time_ms=np.arange(10, 201, 10),
                        population_types=np.array(list(populations)), root_ids=system['ids'], **arrays)
    parameters = {'schema': 'qualitative_type_pair_fit_v1', 'pairs': dict(zip(pairs.names, np.exp(fitted.x).tolist())),
                  'tau_ms_by_type': system['tau_by_type'], 'tonic_bias': 1.,
                  'train_scales': dict(zip(train, scales.tolist())), 'dimensional_units': False}
    write_json(args.output / 'parameters.json', parameters)
    changed = int(np.count_nonzero(weights.data != pairs.base.data))
    assert np.array_equal(weights.indices, pairs.base.indices) and np.array_equal(weights.indptr, pairs.base.indptr)
    assert np.array_equal(np.sign(weights.data), np.sign(pairs.base.data))
    for label in ('initial', 'fitted'):
        assert np.max(np.abs(arrays[label + '_no_input'])) == 0
        assert np.max(np.abs(arrays[label + '_disconnected'][:, 1:])) == 0
    report = {'schema': 'qualitative_polarity_training_v1', 'scope': system['scope'],
              'trained': True, 'raw_recording_fit': False, 'live_policy_changed': False,
              'optimizer': {'success': bool(fitted.success), 'message': str(fitted.message),
                            'iterations': int(fitted.nit), 'evaluations': int(fitted.nfev)},
              'loss_initial': initial_loss, 'loss_final': float(fitted.fun), 'loss_history': history,
              'shared_parameters': len(pairs.names), 'changed_existing_edges': changed,
              'row_bound': float(np.asarray(abs(weights).sum(axis=1)).max()),
              'initial': initial_metrics, 'fitted': final_metrics, 'conditions': conditions,
              'mapping_report_sha256': digest(RESULT / 'report.json', 'sha256'),
              'provenance': system['provenance'], 'reference': json.loads((REFERENCE_ROOT / 'manifest.json').read_text()),
              'source_sha256': {n: digest(Path(__file__).parent / n, 'sha256') for n in
                                ['polarity_training.py', 'physiology_sources.py', 'graded_vision.py', 'retinal_experiment.py']},
              'output_sha256': {n: digest(args.output / n, 'sha256') for n in
                                ['weights.npz', 'traces.npz', 'parameters.json', 'plan.json']},
              'gates': {'structural_zeros_and_signs_preserved': True, 'measured_physiology_validated': False,
                        'whole_brain_integrated': False, 'promoted_to_live': False},
              'limitations': ['Qualitative literature labels only; no measured trace, amplitude or latency fitting',
                              'Dryad archive download returned HTTP 403; API download returned HTTP 401',
                              'Exploratory Mi4 target selected after inspecting prototype responses',
                              'Held-out cell types share the circuit and literature source; not independent animals',
                              'Fixed 20 ms time constants; engineering margin and tonic baseline',
                              'Retinal assignments remain provisional; visual graph omits 46.25% of incoming contacts',
                              'No Laya task loss, descending output, whole-brain integration or live policy promotion']}
    write_json(args.output / 'report.json', report)
    from flydoom.polarity_report import render
    render(args.output)
    print(json.dumps({'report': str(args.output / 'index.html'), 'loss': [initial_loss, float(fitted.fun)],
                      'changed_edges': changed, 'optimizer_success': bool(fitted.success)}, indent=2))


if __name__ == '__main__':
    main()
