"""Restore measured C2/C3 edges and fit a bounded temporal-feedback candidate.

This is a development experiment: low-luminance means were already inspected
in timing-training-v1. They are not a new independent confirmation set.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from flydoom.data import digest
from flydoom.eye_mapping import write_json
from flydoom.graded_vision import GradedVision
from flydoom.retinal_experiment import TYPES
from flydoom.retinal_mapping import RESULT, load_counts
from flydoom.simulation import transmitter_signs
from flydoom.timing_audit import load_network
from flydoom.timing_sources import load_split
from flydoom.timing_training import observe, fit_scales, metrics

PARENT = Path('runs/timing-training-v1')
OUTPUT = Path('runs/retinal-feedback-v1')
DT = 1000 / 120 / 10


def prepare():
    """Embed the exact parent weights, then add only measured C2/C3 edges."""
    old_ids, parent = load_network(PARENT)
    ids, rows, full, provenance = load_counts()
    chosen = np.array([i for i, r in enumerate(rows)
                       if r['side'] == 'left' and r['cell_type'] in (*TYPES, 'C2', 'C3')])
    ids, rows = ids[chosen], [rows[i] for i in chosen]
    types = np.array([r['cell_type'] for r in rows])
    old = np.searchsorted(ids, old_ids)
    if not np.array_equal(ids[old], old_ids):
        raise ValueError('Parent root IDs are absent from expanded graph')
    coo = parent.weights.tocoo()
    base = sparse.csr_matrix((coo.data, (old[coo.row], old[coo.col])), shape=(len(ids), len(ids)))
    counts = full[chosen][:, chosen].tocsr()
    signs, rules = transmitter_signs(rows)
    coo = counts.tocoo()
    new = np.isin(types, ['C2', 'C3'])
    mask = new[coo.row] | new[coo.col]
    added = sparse.csr_matrix((coo.data[mask] * signs[coo.col[mask]],
                              (coo.row[mask], coo.col[mask])), shape=counts.shape, dtype=np.float64)
    # One gain per target type preserves within-type anatomical contact ratios.
    # The budget bounds all added edges jointly while leaving old weights exact.
    existing_mass = np.asarray(abs(base).sum(axis=1)).ravel()
    added_mass = np.asarray(abs(added).sum(axis=1)).ravel()
    target_gains = {}
    for kind in sorted(set(types)):
        active = (types == kind) & (added_mass > 0)
        target_gains[kind] = float(np.min((.9-existing_mass[active]) / added_mass[active])) if active.any() else 0.
    added.data *= np.repeat([target_gains[k] for k in types], np.diff(added.indptr))
    # Each group includes incoming edges to its new cells and its outgoing edges
    # into original cells. C2/C3 cross-edges belong to the postsynaptic new group.
    coo = added.tocoo()
    group = np.where(new[coo.row], types[coo.row], types[coo.col])
    parts = [sparse.csr_matrix((coo.data[group == k], (coo.row[group == k], coo.col[group == k])),
                              shape=base.shape) for k in ('C2', 'C3')]
    tau = np.full(len(ids), 20., dtype=np.float64)
    tau[old] = parent.tau
    receptors = json.loads((RESULT / 'receptors.json').read_text())
    inputs = np.searchsorted(ids, np.array([int(r['root_id']) for r in receptors], dtype=np.uint64))
    if not np.array_equal(ids[inputs], np.array([int(r['root_id']) for r in receptors], dtype=np.uint64)):
        raise ValueError('Receptor root ID mismatch')
    return dict(ids=ids, rows=rows, types=types, base=base, parts=parts, tau=tau,
                inputs=inputs, receptors=receptors, target_gains=target_gains,
                counts=counts, provenance=provenance, sign_rules=rules,
                old_indices=old, parent_weights=parent.weights)


def build(system, parameters, dt_ms=DT):
    parameters = np.asarray(parameters, float)
    if parameters.shape != (4,) or not np.isfinite(parameters).all():
        raise ValueError('Expected two time constants and two feedback gains')
    if np.any(parameters[:2] < 2) or np.any(parameters[:2] > 500) or np.any(parameters[2:] < 0) or np.any(parameters[2:] > 1):
        raise ValueError('Feedback parameters outside declared bounds')
    weights = system['base'] + system['parts'][0] * parameters[2] + system['parts'][1] * parameters[3]
    tau = system['tau'].copy()
    for kind, value in zip(('C2', 'C3'), parameters[:2]):
        tau[system['types'] == kind] = value
    return GradedVision(weights, tau, dt_ms=dt_ms)


def simulate(system, parameters, samples, dt_ms=DT, connected=True):
    model = build(system, parameters, dt_ms)
    groups = [np.flatnonzero(system['types'] == k) for k in ('L1', 'L2')]
    steps = int(round(max(samples) / dt_ms))
    times = np.arange(steps+1) * dt_ms
    result = []
    for sign in (-1, 1):
        model.reset()
        values = np.zeros((steps+1, 2))
        drive = np.zeros(len(system['ids']))
        for tick in range(steps):
            drive.fill(0)
            drive[system['inputs']] = sign * .1 * np.clip((20-times[tick])/dt_ms, 0, 1)
            model.step(drive, connected=connected)
            values[tick+1] = [model.delta[g].mean(dtype=np.float64) for g in groups]
        result.append(observe(times, values, samples, 3.).T)
    return np.stack(result, axis=1)


def load_checkpoint(folder=OUTPUT):
    folder = Path(folder)
    report = json.loads((folder / 'report.json').read_text())
    for name, sha in report['output_sha256'].items():
        if digest(folder / name, 'sha256') != sha:
            raise ValueError('Feedback checkpoint checksum mismatch: ' + name)
    for name, sha in report['source_sha256'].items():
        if digest(Path(__file__).parent / name, 'sha256') != sha:
            raise ValueError('Feedback source checksum mismatch: ' + name)
    data = np.load(folder / 'checkpoint.npz', allow_pickle=False)
    with data:
        ids, tau = data['root_ids'].copy(), data['tau_ms'].copy()
    if ids.dtype != np.uint64 or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid checkpoint neuron identities')
    return ids, GradedVision(sparse.load_npz(folder / 'weights.npz'), tau, dt_ms=DT), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {'schema': 'anatomical_feedback_plan_v1', 'training': 'L1/L2 high-luminance means',
            'comparison': 'Previously seen low-luminance means; development evidence only',
            'trainable': ['C2 tau', 'C3 tau', 'C2 edge-group gain', 'C3 edge-group gain'],
            'bounds': {'tau_ms': [2, 500], 'gain': [0, 1], 'absolute_row_sum_max': .9},
            'frozen': 'All parent edges and original cell time constants', 'dt_ms': DT,
            'starts': [[20, 20, .5, .5], [100, 100, .5, .5]], 'max_nfev': 25,
            'selection': 'Training NRMSE only; no gain refit on comparison data',
            'gate': 'Reference waveform threshold remains 0.5; independent confirmation still required',
            'live_mode': 'Observation only; never promotes a motor policy'}
    write_json(args.output / 'plan.json', plan)
    system = prepare()
    times, target = load_split('train')
    scales = np.maximum(np.sqrt(np.mean(target**2, axis=(1, 2))), 1e-8)
    fits = []
    for start in plan['starts']:
        print('Fitting measured C2/C3 feedback: ' + str(start), flush=True)
        def residual(x):
            pars = [*np.exp(x[:2]), *x[2:]]
            raw = simulate(system, pars, times)
            _, prediction = fit_scales(raw, target)
            return ((prediction-target)/scales[:, None, None]).ravel()
        result = least_squares(residual, [*np.log(start[:2]), *start[2:]],
                               bounds=([np.log(2), np.log(2), 0, 0], [np.log(500), np.log(500), 1, 1]),
                               diff_step=.005, max_nfev=25, ftol=1e-5, xtol=1e-5, gtol=1e-5)
        pars = [*np.exp(result.x[:2]), *result.x[2:]]
        fits.append({'parameters': list(map(float, pars)), 'loss': float(np.mean(result.fun**2)),
                     'success': bool(result.success), 'nfev': result.nfev})
    selected = min(fits, key=lambda f: f['loss'])
    pars = selected['parameters']
    gains, prediction = fit_scales(simulate(system, pars, times), target)
    other_times, comparison = load_split('test')
    if not np.array_equal(times, other_times):
        raise ValueError('Comparison time axes differ')
    fine = simulate(system, pars, times, DT/2) * gains[:, None, None]
    disconnected = simulate(system, pars, times, connected=False)
    if np.any(disconnected):
        raise ValueError('Disconnected downstream cells responded')
    ablated = simulate(system, [*pars[:2], 0, 0], times) * gains[:, None, None]
    model = build(system, pars)
    old = system['old_indices']
    difference = model.weights[old][:, old] - system['parent_weights']
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-8:
        raise ValueError('Frozen parent weights changed')
    np.savez_compressed(args.output/'checkpoint.npz', root_ids=system['ids'], tau_ms=model.tau)
    sparse.save_npz(args.output/'weights.npz', model.weights)
    np.savez_compressed(args.output/'traces.npz', time_ms=times, train=target, comparison=comparison,
                        fitted=prediction, feedback_ablated=ablated, fine=fine)
    write_json(args.output/'parameters.json', {'selected': selected, 'fits': fits,
               'observation_gains': gains.tolist(), 'target_type_added_edge_scales': system['target_gains'],
               'live_control_compatible': False, 'observation_only': True})
    contacts = {}
    for source in ('C2', 'C3'):
        for target_type in ('L1', 'L2', 'Mi1'):
            contacts[source+' -> '+target_type] = int(system['counts'][system['types']==target_type][:, system['types']==source].sum())
    numerical = float(np.max(np.abs(fine-prediction)/scales[:, None, None]))
    report = {'schema': 'anatomical_feedback_v1', 'neurons': len(system['ids']),
              'added_neurons': int(sum(np.isin(system['types'], ['C2', 'C3']))),
              'added_structural_edges': int(sum(part.nnz for part in system['parts'])),
              'contacts': contacts, 'selected': selected,
              'metrics': {'train': metrics(prediction, target, times, scales),
                          'seen_low_luminance': metrics(prediction, comparison, times, scales),
                          'feedback_ablated_train': metrics(ablated, target, times, scales)},
              'normalized_dt_halving_difference': numerical, 'row_bound': model.row_bound,
              'gates': {'parent_edges_preserved': True, 'disconnected_silent': True,
                        'numerical_convergence': numerical < .05,
                        'independent_confirmation': False, 'promoted_to_live_control': False},
              'limitations': ['FAFB coverage remains incomplete; only C2/C3 restored',
                              'Transmitter signs and contraction gains remain model assumptions',
                              'No luminance adaptation or calibrated phototransduction',
                              'Population means do not identify intrinsic membrane time constants',
                              'Low-luminance traces were previously seen; no new held-out success claim'],
              'parent_report_sha256': digest(PARENT/'report.json', 'sha256'),
              'source_sha256': {n:digest(Path(__file__).parent/n, 'sha256') for n in
                                ['retinal_feedback.py', 'graded_vision.py', 'timing_sources.py', 'timing_training.py']},
              'output_sha256': {n:digest(args.output/n, 'sha256') for n in
                                ['plan.json','parameters.json','checkpoint.npz','weights.npz','traces.npz']}}
    write_json(args.output/'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
