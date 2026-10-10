"""Reconstruct and audit the qualitative visual-gain checkpoint offline."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from flydoom.data import digest
from flydoom.eye_mapping import write_json
from flydoom.physiology_sources import REFERENCE_ROOT, reference_targets
from flydoom.polarity_training import PairGains, polarity_metrics, prior_loss, response
from flydoom.retinal_experiment import prepare
from flydoom.retinal_mapping import RESULT


def audit(output):
    report = json.loads((output / 'report.json').read_text())
    for name, sha in report['output_sha256'].items():
        if digest(output / name, 'sha256') != sha:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    for name, sha in report['source_sha256'].items():
        if digest(Path(__file__).parent / name, 'sha256') != sha:
            raise ValueError(f'Source checksum mismatch: {name}')
    if digest(RESULT / 'report.json', 'sha256') != report['mapping_report_sha256']:
        raise ValueError('Retinal mapping changed')
    plan = json.loads((output / 'plan.json').read_text())
    parameters = json.loads((output / 'parameters.json').read_text())
    targets = reference_targets()
    if digest(REFERENCE_ROOT / 'groundtruth_utils.py', 'sha256') != plan['reference_sha256']:
        raise ValueError('Reference checksum mismatch')
    train, held = plan['train_types'], plan['held_out_types']
    if set(train) & set(held) or len(train) != 13 or len(held) != 8:
        raise ValueError('Invalid cell-type label split')
    system = prepare(RESULT)
    pairs = PairGains(system['model'].weights, [r['cell_type'] for r in system['rows']])
    gains = np.log([parameters['pairs'][k] for k in pairs.names])
    reconstructed = pairs.weights(gains)
    saved = sparse.load_npz(output / 'weights.npz')
    np.testing.assert_array_equal(saved.indices, pairs.base.indices)
    np.testing.assert_array_equal(saved.indptr, pairs.base.indptr)
    np.testing.assert_allclose(saved.data, reconstructed.data, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(np.sign(saved.data), np.sign(pairs.base.data))
    if np.asarray(abs(saved).sum(axis=1)).max() >= 1:
        raise ValueError('Contraction bound exceeded')
    changed = np.count_nonzero(saved.data != pairs.base.data)
    if changed != report['changed_existing_edges']:
        raise ValueError('Changed edge count mismatch')
    drive = np.zeros(len(system['ids']))
    drive[system['selected']] = .5
    for label, weights in [('initial', pairs.base), ('fitted', saved)]:
        values = response(weights, drive)
        actual = polarity_metrics(values, system['populations'], targets, train)
        if actual != report[label]:
            raise ValueError('Steady response metrics differ from reconstructed model')
        scales = np.array([parameters['train_scales'][k] for k in train])
        means = np.array([actual[k]['mean_delta'] for k in train])
        signs = np.array([targets[k] for k in train])
        loss = prior_loss(means, signs, scales)
        if label == 'fitted':
            loss += .0001 * np.mean(gains ** 2)
        np.testing.assert_allclose(loss, report['loss_final' if label == 'fitted' else 'loss_initial'], rtol=1e-8)
    with np.load(output / 'traces.npz') as traces:
        np.testing.assert_array_equal(traces['root_ids'], system['ids'])
        np.testing.assert_array_equal(traces['time_ms'], np.arange(10, 201, 10))
        np.testing.assert_array_equal(traces['population_types'], list(system['populations']))
        for key, condition in report['conditions'].items():
            trace = traces[key]
            if trace.shape != (20, len(system['populations'])) or not np.isfinite(trace).all():
                raise ValueError('Invalid transient trace')
            if key.endswith('no_input') and np.any(trace):
                raise ValueError('No-input control was not silent')
            if key.endswith('disconnected') and np.any(trace[:, 1:]):
                raise ValueError('Disconnected downstream control was not silent')
            expected = np.array([condition['population_mean_110_160_ms'][k] for k in system['populations']])
            np.testing.assert_allclose(trace[10:16].mean(axis=0), expected, rtol=1e-12, atol=1e-15)
    result = {'schema': 'polarity_artifact_audit_v1', 'passed': True,
              'weights_reconstructed': True, 'steady_responses_recomputed': True,
              'transients_replayed': False, 'changed_existing_edges': int(changed),
              'report_sha256': digest(output / 'report.json', 'sha256'),
              'note': 'Artifact consistency audit; not experimental physiological validation'}
    write_json(output / 'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/polarity-training-v1'))
    print(json.dumps(audit(parser.parse_args().output), indent=2))
