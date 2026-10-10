"""Reconstruct measured timing inputs and replay the saved candidate."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from flydoom.data import digest
from flydoom.eye_mapping import write_json
from flydoom.graded_vision import GradedVision
from flydoom.polarity_audit import audit as audit_parent
from flydoom.retinal_experiment import prepare
from flydoom.retinal_mapping import RESULT
from flydoom.timing_sources import ROOT, load_split, FILES, verify
from flydoom.timing_training import PARENT, TimingCircuit, metrics, fit_scales, TRAINED_TYPES, DT_MS


def load_network(output):
    """Load a research-only graded checkpoint; never select a live Doom policy."""
    parameters = json.loads((output / 'parameters.json').read_text())
    report = json.loads((output / 'report.json').read_text())
    for name in ('parameters.json', 'checkpoint.npz'):
        if digest(output / name, 'sha256') != report['output_sha256'][name]:
            raise ValueError('Timing checkpoint identity mismatch')
    weight_path = Path(parameters['weights_path'])
    if digest(weight_path, 'sha256') != parameters['weights_sha256']:
        raise ValueError('Parent weight identity mismatch')
    weights = sparse.load_npz(weight_path)
    with np.load(output / 'checkpoint.npz') as data:
        ids, tau = data['root_ids'].copy(), data['tau_ms'].copy()
    if ids.dtype != np.uint64 or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid exact root IDs')
    return ids, GradedVision(weights, tau, dt_ms=DT_MS)


def audit(output):
    report = json.loads((output / 'report.json').read_text())
    for n, sha in report['output_sha256'].items():
        if digest(output / n, 'sha256') != sha:
            raise ValueError('Output checksum mismatch: ' + n)
    for n, sha in report['source_sha256'].items():
        if digest(Path(__file__).parent / n, 'sha256') != sha:
            raise ValueError('Source checksum mismatch: ' + n)
    if digest(ROOT / 'timing_manifest.json', 'sha256') != report['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest changed')
    for n, sha in FILES.items():
        verify((ROOT / n).read_bytes(), sha)
    if digest(PARENT / 'report.json', 'sha256') != report['parent_report_sha256']:
        raise ValueError('Parent report changed')
    audit_parent(PARENT)
    system = prepare(RESULT)
    ids, network = load_network(output)
    np.testing.assert_array_equal(ids, system['ids'])
    tau = [report['selected']['tau_ms'][k] for k in TRAINED_TYPES]
    types = np.array([r['cell_type'] for r in system['rows']])
    expected_tau = np.array([report['selected']['tau_ms'].get(k, 20.) for k in types])
    np.testing.assert_allclose(network.tau, expected_tau, rtol=1e-7)
    weights = sparse.load_npz(PARENT / 'weights.npz')
    circuit = TimingCircuit(weights, types, system['selected'])
    time, train = load_split('train')
    test_time, test = load_split('test')
    np.testing.assert_array_equal(time, test_time)
    scales = np.maximum(np.sqrt(np.mean(train**2, axis=(1, 2))), 1e-8)
    gain, prediction = fit_scales(circuit.simulate(tau, time), train)
    initial_gain, initial = fit_scales(circuit.simulate([20,20,20], time), train)
    np.testing.assert_allclose(gain, report['selected']['gain'], rtol=1e-10)
    with np.load(output / 'traces.npz') as saved:
        for key, expected in [('time_ms',time),('train',train),('test',test),('initial',initial),('fitted',prediction)]:
            np.testing.assert_allclose(saved[key], expected, rtol=1e-10, atol=1e-12)
    for name, predicted, target in [('initial_train',initial,train),('initial_test',initial,test),
                                    ('fitted_train',prediction,train),('fitted_test',prediction,test)]:
        actual = metrics(predicted, target, time, scales)
        np.testing.assert_allclose(actual['nrmse'], report['metrics'][name]['nrmse'], rtol=1e-10)
        if actual['traces'] != report['metrics'][name]['traces']:
            raise ValueError('Timing metric replay differs: ' + name)
    # Check the efficient linear simulator against the actual saved graded kernel,
    # not just a second call to the same fitting simulator.
    delta_error = 0.
    linear = np.zeros(len(ids))
    decay = np.exp(-DT_MS/expected_tau)
    for tick in range(240):
        drive = np.zeros(len(ids), dtype=np.float32)
        drive[system['selected']] = .1*np.clip((20-tick*DT_MS)/DT_MS, 0, 1)
        linear = decay*linear+(1-decay)*(weights@linear+drive)
        network.step(drive)
        delta_error = max(delta_error, float(np.max(np.abs(linear-network.delta))))
    if delta_error > 1e-5:
        raise ValueError('Linear fit does not match the saved graded network')
    result = {'schema':'timing_audit_v1','passed':True,'measured_targets_reconstructed':True,
              'candidate_replayed':True,'exact_root_order_checked':True,
              'max_kernel_difference':delta_error,'report_sha256':digest(output/'report.json','sha256'),
              'biological_validation':False}
    write_json(output/'audit.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/timing-training-v1'))
    print(json.dumps(audit(parser.parse_args().output),indent=2))
