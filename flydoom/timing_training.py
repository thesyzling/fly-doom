"""Calibrate visual pathway dynamics against measured L1/L2 temporal responses.

Time constants are conditional model fits, not unique intrinsic membrane
measurements. The observation model separates indicator lag and bin averaging.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from flydoom.data import digest
from flydoom.eye_mapping import write_json
from flydoom.polarity_audit import audit as audit_parent
from flydoom.retinal_experiment import prepare
from flydoom.retinal_mapping import RESULT
from flydoom.timing_sources import ROOT, load_split

PARENT = Path('runs/polarity-training-v1')
TRAINED_TYPES = ('R1-6', 'L1', 'L2')
DT_MS = 1000 / 120 / 20
PULSE_MS = 20.


def observe(time_ms, voltage, sample_ms, sensor_tau_ms, bin_ms=1000 / 120):
    """First-order sensor followed by causal trailing-bin average.

Sensor time constants are sensitivity scenarios, not calibrated properties of
this animal preparation. No target-dependent shifting or amplitude fitting here.
"""
    time_ms, voltage, sample_ms = map(np.asarray, (time_ms, voltage, sample_ms))
    if voltage.shape[0] != len(time_ms) or np.any(np.diff(time_ms) <= 0):
        raise ValueError('Invalid observation timeline')
    if sensor_tau_ms < 0 or bin_ms <= 0:
        raise ValueError('Invalid sensor lag or averaging window')
    filtered = voltage.copy()
    if sensor_tau_ms:
        for i in range(1, len(time_ms)):
            decay = np.exp(-(time_ms[i] - time_ms[i-1]) / sensor_tau_ms)
            filtered[i] = decay * filtered[i-1] + (1-decay) * voltage[i]
    dt = np.diff(time_ms)
    integral = np.vstack([np.zeros(voltage.shape[1]), np.cumsum((filtered[:-1] + filtered[1:]) * dt[:, None] / 2, axis=0)])
    return np.column_stack([
        (np.interp(sample_ms, time_ms, integral[:, j]) -
         np.interp(sample_ms - bin_ms, time_ms, integral[:, j], left=0)) / bin_ms
        for j in range(voltage.shape[1])])


class TimingCircuit:
    """Exact linear regime of the existing graded graph for small retinal pulses."""

    def __init__(self, weights, types, input_indices, *, dt_ms=DT_MS):
        self.weights = sparse.csr_matrix(weights, dtype=np.float64)
        self.types = np.asarray(types)
        self.groups = [np.flatnonzero(self.types == k) for k in TRAINED_TYPES]
        self.inputs = np.asarray(input_indices, dtype=int)
        self.dt_ms = dt_ms
        self.row_bound = float(np.asarray(abs(self.weights).sum(axis=1)).max())
        if self.row_bound >= 1 or any(len(g) == 0 for g in self.groups):
            raise ValueError('Need a contractive graph with all timing populations')
        baseline = np.ones(len(types))
        for _ in range(200):
            updated = 1 + self.weights @ np.maximum(baseline, 0)
            if np.max(np.abs(updated - baseline)) < 1e-10:
                break
            baseline = updated
        self.baseline = updated
        self.amplitude = .1
        # This bound guarantees neither pulse polarity crosses the release rectifier.
        if self.baseline.min() <= self.amplitude / (1-self.row_bound):
            raise ValueError('Small-signal linear regime is not guaranteed')

    def simulate(self, taus_ms, sample_ms, sensor_tau_ms=3., *, connected=True):
        taus_ms = np.asarray(taus_ms, dtype=float)
        if taus_ms.shape != (3,) or not np.isfinite(taus_ms).all() or np.any(taus_ms < 2) or np.any(taus_ms > 100):
            raise ValueError('Timing constants must lie in [2, 100] ms')
        tau = np.full(len(self.types), 20.)
        for group, value in zip(self.groups, taus_ms):
            tau[group] = value
        decay = np.exp(-self.dt_ms / tau)
        state = np.zeros(len(tau))
        duration = float(max(sample_ms))
        steps = int(round(duration / self.dt_ms))
        times = np.arange(steps+1) * self.dt_ms
        values = np.zeros((steps+1, 2))
        drive = np.zeros_like(state)
        for tick in range(steps):
            # Integrate the fraction of a step inside the physical 20 ms flash.
            fraction = np.clip((PULSE_MS-times[tick]) / self.dt_ms, 0, 1)
            drive.fill(0)
            drive[self.inputs] = self.amplitude * fraction
            coupling = self.weights @ state if connected else 0
            state = decay * state + (1-decay) * (coupling + drive)
            values[tick+1] = [state[group].mean() for group in self.groups[1:]]
        observed = observe(times, values, sample_ms, sensor_tau_ms)
        # Source export ordering is dark then light; the exact small-signal
        # circuit has sign-inverted responses. Retain that limitation explicitly.
        return np.stack([-observed.T, observed.T], axis=1)


def fit_scales(prediction, target):
    """One nonnegative observation gain per cell type, fit on training data only."""
    numerator = np.sum(prediction * target, axis=(1, 2))
    denominator = np.sum(prediction * prediction, axis=(1, 2))
    gains = np.maximum(0, numerator / np.maximum(denominator, 1e-30))
    return gains, prediction * gains[:, None, None]


def metrics(prediction, target, time_ms, scales):
    rows = []
    for j, cell in enumerate(('L1', 'L2')):
        for k, contrast in enumerate(('dark', 'light')):
            sign = 1 if contrast == 'dark' else -1
            early = np.flatnonzero((time_ms > 0) & (time_ms <= 100))
            measured_peak = int(early[np.argmax(sign * target[j, k, early])])
            predicted_peak = int(early[np.argmax(sign * prediction[j, k, early])])
            rows.append({'cell_type': cell, 'contrast': contrast,
                         'nrmse': float(np.sqrt(np.mean((prediction[j, k]-target[j, k])**2)) / scales[j]),
                         'measured_first_peak_ms': float(time_ms[measured_peak]),
                         'predicted_first_peak_ms': float(time_ms[predicted_peak]),
                         'first_peak_error_ms': float(abs(time_ms[measured_peak]-time_ms[predicted_peak])),
                         'late_rmse': float(np.sqrt(np.mean((prediction[j, k, time_ms>=50]-target[j, k, time_ms>=50])**2)))})
    return {'nrmse': float(np.sqrt(np.mean(((prediction-target)/scales[:, None, None])**2))), 'traces': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/timing-training-v1'))
    args = parser.parse_args()
    if (args.output / 'report.json').exists():
        raise ValueError('Choose a new output directory; previous evidence is retained')
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {'schema': 'measured_timing_plan_v1', 'train': 'L1/L2 highLum means; both contrasts; all 63 time samples',
            'test': 'L1/L2 lowLum means; evaluated only after training selection; no test gain refit',
            'split_limitation': 'Luminance condition holdout; control files lack animal IDs and individual traces',
            'trained_types': list(TRAINED_TYPES), 'tau_bounds_ms': [2, 100], 'other_tau_ms': 20,
            'weights': 'Frozen polarity-training-v1 existing edges', 'input': 'Mapped R1-6; +/-0.1 dimensionless',
            'stimulus': {'onset_ms': 0, 'duration_ms': PULSE_MS, 'source': '20 ms flash in primary methods'},
            'source_notebook_discrepancy': 'Author model approximates input at indices 2:5; use experimental onset and duration with explicit observation lag instead',
            'observation': 'Negative ASAP2f dF/F; subtract source t=0; trailing 8.333 ms bin; positive per-type amplitude gain',
            'peak_metric': 'Largest first-polarity response in (0, 100] ms; no forced early peak',
            'sensor_tau_ms_primary': 3., 'sensor_tau_ms_sensitivity': [0., 8.],
            'sensor_note': 'Sensitivity assumptions, not independent calibration of ASAP2f in this preparation',
            'starts_ms': [[20, 20, 20], [5, 40, 40]], 'selection': 'Lowest training loss; no held-out selection',
            'optimizer': 'Bounded least_squares on log tau; max_nfev=45; diff_step=0.002',
            'dt_ms': DT_MS, 'validation_dt_ms': DT_MS/2,
            'gate': {'test_nrmse_max': .5, 'test_first_peak_error_ms_max': 1000/120,
                     'source_timing_identified_required_for_live': True},
            'intrinsic_tau_claim': False, 'live_promotion': False}
    write_json(args.output / 'plan.json', plan)
    audit_parent(PARENT)
    system = prepare(RESULT)
    weights = sparse.load_npz(PARENT / 'weights.npz')
    circuit = TimingCircuit(weights, [r['cell_type'] for r in system['rows']], system['selected'])
    time_ms, target = load_split('train')
    scales = np.maximum(np.sqrt(np.mean(target**2, axis=(1, 2))), 1e-8)
    initial = circuit.simulate([20, 20, 20], time_ms)
    initial_gains, initial_prediction = fit_scales(initial, target)
    fits = []
    for sensor in (3., 0., 8.):
        for start in (plan['starts_ms'] if sensor == 3 else [plan['starts_ms'][0]]):
            def residual(log_tau):
                prediction = circuit.simulate(np.clip(np.exp(log_tau), 2, 100), time_ms, sensor)
                _, prediction = fit_scales(prediction, target)
                return ((prediction-target) / scales[:, None, None]).ravel()
            print(f'Fitting sensor={sensor} ms, initial tau={start}', flush=True)
            result = least_squares(residual, np.log(start), bounds=(np.log(2), np.log(100)),
                                   diff_step=.002, max_nfev=45, ftol=1e-7, xtol=1e-6, gtol=1e-6)
            tau = np.clip(np.exp(result.x), 2, 100)
            gains, prediction = fit_scales(circuit.simulate(tau, time_ms, sensor), target)
            singular = np.linalg.svd(result.jac, compute_uv=False)
            fit = {'sensor_tau_ms': sensor, 'start_ms': start, 'tau_ms': dict(zip(TRAINED_TYPES, tau.tolist())),
                   'gain': gains.tolist(), 'train_nrmse': float(np.sqrt(np.mean(result.fun**2))),
                   'success': bool(result.success), 'message': result.message, 'nfev': int(result.nfev),
                   'jacobian_singular_values': singular.tolist(),
                   'at_bound': bool(np.any(tau < 2.02) or np.any(tau > 99.9))}
            fits.append(fit)
            print(json.dumps(fit), flush=True)
    selected = min((f for f in fits if f['sensor_tau_ms']==3), key=lambda f:f['train_nrmse'])
    tau = np.array([selected['tau_ms'][k] for k in TRAINED_TYPES])
    gains = np.array(selected['gain'])
    fitted_prediction = circuit.simulate(tau, time_ms) * gains[:, None, None]
    # Load the reserved luminance condition only after choosing the candidate.
    test_time, test = load_split('test')
    np.testing.assert_array_equal(test_time, time_ms)
    comparisons = {
        'initial_train': metrics(initial_prediction, target, time_ms, scales),
        'fitted_train': metrics(fitted_prediction, target, time_ms, scales),
        'initial_test': metrics(initial_prediction, test, time_ms, scales),
        'fitted_test': metrics(fitted_prediction, test, time_ms, scales)}
    fine = TimingCircuit(weights, circuit.types, circuit.inputs, dt_ms=DT_MS/2)
    fine_prediction = fine.simulate(tau, time_ms) * gains[:, None, None]
    numerical_error = float(np.max(np.abs(fine_prediction-fitted_prediction) / scales[:, None, None]))
    disconnected = circuit.simulate(tau, time_ms, connected=False)
    if np.any(disconnected):
        raise ValueError('Disconnected downstream response was not silent')
    tau_table = {k: selected['tau_ms'].get(k, 20.) for k in system['populations']}
    tau_per_cell = np.array([tau_table[k] for k in circuit.types])
    np.savez_compressed(args.output / 'checkpoint.npz', root_ids=system['ids'], tau_ms=tau_per_cell)
    np.savez_compressed(args.output / 'traces.npz', time_ms=time_ms, train=target, test=test,
                        initial=initial_prediction, fitted=fitted_prediction, fine=fine_prediction)
    parameters = {'schema': 'conditional_measured_timing_v1', 'tau_ms_by_type': tau_table,
                  'sensor_tau_ms': 3., 'observation_gain_by_type': dict(zip(('L1','L2'), gains.tolist())),
                  'weights_path': str(PARENT / 'weights.npz'), 'weights_sha256': digest(PARENT / 'weights.npz', 'sha256'),
                  'intrinsic_membrane_times_identified': False, 'live_compatible': False}
    write_json(args.output / 'parameters.json', parameters)
    test_pass = comparisons['fitted_test']['nrmse'] <= .5 and all(r['first_peak_error_ms']<=1000/120+1e-9 for r in comparisons['fitted_test']['traces'])
    report = {'schema': 'measured_timing_training_v1', 'trained_on_measured_traces': True,
              'selected': selected, 'fits': fits, 'metrics': comparisons, 'scope': system['scope'],
              'observation_units': 'Baseline-centered negative ASAP2f dF/F; not calibrated millivolts',
              'sensor_sensitivity_tau_range_ms': {k:[min(f['tau_ms'][k] for f in fits),max(f['tau_ms'][k] for f in fits)] for k in TRAINED_TYPES},
              'max_normalized_dt_halving_difference': numerical_error, 'disconnected_silent': True,
              'parent_report_sha256': digest(PARENT / 'report.json', 'sha256'),
              'dataset_manifest_sha256': digest(ROOT / 'timing_manifest.json', 'sha256'),
              'source_sha256': {n:digest(Path(__file__).parent / n, 'sha256') for n in ['timing_sources.py','timing_training.py','graded_vision.py']},
              'output_sha256': {n:digest(args.output / n, 'sha256') for n in ['plan.json','parameters.json','checkpoint.npz','traces.npz']},
              'gates': {'heldout_temporal_threshold_passed': test_pass,
                        'numerical_convergence_passed': numerical_error < .05,
                        'intrinsic_times_identified': False, 'whole_brain_integrated': False, 'promoted_to_live': False},
              'limitations': ['Only R1-6/L1/L2 timing parameters fit; other types retain 20 ms',
                              'Sensor lag and upstream dynamics can trade off with fitted tau; no unique membrane-constant claim',
                              'Population means and a luminance holdout, not independent identified animals',
                              'Small-signal graph predicts sign-inverted flashes; measured biphasic/asymmetric responses may remain unexplained',
                              'Indicator scenarios are assumptions; no preparation-specific sensor calibration',
                              'Original visual graph omits substantial lamina/feedback input; geometry remains provisional',
                              'No Laya/Doom training or live checkpoint promotion in this experiment']}
    write_json(args.output / 'report.json', report)
    from flydoom.timing_report import render
    render(args.output)
    print(json.dumps({'report': str(args.output/'index.html'), 'selected_tau':selected['tau_ms'], 'gates':report['gates']}, indent=2))


if __name__ == '__main__':
    main()
