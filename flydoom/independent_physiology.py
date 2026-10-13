"""Read a pinned independent calcium dataset without executing author code.

The uniform-field probe and assumed calcium filter are descriptive diagnostics,
not a matched stimulus reconstruction or identification of membrane time constants.
"""
import argparse
import hashlib
import io
import pickle
from pathlib import Path

import numpy as np

from flydoom.data import digest
from flydoom.research_registry import atomic_json
from flydoom.retinal_policy import WINDOW_MS
from flydoom.self_learning import verified_pointer
from flydoom.stable_vision import load_any

REVISION = '77f16072de363dac2e079b07be93d2b08d4f8fb7'
ROOT = Path('data/raw/pirogova-timecourse')
FILES = {'L1': ('L1_FigS2.pkl', 'ca211eb40393865f6cf8dac0481ef8c5b3c6934a'),
         'Mi1': ('Mi1_Fig3.pkl', 'b27f6c78c35c7b1a0f548b86cc07db19d185d6bd'),
         'Tm3': ('Tm3_Fig3.pkl', 'afa24f913eb694a7c79665ceff9a99c5c1e223e0')}


class DataBox:
    """Inert storage for the old pandas object layout, not a pandas instance."""
    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls); obj.args = args; return obj
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        self.state = state


def index_box(cls, state):
    obj = DataBox(); obj.state = state; return obj


class RestrictedReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('__builtin__', 'slice'): return slice
        if (module, name) == ('numpy.core.multiarray', '_reconstruct'): return np._core.multiarray._reconstruct
        if module == 'numpy' and name in ('dtype', 'ndarray'): return getattr(np, name)
        if (module, name) == ('pandas.core.indexes.frozen', 'FrozenNDArray'): return np.ndarray
        if (module, name) == ('pandas.core.indexes.base', '_new_Index'): return index_box
        if (module, name) in {('pandas.core.frame', 'DataFrame'), ('pandas.core.indexes.base', 'Index'),
            ('pandas.core.indexes.multi', 'MultiIndex'), ('pandas.core.indexes.numeric', 'Float64Index'),
            ('pandas.core.indexes.numeric', 'Int64Index'), ('pandas.core.internals', 'BlockManager')}:
            return DataBox
        raise ValueError('Unsupported serialized global: '+module+'.'+name)


def axis_values(axis):
    state = axis.state
    return {name: np.asarray(level.state['data'])[np.asarray(labels, int)]
            for name, level, labels in zip(state['names'], state['levels'], state['labels'])}


def measured(cell):
    name, sha = FILES[cell]; body = (ROOT/name).read_bytes()
    if len(body) > 4_000_000 or hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest() != sha:
        raise ValueError('Independent physiology source identity changed')
    data = RestrictedReader(io.BytesIO(body), encoding='latin1').load()
    axes, blocks, items, _ = data.state['_data'].state
    row = axis_values(axes[1]); output = {}
    for condition in ('gray, 100%', 'mov_grat, 100%'):
        selected = np.flatnonzero((row['step_dts'] == 1) & (row['velocities'] == condition)
                                  & (row['time'] >= -.2) & (row['time'] <= 2))
        selected = selected[np.argsort(row['time'][selected])]
        time_s = np.asarray(row['time'][selected], float)
        if len(time_s) < 15 or np.any(np.diff(time_s) <= 0): raise ValueError('Invalid measurement time axis')
        baseline = (time_s >= -.1) & (time_s <= 0); pulse = (time_s >= 0) & (time_s <= 1)
        recordings = []; ids = []; rejected = 0
        for block_id, (block, columns) in enumerate(zip(blocks, items)):
            col = axis_values(columns)
            if block.shape != (len(col['cell']), len(row['time'])): raise ValueError('Unexpected source matrix shape')
            for cell_id in np.unique(col['cell']):
                trials = np.asarray(block[col['cell'] == cell_id][:, selected], float)
                counts = np.isfinite(trials).sum(axis=0)
                if not np.all(counts): rejected += 1; continue
                mean = np.nansum(trials, axis=0)/counts
                mean -= mean[baseline].mean()
                peak = mean[pulse].max()
                if peak <= 1e-8: rejected += 1; continue
                recordings.append(mean/peak); ids.append(f'block{block_id}:{cell_id}')
        if not recordings: raise ValueError('No valid independent recordings')
        recordings = np.array(recordings)
        output[condition] = {'time_s': time_s, 'normalized': recordings, 'recording_ids': np.array(ids),
                             'rejected_records': rejected}
    return output


def shape_metrics(time_s, observed, predicted):
    window = (time_s >= 0) & (time_s <= 1)
    indices = np.flatnonzero(window)
    error = float(np.sqrt(np.mean((observed[window]-predicted[window])**2)))
    actual_peak = float(time_s[indices[np.argmax(observed[window])]])
    prediction_peak = float(time_s[indices[np.argmax(predicted[window])]])
    return {'normalized_rmse_during_step': error, 'measured_peak_ms': actual_peak*1000,
            'predicted_peak_ms': prediction_peak*1000, 'peak_error_ms': abs(actual_peak-prediction_peak)*1000}


def native_probe(encoder):
    groups = {name: np.flatnonzero(encoder.types == name) for name in FILES}
    if any(not len(v) for v in groups.values()): raise ValueError('Required measured cell population absent')
    dt = WINDOW_MS/encoder.steps
    time_ms = np.arange(int(np.ceil(2000/dt))+1)*dt
    result = {}
    for polarity in (-1, 1):
        encoder.reset(); values = np.zeros((len(time_ms), len(groups)))
        for i in range(1, len(time_ms)):
            fraction = np.clip((1000-time_ms[i-1])/dt, 0, 1)
            encoder.advance(np.full(len(encoder.inputs), polarity*fraction, np.float32), dt)
            values[i] = [encoder.model.delta[g].mean() for g in groups.values()]
        # Author modelling uses a 200 ms calcium low-pass; it is not an in-vivo calibration.
        decay = np.exp(-dt/200); filtered = np.zeros_like(values)
        for i in range(1, len(values)):
            filtered[i] = decay*filtered[i-1]+(1-decay)*values[i]
        for j, cell in enumerate(groups):
            if polarity == (-1 if cell == 'L1' else 1):
                peak = filtered[time_ms <= 1000, j].max()
                result[cell] = filtered[:, j]/peak if peak > 1e-12 else np.zeros(len(time_ms))
    return time_ms/1000, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/independent-physiology-v1'))
    args = parser.parse_args(); root = args.output
    root.mkdir(exist_ok=False)
    parent = verified_pointer(Path('runs/self-learning/cursor.json'))
    e, _, _ = load_any(parent)
    atomic_json(root/'plan.json', {'source_revision': REVISION, 'fitting': False, 'checkpoint': str(parent),
        'protocol': 'One-second preferred-polarity uniform receptor step, inherited dynamics, assumed 200 ms calcium filter. Author experiment uses a 5-degree RF-centred spot, 30-degree annulus and surround stimuli; spatial protocol is not matched.',
        'observation': 'GCaMP6f fluorescence, independently normalized per recording after trial averaging; baseline -0.1 to 0 s. Not membrane voltage.',
        'independent_study': True, 'independent_animals_identifiable': False,
        'acceptance': 'No biological promotion without matched spatial protocol and calibrated observation mapping. Waveform errors are descriptive.'})
    probe_time, probe = native_probe(e)
    metrics = {}; arrays = {'probe_time_s': probe_time}
    for cell in FILES:
        metrics[cell] = {}; arrays[cell+'_probe'] = probe[cell]
        for name, d in measured(cell).items():
            time_s = d['time_s']; mean = d['normalized'].mean(axis=0)
            predicted = np.interp(time_s, probe_time, probe[cell], left=0)
            key = cell+('_gray' if name.startswith('gray') else '_moving')
            arrays[key+'_time_s'] = time_s; arrays[key+'_records'] = d['normalized']; arrays[key+'_ids'] = d['recording_ids']
            metrics[cell][name] = {'recordings': len(d['normalized']), 'excluded_records': d['rejected_records'],
                'sample_spacing_ms': float(np.median(np.diff(time_s))*1000),
                **shape_metrics(time_s, mean, predicted),
                'stimulus_matched': False, 'probe_has_surround_normalization': False}
            np.savetxt(root/(key+'.csv'), np.c_[time_s, mean, predicted], delimiter=',',
                       header='time_seconds,measured_normalized_mean,uniform_field_probe', comments='')
    np.savez_compressed(root/'traces.npz', **arrays)
    atomic_json(root/'report.json', {'schema': 'independent_calcium_audit_v1', 'status': 'completed',
        'checkpoint': str(parent), 'checkpoint_sha256': digest(parent/'report.json', 'sha256'),
        'paper': 'https://doi.org/10.1371/journal.pone.0285686',
        'repository': 'https://github.com/nopirogova/paper_signal_time-course', 'revision': REVISION,
        'data': {cell: {'file': name, 'git_blob': sha, 'sha256': digest(ROOT/name, 'sha256')} for cell, (name, sha) in FILES.items()},
        'metrics': metrics, 'biological_validation_passed': False, 'timing_deployed': False,
        'blocking_gaps': ['Spatial centre/surround protocol is not reconstructed in the game eye mapping.',
            'Calcium indicator and voltage-to-fluorescence mapping are not independently calibrated.',
            'Exported cell/trial labels do not identify animals for animal-level uncertainty.'],
        'scope': 'Independent measured time-course import and descriptive frozen-circuit probe. No parameter fitting, intrinsic membrane-time claim or live timing update.',
        'source_sha256': digest(Path(__file__), 'sha256'),
        'output_sha256': {p.name: digest(p, 'sha256') for p in root.iterdir() if p.is_file()}})
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
