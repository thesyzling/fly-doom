"""Reconstruct retinal paths and independently verify recorded graded metrics."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from flydoom.data import digest
from flydoom.eye_mapping import OUTPUT, write_json
from flydoom.retinal_mapping import RESULT, infer_paths, load_counts
from flydoom.retinal_experiment import CONDITIONS, TYPES
from flydoom.simulation import transmitter_signs


def audit(output, mapping):
    report = json.loads((output / 'report.json').read_text())
    mapped = json.loads((mapping / 'report.json').read_text())
    for folder, value in [(output, report), (mapping, mapped)]:
        for name, expected in value['output_sha256'].items():
            if digest(folder / name, 'sha256') != expected:
                raise ValueError(f'Artifact changed: {folder / name}')
        for name, expected in value['source_sha256'].items():
            if digest(Path(__file__).parent / name, 'sha256') != expected:
                raise ValueError(f'Source changed: {name}')
    if digest(output / 'plan.json', 'sha256') != report['plan_sha256']:
        raise ValueError('Experiment plan changed')
    if digest(mapping / 'report.json', 'sha256') != report['mapping_report_sha256']:
        raise ValueError('Mapping report changed')
    if digest(OUTPUT / 'report.json', 'sha256') != mapped['prior_report_sha256']:
        raise ValueError('Prior mapping changed')
    checks = json.loads((mapping / 'heldout.json').read_text())
    prior = json.loads((OUTPUT / 'columns.json').read_text())
    accepted = []
    for original, checked in zip(prior, checks, strict=True):
        expected = (original['geometry_supported'] and checked['geometry_supported']
                    and original['candidate_root_id'] == checked['candidate_root_id'])
        if bool(expected) != checked['heldout_supported']:
            raise ValueError('Inconsistent held-out support')
        if expected:
            accepted.append(original)
    ids, rows, counts, provenance = load_counts()
    columns, receptors = infer_paths(ids, rows, counts, accepted)
    if columns != json.loads((mapping / 'columns.json').read_text()) or receptors != json.loads((mapping / 'receptors.json').read_text()):
        raise ValueError('Retinal paths do not reproduce from actual graph counts')
    if provenance != report['provenance'] or provenance != mapped['provenance']:
        raise ValueError('Prepared graph provenance changed')
    side = report['scope']['side']
    selected = np.array([i for i, r in enumerate(rows) if r['side'] == side and r['cell_type'] in TYPES])
    graph = counts[selected][:, selected].astype(np.float32).tocsr()
    rows = [rows[i] for i in selected]
    ids = ids[selected]
    signs, _ = transmitter_signs(rows)
    recorded_weights = sparse.load_npz(output / 'graded_weights.npz')
    gains = report['parameters']['target_type_initial_gain']
    graph.data *= signs[graph.indices]
    graph.data *= np.repeat(np.array([gains[r['cell_type']] for r in rows], dtype=np.float32), np.diff(graph.indptr))
    for name in ['data', 'indices', 'indptr']:
        np.testing.assert_array_equal(getattr(graph, name), getattr(recorded_weights, name))
    indices = {k: np.array([i for i, r in enumerate(rows) if r['cell_type'] == k]) for k in report['population_sizes']}
    input_ids = [r['root_id'] for r in receptors]
    results = {}
    if tuple(r['condition'] for r in report['conditions']) != CONDITIONS:
        raise ValueError('Incomplete or reordered condition set')
    for condition in report['conditions']:
        name = condition['condition']
        with np.load(output / (name + '.npz'), allow_pickle=False) as data:
            np.testing.assert_array_equal(data['root_ids'], ids)
            if list(map(str, data['input_root_ids'])) != input_ids:
                raise ValueError('Input identities changed')
            g, v, s = data['graded_delta'], data['lif_delta_mv'], data['lif_spike_counts']
            if g.shape != (20, len(ids)) or v.shape != g.shape or not np.isfinite(g).all() or not np.isfinite(v).all():
                raise ValueError('Invalid recording shape or state')
            if [t['time_ms'] for t in condition['trace']] != list(range(10, 201, 10)):
                raise ValueError('Recorded sample times changed')
            for pop, idx in indices.items():
                mean_g, mean_v = g[:, idx].mean(axis=1, dtype=np.float64), v[:, idx].mean(axis=1, dtype=np.float64)
                m = condition['metrics'][pop]
                np.testing.assert_allclose([np.abs(mean_g).max(), np.abs(mean_v).max()],
                                           [m['graded_peak_abs_mean_delta'], m['lif_peak_abs_mean_delta_mv']], rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(mean_g, [t['populations'][pop]['graded_delta'] for t in condition['trace']], rtol=1e-5, atol=1e-7)
                if int(s[idx].sum()) != m['lif_spikes']:
                    raise ValueError('LIF spike count mismatch')
            direct = np.isin(ids, np.array([int(n) for n in input_ids], dtype=np.uint64))
            if name == 'no_input' and (g.any() or v.any() or s.any()):
                raise ValueError('No-input baseline drift')
            if name == 'retina_disconnected' and (g[:, ~direct].any() or v[:, ~direct].any() or s[~direct].any()):
                raise ValueError('Disconnected transmission')
            results[name] = float(np.abs(g[:, indices['Mi1']].mean(axis=1, dtype=np.float64)).max())
    attenuation = 1 - results['retina_bright_L1_block'] / results['retina_bright']
    np.testing.assert_allclose(attenuation, report['Mi1_bright_peak_attenuation_after_L1_evoked_block'])
    result = {'schema': 'retinal_audit_v1', 'passed': True, 'conditions': len(results),
              'reconstructed_receptor_paths': len(receptors), 'heldout_supported': len(accepted),
              'reconstructed_weight_entries': graph.nnz, 'Mi1_peak_attenuation': attenuation,
              'physiology_validated': False, 'whole_brain_integrated': False,
              'scope': 'Artifact/source identities, anatomical path reconstruction, full weight reconstruction, recorded metrics and controls'}
    per_column = Counter(r['column_index'] for r in receptors)
    result['receptors_per_column_histogram'] = dict(sorted(Counter(per_column.values()).items()))
    result['columns_with_more_than_six_candidates_require_review'] = sum(n > 6 for n in per_column.values())
    result['coverage_note'] = 'Dominant paths do not establish complete or canonical cartridge membership; incomplete and excess candidate counts need anatomical review.'
    write_json(output / 'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/graded-retina-v1'))
    parser.add_argument('--mapping', type=Path, default=RESULT)
    args = parser.parse_args()
    print(json.dumps(audit(args.output, args.mapping), indent=2))
