"""Compare provisional retinal input through graded and LIF visual subcircuits.

No parameters are trained. Both variants retain the same induced structural
graph, with explicitly different gains, units and tonic-state assumptions.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from flydoom.data import digest
from flydoom.eye_mapping import write_json
from flydoom.graded_vision import GradedVision, type_scaled_weights
from flydoom.retinal_mapping import RESULT, load_counts
from flydoom.simulation import LIFNetwork, LIFParameters, signed_weights, transmitter_signs
from flydoom.visual_probe import stimulus

TYPES = ('R1-6', 'L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Mi4', 'Mi9',
         'Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9', 'T4a', 'T4b', 'T4c', 'T4d',
         'T5a', 'T5b', 'T5c', 'T5d')
CONDITIONS = ('no_input', 'retina_bright', 'retina_dark', 'retina_disconnected',
              'retina_bright_L1_block', 'retina_x_positive', 'retina_x_negative',
              'retina_z_positive', 'retina_z_negative')


def drive_profile(condition, time_ms, directions):
    if condition not in CONDITIONS:
        raise ValueError('Unknown retinal condition')
    if condition == 'no_input':
        return stimulus('no_input', time_ms, directions)
    if condition in CONDITIONS[1:5]:
        return stimulus('Mi1_flash', time_ms, directions) * (-1 if condition == 'retina_dark' else 1)
    return stimulus(condition.replace('retina_', 'Mi1_'), time_ms, directions)


def prepare(mapping):
    report = json.loads((mapping / 'report.json').read_text())
    for name, sha in report['output_sha256'].items():
        if digest(mapping / name, 'sha256') != sha:
            raise ValueError('Retinal mapping checksum mismatch')
    receptors = json.loads((mapping / 'receptors.json').read_text())
    columns = json.loads((mapping / 'columns.json').read_text())
    if not receptors or len({r['root_id'] for r in receptors}) != len(receptors):
        raise ValueError('Need unique provisional receptor assignments')
    sides = {r['side'] for r in receptors}
    if len(sides) != 1:
        raise ValueError('Expected a single anatomical hemisphere')
    side = sides.pop()
    ids, rows, full, provenance = load_counts()
    chosen = np.array([i for i, r in enumerate(rows) if r['side'] == side and r['cell_type'] in TYPES], dtype=int)
    counts = full[chosen][:, chosen].tocsr()
    retained = int(counts.sum())
    total_input = int(full[chosen].sum())
    rows = [rows[i] for i in chosen]
    ids = ids[chosen]
    lookup = {str(root): i for i, root in enumerate(ids)}
    selected = np.array([lookup[r['root_id']] for r in receptors], dtype=int)
    directions = np.array([r['optical_direction'] for r in receptors])
    types = np.array([r['cell_type'] for r in rows])
    signs, rules = transmitter_signs(rows)
    weights, gains = type_scaled_weights(counts, signs, types)
    # Initial cell-type parameter table; equal starting values are not fits.
    tau_by_type = {kind: 20. for kind in sorted(set(types))}
    tau = np.array([tau_by_type[k] for k in types])
    model = GradedVision(weights, tau, dt_ms=.5)
    populations = {kind: np.flatnonzero(types == kind) for kind in TYPES if np.any(types == kind)}
    return dict(ids=ids, rows=rows, counts=counts, signs=signs, model=model, selected=selected,
                directions=directions, populations=populations, receptors=receptors, columns=columns,
                provenance=provenance, gains=gains, tau_by_type=tau_by_type, sign_rules=rules,
                scope={'side': side, 'neurons': len(ids), 'structural_edges': counts.nnz,
                       'retained_incoming_contacts': retained, 'all_incoming_contacts': total_input,
                       'omitted_incoming_contact_fraction': 1 - retained / total_input})


def run(system, condition):
    graded = system['model']
    graded.reset()
    graded.blocked_release[:] = False
    if condition == 'retina_bright_L1_block':
        graded.blocked_release[system['populations']['L1']] = True
    params = LIFParameters(dt_ms=.5, mv_per_contact=.03)
    weights = signed_weights(system['counts'], system['signs'], params.mv_per_contact)
    if condition == 'retina_bright_L1_block':
        weights.data[np.isin(weights.indices, system['populations']['L1'])] = 0
    lif = LIFNetwork(weights, params)
    counts = np.zeros(len(system['ids']), dtype=np.int64)
    traces, graded_states, lif_states = [], [], []
    connected = condition != 'retina_disconnected'
    for tick in range(params.steps(200)):
        profile = drive_profile(condition, tick * params.dt_ms, system['directions'])
        drive = np.zeros(len(system['ids']), dtype=np.float32)
        drive[system['selected']] = profile
        graded.step(.5 * drive, connected=connected)
        counts += lif.step(20 * drive, connected=connected)
        if np.max(np.abs(graded.delta)) > 2 or lif.voltage.min() < -250:
            raise FloatingPointError('Experiment exceeded its predeclared activity bounds')
        if (tick + 1) % params.steps(10) == 0:
            traces.append({'time_ms': (tick + 1) * params.dt_ms,
                           'populations': {k: {'graded_delta': float(graded.delta[idx].mean(dtype=np.float64)),
                                              'lif_delta_mv': float((lif.voltage[idx] - params.rest_mv).mean(dtype=np.float64))}
                                           for k, idx in system['populations'].items()}})
            graded_states.append(graded.delta.copy())
            lif_states.append(lif.voltage.copy() - params.rest_mv)
    graded_states, lif_states = np.stack(graded_states), np.stack(lif_states)
    metrics = {k: {'graded_peak_abs_mean_delta': float(np.max(np.abs(graded_states[:, idx].mean(axis=1, dtype=np.float64)))),
                   'lif_peak_abs_mean_delta_mv': float(np.max(np.abs(lif_states[:, idx].mean(axis=1, dtype=np.float64)))),
                   'lif_spikes': int(counts[idx].sum())} for k, idx in system['populations'].items()}
    return {'condition': condition, 'completed': True, 'trace': traces, 'metrics': metrics}, graded_states, lif_states, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, default=RESULT)
    parser.add_argument('--output', type=Path, default=Path('runs/graded-retina-v1'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {'schema': 'graded_retina_plan_v1', 'conditions': list(CONDITIONS), 'duration_ms': 200,
            'stimulus_interval_ms': [40, 160], 'dt_ms': .5, 'graded_input_amplitude': .5,
            'lif_input_amplitude_mv': 20, 'type_tau_ms_initial': 20, 'row_contraction_bound': .6,
            'graded_activity_limit': 2, 'lif_voltage_floor_mv': -250,
            'training': False, 'physiological_fit': False,
            'comparison': 'Same induced connectivity; different dynamics, scaling, state units and tonic baseline'}
    write_json(args.output / 'plan.json', plan)
    system = prepare(args.mapping)
    print(json.dumps(system['scope']), flush=True)
    results, output_hashes, recordings = [], {}, {}
    for condition in CONDITIONS:
        print('Running ' + condition, flush=True)
        result, g, v, spikes = run(system, condition)
        filename = condition + '.npz'
        np.savez_compressed(args.output / filename, root_ids=system['ids'], graded_delta=g,
                            lif_delta_mv=v, lif_spike_counts=spikes,
                            input_root_ids=system['ids'][system['selected']])
        output_hashes[filename] = digest(args.output / filename, 'sha256')
        results.append(result)
        recordings[condition] = g
    sparse.save_npz(args.output / 'graded_weights.npz', system['model'].weights)
    output_hashes['graded_weights.npz'] = digest(args.output / 'graded_weights.npz', 'sha256')
    parameters = {'target_type_initial_gain': system['gains'], 'tau_ms_by_type': system['tau_by_type'],
                  'tonic_bias': 1., 'units': 'Dimensionless state and graded release; not firing rate or millivolts',
                  'sign_rules': system['sign_rules'], 'lif_parameters': asdict(LIFParameters(dt_ms=.5, mv_per_contact=.03))}
    write_json(args.output / 'parameters.json', parameters)
    output_hashes['parameters.json'] = digest(args.output / 'parameters.json', 'sha256')
    by_name = {r['condition']: r for r in results}
    mi1 = by_name['retina_bright']['metrics']['Mi1']['graded_peak_abs_mean_delta']
    blocked = by_name['retina_bright_L1_block']['metrics']['Mi1']['graded_peak_abs_mean_delta']
    report = {'schema': 'graded_retina_experiment_v1', 'trained': False, 'live_policy_changed': False,
              'scope': system['scope'], 'mapping_report_sha256': digest(args.mapping / 'report.json', 'sha256'),
              'provenance': system['provenance'], 'plan_sha256': digest(args.output / 'plan.json', 'sha256'),
              'parameters': parameters, 'tonic_baseline_residual': system['model'].baseline_residual,
              'effective_row_bound': system['model'].row_bound,
              'source_sha256': {n: digest(Path(__file__).parent / n, 'sha256') for n in
                                ['graded_vision.py', 'retinal_experiment.py', 'retinal_mapping.py', 'simulation.py', 'visual_probe.py']},
              'output_sha256': output_hashes, 'conditions': results,
              'population_sizes': {k: len(idx) for k, idx in system['populations'].items()},
              'Mi1_bright_peak_attenuation_after_L1_evoked_block': (1 - blocked / mi1) if mi1 > 0 else None,
              'gates': {'bounded_experiment_completed': True, 'graded_dynamics_implemented': True,
                        'retinal_identity_verified': False, 'physiology_validated': False,
                        'whole_brain_integrated': False, 'ready_for_task_training': False},
              'limitations': ['Induced visual subcircuit only; no descending cells, Laya training or motor output',
                              'R1-6 optical assignments remain provisional; incomplete retinal coverage',
                              'Shared type time constants start at 20 ms; no physiological parameter fitting',
                              'Contractive type gains and tonic bias are engineering assumptions',
                              'L1 block removes evoked release while preserving tonic baseline in the graded model',
                              'LIF/graded comparisons differ in scales and baseline; no quantitative biological superiority claim',
                              'Sign rules omit receptor-specific and cotransmitter effects; CT1 compartmentalization is absent',
                              'Motion inputs use author direction cosines, not calibrated retinal movies']}
    write_json(args.output / 'report.json', report)
    from flydoom.retinal_report import render
    render(args.output, args.mapping, system, recordings)
    print(json.dumps({'report': str(args.output / 'index.html'), 'Mi1_peak': mi1,
                      'Mi1_peak_L1_block': blocked, 'gates': report['gates']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
