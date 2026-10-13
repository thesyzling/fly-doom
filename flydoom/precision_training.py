"""Evaluate cancellation-free neural transmission and offline Laya readout refits."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from flydoom.adaptive_policy import ACTIONS
from flydoom.balanced_distillation import fit_candidates
from flydoom.data import digest
from flydoom.laya_brain_fit import targets
from flydoom.recovery_training_v2 import comparison, replay
from flydoom.research_cycle import summarize
from flydoom.research_registry import CycleLock, atomic_json
from flydoom.reward_training_v3 import reserve
from flydoom.sampling_evaluation import evaluate
from flydoom.sensorimotor import save_candidate
from flydoom.stable_vision import MODE, SCHEMA, load_any, upgrade

PARENT = Path('runs/reward-student/cycle-20261011-050208-412071')
DATA = Path('runs/balanced-student/cycle-20261011-065923-532906')
ROOT = Path('runs/precision-student')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def select_candidate(results):
    """Select corrected arithmetic variants on development games only."""
    seeds = [row['seed'] for row in results['legacy_parent']]
    if any([r['seed'] for r in rows] != seeds for rows in results.values()):
        raise ValueError('Development conditions must share ordered starts')
    names = [name for name in results if name != 'legacy_parent']
    # Insertion order breaks exact ties in favor of the unchanged readout.
    selected = max(names, key=lambda name: (sum(r['kills'] > 0 for r in results[name]),
                                            sum(r['return'] for r in results[name])))
    return selected


def acceptance(results):
    paired = comparison(results['corrected_connected'], results['legacy_parent'])
    vision = comparison(results['corrected_connected'], results['corrected_disconnected'])
    reasons = []
    if paired['net_additional_hits'] < 4:
        reasons.append('Fewer than four additional hits versus the sampled parent')
    if paired['lost_parent_successes'] > 2:
        reasons.append('More than two lost parent successes')
    if paired['mean_return_change'] < 0:
        reasons.append('Mean return regressed versus the parent')
    if vision['net_additional_hits'] < 4:
        reasons.append('Fewer than four additional hits versus disconnected vision')
    return {'passed': not reasons, 'reasons': reasons, 'paired_parent': paired, 'paired_vision': vision}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-episodes', type=int, default=32)
    args = parser.parse_args(argv)
    if not 32 <= args.test_episodes <= 64:
        parser.error('Use 32-64 final episodes')
    encoder, parent, inherited = load_any(PARENT)
    load_any(DATA)
    rows = read(DATA/'experience.json')
    train = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    mask = np.array([r['png_sha256'] not in {t['png_sha256'] for t in train} for r in validation])
    with CycleLock(ROOT):
        folder = ROOT/datetime.now().strftime('cycle-%Y%m%d-%H%M%S-%f')
        folder.mkdir()
        def progress(message):
            print(message, flush=True)
            atomic_json(ROOT/'progress.json', {'status': 'running', 'folder': str(folder), 'message': message})
        try:
            sources = {**inherited['source_sha256'], **{name: digest(Path(__file__).with_name(name), 'sha256')
                for name in ('stable_vision.py', 'precision_training.py', 'balanced_distillation.py',
                             'recovery_training_v2.py', 'sampling_evaluation.py', 'policy_sampling.py',
                             'reward_training_v3.py', 'reward_environment.py', 'recovery_training.py')}}
            progress('Reserve fresh matched starts before fitting or examining outcomes')
            plan = reserve(8, args.test_episodes, training_steps=75, training_tics=300, progress=progress)
            plan['splits']['development'] = plan['splits'].pop('reward_train')
            # These eight starts remain reserved and unused; no extra acceptance tuning.
            plan['splits']['unused_gate'] = plan['splits'].pop('gate')
            plan.update(schema='stable_neural_precision_v1', parent=str(PARENT.resolve()),
                data=str(DATA.resolve()), neural_numerics=MODE,
                training='Re-encode all 833 TRAIN images through the corrected circuit. Fixed 80-epoch balanced soft Laya fitting with parent-policy rehearsal; fractions .25, .5, 1.0. Include unchanged decoder.',
                selection='Eight fresh development games choose the corrected variant by hits, then return; ties prefer unchanged decoder. Final games never select weights.',
                acceptance='At least four more hits versus sampled parent and versus disconnected candidate; at most two lost parent successes; nondecreasing mean return versus parent. No default registry promotion.',
                policy='Every condition samples its learned softmax at temperature 1 with RNG seeded from the same opening. No teacher or engine annotations at evaluation.',
                graph_weights_changed=False, response_times_changed=False,
                scope='Numerical correction plus optional readout refit, not new synaptic learning or biological validation.')
            atomic_json(folder/'plan.json', plan)
            atomic_json(folder/'experience.json', rows)
            atomic_json(folder/'numerical-audit.json', read('runs/neural-precision-audit-v1/report.json'))
            upgrade(encoder)
            progress('Replay verified TRAIN and validation images with cancellation-free release')
            raw, vraw = replay(encoder, train), replay(encoder, validation)
            np.savez_compressed(folder/'neural-replay.npz', train=raw, validation=vraw, validation_mask=mask)
            candidates, fit = fit_candidates(raw, train, parent, (vraw[mask], targets(validation)[mask]))
            candidates = {'stable_frozen': parent, **candidates}
            atomic_json(folder/'fit.json', fit)
            for name, actor in candidates.items():
                actor.save(folder/(name+'-actor.npz'))
            development = {}
            legacy_encoder, _, _ = load_any(PARENT)
            development['legacy_parent'] = evaluate(legacy_encoder, parent, plan['splits']['development'], connected=True, label='legacy development')
            for name, actor in candidates.items():
                progress('Corrected development gameplay: '+name)
                development[name] = evaluate(encoder, actor, plan['splits']['development'], connected=True, label=name+' development')
                atomic_json(folder/'development.json', development)
            selected = select_candidate(development)
            actor = candidates[selected]
            atomic_json(folder/'selection.json', {'selected': selected, 'selected_before_final_test': True,
                'development': {name: summarize(r) for name, r in development.items()}})
            tests = {}
            for name, e, a, connected in [('legacy_parent', legacy_encoder, parent, True),
                                          ('corrected_connected', encoder, actor, True),
                                          ('corrected_disconnected', encoder, actor, False)]:
                progress(f'Locked final test: {name}; selected={selected}')
                tests[name] = evaluate(e, a, plan['splits']['test'], connected=connected, label=name)
                atomic_json(folder/'test.json', tests)
            audit = acceptance(tests)
            scores = {name: summarize(r) for name, r in tests.items()}
            detail = {'selected': selected, 'train_frames': len(train), 'validation_frames': int(mask.sum()),
                'neural_numerics': MODE, 'decoder_change_l2': float(np.linalg.norm(actor.weights-parent.weights)),
                'graph_weights_changed': False, 'timing_changed': False, 'acceptance': audit,
                'acceptance_rule': plan['acceptance'], 'scope': plan['scope'],
                'development': {name: summarize(r) for name, r in development.items()}}
            atomic_json(folder/'precision-training.json', detail)
            save_candidate(folder, encoder, actor)
            for name in ('physiology.json', 'ablation.json'):
                (folder/name).write_bytes((PARENT/name).read_bytes())
            ablation = read(folder/'ablation.json')
            ablation['finding'] = 'Historical parent evidence only. Corrected arithmetic and visual disconnection are measured in the current game table.'
            atomic_json(folder/'ablation.json', ablation)
            report = {'schema': SCHEMA, 'status': 'completed', 'parent': str(PARENT.resolve()),
                'parent_report_sha256': digest(PARENT/'report.json', 'sha256'),
                'neurons': len(encoder.ids), 'readout_cells': len(actor.root_ids), 'actions': ACTIONS, 'tasks': ['basic'],
                'laya_training': inherited['laya_training'], 'inherited_reward_training': inherited.get('reward_training'),
                'neural_numerics': MODE, 'decision_sampling': 'trained_softmax_v1', 'precision_training': detail,
                'plasticity': {'changed_edges': 0, 'changed_groups': 0, 'timing_trained': False},
                'four_steps': {'numerical_reference': 'completed', 'corrected_teacher_replay': 'completed',
                               'matched_game_comparison': 'completed', 'independent_physiology': 'not_validated'},
                'test': scores, 'test_by_task': {name: {'basic': s} for name, s in scores.items()},
                'improvement_gate_passed': audit['passed'],
                'game_gate_passed': bool(audit['passed'] and scores['corrected_connected']['kill_episode_rate'] >= .8),
                'biological_validation_passed': False, 'promoted': False, 'source_sha256': sources,
                'scope': plan['scope'], 'input_sha256': {str((DATA/'report.json').resolve()): digest(DATA/'report.json', 'sha256')}}
            names = ['plan.json', 'experience.json', 'numerical-audit.json', 'neural-replay.npz', 'fit.json',
                'development.json', 'selection.json', 'test.json', 'precision-training.json', 'weights.npz',
                'circuit.npz', 'circuit.json', 'actor.npz', 'physiology.json', 'ablation.json']
            names += [name+'-actor.npz' for name in candidates]
            report['output_sha256'] = {name: digest(folder/name, 'sha256') for name in names}
            atomic_json(folder/'report.json', report)
            load_any(folder)
            pointer = lambda p: {'path': str(p.resolve()), 'sha256': digest(p/'report.json', 'sha256')}
            atomic_json(ROOT/'latest.json', pointer(folder))
            atomic_json(ROOT/'selected.json', {**pointer(folder if audit['passed'] else PARENT),
                                               'accepted': audit['passed'], 'experiment': str(folder.resolve())})
            atomic_json(ROOT/'progress.json', {'status': 'completed', 'folder': str(folder), 'accepted': audit['passed']})
            print(json.dumps({'folder': str(folder), 'selected': selected, 'test': scores, 'acceptance': audit}, indent=2), flush=True)
        except BaseException as error:
            atomic_json(ROOT/'progress.json', {'status': 'failed', 'folder': str(folder), 'error': str(error)})
            raise


if __name__ == '__main__':
    main()
