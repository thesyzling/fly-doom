"""Compare trained-policy sampling with greedy play and disconnected visual input."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from flydoom.adaptive_policy import ACTIONS, SCHEMA, action, load_any
from flydoom.data import digest
from flydoom.laya_brain_training import open_game
from flydoom.policy_sampling import choose_action
from flydoom.recovery_training_v2 import comparison
from flydoom.research_cycle import frame_hash, summarize
from flydoom.research_registry import CycleLock, atomic_json
from flydoom.sensorimotor import save_candidate

ROOT = Path('runs/sampling-evaluation-v1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def evaluate(encoder, actor, openings, *, connected, label):
    results = []
    for opening in openings:
        game = open_game(opening)
        encoder.reset()
        actor.reset()
        rng = np.random.default_rng(opening['seed'])
        initial = game.get_total_reward()
        trace = []
        try:
            if frame_hash(game.get_state().screen_buffer) != opening['opening_sha256']:
                raise ValueError('Opening changed')
            for step in range(75):
                if game.is_episode_finished():
                    break
                frame = game.get_state().screen_buffer
                raw, _ = encoder.encode(frame, connected=connected)
                if not connected and np.any(raw):
                    raise ValueError('Disconnected input produced descending activity')
                choice, p, _ = choose_action(actor, raw, rng, sample=True)
                effect, _ = action(game, choice)
                trace.append({'step': step, 'action': ACTIONS[choice], 'probabilities': p.tolist(),
                              'frame_sha256': frame_hash(frame), 'reward': effect['reward_delta'], 'kills': effect['kill_delta']})
            result = {**opening, 'return': float(game.get_total_reward()-initial),
                      'kills': sum(row['kills'] for row in trace), 'decisions': len(trace), 'trace': trace,
                      'teacher_controls_action': False, 'labels_buffer_enabled': game.is_labels_buffer_enabled(),
                      'sampling_seed': opening['seed'], 'eye_connected': connected}
            results.append(result)
            print(f'{label} {len(results)}/{len(openings)}: kills={result["kills"]}, return={result["return"]:.2f}', flush=True)
        finally:
            game.close()
    return results


def main():
    plan = read(ROOT/'plan.json')
    parent = Path(plan['parent'])
    reference = Path(plan['paired_reference'])
    if plan['final_parent_results_seen'] or digest(reference/'plan.json', 'sha256') != plan['paired_plan_sha256']:
        raise ValueError('The comparison must be locked before final reference outcomes')
    encoder, actor, inherited = load_any(parent)
    sources = {**inherited['source_sha256'], **{name: digest(Path(__file__).with_name(name), 'sha256')
               for name in ('sampling_evaluation.py', 'policy_sampling.py', 'recovery_training_v2.py', 'balanced_distillation.py',
                            'reward_training_v3.py', 'reward_environment.py', 'recovery_training.py')}}
    with CycleLock(ROOT):
        def progress(message):
            print(message, flush=True)
            atomic_json(ROOT/'progress.json', {'status': 'running', 'message': message})
        try:
            dev = {'greedy_connected': read(reference/'development.json')['parent']}
            for name, connected in [('sampled_connected', True), ('sampled_disconnected', False)]:
                progress('Development '+name)
                dev[name] = evaluate(encoder, actor, plan['splits']['development'], connected=connected, label=name+' development')
                atomic_json(ROOT/'development.json', dev)
            ds = {name: summarize(rows) for name, rows in dev.items()}
            development_passed = ds['sampled_connected']['kills'] > max(ds['greedy_connected']['kills'], ds['sampled_disconnected']['kills'])
            atomic_json(ROOT/'development-summary.json', {'scores': ds, 'passed': development_passed})
            progress('Verify the shared frozen reference report before using its paired final results')
            while not (reference/'report.json').exists():
                state = read(reference.parent/'progress.json')
                if state.get('status') == 'failed':
                    raise RuntimeError('Reference experiment failed')
                time.sleep(2)
            load_any(reference)
            reference_report = read(reference/'report.json')
            tests = {'greedy_connected': read(reference/'test.json')['continued_parent']}
            for name, connected in [('sampled_connected', True), ('sampled_disconnected', False)]:
                progress('Locked final comparison '+name)
                tests[name] = evaluate(encoder, actor, plan['splits']['test'], connected=connected, label=name+' test')
                atomic_json(ROOT/'test.json', tests)
            scores = {name: summarize(rows) for name, rows in tests.items()}
            paired = comparison(tests['sampled_connected'], tests['greedy_connected'])
            accepted = bool(development_passed and paired['net_additional_hits'] >= 4 and paired['lost_parent_successes'] <= 2
                and paired['mean_return_change'] >= 0
                and scores['sampled_connected']['kills'] >= scores['sampled_disconnected']['kills']+4)
            evidence = {'method': 'Deploy the original categorical reward-trained policy instead of replacing it with greedy argmax.',
                'parent': str(parent), 'weights_changed': False, 'temperature': 1., 'decision_sampling': 'trained_softmax_v1',
                'development': ds, 'development_passed': development_passed, 'paired_final': paired,
                'acceptance_passed': accepted, 'acceptance_rule': plan['acceptance'],
                'scope': 'One predeclared action RNG stream per environment start. This tests deployment and visual-signal dependence, not new learning or biological topology superiority.'}
            atomic_json(ROOT/'sampling-evaluation.json', evidence)
            encoder.reset()
            save_candidate(ROOT, encoder, actor)
            for name in ('physiology.json', 'ablation.json'):
                (ROOT/name).write_bytes((parent/name).read_bytes())
            ablation = read(ROOT/'ablation.json')
            ablation['finding'] = 'Inherited parent-only teaching ablation. The current visual-disconnection game control is in the main results table.'
            atomic_json(ROOT/'ablation.json', ablation)
            report = {'schema': SCHEMA, 'status': 'completed', 'parent': str(parent),
                'parent_report_sha256': digest(parent/'report.json', 'sha256'),
                'neurons': len(encoder.ids), 'readout_cells': len(actor.root_ids), 'actions': ACTIONS, 'tasks': ['basic'],
                'laya_training': inherited['laya_training'], 'inherited_reward_training': inherited.get('reward_training'),
                'sampling_evaluation': evidence, 'decision_sampling': 'trained_softmax_v1',
                'plasticity': {'changed_edges': 0, 'changed_groups': 0, 'timing_trained': False},
                'four_steps': {'policy_sampling': 'completed', 'matched_game_test': 'completed',
                               'visual_disconnection': 'completed', 'independent_physiology': 'not_validated'},
                'test': scores, 'test_by_task': {name: {'basic': score} for name, score in scores.items()},
                'game_gate_passed': bool(accepted and scores['sampled_connected']['kill_episode_rate'] >= .8),
                'improvement_gate_passed': accepted, 'biological_validation_passed': False, 'promoted': False,
                'scope': evidence['scope'], 'source_sha256': sources,
                'shared_reference_report_sha256': digest(reference/'report.json', 'sha256'),
                'completed_at': datetime.now(timezone.utc).isoformat()}
            names = ['plan.json', 'development.json', 'development-summary.json', 'test.json', 'sampling-evaluation.json',
                     'weights.npz', 'circuit.npz', 'circuit.json', 'actor.npz', 'physiology.json', 'ablation.json']
            report['output_sha256'] = {name: digest(ROOT/name, 'sha256') for name in names}
            for name, sha in sources.items():
                if digest(Path(__file__).with_name(name), 'sha256') != sha:
                    raise ValueError('Evaluation implementation changed: '+name)
            atomic_json(ROOT/'report.json', report)
            load_any(ROOT)
            chosen = ROOT if accepted else parent
            atomic_json(ROOT/'selected.json', {'path': str(chosen.resolve()), 'sha256': digest(chosen/'report.json','sha256'),
                        'accepted': accepted, 'experiment': str(ROOT.resolve())})
            atomic_json(ROOT/'progress.json', {'status': 'completed', 'accepted': accepted})
            print(json.dumps({'test': scores, 'paired': paired, 'accepted': accepted}, indent=2), flush=True)
        except BaseException as error:
            atomic_json(ROOT/'progress.json', {'status': 'failed', 'error': str(error)})
            raise


if __name__ == '__main__':
    main()
