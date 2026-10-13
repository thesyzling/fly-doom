"""Finite reward-learning cycles with corrected neural replay and guarded selection.

The exploration cursor can continue a rejected candidate. The preferred model
changes only after independent matched gate and final comparisons pass.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from flydoom.adaptive_policy import ACTIONS, action
from flydoom.bridge_training import ordered
from flydoom.data import digest
from flydoom.laya_brain_fit import targets
from flydoom.precision_training import comparison
from flydoom.research_cycle import frame_hash, summarize
from flydoom.research_registry import CycleLock, atomic_json
from flydoom.reward_environment import open_game
from flydoom.reward_student_v2 import RewardStudent, discounted_returns
from flydoom.reward_training_v3 import reserve, restore_critic
from flydoom.sampling_evaluation import evaluate
from flydoom.sensorimotor import save_candidate
from flydoom.stable_bridge import StableBridge, cache_rehearsal, record_frame, verify_replay
from flydoom.stable_vision import MODE, SCHEMA, load_any

ROOT = Path('runs/self-learning')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pointer(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path/'report.json', 'sha256')}


def verified_pointer(path):
    item = read(path)
    folder = Path(item['path'])
    if digest(folder/'report.json', 'sha256') != item['sha256']:
        raise ValueError('Checkpoint pointer changed')
    load_any(folder)
    return folder


def assess(gate, final):
    """Predeclared game criteria; training-objective gains cannot promote a model."""
    reasons = []
    for split, results in [('gate', gate), ('test', final)]:
        candidate = results['candidate']
        for name, rows in results.items():
            comparison(candidate, rows)
        parent = comparison(candidate, results['incumbent'])
        decoder = comparison(candidate, results['decoder_only'])
        blind = comparison(candidate, results['disconnected'])
        if parent['mean_return_change'] < 0:
            reasons.append(split+': mean return below incumbent')
        if parent['net_additional_hits'] < (2 if split == 'test' else 0):
            reasons.append(split+': insufficient additional incumbent hits')
        if parent['lost_parent_successes'] > (2 if split == 'test' else 1):
            reasons.append(split+': too many lost incumbent successes')
        if decoder['net_additional_hits'] < 0:
            reasons.append(split+': fewer hits than decoder-only learning')
        if blind['net_additional_hits'] < (4 if split == 'test' else 1):
            reasons.append(split+': insufficient advantage over disconnection')
    return {'passed': not reasons, 'reasons': reasons,
            'paired_incumbent': comparison(final['candidate'], final['incumbent']),
            'paired_decoder': comparison(final['candidate'], final['decoder_only']),
            'paired_disconnected': comparison(final['candidate'], final['disconnected'])}


def publish(root, folder, incumbent, audit):
    """Keep exploration and preferred pointers distinct, including after rejection."""
    selected = folder if audit['passed'] else incumbent
    event = {'completed_at': datetime.now(timezone.utc).isoformat(), 'candidate': pointer(folder),
             'previous_selected': pointer(incumbent), 'selected': pointer(selected), 'acceptance': audit}
    history_path = root/'history.json'
    history = read(history_path) if history_path.exists() else []
    history.append(event)
    atomic_json(history_path, history)
    atomic_json(root/'latest.json', pointer(folder))
    atomic_json(root/'cursor.json', pointer(folder))
    atomic_json(root/'selected.json', {**pointer(selected), 'accepted': audit['passed'],
                                      'experiment': str(folder.resolve())})
    return Path(selected)


def teacher_source(parent, report):
    source = Path(report['reward_training']['teacher_records_from']) if report.get('reward_training', {}).get('version') == 4 else parent
    load_any(source)
    rows = ordered([r for r in read(source/'experience.json') if r.get('split') == 'train'])
    if not rows:
        raise ValueError('A verified Laya TRAIN source is required')
    # Keep complete causal episodes within a finite rehearsal budget.
    ends = np.flatnonzero(np.r_[[r['step'] == 0 for r in rows][1:], True])+1
    count = int(ends[np.searchsorted(ends, min(64, len(rows)))])
    return source, rows[:count]


def collect(learner, opening, path):
    b, a = learner.bridge, learner.actor
    b.apply(learner.gains)
    game = open_game(opening)
    rng = np.random.default_rng(opening['seed'])
    drive = np.empty((75, b.encoder.steps, 2, len(b.nodes)), np.float32)
    frames, raw, choices, probabilities, rewards, kills = [], [], [], [], [], []
    try:
        if game.is_labels_buffer_enabled() or frame_hash(game.get_state().screen_buffer) != opening['opening_sha256']:
            raise ValueError('Training observation protocol changed')
        for step in range(75):
            if game.is_episode_finished():
                break
            frame = game.get_state().screen_buffer.copy()
            r = record_frame(b, frame, drive[step])
            choice, p, _ = a.decide(r, rng)
            effect, _ = action(game, choice)
            frames.append(frame); raw.append(r); choices.append(choice); probabilities.append(p)
            rewards.append(effect['reward_delta']); kills.append(effect['kill_delta'])
        termination = 'target_eliminated' if sum(kills) else 'time_limit' if game.is_episode_finished() else 'finite_training_horizon'
    finally:
        game.close()
    raw = np.asarray(raw)
    resets = np.zeros(len(raw), bool); resets[0] = True
    drive = drive[:len(raw)]
    audit = verify_replay(b, drive, resets, learner.gains, raw, a)
    features = np.array([a.features(r) for r in raw])
    np.savez_compressed(path, frames=frames, raw=raw, actions=choices, probabilities=probabilities,
        rewards=rewards, kill_deltas=kills, returns=discounted_returns(rewards)/100,
        values=features@learner.critic, gains=learner.gains, weights=a.weights,
        seed=opening['seed'], termination=termination, decision_limit=75, timeout_tics=300)
    return {'raw': raw, 'drive': drive, 'resets': resets, 'actions': np.asarray(choices),
        'probabilities': np.asarray(probabilities), 'rewards': np.asarray(rewards), 'kills': sum(kills),
        'path': str(path.resolve()), 'sha256': digest(path, 'sha256'), 'seed': opening['seed'],
        'termination': termination, 'replay_max_error': audit['max_absolute_error'], 'replay_audit': audit}


def inherited_distillation(parent):
    seen = set()
    while parent not in seen:
        seen.add(parent)
        _, _, report = load_any(parent)
        if (parent/'distillation.json').exists():
            return parent, (parent/'distillation.json').read_bytes()
        next_parent = Path(report['parent'])
        if digest(next_parent/'report.json', 'sha256') != report['parent_report_sha256']:
            raise ValueError('Inherited teacher history identity changed')
        parent = next_parent
    raise ValueError('Cyclic teacher history')


def run_cycle(parent, incumbent, args, progress):
    parent_encoder, parent_actor, inherited = load_any(parent)
    if inherited.get('neural_numerics') != MODE:
        raise ValueError('Start from an explicitly corrected-arithmetic checkpoint')
    folder = args.output/('cycle-'+str(time.time_ns()))
    folder.mkdir()
    progress('Reserve fresh training, gate and final starts', folder)
    names = ('self_learning.py', 'stable_bridge.py', 'reward_student_v2.py', 'reward_training_v3.py',
             'reward_environment.py', 'sampling_evaluation.py', 'policy_sampling.py')
    sources = {**inherited['source_sha256'], **{n: digest(Path(__file__).with_name(n), 'sha256') for n in names}}
    source, rows = teacher_source(parent, inherited)
    plan = reserve(args.batches*args.episodes_per_batch, args.test_episodes, training_steps=75,
                   training_tics=300, progress=lambda m: progress(m, folder))
    plan.update(schema='stable_reward_cycle_v1', parent=str(parent.resolve()), incumbent=str(incumbent.resolve()),
        teacher_records_from=str(source.resolve()), batches=args.batches, episodes_per_batch=args.episodes_per_batch,
        selection='No epoch or seed tuning. Gate: no hit/return regression against incumbent, at most one lost hit, no fewer hits than decoder-only, at least one extra hit over disconnected. Final: at least two extra incumbent hits, at most two lost hits, nondecreasing return, no fewer hits than decoder-only, at least four extra hits over disconnected. Both required.',
        deployment='All conditions use the learned categorical distribution, temperature one, RNG seeded by the shared environment start.',
        continuation='Continue the research cursor even when rejected; change the preferred pointer only after both game checks pass. Every cycle reserves fresh evaluation starts.',
        trainable='Existing relay connection magnitudes by target type and signed channel, plus action readout. Anatomy, visual rows, signs and response times fixed.',
        source_sha256=sources, created_at=datetime.now(timezone.utc).isoformat())
    atomic_json(folder/'plan.json', plan)
    atomic_json(folder/'teacher-rehearsal.json', rows)
    visual = load_any('runs/retinal-control-v1')[0].ids
    progress(f'Rebuild {len(rows)} corrected teacher rehearsal frames', folder)
    bridge = StableBridge(parent_encoder, visual)
    drive, resets, cache_audit = cache_rehearsal(bridge, parent_actor, rows, folder/'rehearsal-drive.npy')
    cache_audit.update(parent_weights_sha256=digest(parent/'weights.npz', 'sha256'), teacher_source=str(source))
    atomic_json(folder/'rehearsal-audit.json', cache_audit)
    learners = {}
    for name in ('candidate', 'decoder_only'):
        e, a, _ = load_any(parent)
        learners[name] = RewardStudent(StableBridge(e, visual), a, drive, resets, targets(rows),
            train_relays=name == 'candidate', seed=131026, relay_directions=4)
    critic_restored = restore_critic(parent, inherited, learners)
    history = {name: [] for name in learners}; manifest = []
    for batch in range(args.batches):
        openings = plan['splits']['reward_train'][batch*args.episodes_per_batch:(batch+1)*args.episodes_per_batch]
        for name, learner in learners.items():
            episodes = []
            for opening in openings:
                progress(f'{name}: train batch {batch+1}/{args.batches}, episode {len(episodes)+1}/{len(openings)}', folder)
                episode = collect(learner, opening, folder/f'{name}-train-{opening["seed"]}.npz')
                episodes.append(episode)
                manifest.append({k: episode[k] for k in ('path', 'sha256', 'seed', 'kills', 'termination', 'replay_max_error', 'replay_audit')} | {'policy': name, 'split': 'reward_train'})
            progress(f'{name}: delayed-reward update {batch+1}/{args.batches}', folder)
            history[name].append(learner.update(episodes))
            np.savez_compressed(folder/f'{name}-batch-{batch+1:02d}.npz', weights=learner.actor.weights, gains=learner.gains, critic=learner.critic)
            atomic_json(folder/'learning.json', history); atomic_json(folder/'experience.json', manifest)
            print(json.dumps(history[name][-1]), flush=True)
            del episodes, episode
    incumbent_encoder, incumbent_actor, _ = load_any(incumbent)
    joint = learners['candidate']; decoder = learners['decoder_only']
    results = {}
    for split in ('gate', 'test'):
        results[split] = {}
        for name, e, a, connected in [('candidate', joint.bridge.encoder, joint.actor, True),
            ('decoder_only', decoder.bridge.encoder, decoder.actor, True),
            ('incumbent', incumbent_encoder, incumbent_actor, True),
            ('disconnected', joint.bridge.encoder, joint.actor, False)]:
            progress(f'Frozen {split}: {name}', folder)
            results[split][name] = evaluate(e, a, plan['splits'][split], connected=connected, label=f'{split} {name}')
            atomic_json(folder/(split+'.json'), results[split])
    audit = assess(results['gate'], results['test'])
    e, a = joint.bridge.encoder, joint.actor
    joint.bridge.apply(joint.gains)
    np.testing.assert_array_equal(e.model.tau, parent_encoder.model.tau)
    np.testing.assert_array_equal(e.model.weights.indices, parent_encoder.model.weights.indices)
    np.testing.assert_array_equal(e.model.weights.indptr, parent_encoder.model.weights.indptr)
    np.testing.assert_array_equal(np.sign(e.model.weights.data), np.sign(parent_encoder.model.weights.data))
    np.testing.assert_array_equal(e.model.weights[joint.bridge.visual].data, parent_encoder.model.weights[joint.bridge.visual].data)
    changed = int(np.count_nonzero(e.model.weights.data != joint.bridge.base_weights))
    save_candidate(folder, e, a)
    decoder.actor.save(folder/'decoder-only-actor.npz')
    for name in ('physiology.json',):
        (folder/name).write_bytes((parent/name).read_bytes())
    teacher_history, teaching_bytes = inherited_distillation(parent)
    (folder/'distillation.json').write_bytes(teaching_bytes)
    ablation = read(parent/'ablation.json')
    ablation['finding'] = 'Historical teacher evidence only. Current evidence is matched joint, decoder-only, incumbent and disconnected gameplay.'
    atomic_json(folder/'ablation.json', ablation)
    reward = {'version': 4, 'method': 'Corrected on-policy reward-to-go, learned value baseline, bounded relay finite-difference gradients and analytic readout gradients with Laya TRAIN rehearsal',
        'parent': str(parent.resolve()), 'batches': args.batches, 'training_episodes_per_policy': args.batches*args.episodes_per_batch,
        'continuation': 'Warm start from the research cursor; preferred model is independently guarded.',
        'teacher_records_from': str(source.resolve()), 'critic_restored': critic_restored,
        'training_decisions': 75, 'training_timeout_tics': 300, 'evaluation_decisions': 75, 'evaluation_timeout_tics': 300,
        'rehearsal_frames': len(rows), 'teacher_live_calls': 0, 'gamma': .99, 'reward_scale': .01,
        'relay_active_groups': int(joint.active.sum()), 'changed_edges': changed,
        'decoder_change_l2': float(np.linalg.norm(a.weights-parent_actor.weights)),
        'accepted_batches': sum(r['accepted'] for r in history['candidate']), 'history': history,
        'acceptance': audit, 'acceptance_rule': plan['selection'], 'frozen_game_tests': True,
        'retained_timing': True, 'scope': 'Finite offline reward learning on basic. Shared edge gains learn, not each synapse independently. No live teacher, online Run updates or biological dopamine claim.'}
    atomic_json(folder/'reward-training.json', reward)
    scores = {name: summarize(v) for name, v in results['test'].items()}
    report = {'schema': SCHEMA, 'status': 'completed', 'parent': str(parent.resolve()),
        'parent_report_sha256': digest(parent/'report.json', 'sha256'), 'neurons': len(e.ids), 'readout_cells': len(a.root_ids),
        'actions': ACTIONS, 'tasks': ['basic'], 'laya_training': inherited['laya_training'], 'reward_training': reward,
        'neural_numerics': MODE, 'decision_sampling': 'trained_softmax_v1', 'distillation_inherited_from': str(teacher_history),
        'four_steps': {'stable_reward_replay': 'completed', 'joint_reward_training': 'completed',
                       'independent_game_checks': 'completed', 'guarded_selection': 'passed' if audit['passed'] else 'parent_retained'},
        'test': scores, 'test_by_task': {n: {'basic': s} for n, s in scores.items()},
        'game_gate_passed': bool(audit['passed'] and scores['candidate']['kill_episode_rate'] >= .8),
        'improvement_gate_passed': audit['passed'], 'biological_validation_passed': False, 'promoted': False,
        'source_sha256': sources, 'scope': reward['scope'],
        'input_sha256': {str(p.resolve()): digest(p, 'sha256') for p in (parent/'report.json', incumbent/'report.json', source/'report.json')}}
    artifacts = [p for p in folder.iterdir() if p.is_file()]
    report['output_sha256'] = {p.name: digest(p, 'sha256') for p in artifacts}
    atomic_json(folder/'report.json', report)
    load_any(folder)
    return folder, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--batches', type=int, default=2)
    parser.add_argument('--episodes-per-batch', type=int, default=2)
    parser.add_argument('--test-episodes', type=int, default=32)
    args = parser.parse_args(argv)
    if not 1 <= args.cycles <= 5 or not 1 <= args.batches <= 8 or not 2 <= args.episodes_per_batch <= 4 or not 32 <= args.test_episodes <= 64:
        parser.error('Use 1-5 cycles, 1-8 batches, 2-4 episodes per batch and 32-64 final games')
    with CycleLock(args.output):
        parent = args.parent or verified_pointer(args.output/'cursor.json' if (args.output/'cursor.json').exists() else Path('runs/precision-student/latest.json'))
        incumbent = verified_pointer(args.output/'selected.json') if (args.output/'selected.json').exists() else parent
        load_any(parent); load_any(incumbent)
        def progress(message, folder):
            atomic_json(args.output/'progress.json', {'status': 'running', 'folder': str(folder), 'message': message})
            print(message, flush=True)
        try:
            for cycle in range(args.cycles):
                print(f'Autonomous cycle {cycle+1}/{args.cycles}; cursor={parent}; preferred={incumbent}', flush=True)
                folder, audit = run_cycle(parent, incumbent, args, progress)
                incumbent = publish(args.output, folder, incumbent, audit)
                parent = folder
                print(json.dumps({'cycle': cycle+1, 'acceptance': audit, 'preferred': str(incumbent)}, indent=2), flush=True)
            atomic_json(args.output/'progress.json', {'status': 'completed', 'cycles': args.cycles,
                'folder': str(parent), 'selected': str(incumbent), 'message': 'Finite budget completed; Run remains frozen.'})
        except BaseException as error:
            atomic_json(args.output/'progress.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                'error': str(error), 'cursor': str(parent), 'selected': str(incumbent),
                'message': 'Completed cycles remain usable; partial cycle is not published.'})
            raise


if __name__ == '__main__':
    main()
