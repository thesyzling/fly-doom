"""Continue learned brain weights with longer training and unchanged game tests."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import secrets

import numpy as np

from flydoom.adaptive_policy import ACTIONS, SCHEMA, load_any, action
from flydoom.bridge_training import Bridge, ordered, cache_drive
from flydoom.data import digest
from flydoom.laya_brain_fit import targets
from flydoom.laya_brain_training import evaluate
from flydoom.reward_environment import open_game
from flydoom.research_cycle import frame_hash, summarize
from flydoom.research_registry import CycleLock, atomic_json
from flydoom.retinal_policy import WINDOW_MS
from flydoom.reward_student_v2 import RewardStudent, discounted_returns
from flydoom.sensorimotor import save_candidate
from flydoom.vision_validation import collect_seeds
from flydoom.visual_observer import sample_frame

ROOT = Path('runs/reward-student')


def reserve(training_episodes, test_episodes, *, training_steps=150, training_tics=600, progress=print):
    used, hashes = set(), set()
    for path in Path('runs').rglob('plan.json'):
        plan = json.loads(path.read_text(encoding='utf-8'))
        used.update(collect_seeds(plan))
        for rows in plan.get('splits', {}).values():
            if isinstance(rows, list):
                hashes.update(r['opening_sha256'] for r in rows if isinstance(r, dict) and 'opening_sha256' in r)
    splits = {}
    for split, count in [('test', test_episodes), ('gate', 8), ('reward_train', training_episodes)]:
        splits[split] = []
        for index in range(count):
            progress(f'Reserve {split} opening {index+1}/{count}')
            for attempt in range(10000):
                seed = secrets.randbelow(2**31-1)
                if seed in used:
                    continue
                row = {'seed': seed, 'task': 'basic', 'timeout_tics': training_tics if split == 'reward_train' else 300}
                game = open_game(row)
                try:
                    sha = frame_hash(game.get_state().screen_buffer)
                finally:
                    game.close()
                if sha not in hashes:
                    break
                if (attempt+1) % 20 == 0:
                    progress(f'Reserve {split} opening {index+1}/{count}: rejected {attempt+1} repeated starts')
            else:
                raise ValueError('Could not reserve a fresh opening')
            used.add(seed)
            hashes.add(sha)
            splits[split].append({**row, 'opening_sha256': sha})
    return {'schema': 'laya_then_reward_v1', 'splits': splits, 'actions': ACTIONS,
            'scenario': 'basic', 'training_steps': training_steps, 'training_timeout_tics': training_tics,
            'evaluation_steps': 75, 'evaluation_timeout_tics': 300,
            'warmup': 'Six seeded strafe actions; warmup rewards excluded',
            'reward': 'Unmodified ViZDoom scenario reward, divided by 100 during optimization only',
            'selection': 'Fixed batch budget; no evaluation-based tuning. Gate >=80% hit episodes and no kill/return regression vs parent and decoder-only RL. Final tests never select weights.',
            'terminal': 'Target hit, engine timeout or the declared decision horizon ends an episode; training proceeds to the next episode. No reward credit crosses resets.',
            'teacher': 'No live teacher inference. Only retained TRAIN probabilities are rehearsed.',
            'policy_inputs': 'Eye-driven descending neuron states only; no engine annotations, positions or rewards'}


def collect(learner, opening, path, max_decisions=150):
    """Record causal visual drive during the actual full neural simulation."""
    b, actor = learner.bridge, learner.actor
    e = b.encoder
    b.apply(learner.gains)
    game = open_game(opening)
    rng = np.random.default_rng(opening['seed'])
    drive = np.empty((max_decisions, e.steps, 2, len(b.nodes)), np.float32)
    frames, raw, actions, probabilities, rewards, kills = [], [], [], [], [], []
    try:
        if game.is_labels_buffer_enabled():
            raise ValueError('Training policy must use pixels only')
        if frame_hash(game.get_state().screen_buffer) != opening['opening_sha256']:
            raise ValueError('Reserved opening changed')
        for step in range(max_decisions):
            if game.is_episode_finished():
                break
            frame = game.get_state().screen_buffer.copy()
            contrast = sample_frame(frame, e.uv, e.visible)
            for tick in range(e.steps):
                release = np.maximum(e.model.baseline+e.model.delta, 0)-np.maximum(e.model.baseline, 0)
                drive[step, tick, 0] = b.positive @ release[b.visual]
                drive[step, tick, 1] = b.negative @ release[b.visual]
                e.advance(contrast, WINDOW_MS/e.steps)
            r = e.model.delta[e.outputs].copy()
            choice, p, _ = actor.decide(r, rng)
            effect, _ = action(game, choice)
            frames.append(frame)
            raw.append(r)
            actions.append(choice)
            probabilities.append(p)
            rewards.append(effect['reward_delta'])
            kills.append(effect['kill_delta'])
        termination = ('target_eliminated' if sum(kills) else
                       'time_limit' if game.is_episode_finished() and game.get_episode_time() >= game.get_episode_timeout() else
                       'engine_finished' if game.is_episode_finished() else 'finite_training_horizon')
    finally:
        game.close()
    resets = np.zeros(len(raw), bool)
    resets[0] = True
    raw = np.asarray(raw)
    drive = drive[:len(raw)]
    replay = b.replay(drive, resets, learner.gains)
    if not np.allclose(replay, raw, rtol=2e-3, atol=2e-7):
        raise ValueError('Cached relay trajectory differs from deployed neural computation')
    x = np.array([actor.features(r) for r in raw])
    np.savez_compressed(path, frames=np.asarray(frames), raw=raw, actions=actions, probabilities=probabilities,
                        rewards=rewards, kill_deltas=kills, returns=discounted_returns(rewards)/100,
                        values=x @ learner.critic, gains=learner.gains, weights=actor.weights,
                        seed=opening['seed'], termination=termination, decision_limit=max_decisions,
                        timeout_tics=opening.get('timeout_tics', 300))
    return {'raw': raw, 'drive': drive, 'resets': resets, 'actions': np.asarray(actions),
            'probabilities': np.asarray(probabilities), 'rewards': np.asarray(rewards), 'kills': sum(kills),
            'path': str(path), 'sha256': digest(path, 'sha256'), 'seed': opening['seed'],
            'replay_max_error': float(np.max(abs(replay-raw))), 'termination': termination}


def teaching_checkpoint(parent):
    """Find verified original teacher records without relabelling RL frames."""
    seen = set()
    parent = Path(parent).resolve()
    while True:
        if parent in seen:
            raise ValueError('Checkpoint ancestry contains a cycle')
        seen.add(parent)
        review = parent/'review.json'
        if review.exists() and json.loads(review.read_text()).get('eligible_for_use') is False:
            raise ValueError('Cannot continue a superseded checkpoint')
        _, _, report = load_any(parent)
        if not report.get('laya_training'):
            raise ValueError('Require a Laya-taught ancestor')
        if not report.get('reward_training'):
            return parent
        parent = Path(report['reward_training']['parent']).resolve()


def restore_critic(parent, report, learners):
    """Warm-start the common value baseline; this is a fresh experiment, not exact process resume."""
    reward = report.get('reward_training')
    if not reward:
        return False
    name = f'candidate-batch-{reward["batches"]:02d}.npz'
    if digest(parent/name, 'sha256') != report['output_sha256'][name]:
        raise ValueError('Previous optimizer snapshot changed')
    with np.load(parent/name, allow_pickle=False) as saved:
        critic = saved['critic'].copy()
    for learner in learners.values():
        if critic.shape != learner.critic.shape or not np.isfinite(critic).all():
            raise ValueError('Previous value baseline is incompatible')
        learner.critic = critic.copy()
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--episodes-per-batch', type=int, default=4)
    parser.add_argument('--test-episodes', type=int, default=24)
    parser.add_argument('--training-decisions', type=int, default=150, help='Maximum decisions per training episode; tests remain at 75')
    parser.add_argument('--training-tics', type=int, default=600, help='Engine timeout for training only; tests remain at 300 tics')
    parser.add_argument('--relay-directions', type=int, default=8, help='Antithetic directions per relay gradient estimate')
    args = parser.parse_args(argv)
    if not 1 <= args.batches <= 32 or not 2 <= args.episodes_per_batch <= 8 or not 8 <= args.test_episodes <= 100:
        parser.error('Use 1-32 batches, 2-8 episodes per batch and 8-100 test episodes')
    if not 75 <= args.training_decisions <= 300 or not 300 <= args.training_tics <= 1200 or not 4 <= args.relay_directions <= 32:
        parser.error('Use 75-300 training decisions, 300-1200 training tics and 4-32 relay directions')
    parent = args.parent
    if parent is None:
        pointer_path = ROOT/'latest.json'
        if not pointer_path.exists():
            pointer_path = Path('runs/laya-brain-learning/latest.json')
        pointer = json.loads(pointer_path.read_text())
        parent = Path(pointer['path'])
        if digest(parent/'report.json', 'sha256') != pointer['sha256']:
            raise ValueError('Parent checkpoint identity changed')
    parent_encoder, parent_actor, parent_report = load_any(parent)
    teacher_parent = teaching_checkpoint(parent)
    print(f'Continue learned weights from: {parent}', flush=True)
    print(f'Training: {args.batches} batches x {args.episodes_per_batch} episodes x 2 policies; '
          f'up to {args.training_decisions} decisions / {args.training_tics} tics per episode.', flush=True)
    print(f'Tests stay at 75 decisions / 300 tics. This command trains without opening a browser. '
          'To play the last completed model: python -m flydoom --reward-student', flush=True)
    with CycleLock(ROOT):
        folder = ROOT/datetime.now().strftime('cycle-%Y%m%d-%H%M%S-%f')
        folder.mkdir()
        def progress(stage, message):
            atomic_json(ROOT/'progress.json', {'status': 'running', 'stage': stage, 'folder': str(folder), 'message': message})
            print(message, flush=True)
        try:
            sources = {**parent_report['source_sha256'], **{name: digest(Path(__file__).with_name(name), 'sha256')
                       for name in ('reward_student_v2.py', 'reward_training_v3.py', 'reward_environment.py')}}
            progress(1, 'Reserve untouched games; load Laya training-only rehearsal data')
            plan = reserve(args.batches*args.episodes_per_batch, args.test_episodes,
                           training_steps=args.training_decisions, training_tics=args.training_tics,
                           progress=lambda message: progress(1, message))
            plan.update(parent=str(parent.resolve()), teacher_records_from=str(teacher_parent),
                        batches=args.batches, episodes_per_batch=args.episodes_per_batch,
                        relay_directions=args.relay_directions)
            atomic_json(folder/'plan.json', plan)
            rows = ordered([r for r in json.loads((teacher_parent/'experience.json').read_text()) if r['split'] == 'train'])
            all_resets = np.array([r['step'] == 0 for r in rows])
            ends = np.flatnonzero(np.r_[all_resets[1:], True])+1
            count = int(ends[np.searchsorted(ends, min(64, len(rows)))])
            visual = load_any('runs/retinal-control-v1')[0].ids
            # Incoming drive includes synaptic weights: a cache from BEFORE Laya
            # relay fitting cannot be reused with the taught graph as a new base.
            progress(1, f'Rebuild {count} teacher rehearsal frames against the continued graph')
            demo_encoder, _, _ = load_any(parent)
            demo_bridge = Bridge(demo_encoder, visual)
            demo_drive, demo_resets, demo_raw, demo_audit = cache_drive(
                demo_bridge, rows[:count], folder/'rehearsal-drive.npy')
            demo_audit['parent_weights_sha256'] = digest(parent/'weights.npz', 'sha256')
            demo_audit['cache_base'] = 'Verified Laya-trained parent graph, not the earlier teaching cache'
            atomic_json(folder/'rehearsal-audit.json', demo_audit)
            demo_targets = targets(rows[:count])
            del demo_encoder, demo_bridge, demo_raw
            learners = {}
            for name in ('candidate', 'decoder_only'):
                e, a, _ = load_any(parent)
                learners[name] = RewardStudent(Bridge(e, visual), a, demo_drive, demo_resets, demo_targets,
                                               train_relays=name == 'candidate', seed=121026,
                                               relay_directions=args.relay_directions)
            critic_restored = restore_critic(parent, parent_report, learners)
            manifest = []
            history = {name: [] for name in learners}
            progress(2, 'Learn from actual game rewards; Laya remains offline')
            for batch in range(args.batches):
                openings = plan['splits']['reward_train'][batch*args.episodes_per_batch:(batch+1)*args.episodes_per_batch]
                for name, learner in learners.items():
                    episodes = []
                    for episode_index, opening in enumerate(openings):
                        progress(2, f'{name}: batch {batch+1}/{args.batches}, episode {episode_index+1}/{len(openings)}')
                        path = folder/f'{name}-train-{opening["seed"]}.npz'
                        episode = collect(learner, opening, path, args.training_decisions)
                        episodes.append(episode)
                        manifest.append({k: episode[k] for k in ('path', 'sha256', 'seed', 'kills', 'replay_max_error', 'termination')} | {'policy': name, 'split': 'reward_train'})
                        print(f'{name} batch {batch+1}/{args.batches}: return={sum(episode["rewards"]):.1f}, kills={episode["kills"]}', flush=True)
                    progress(2, f'{name}: updating batch {batch+1}/{args.batches}; estimating relay and decoder gradients')
                    result = learner.update(episodes)
                    history[name].append(result)
                    np.savez_compressed(folder/f'{name}-batch-{batch+1:02d}.npz', weights=learner.actor.weights,
                                        gains=learner.gains, critic=learner.critic)
                    atomic_json(folder/'learning.json', history)
                    atomic_json(folder/'experience.json', manifest)
                    print(f'{name} update: accepted={result["accepted"]}, relay={result["relay_change_l2"]:.5f}, decoder={result["decoder_change_l2"]:.5f}', flush=True)
                    del episodes, episode
            progress(3, 'Freeze all learners; run matched gates and untouched final tests')
            policies = {name: (l.bridge.encoder, l.actor) for name, l in learners.items()}
            policies['continued_parent'] = (parent_encoder, parent_actor)
            gate, tests = {}, {}
            for split, results in [('gate', gate), ('test', tests)]:
                for name, (e, a) in policies.items():
                    results[name] = evaluate(e, a, plan['splits'][split], name+' '+split)
                    atomic_json(folder/(split+'.json'), results)
            scores = {k: summarize(v) for k, v in gate.items()}
            c = scores['candidate']
            passed = c['kill_episode_rate'] >= .8 and all(c['kills'] >= r['kills'] and c['mean_return'] >= r['mean_return']
                     for name, r in scores.items() if name != 'candidate')
            learner = learners['candidate']
            e, a = learner.bridge.encoder, learner.actor
            learner.bridge.apply(learner.gains)
            changed = int(np.count_nonzero(e.model.weights.data != learner.bridge.base_weights))
            if not np.array_equal(e.model.tau, learner.bridge.base_tau):
                raise ValueError('Reward training altered physiological time constants')
            if (e.model.weights[learner.bridge.visual] != parent_encoder.model.weights[learner.bridge.visual]).nnz:
                raise ValueError('Reward training altered the visual circuit')
            e.reset()
            disconnected, _ = e.encode(np.full((240, 320, 3), 255, np.uint8), connected=False)
            if np.any(disconnected):
                raise ValueError('Policy output bypasses anatomical transmission')
            save_candidate(folder, e, a)
            for name in ('physiology.json', 'distillation.json'):
                (folder/name).write_bytes((parent/name).read_bytes())
            ablation = json.loads((parent/'ablation.json').read_text())
            ablation['finding'] = 'Inherited initialization validation only; not a post-continuation measurement. See matched decoder-only and unchanged-parent game tests.'
            ablation['inherited_from'] = str(parent)
            atomic_json(folder/'ablation.json', ablation)
            reward_training = {'method': 'On-policy reward-to-go with a learned linear value baseline; finite-difference relay policy gradients and analytic decoder gradients',
                'parent': str(parent.resolve()), 'batches': args.batches, 'training_episodes_per_policy': args.batches*args.episodes_per_batch,
                'continuation': 'Warm-start learned graph, decoder and value baseline; fresh rollout RNG and trust anchor. Not exact interrupted-process resume.',
                'critic_restored': critic_restored, 'teacher_records_from': str(teacher_parent),
                'training_decisions': args.training_decisions, 'training_timeout_tics': args.training_tics,
                'evaluation_decisions': 75, 'evaluation_timeout_tics': 300,
                'relay_gradient_directions': args.relay_directions,
                'rehearsal_frames': count, 'rehearsal_cache_base': 'verified Laya-trained parent', 'teacher_live_calls': 0, 'gamma': .99, 'reward_scale': .01,
                'relay_active_groups': int(learner.active.sum()), 'changed_edges': changed,
                'decoder_change_l2': float(np.linalg.norm(a.weights-parent_actor.weights)),
                'accepted_batches': sum(r['accepted'] for r in history['candidate']),
                'history': history, 'frozen_game_tests': True, 'retained_timing': True,
                'scope': 'Basic target shooting only. Reward credit uses discounted outcomes on finite-horizon training trajectories. The value baseline is not used for deployed actions. No live teacher or biological dopamine mechanism.'}
            atomic_json(folder/'reward-training.json', reward_training)
            atomic_json(folder/'gate-summary.json', {'scores': scores, 'passed': bool(passed)})
            report = {'schema': SCHEMA, 'status': 'completed', 'parent': str(parent),
                'parent_report_sha256': digest(parent/'report.json', 'sha256'), 'neurons': len(e.ids),
                'readout_cells': len(a.root_ids), 'actions': ACTIONS, 'tasks': ['basic'],
                'laya_training': parent_report['laya_training'], 'reward_training': reward_training,
                'four_steps': {'laya_initialization': 'completed', 'reward_credit_training': 'completed',
                               'frozen_game_test': 'completed', 'independent_physiology': 'not_validated'},
                'test': {k: summarize(v) for k, v in tests.items()},
                'test_by_task': {k: {'basic': summarize(v)} for k, v in tests.items()},
                'game_gate_passed': bool(passed), 'biological_validation_passed': False, 'promoted': False,
                'scope': 'Laya-initialized eye-driven brain, subsequently trained using real game rewards and retained teacher examples. Relay magnitudes and decoder weights learn; anatomy, signs, visual rows and time constants remain fixed. Experimental; no automatic promotion.'}
            names = ['weights.npz', 'circuit.npz', 'circuit.json', 'actor.npz', 'plan.json', 'experience.json',
                     'learning.json', 'reward-training.json', 'gate-summary.json', 'gate.json', 'test.json',
                     'physiology.json', 'distillation.json', 'ablation.json', 'rehearsal-audit.json']
            names += [p.name for p in folder.glob('*-batch-*.npz')]
            report['output_sha256'] = {n: digest(folder/n, 'sha256') for n in names}
            for name, sha in sources.items():
                if digest(Path(__file__).with_name(name), 'sha256') != sha:
                    raise ValueError('Implementation changed during training: '+name)
            report['source_sha256'] = sources
            atomic_json(folder/'report.json', report)
            load_any(folder)
            atomic_json(ROOT/'latest.json', {'path': str(folder.resolve()), 'sha256': digest(folder/'report.json', 'sha256')})
            atomic_json(ROOT/'progress.json', {'status': 'completed', 'folder': str(folder)})
            print(json.dumps({'folder': str(folder), 'test': report['test'], 'changed_edges': changed, 'gate_passed': bool(passed)}, indent=2), flush=True)
        except BaseException as error:
            atomic_json(ROOT/'progress.json', {'status': 'failed', 'folder': str(folder), 'error': str(error)})
            raise


if __name__ == '__main__':
    main()
