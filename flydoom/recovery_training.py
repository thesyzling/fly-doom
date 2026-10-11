"""Laya correction on student-visited states, with fresh frozen game controls.

A bounded DAgger-style aggregation pass. The teacher labels observations but
never controls collection actions or frozen student tests. Retired diagnostic
test trajectories are excluded from the optimization data.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from PIL import Image

from flydoom.adaptive_policy import ACTIONS, SCHEMA, load_any
from flydoom.bridge_training import Bridge, ordered, cache_drive
from flydoom.data import digest
from flydoom.laya_brain_fit import active_groups, fit, joint_fit, targets
from flydoom.laya_brain_teacher import LayaBrainTeacher
from flydoom.laya_brain_training import collect, evaluate
from flydoom.research_cycle import summarize
from flydoom.research_registry import CycleLock, atomic_json
from flydoom.retinal_train import metrics
from flydoom.reward_training_v3 import reserve, teaching_checkpoint
from flydoom.sensorimotor import save_candidate

ROOT = Path('runs/recovery-student')


def training_only_rehearsal(checkpoint, minimum=64):
    """Keep complete original TRAIN trajectories; never import past tests."""
    teacher = teaching_checkpoint(checkpoint)
    rows = ordered([r for r in json.loads((teacher/'experience.json').read_text(encoding='utf-8'))
                    if r['split'] == 'train'])
    resets = np.array([r['step'] == 0 for r in rows])
    ends = np.flatnonzero(np.r_[resets[1:], True])+1
    count = int(ends[np.searchsorted(ends, min(minimum, len(rows)))])
    return rows[:count], teacher


def validation_mask(train, validation):
    """Exclude identical images while retaining the full causal replay prefix."""
    seeds = {r['seed'] for r in train}
    if any(r['seed'] in seeds for r in validation):
        raise ValueError('Training and validation episodes overlap')
    hashes = {r['teacher']['frame_sha256'] for r in train}
    mask = np.array([r['teacher']['frame_sha256'] not in hashes for r in validation])
    if not mask.any():
        raise ValueError('No distinct validation observations remain')
    return mask


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--train-episodes', type=int, default=8)
    parser.add_argument('--validation-episodes', type=int, default=4)
    parser.add_argument('--test-episodes', type=int, default=24)
    parser.add_argument('--relay-pairs', type=int, default=4)
    args = parser.parse_args(argv)
    if not 4 <= args.train_episodes <= 24 or not 2 <= args.validation_episodes <= 8 or not 16 <= args.test_episodes <= 100 or not 1 <= args.relay_pairs <= 8:
        parser.error('Use 4-24 training, 2-8 validation, 16-100 test episodes and 1-8 relay pairs')
    parent = args.parent.resolve()
    e, actor_parent, parent_report = load_any(parent)
    retained, teacher_parent = training_only_rehearsal(parent)
    with CycleLock(ROOT):
        folder = ROOT/datetime.now().strftime('cycle-%Y%m%d-%H%M%S-%f')
        folder.mkdir()
        def progress(stage, message):
            atomic_json(ROOT/'progress.json', {'status': 'running', 'folder': str(folder), 'stage': stage, 'message': message})
            print(message, flush=True)
        try:
            names = ('recovery_training.py', 'laya_brain_fit.py', 'laya_brain_training.py',
                     'laya_brain_teacher.py', 'bridge_training.py', 'reward_training_v3.py', 'reward_environment.py')
            sources = {**parent_report['source_sha256'], **{n: digest(Path(__file__).with_name(n), 'sha256') for n in names}}
            progress(1, 'Reserve fresh openings; previous diagnostic tests remain excluded from training')
            plan = reserve(args.train_episodes+args.validation_episodes, args.test_episodes,
                           training_steps=75, training_tics=300, progress=lambda m: progress(1, m))
            openings = plan['splits'].pop('reward_train')
            plan['splits']['train'] = openings[:args.train_episodes]
            plan['splits']['validation'] = openings[args.train_episodes:]
            plan.update(schema='student_visited_laya_correction_v1', parent=str(parent),
                teacher_records_from=str(teacher_parent), relay_pairs=args.relay_pairs,
                collection='Frozen parent student acts greedily; Laya only labels its visited images. Complete episodes up to 75 decisions.',
                selection='Relay proposals use TRAIN teacher KL only. Readout epoch uses distinct validation teacher KL, including epoch zero. Gate >=80% and no return/hit regression versus unchanged parent and fixed-relay correction. Final tests never choose parameters.',
                correction_passes=1, diagnostic_tests_used_for_training=False)
            atomic_json(folder/'plan.json', plan)
            frames = folder/'frames'
            frames.mkdir()
            fresh = []
            teacher = LayaBrainTeacher(folder/'teacher.log')
            try:
                atomic_json(folder/'teacher.json', teacher.metadata)
                for split in ('train', 'validation'):
                    for i, opening in enumerate(plan['splits'][split]):
                        progress(2, f'Laya correction {split} {i+1}/{len(plan["splits"][split])}; student controls the game')
                        fresh.extend(collect(e, actor_parent, teacher, opening, frames, split, 0., 75))
                        atomic_json(folder/'experience.json', fresh+retained)
            finally:
                teacher.close()
            if teacher.process.poll() is None:
                raise RuntimeError('Teacher must exit before fitting and frozen tests')
            train = ordered([r for r in fresh if r['split'] == 'train']+retained)
            validation = ordered([r for r in fresh if r['split'] == 'validation'])
            mask = validation_mask(train, validation)
            visual = load_any('runs/retinal-control-v1')[0].ids
            bridge = Bridge(e, visual)
            progress(3, 'Replay complete eye/brain histories; fit recovery decisions against Laya labels')
            drive, resets, raw, train_audit = cache_drive(bridge, train, folder/'train-drive.npy')
            vd, vr, vraw, validation_audit = cache_drive(bridge, validation, folder/'validation-drive.npy')
            atomic_json(folder/'cache-audit.json', {'train': train_audit, 'validation': validation_audit,
                        'parent_weights_sha256': digest(parent/'weights.npz', 'sha256')})
            y, vy = targets(train), targets(validation)[mask]
            fixed, fixed_fit = fit(raw, y, actor_parent, (vraw[mask], vy))
            fixed.save(folder/'fixed-biology-actor.npz')
            atomic_json(folder/'fixed-biology-fit.json', fixed_fit)
            progress(4, 'Correct shared relay gains and decoder; preserve anatomy and physiological timing')
            gains, optimization = joint_fit(bridge, drive, resets, train, actor_parent, pairs=args.relay_pairs)
            learned = bridge.replay(drive, resets, gains)
            vlearned = bridge.replay(vd, vr, gains)
            actor, distillation = fit(learned, y, actor_parent, (vlearned[mask], vy))
            bridge.apply(gains)
            reference = []
            for row in train[:12]:
                if row['step'] == 0:
                    e.reset()
                with Image.open(row['path']) as image:
                    value, _ = e.encode(np.asarray(image.convert('RGB')))
                reference.append(value)
            if not np.allclose(reference, learned[:12], rtol=2e-3, atol=2e-7):
                raise ValueError('Full eye/brain deployment differs from fitting replay')
            if not np.array_equal(e.model.tau, bridge.base_tau):
                raise ValueError('Correction changed response timing')
            e.reset()
            disconnected, _ = e.encode(np.full((240, 320, 3), 255, np.uint8), connected=False)
            if np.any(disconnected):
                raise ValueError('Descending output bypasses anatomical transmission')
            changed = int(np.count_nonzero(e.model.weights.data != bridge.base_weights))
            plasticity = {**optimization, 'gains': gains.tolist(), 'changed_edges': changed,
                'changed_groups': int(np.count_nonzero(gains != 1)), 'visual_rows_unchanged': True,
                'disconnected_output_zero': True, 'deployment_max_absolute_error': float(np.max(abs(reference-learned[:12])))}
            atomic_json(folder/'plasticity.json', plasticity)
            atomic_json(folder/'distillation.json', distillation)
            ablation = {'frames': int(mask.sum()), 'development_only': True,
                'active_transmission_groups': int(active_groups(bridge).sum()),
                'prior_empty_transmission_parameters': int(3*len(bridge.types)-active_groups(bridge).sum()),
                'finding': 'Fresh student-visited Laya validation; teacher agreement is separate from gameplay.',
                'matrix': {condition: {name: metrics(a, values[mask], vy) for name, a in [('fixed_decoder', fixed), ('joint_decoder', actor)]}
                           for condition, values in [('neutral', vraw), ('learned_connections', vlearned)]}}
            atomic_json(folder/'ablation.json', ablation)
            (folder/'physiology.json').write_bytes((parent/'physiology.json').read_bytes())
            baseline_e, baseline_a, _ = load_any(parent)
            progress(5, 'Teacher is closed; compare frozen corrected policies and unchanged parent on identical starts')
            gate, tests = {}, {}
            for split, results in [('gate', gate), ('test', tests)]:
                for name, encoder, decoder in [('candidate', e, actor), ('fixed_biology', e, fixed), ('continued_parent', baseline_e, baseline_a)]:
                    if name != 'continued_parent':
                        bridge.apply(gains if name == 'candidate' else bridge.neutral())
                    progress(5, f'{name}: {split}, {len(plan["splits"][split])} matched episodes')
                    results[name] = evaluate(encoder, decoder, plan['splits'][split], name+' '+split)
                    atomic_json(folder/(split+'.json'), results)
            scores = {k: summarize(v) for k, v in gate.items()}
            c = scores['candidate']
            passed = c['kill_episode_rate'] >= .8 and all(c['kills'] >= r['kills'] and c['mean_return'] >= r['mean_return']
                     for k, r in scores.items() if k != 'candidate')
            bridge.apply(gains)
            save_candidate(folder, e, actor)
            recovery = {'method': 'One DAgger-style aggregation pass on fresh greedy student trajectories, retaining original TRAIN examples; joint anatomical relay/readout distillation',
                'parent': str(parent), 'teacher_records_from': str(teacher_parent),
                'fresh_train_frames': sum(r['split'] == 'train' for r in fresh), 'retained_train_frames': len(retained),
                'validation_frames': int(mask.sum()), 'correction_passes': 1,
                'collection_teacher_controls_action': False, 'engine_annotation_targets': False,
                'diagnostic_test_data_used': False, 'teacher_live_calls': 0, 'changed_edges': changed,
                'decoder_change_l2': float(np.linalg.norm(actor.weights-actor_parent.weights)),
                'teacher_fit': distillation, 'relay_fit': optimization,
                'scope': 'Basic target shooting; no pixel or teacher shortcut at deployment. Previous reward learning is inherited, not rerun here.'}
            atomic_json(folder/'recovery-training.json', recovery)
            atomic_json(folder/'gate-summary.json', {'scores': scores, 'passed': bool(passed)})
            report = {'schema': SCHEMA, 'status': 'completed', 'parent': str(parent),
                'parent_report_sha256': digest(parent/'report.json', 'sha256'), 'neurons': len(e.ids),
                'readout_cells': len(actor.root_ids), 'actions': ACTIONS, 'tasks': ['basic'],
                'laya_training': {**parent_report['laya_training'], 'target_source': 'Laya probabilities on student-visited images plus retained original TRAIN examples'},
                'inherited_reward_training': parent_report.get('reward_training'), 'recovery_training': recovery,
                'plasticity': plasticity, 'training_frames': len(train), 'validation_frames': int(mask.sum()),
                'four_steps': {'failure_diagnosis': 'completed', 'student_state_correction': 'completed',
                               'frozen_game_test': 'completed', 'independent_physiology': 'not_validated'},
                'test': {k: summarize(v) for k, v in tests.items()},
                'test_by_task': {k: {'basic': summarize(v)} for k, v in tests.items()},
                'game_gate_passed': bool(passed), 'biological_validation_passed': False, 'promoted': False,
                'scope': recovery['scope']}
            artifacts = ['plan.json', 'experience.json', 'teacher.json', 'cache-audit.json', 'fixed-biology-actor.npz',
                'fixed-biology-fit.json', 'plasticity.json', 'distillation.json', 'ablation.json', 'physiology.json',
                'weights.npz', 'circuit.npz', 'circuit.json', 'actor.npz', 'gate.json', 'test.json',
                'gate-summary.json', 'recovery-training.json']
            report['output_sha256'] = {n: digest(folder/n, 'sha256') for n in artifacts}
            for name, sha in sources.items():
                if digest(Path(__file__).with_name(name), 'sha256') != sha:
                    raise ValueError('Implementation changed during correction: '+name)
            report['source_sha256'] = sources
            atomic_json(folder/'report.json', report)
            load_any(folder)
            atomic_json(ROOT/'latest.json', {'path': str(folder.resolve()), 'sha256': digest(folder/'report.json', 'sha256')})
            atomic_json(ROOT/'progress.json', {'status': 'completed', 'folder': str(folder)})
            print(json.dumps({'folder': str(folder), 'test': report['test'], 'gate_passed': bool(passed)}, indent=2), flush=True)
        except BaseException as error:
            atomic_json(ROOT/'progress.json', {'status': 'failed', 'folder': str(folder), 'error': str(error)})
            raise


if __name__ == '__main__':
    main()
