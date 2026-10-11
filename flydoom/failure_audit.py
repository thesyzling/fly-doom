"""Replay retired test episodes as development diagnostics, never training data."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from flydoom.adaptive_policy import ACTIONS, action, load_any, make_game
from flydoom.aim_training import target
from flydoom.data import digest
from flydoom.laya_brain_teacher import LayaBrainTeacher
from flydoom.research_cycle import frame_hash
from flydoom.research_registry import atomic_json
from flydoom.visual_observer import sample_frame


def audit(checkpoint, output):
    output.mkdir(parents=True, exist_ok=False)
    e, actor, report = load_any(checkpoint)
    old = json.loads((checkpoint/'test.json').read_text(encoding='utf-8'))['candidate']
    selected = [r for r in old if not r['kills']][:2]+[r for r in old if r['kills']][:1]
    audit_rows, raw_rows, frames = [], [], []
    for episode in selected:
        game = make_game(episode['seed'], labels=True)
        rng = np.random.default_rng(episode['seed'])
        e.reset()
        actor.reset()
        try:
            for _ in range(6):
                action(game, int(rng.choice([1, 2])))
            assert frame_hash(game.get_state().screen_buffer) == episode['opening_sha256']
            for step, saved in enumerate(episode['trace']):
                state = game.get_state()
                frame = state.screen_buffer.copy()
                annotation = target(state)
                contrast = sample_frame(frame, e.uv, e.visible)
                raw, _ = e.encode(frame)
                choice, p, features = actor.decide(raw)
                assert ACTIONS[choice] == saved['action'], 'Diagnostic replay changed the policy trajectory'
                n = 0
                if annotation:
                    x, y, w, h = annotation['bbox']
                    uv = e.uv*[319, 239]
                    n = int(np.sum(e.visible & (uv[:, 0] >= x) & (uv[:, 0] < x+w) &
                                   (uv[:, 1] >= y) & (uv[:, 1] < y+h)))
                row = {'seed': episode['seed'], 'step': step, 'episode_kills': episode['kills'],
                       'student_action': ACTIONS[choice], 'probabilities': p.tolist(),
                       'target_diagnostic': annotation, 'receptors_in_target_box': n,
                       'eye_rms': float(np.sqrt(np.mean(contrast**2))),
                       'descending_rms': float(np.sqrt(np.mean(raw**2))),
                       'saturated_feature_fraction': float(np.mean(abs(features[:-1]) > .995)),
                       'frame_sha256': frame_hash(frame)}
                audit_rows.append(row)
                raw_rows.append(raw)
                frames.append(frame)
                effect, _ = action(game, choice)
                assert effect['reward_delta'] == saved['reward']
            print(f'Diagnostic replay seed {episode["seed"]}: {len(episode["trace"])} exact decisions', flush=True)
        finally:
            game.close()
    np.savez_compressed(output/'trajectories.npz', raw=np.array(raw_rows), frames=np.array(frames))
    atomic_json(output/'decisions.json', audit_rows)
    teacher = LayaBrainTeacher(output/'teacher.log')
    try:
        for episode in selected:
            indices = [i for i, r in enumerate(audit_rows) if r['seed'] == episode['seed']]
            for index in np.unique(np.linspace(0, len(indices)-1, 6).astype(int)):
                i = indices[index]
                row = audit_rows[i]
                answer = teacher.predict(frames[i])
                assert answer['frame_sha256'] == row['frame_sha256']
                row['teacher'] = answer
                Image.fromarray(frames[i]).save(output/f'{row["seed"]}-{row["step"]:03d}.png')
                print(f'Audit Laya {row["seed"]}/{row["step"]}: student={row["student_action"]}, teacher={ACTIONS[int(np.argmax(answer["probabilities"]))]}', flush=True)
        identity = teacher.metadata
    finally:
        teacher.close()
    atomic_json(output/'decisions.json', audit_rows)
    summaries = []
    for episode in selected:
        rows = [r for r in audit_rows if r['seed'] == episode['seed']]
        actions = [r['student_action'] for r in rows]
        queried = [r for r in rows if 'teacher' in r]
        annotated = [r for r in rows if r['target_diagnostic']]
        summaries.append({'seed': episode['seed'], 'kills': episode['kills'], 'decisions': len(rows),
            'action_counts': dict(Counter(actions)),
            'left_right_reversals': sum({a, b} == {'MOVE_LEFT', 'MOVE_RIGHT'} for a, b in zip(actions, actions[1:])),
            'visible_target_frames': len(annotated),
            'minimum_receptors_in_target_box': min((r['receptors_in_target_box'] for r in annotated), default=0),
            'teacher_queries': len(queried),
            'teacher_student_agreement': float(np.mean([r['student_action'] == ACTIONS[int(np.argmax(r['teacher']['probabilities']))] for r in queried])),
            'student_geometric_direction_agreement': float(np.mean([r['student_action'] == ACTIONS[r['target_diagnostic']['action']] for r in annotated]))})
    result = {'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': digest(checkpoint/'report.json', 'sha256'),
              'episodes': summaries, 'teacher': identity,
              'scope': 'Previously reported tests are now retired development diagnostics. Engine boxes are diagnostic only and never policy inputs or teaching labels. Exact original actions/rewards were replayed. Six same-image teacher queries per selected episode are descriptive, not a representative accuracy estimate.'}
    atomic_json(output/'report.json', result)
    print(json.dumps(result['episodes'], indent=2), flush=True)


def occlusion_audit(output):
    """Intervene on one input patch from identical replayed neural histories."""
    report = json.loads((output/'report.json').read_text(encoding='utf-8'))
    rows = json.loads((output/'decisions.json').read_text(encoding='utf-8'))
    with np.load(output/'trajectories.npz', allow_pickle=False) as data:
        frames, saved_raw = data['frames'], data['raw']
    e, actor, _ = load_any(report['checkpoint'])
    checks = []
    for episode in report['episodes']:
        indices = [i for i, row in enumerate(rows) if row['seed'] == episode['seed']]
        selected = indices[len(indices)//2]
        e.reset()
        for index in indices:
            if index == selected:
                break
            e.encode(frames[index])
        delta, adaptation, decisions = e.model.delta.copy(), e.adaptation.copy(), e.decisions
        actual, _ = e.encode(frames[selected])
        original_p = actor.decide(actual)[1]
        e.model.delta[:] = delta
        e.adaptation[:] = adaptation
        e.decisions = decisions
        altered = frames[selected].copy()
        x, y, w, h = rows[selected]['target_diagnostic']['bbox']
        altered[y:y+h, x:x+w] = altered[y:y+h, max(0, x-1):max(0, x-1)+1]
        counter, _ = e.encode(altered)
        counter_p = actor.decide(counter)[1]
        np.testing.assert_array_equal(actual, saved_raw[selected])
        eye_before = sample_frame(frames[selected], e.uv, e.visible)
        eye_after = sample_frame(altered, e.uv, e.visible)
        checks.append({'seed': episode['seed'], 'step': rows[selected]['step'],
            'eye_changed_receptors': int(np.count_nonzero(eye_before != eye_after)),
            'descending_changed_cells': int(np.count_nonzero(actual != counter)),
            'descending_change_l2': float(np.linalg.norm(actual-counter)),
            'action_probability_change_l1': float(np.sum(abs(original_p-counter_p)))})
    result = {'checks': checks, 'audit_source_sha256': digest(Path(__file__), 'sha256'),
        'scope': 'Diagnostic target-box occlusion from identical replayed neural states. Adjacent background substitution is an artificial intervention, not a natural scene or training target. Sensitivity proves transmission of this image perturbation, not correct target localization.'}
    atomic_json(output/'occlusion-audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--occlusion', action='store_true', help='Use an existing audit to test target-patch transmission without invoking Laya')
    args = parser.parse_args()
    if args.occlusion:
        return occlusion_audit(args.output)
    checkpoint = args.checkpoint or Path(json.loads(Path('runs/reward-student/latest.json').read_text())['path'])
    audit(checkpoint, args.output)


if __name__ == '__main__':
    main()
