"""Frozen topology interventions and transfer tests; never train on these games."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import secrets
import time

import numpy as np
import vizdoom as vzd

from flydoom.adaptive_policy import ACTIONS, action
from flydoom.data import digest
from flydoom.research_cycle import frame_hash, summarize
from flydoom.research_registry import atomic_json
from flydoom.reward_training_v3 import reserve
from flydoom.sampling_evaluation import evaluate
from flydoom.self_learning import verified_pointer
from flydoom.stable_vision import load_any


def rewire_relays(weights, visual, seed, attempts_per_edge=5):
    """Signed directed double-edge swaps, preserving degree and target weight mass.

    Visual rows and the visual/relay partition stay fixed. Per-target signed
    weight multisets stay fixed; source weighted strengths need not stay fixed.
    This is an intervention with a frozen decoder, not retrained null anatomy.
    """
    w = weights.tocsr(copy=True)
    if not w.has_canonical_format or np.any(w.data == 0):
        raise ValueError('Require canonical nonzero sparse weights')
    n = w.shape[0]; rows = np.repeat(np.arange(n), np.diff(w.indptr))
    is_visual = np.zeros(n, bool); is_visual[visual] = True
    edges = np.flatnonzero(~is_visual[rows]); rng = np.random.default_rng(seed)
    present = [set(w.indices[w.indptr[i]:w.indptr[i+1]]) for i in range(n)]
    accepted = 0
    original = {(int(rows[i]), int(w.indices[i])) for i in edges}
    for internal in (False, True):
        for positive in (False, True):
            group = edges[((~is_visual[w.indices[edges]]) == internal) & ((w.data[edges] > 0) == positive)]
            if len(group) < 2:
                continue
            for i, j in rng.choice(group, (attempts_per_edge*len(group), 2)):
                a, b = rows[i], rows[j]; c, d = w.indices[i], w.indices[j]
                if a == b or c == d or a == d or b == c or d in present[a] or c in present[b]:
                    continue
                present[a].remove(c); present[b].remove(d)
                present[a].add(d); present[b].add(c)
                w.indices[i], w.indices[j] = d, c
                accepted += 1
    changed = sum((int(rows[i]), int(w.indices[i])) not in original for i in edges)
    w.has_sorted_indices = False; w.sort_indices()
    np.testing.assert_array_equal(np.diff(w.indptr), np.diff(weights.indptr))
    for sign in (-1, 1):
        np.testing.assert_array_equal(np.bincount(w.indices[np.sign(w.data) == sign], minlength=n),
                                      np.bincount(weights.indices[np.sign(weights.data) == sign], minlength=n))
    np.testing.assert_allclose(np.asarray(abs(w).sum(axis=1)), np.asarray(abs(weights).sum(axis=1)), rtol=1e-6, atol=1e-7)
    if (w[visual] != weights[visual]).nnz:
        raise ValueError('Rewiring changed the visual circuit')
    return w, {'seed': seed, 'accepted_swaps': accepted, 'relay_edges': len(edges),
               'new_contacts': changed, 'changed_fraction': changed/max(1, len(edges)),
               'signed_degrees_preserved': True, 'target_weight_mass_preserved': True,
               'visual_rows_preserved': True, 'source_weighted_strength_preserved': False}


def open_task(opening):
    task = opening['task']
    if task not in ('predict_position', 'deadly_corridor'):
        raise ValueError('Unsupported transfer task')
    package = Path(os.path.relpath(Path(vzd.__file__).parent)); game = vzd.DoomGame()
    try:
        game.load_config(str(package/'scenarios'/f'{task}.cfg'))
        game.set_vizdoom_path(str(package/('vizdoom.exe' if os.name == 'nt' else 'vizdoom')))
        game.set_doom_game_path(str(package/'freedoom2.wad'))
        game.set_available_buttons([getattr(vzd.Button, a) for a in ACTIONS[1:]])
        game.set_available_game_variables([vzd.GameVariable.KILLCOUNT, vzd.GameVariable.POSITION_X,
                                          vzd.GameVariable.POSITION_Y, vzd.GameVariable.HEALTH])
        game.set_screen_format(vzd.ScreenFormat.RGB24); game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.set_mode(vzd.Mode.PLAYER); game.set_window_visible(False); game.set_render_hud(False)
        game.set_labels_buffer_enabled(False); game.set_sound_enabled(False); game.set_console_enabled(False)
        game.set_seed(opening['seed']); game.init(); game.new_episode()
        rng = np.random.default_rng(opening['seed'])
        for _ in range(6):
            action(game, int(rng.choice([1, 2])))
        if game.is_episode_finished():
            raise ValueError('Transfer warmup terminated')
        return game
    except BaseException:
        game.close(); raise


def reserve_transfer(count):
    used = set(); hashes = set()
    def visit(x):
        if isinstance(x, dict):
            if 'seed' in x and isinstance(x['seed'], int): used.add(x['seed'])
            if 'opening_sha256' in x: hashes.add(x['opening_sha256'])
            for y in x.values(): visit(y)
        elif isinstance(x, list):
            for y in x: visit(y)
    for path in Path('runs').rglob('plan.json'):
        visit(json.loads(path.read_text(encoding='utf-8')))
    result = {}
    for task in ('predict_position', 'deadly_corridor'):
        result[task] = []
        for i in range(count):
            for _ in range(10000):
                seed = secrets.randbelow(2**31-1)
                if seed in used: continue
                row = {'seed': seed, 'task': task}
                game = open_task(row)
                try:
                    sha = frame_hash(game.get_state().screen_buffer)
                    timeout = game.get_episode_timeout()
                finally: game.close()
                if sha not in hashes: break
            else: raise ValueError('No fresh transfer opening found')
            used.add(seed); hashes.add(sha)
            result[task].append({**row, 'opening_sha256': sha, 'timeout_tics': timeout})
            print(f'Reserve transfer {task} {i+1}/{count}', flush=True)
    return result


def evaluate_transfer(e, actor, openings, *, connected, label):
    results = []
    for opening in openings:
        game = open_task(opening); e.reset(); actor.reset()
        rng = np.random.default_rng(opening['seed']); initial = game.get_total_reward()
        trace = []; start_x = game.get_game_variable(vzd.GameVariable.POSITION_X)
        try:
            if frame_hash(game.get_state().screen_buffer) != opening['opening_sha256']:
                raise ValueError('Transfer opening identity changed')
            for step in range(int(np.ceil(opening['timeout_tics']/4))):
                if game.is_episode_finished(): break
                frame = game.get_state().screen_buffer
                raw, _ = e.encode(frame, connected=connected)
                choice, p, _ = actor.decide(raw, rng)
                sha = frame_hash(frame)
                effect, _ = action(game, choice)
                trace.append({'step': step, 'action': ACTIONS[choice], 'probabilities': p.tolist(),
                              'frame_sha256': sha, 'reward': effect['reward_delta'], 'kills': effect['kill_delta']})
            dead = bool(game.is_player_dead()); ticks = game.get_episode_time()
            results.append({**opening, 'return': float(game.get_total_reward()-initial),
                'kills': sum(t['kills'] for t in trace), 'decisions': len(trace), 'trace': trace,
                'dead': dead, 'engine_tics': ticks,
                'engine_finished_alive_before_timeout': bool(game.is_episode_finished() and not dead and ticks < opening['timeout_tics']),
                'forward_displacement': float(game.get_game_variable(vzd.GameVariable.POSITION_X)-start_x),
                'teacher_controls_action': False, 'labels_buffer_enabled': game.is_labels_buffer_enabled(),
                'eye_connected': connected})
            print(f'{label} {len(results)}/{len(openings)}: kills={results[-1]["kills"]}, return={results[-1]["return"]:.2f}, dead={dead}', flush=True)
        finally: game.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('runs/structural-transfer-v1'))
    args = parser.parse_args(); root = args.output
    root.mkdir(exist_ok=True)
    if args.prepare:
        if (root/'plan.json').exists(): raise ValueError('Preserve existing reserved starts')
        basic = reserve(0, 8)
        atomic_json(root/'plan.json', {'schema': 'frozen_structure_transfer_v1', 'basic': basic,
            'transfer': reserve_transfer(8), 'rewiring_seeds': [101, 202, 303],
            'protocol': 'Frozen decoder; signed degree-preserving relay swaps; no retraining or biological-superiority claim. Reserved basic gate starts unused. Transfer uses native scenario timeout and eight-action interface.',
            'policy_inputs': 'Pixels through neural circuit only. Health and position are diagnostics.',
            'sources': {n: digest(Path(__file__).with_name(n), 'sha256') for n in ('structural_validation.py', 'stable_vision.py')}})
        return
    if (root/'report.json').exists(): raise ValueError('Preserve the completed audit')
    plan = json.loads((root/'plan.json').read_text())
    for n, sha in plan['sources'].items():
        if digest(Path(__file__).with_name(n), 'sha256') != sha: raise ValueError('Reserved audit implementation changed')
    checkpoint = args.checkpoint or verified_pointer(Path('runs/self-learning-expanded/latest.json'))
    parent = verified_pointer(Path('runs/self-learning/selected.json'))
    visual_ids = load_any('runs/retinal-control-v1')[0].ids
    topology = {}; swaps = {}
    for name, seed in [('anatomical', None)]+[(f'rewired_{s}', s) for s in plan['rewiring_seeds']]:
        e, a, _ = load_any(checkpoint)
        if seed is not None:
            w, audit = rewire_relays(e.model.weights, np.flatnonzero(np.isin(e.ids, visual_ids)), seed)
            if audit['changed_fraction'] < .2: raise ValueError('Null intervention changed too few contacts')
            e.model.weights = w; swaps[name] = audit
        topology[name] = evaluate(e, a, plan['basic']['splits']['test'], connected=True, label=name)
        atomic_json(root/'topology.json', topology); atomic_json(root/'swaps.json', swaps)
    transfer = {}
    for name, folder, connected in [('candidate', checkpoint, True), ('incumbent', parent, True), ('disconnected', checkpoint, False)]:
        transfer[name] = {}
        for task, openings in plan['transfer'].items():
            e, a, _ = load_any(folder)
            transfer[name][task] = evaluate_transfer(e, a, openings, connected=connected, label=f'{name} {task}')
            atomic_json(root/'transfer.json', transfer)
    scores = {name: {} for name in transfer}
    for name, tasks in transfer.items():
        for task, rows in tasks.items():
            scores[name][task] = {**summarize(rows), 'deaths': sum(r['dead'] for r in rows),
                'engine_completions_alive': sum(r['engine_finished_alive_before_timeout'] for r in rows),
                'mean_forward_displacement': float(np.mean([r['forward_displacement'] for r in rows])),
                'action_counts': dict(Counter(t['action'] for r in rows for t in r['trace']))}
    atomic_json(root/'report.json', {'schema': 'frozen_structure_transfer_v1', 'checkpoint': str(checkpoint),
        'checkpoint_sha256': digest(checkpoint/'report.json', 'sha256'), 'incumbent': str(parent),
        'topology': {n: summarize(r) for n, r in topology.items()}, 'rewiring': swaps, 'transfer': scores,
        'trained_on_transfer': False, 'biological_topology_superiority_proven': False,
        'scope': 'Frozen interventions measure sensitivity to relay topology; null graphs were not trained. Transfer scores are descriptive, not model-selection data; kills do not establish navigation.',
        'output_sha256': {p.name: digest(p, 'sha256') for p in root.iterdir() if p.is_file()}})


if __name__ == '__main__':
    main()
