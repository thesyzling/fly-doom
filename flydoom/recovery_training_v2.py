"""Balanced correction with gameplay selection, retention checks and parent fallback."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from PIL import Image

from flydoom.adaptive_policy import ACTIONS, SCHEMA, load_any
from flydoom.balanced_distillation import fit_candidates, select_development
from flydoom.bridge_training import ordered
from flydoom.data import digest
from flydoom.laya_brain_fit import targets
from flydoom.laya_brain_training import evaluate
from flydoom.recovery_training import validation_mask
from flydoom.research_cycle import frame_hash, summarize
from flydoom.research_registry import CycleLock, atomic_json
from flydoom.reward_training_v3 import reserve
from flydoom.sensorimotor import save_candidate

ROOT = Path('runs/balanced-student')
PARENT = Path('runs/reward-student/cycle-20261011-050208-412071')
DATA = Path('runs/recovery-student/cycle-20261011-061526-313894')
ORIGINAL = Path('runs/laya-brain-learning/cycle-20261011-035639-088292')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def replay(encoder, rows):
    values = []
    for index, row in enumerate(rows):
        if row['step'] == 0:
            encoder.reset()
        path = Path(row['path'])
        if digest(path, 'sha256') != row['png_sha256']:
            raise ValueError('Teaching PNG changed')
        with Image.open(path) as image:
            frame = np.asarray(image.convert('RGB'))
        if frame_hash(frame) != row['teacher']['frame_sha256']:
            raise ValueError('Teaching observation mismatch')
        raw, _ = encoder.encode(frame)
        values.append(raw)
        if (index+1) % 100 == 0:
            print(f'Native neural replay {index+1}/{len(rows)}', flush=True)
    return np.asarray(values)


def comparison(candidate, parent):
    if [r['seed'] for r in candidate] != [r['seed'] for r in parent]:
        raise ValueError('Comparison starts differ')
    lost = sum(p['kills'] > 0 and c['kills'] == 0 for c, p in zip(candidate, parent))
    gained = sum(c['kills'] > 0 and p['kills'] == 0 for c, p in zip(candidate, parent))
    return {'lost_parent_successes': lost, 'gained_successes': gained,
            'net_additional_hits': gained-lost,
            'mean_return_change': float(np.mean([c['return']-p['return'] for c, p in zip(candidate, parent)]))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-episodes', type=int, default=32)
    args = parser.parse_args(argv)
    if not 24 <= args.test_episodes <= 64:
        parser.error('Use 24-64 final episodes')
    encoder, parent, inherited = load_any(PARENT)
    # Verify the old data sources without changing either model or any old cache.
    load_any(DATA)
    load_any(ORIGINAL)
    previous = read(DATA/'experience.json')
    original = read(ORIGINAL/'experience.json')
    original_seeds = {r['seed'] for r in original if r['split'] == 'train'}
    train = ordered([r for r in previous if r['split'] == 'train' and r['seed'] not in original_seeds]
                    + [r for r in original if r['split'] == 'train'])
    validation = ordered([r for r in previous if r['split'] == 'validation'])
    mask = validation_mask(train, validation)
    with CycleLock(ROOT):
        folder = ROOT/datetime.now().strftime('cycle-%Y%m%d-%H%M%S-%f')
        folder.mkdir()
        def progress(message):
            atomic_json(ROOT/'progress.json', {'status': 'running', 'folder': str(folder), 'message': message})
            print(message, flush=True)
        try:
            names = ('recovery_training_v2.py', 'balanced_distillation.py', 'recovery_training.py',
                     'laya_brain_fit.py', 'laya_brain_training.py', 'reward_training_v3.py', 'reward_environment.py')
            sources = {**inherited['source_sha256'], **{n: digest(Path(__file__).with_name(n), 'sha256') for n in names}}
            progress('Reserve untouched development, acceptance and final game starts')
            plan = reserve(8, args.test_episodes, training_steps=75, training_tics=300, progress=progress)
            plan['splits']['development'] = plan['splits'].pop('reward_train')
            plan.update(schema='balanced_laya_correction_v2', parent=str(PARENT.resolve()),
                train_sources=[str(DATA.resolve()), str(ORIGINAL.resolve())],
                teacher='Reuse verified TRAIN Laya probabilities; no new teacher calls or engine annotations.',
                training='All complete TRAIN episodes, balanced by episode then teacher action. Parent-policy rehearsal gives successful TRAIN episodes fivefold priority. Fixed 80 epochs; fractions 0.25, 0.5 and 1.0.',
                selection='Eight fresh DEVELOPMENT games choose among three fractions and unchanged parent. Require zero lost parent hits, strictly more hits, and nondecreasing return; otherwise choose parent.',
                acceptance='Independent eight-game gate: no lost parent hits and nondecreasing return. Final test: at least two additional hits, at most one lost parent hit and nondecreasing return. Both required for preferred experimental pointer; readiness threshold remains 80% hits.',
                policy_inputs='Descending neuronal states only. Graph, timing, neuron identities and normalization remain inherited.',
                final_test_used_for_parameter_selection=False, graph_updates=False)
            atomic_json(folder/'plan.json', plan)
            atomic_json(folder/'experience.json', train+validation)
            progress(f'Replay {len(train)} TRAIN and {len(validation)} validation frames through the verified parent brain')
            raw = replay(encoder, train)
            vraw = replay(encoder, validation)
            np.savez_compressed(folder/'neural-replay.npz', train=raw, validation=vraw, validation_mask=mask)
            progress('Fit balanced Laya targets while anchoring successful parent behavior')
            candidates, fit = fit_candidates(raw, train, parent, (vraw[mask], targets(validation)[mask]))
            atomic_json(folder/'fit.json', fit)
            for name, actor in candidates.items():
                actor.save(folder/(name+'-actor.npz'))
            development = {}
            for name, actor in [('parent', parent), *candidates.items()]:
                progress(f'Development gameplay: {name}; eight fixed starts')
                development[name] = evaluate(encoder, actor, plan['splits']['development'], name+' development')
                atomic_json(folder/'development.json', development)
            selected, selection = select_development(development)
            actor = parent if selected == 'parent' else candidates[selected]
            atomic_json(folder/'selection.json', {'selected': selected, 'audit': selection,
                        'criterion': plan['selection'], 'selected_before_gate_and_final_test': True})
            results = {}
            for split in ('gate', 'test'):
                results[split] = {}
                for name, decoder in [('continued_parent', parent), ('candidate', actor)]:
                    progress(f'Frozen {split}: {name}; selected fraction={selected}')
                    if selected == 'parent' and name == 'candidate':
                        results[split][name] = results[split]['continued_parent']
                    else:
                        results[split][name] = evaluate(encoder, decoder, plan['splits'][split], name+' '+split)
                    atomic_json(folder/(split+'.json'), results[split])
            gate = comparison(results['gate']['candidate'], results['gate']['continued_parent'])
            final = comparison(results['test']['candidate'], results['test']['continued_parent'])
            accepted = (selected != 'parent' and gate['lost_parent_successes'] == 0 and gate['mean_return_change'] >= 0
                        and final['net_additional_hits'] >= 2 and final['lost_parent_successes'] <= 1
                        and final['mean_return_change'] >= 0)
            scores = {name: summarize(rows) for name, rows in results['test'].items()}
            acceptance = {'passed': bool(accepted), 'gate': gate, 'final': final,
                          'criterion': plan['acceptance'], 'parent_fallback': not accepted,
                          'identical_parent_results_reused': selected == 'parent'}
            atomic_json(folder/'acceptance.json', acceptance)
            save_candidate(folder, encoder, actor)
            for name in ('physiology.json', 'ablation.json'):
                (folder/name).write_bytes((PARENT/name).read_bytes())
            # The inherited ablation describes the parent, not this readout correction.
            ablation = read(folder/'ablation.json')
            ablation['finding'] = 'Inherited parent-only ablation; not a measurement of this balanced readout.'
            atomic_json(folder/'ablation.json', ablation)
            detail = fit['candidates'].get(selected)
            recovery = {'version': 2, 'parent': str(PARENT.resolve()), 'method': fit['method'],
                'selected_fraction': selected, 'reused_train_frames': len(train),
                'validation_frames': int(mask.sum()), 'teacher_live_calls': 0,
                'teacher_fit': detail, 'changed_edges': 0,
                'decoder_change_l2': float(np.linalg.norm(actor.weights-parent.weights)),
                'development_selection': selection, 'acceptance': acceptance,
                'scope': 'Offline readout correction on existing Laya labels. The inherited reward-trained anatomical circuit and response times remain fixed.'}
            atomic_json(folder/'recovery-training.json', recovery)
            report = {'schema': SCHEMA, 'status': 'completed', 'parent': str(PARENT.resolve()),
                'parent_report_sha256': digest(PARENT/'report.json', 'sha256'),
                'neurons': len(encoder.ids), 'readout_cells': len(actor.root_ids), 'actions': ACTIONS, 'tasks': ['basic'],
                'laya_training': inherited['laya_training'], 'inherited_reward_training': inherited.get('reward_training'),
                'recovery_training': recovery, 'training_frames': len(train), 'validation_frames': int(mask.sum()),
                'plasticity': {'changed_edges': 0, 'changed_groups': 0, 'timing_trained': False},
                'four_steps': {'balanced_teaching': 'completed', 'parent_retention_check': 'completed',
                               'frozen_game_test': 'completed', 'independent_physiology': 'not_validated'},
                'test': scores, 'test_by_task': {name: {'basic': score} for name, score in scores.items()},
                'game_gate_passed': bool(accepted and scores['candidate']['kill_episode_rate'] >= .8),
                'improvement_gate_passed': bool(accepted), 'biological_validation_passed': False,
                'promoted': False, 'scope': recovery['scope'], 'source_sha256': sources,
                'input_sha256': {str(p.resolve()): digest(p, 'sha256') for p in
                                [PARENT/'report.json', DATA/'report.json', ORIGINAL/'report.json']}}
            artifacts = ['plan.json', 'experience.json', 'neural-replay.npz', 'fit.json', 'development.json',
                'selection.json', 'gate.json', 'test.json', 'acceptance.json', 'recovery-training.json',
                'weights.npz', 'circuit.npz', 'circuit.json', 'actor.npz', 'physiology.json', 'ablation.json',
                'conservative-actor.npz', 'balanced-actor.npz', 'full-actor.npz']
            report['output_sha256'] = {name: digest(folder/name, 'sha256') for name in artifacts}
            for name, sha in sources.items():
                if digest(Path(__file__).with_name(name), 'sha256') != sha:
                    raise ValueError('Training implementation changed: '+name)
            atomic_json(folder/'report.json', report)
            load_any(folder)
            pointer = lambda path: {'path': str(path.resolve()), 'sha256': digest(path/'report.json', 'sha256')}
            atomic_json(ROOT/'latest.json', pointer(folder))
            atomic_json(ROOT/'selected.json', {**pointer(folder if accepted else PARENT),
                        'experiment': str(folder.resolve()), 'accepted': bool(accepted)})
            atomic_json(ROOT/'progress.json', {'status': 'completed', 'folder': str(folder), 'accepted': bool(accepted)})
            print(json.dumps({'folder': str(folder), 'selected': selected, 'acceptance': acceptance, 'test': scores}, indent=2), flush=True)
        except BaseException as error:
            atomic_json(ROOT/'progress.json', {'status': 'failed', 'folder': str(folder), 'error': str(error)})
            raise


if __name__ == '__main__':
    main()
