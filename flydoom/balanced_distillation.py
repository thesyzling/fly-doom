"""Episode/action-balanced teacher fitting with successful-parent rehearsal."""
from collections import Counter

import numpy as np

from flydoom.adaptive_policy import AdaptiveActor
from flydoom.laya_brain_fit import targets
from flydoom.retinal_train import metrics


def probabilities(weights, features):
    z = features @ weights.T
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def balanced_weights(rows):
    """Equalize episode contributions first, then teacher argmax action mass."""
    if not rows or any(row['split'] != 'train' for row in rows):
        raise ValueError('Fitting accepts TRAIN observations only')
    lengths = Counter(row['seed'] for row in rows)
    labels = targets(rows).argmax(1)
    weights = np.array([1/lengths[row['seed']] for row in rows])
    mass = np.bincount(labels, weights=weights, minlength=8)
    weights /= mass[labels]
    weights /= weights.sum()
    return weights


def fit_candidates(raw, rows, parent, validation):
    """Fit a fixed budget; no validation or game outcomes select an epoch."""
    y = targets(rows)
    x = np.array([parent.features(r) for r in raw])
    teacher_weight = balanced_weights(rows)
    old = probabilities(parent.weights, x)
    lengths = Counter(row['seed'] for row in rows)
    successful = {row['seed'] for row in rows if row.get('behavior') == 'parent_brain'
                  and row.get('effect', {}).get('kill_delta', 0) > 0}
    retain_weight = np.array([(5 if row['seed'] in successful else 1)/lengths[row['seed']]
                              for row in rows], dtype=float)
    retain_weight /= retain_weight.sum()
    w = parent.weights.copy()
    m, v = np.zeros_like(w), np.zeros_like(w)
    for epoch in range(1, 81):
        p = probabilities(w, x)
        # Preserve the prior policy, especially its complete successful trajectories.
        gradient = ((p-y)*teacher_weight[:, None] + 2*(p-old)*retain_weight[:, None]).T @ x
        gradient += .002*(w-parent.weights)
        m = .9*m + .1*gradient
        v = .999*v + .001*gradient**2
        w -= .002*m/(1-.9**epoch)/(np.sqrt(v/(1-.999**epoch))+1e-8)
    candidates, evidence = {}, {}
    success_mask = np.array([row['seed'] in successful for row in rows])
    for name, fraction in [('conservative', .25), ('balanced', .5), ('full', 1.)]:
        actor = AdaptiveActor(parent.indices, parent.root_ids, parent.mean, parent.scale,
                              parent.weights + fraction*(w-parent.weights))
        p = probabilities(actor.weights, x)
        retention = float(np.mean(p.argmax(1)[success_mask] == old.argmax(1)[success_mask])) if success_mask.any() else None
        candidates[name] = actor
        evidence[name] = {'fraction': fraction, 'epochs': 80,
            'initial': {'train': metrics(parent, raw, y), 'validation': metrics(parent, *validation)},
            'final': {'train': metrics(actor, raw, y), 'validation': metrics(actor, *validation)},
            'weighted_train_kl': float(np.sum(teacher_weight*np.sum(y*np.log(np.maximum(y, 1e-15)/np.maximum(p, 1e-15)), axis=1))),
            'successful_parent_action_retention': retention,
            'successful_parent_episodes': len(successful),
            'decoder_change_l2': float(np.linalg.norm(actor.weights-parent.weights))}
    return candidates, {'candidates': evidence,
        'teacher_action_weight_mass': np.bincount(y.argmax(1), weights=teacher_weight, minlength=8).tolist(),
        'method': 'Episode-normalized, action-balanced soft Laya supervision; parent-policy KL rehearsal '
                  'with fivefold successful-episode priority; fixed 80 epochs and three predeclared step fractions.'}


def select_development(results):
    """Require every parent success to survive before considering additional hits."""
    parent = results['parent']
    seeds = [row['seed'] for row in parent]
    score = lambda rows: (sum(row['kills'] > 0 for row in rows), sum(row['return'] for row in rows))
    best, best_score = 'parent', score(parent)
    audits = {}
    for name, rows in results.items():
        if [row['seed'] for row in rows] != seeds:
            raise ValueError('Development policies require identical ordered starts')
        lost = sum(a['kills'] > 0 and b['kills'] == 0 for a, b in zip(parent, rows))
        current = score(rows)
        eligible = lost == 0 and current[0] > score(parent)[0] and current[1] >= score(parent)[1]
        audits[name] = {'hits': current[0], 'return_sum': current[1], 'lost_parent_successes': lost, 'eligible': eligible}
        if eligible and current > best_score:
            best, best_score = name, current
    return best, audits
