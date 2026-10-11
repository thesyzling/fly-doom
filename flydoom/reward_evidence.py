"""Read retained RL credit records without changing the live game."""
import json
from pathlib import Path

import numpy as np

from flydoom.adaptive_policy import ACTIONS
from flydoom.data import digest


def episode_evidence(checkpoint, index):
    checkpoint = Path(checkpoint).resolve()
    report = json.loads((checkpoint/'report.json').read_text())
    if not report.get('reward_training'):
        raise ValueError('This checkpoint has no reward training records')
    manifest = checkpoint/'experience.json'
    if digest(manifest, 'sha256') != report['output_sha256']['experience.json']:
        raise ValueError('Training manifest changed')
    rows = json.loads(manifest.read_text())
    if type(index) is not int or not 0 <= index < len(rows):
        raise ValueError('Unknown training episode')
    row = rows[index]
    path = Path(row['path']).resolve()
    if path.parent != checkpoint or digest(path, 'sha256') != row['sha256']:
        raise ValueError('Training record identity changed')
    with np.load(path, allow_pickle=False) as data:
        result = [{'decision': i+1, 'action': ACTIONS[int(a)], 'reward': float(data['rewards'][i]),
                   'discounted_return': float(data['returns'][i]), 'predicted_value': float(data['values'][i]),
                   'advantage_before_scaling': float(data['returns'][i]-data['values'][i]),
                   'probability': float(data['probabilities'][i, a])}
                  for i, a in enumerate(data['actions'])]
    return {'index': index, 'episodes': len(rows), 'seed': row['seed'], 'policy': row['policy'],
            'kills': row['kills'], 'termination': row['termination'], 'records': result,
            'scope': 'Retained training episode, not the live run. Discounted returns and values use reward / 100. Advantage is shown before batch normalization and clipping.'}
