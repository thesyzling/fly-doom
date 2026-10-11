"""Descriptive teaching coverage; never an action policy or training target."""
from collections import Counter

from flydoom.adaptive_policy import ACTIONS


def teacher_coverage(rows):
    """Use a TRAIN-chosen constant action to contextualize validation agreement."""
    train = [row for row in rows if row['split'] == 'train']
    hashes = {row['teacher']['frame_sha256'] for row in train}
    validation = [row for row in rows if row['split'] == 'validation'
                  and row['teacher']['frame_sha256'] not in hashes]
    def counts(items):
        return Counter(max(range(len(ACTIONS)), key=lambda i: row['teacher']['probabilities'][i])
                       for row in items)
    train_counts, validation_counts = counts(train), counts(validation)
    if not train or not validation:
        raise ValueError('Require training and distinct validation teacher observations')
    choice = max(range(len(ACTIONS)), key=lambda i: train_counts[i])
    return {'train_frames': len(train), 'validation_scored_frames': len(validation),
            'train_action_counts': {name: train_counts[i] for i, name in enumerate(ACTIONS)},
            'validation_action_counts': {name: validation_counts[i] for i, name in enumerate(ACTIONS)},
            'constant_action_chosen_from_train': ACTIONS[choice],
            'constant_action_validation_agreement': validation_counts[choice]/len(validation),
            'scope': 'A descriptive teacher-label baseline, not a deployed policy or gameplay score. '
                     'Unequal episode lengths and action frequencies can inflate frame-level agreement.'}
