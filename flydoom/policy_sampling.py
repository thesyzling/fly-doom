"""Explicit frozen deployment of the categorical policy used during reward training."""


def sampling_enabled(report):
    mode = report.get('decision_sampling')
    if mode not in (None, 'trained_softmax_v1'):
        raise ValueError('Unknown action sampling mode')
    return mode == 'trained_softmax_v1'


def choose_action(actor, raw, rng, *, sample):
    """Sample the learned distribution without changing weights or probabilities."""
    if sample and rng is None:
        raise ValueError('Sampling requires the episode random generator')
    return actor.decide(raw, rng if sample else None)
