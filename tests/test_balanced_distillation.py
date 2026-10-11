import numpy as np
import pytest

from flydoom.balanced_distillation import balanced_weights, select_development


def test_balancing_reduces_long_episode_dominance_and_equalizes_actions():
    def row(seed, label):
        return {'seed': seed, 'split': 'train', 'teacher': {'probabilities': np.eye(8)[label].tolist()}}
    rows = [row(1, 1)]*8 + [row(2, 1)]*2 + [row(3, 2)]*4
    weight = balanced_weights(rows)
    assert weight[:8].sum() == pytest.approx(weight[8:10].sum())
    assert weight[:10].sum() == pytest.approx(.5)
    assert weight[10:].sum() == pytest.approx(.5)
    with pytest.raises(ValueError, match='TRAIN'):
        balanced_weights([{**rows[0], 'split': 'test'}])


def test_selection_rejects_higher_hit_count_when_parent_success_is_lost():
    def rows(hits):
        return [{'seed': i, 'kills': hit, 'return': 100 if hit else -300} for i, hit in enumerate(hits)]
    name, audit = select_development({'parent': rows([1, 0, 0]), 'forgetting': rows([0, 1, 1]), 'retained': rows([1, 1, 0])})
    assert name == 'retained'
    assert not audit['forgetting']['eligible']
    assert audit['forgetting']['lost_parent_successes'] == 1
    assert select_development({'parent': rows([1, 0]), 'worse': rows([0, 0])})[0] == 'parent'
