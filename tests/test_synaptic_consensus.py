import numpy as np
import pytest

from flydoom.synaptic_consensus import consensus_direction, episode_losses, objective


def test_consensus_requires_multiple_episodes_and_rejects_split_signs():
    gradients = np.array([[1, 1, 1, 0], [1, 1, -1, 0], [1, -1, 1, 0], [0, -1, -1, 10]])
    direction, info = consensus_direction(gradients)
    assert direction[0] < 0
    assert np.count_nonzero(direction) == 1
    assert info["selected_edges"] == 1


def test_minority_magnitude_cannot_reverse_sign_consensus():
    direction, _ = consensus_direction([[1], [1], [1], [-100]])
    assert direction[0] < 0


def test_sparse_cap_and_zero_gradients():
    direction, info = consensus_direction(np.ones((4, 10)), limit=2)
    assert np.flatnonzero(direction).tolist() == [0, 1]
    assert info["supported_edges"] == 10
    empty, info = consensus_direction(np.zeros((4, 10)))
    assert not empty.any() and info["selected_edges"] == 0


def test_equal_episode_loss_and_duplicate_exclusion():
    episodes = [[{"seed": 1, "teacher": {"probabilities": [1., 0.]}},
                 {"seed": 1, "teacher": {"probabilities": [1., 0.]}, "score": False}],
                [{"seed": 2, "teacher": {"probabilities": [0., 1.]}}]]
    rows = episode_losses(episodes, {"probabilities": [[.5, .5], [.01, .99], [.1, .9]]})
    assert rows[0]["kl"] == pytest.approx(np.log(2))
    assert rows[0]["samples"] == 1
    assert objective(rows, [{"kl": 2.}]) == pytest.approx(.75 * (np.log(2) - np.log(.9)) / 2 + .5)


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError): consensus_direction([[np.nan]])
    with pytest.raises(ValueError): episode_losses([[]], {"probabilities": []})
    with pytest.raises(ValueError): episode_losses([[{}]], {"probabilities": []})
