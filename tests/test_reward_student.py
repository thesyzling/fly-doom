import numpy as np
import pytest

from flydoom.adaptive_policy import AdaptiveActor
from flydoom.bridge_training import Bridge
from flydoom.reward_student import RewardStudent, discounted_returns, distribution
from test_bridge_training import circuit


def learner(train_relays=True):
    bridge = Bridge(circuit(), [11])
    actor = AdaptiveActor([0], [33], [0], [.1])
    actor.weights[1, 0] = 1
    drive = np.zeros((4, 4, 2, 2), np.float32)
    drive[:, :, 0, :] = np.arange(1, 5)[:, None, None]*.03
    resets = np.array([True, False, False, False])
    raw = bridge.replay(drive, resets, bridge.neutral())
    p = distribution(actor, raw)[1]
    student = RewardStudent(bridge, actor, drive, resets, p, train_relays=train_relays, seed=17)
    episode = {'drive': drive, 'resets': resets, 'raw': raw, 'actions': np.ones(4, int),
               'probabilities': p, 'rewards': np.array([0., 0., 0., 100.]), 'kills': 1}
    return student, episode


def test_delayed_reward_reaches_earlier_actions_without_reordering():
    np.testing.assert_allclose(discounted_returns([0, 0, 100], .5), [25, 50, 100])
    np.testing.assert_array_equal(discounted_returns([1, 2], 0), [1, 2])
    with pytest.raises(ValueError):
        discounted_returns([np.nan])


def test_opposite_outcomes_reverse_decoder_update_with_identical_teacher_targets():
    positive, ep = learner(False)
    negative, en = learner(False)
    en['rewards'] = -ep['rewards']
    before = distribution(positive.actor, ep['raw'])[1][:, 1].mean()
    rp, rn = positive.update([ep]), negative.update([en])
    assert rp['accepted'] and rn['accepted']
    assert distribution(positive.actor, ep['raw'])[1][:, 1].mean() > before
    assert distribution(negative.actor, en['raw'])[1][:, 1].mean() < before
    assert rp['relay_change_l2'] == rn['relay_change_l2'] == 0


def test_reward_update_preserves_anatomy_visual_rows_timing_and_policy_trust_region():
    student, episode = learner()
    weights = student.bridge.encoder.model.weights.copy()
    tau = student.bridge.encoder.model.tau.copy()
    result = student.update([episode])
    assert result['accepted'] and result['relay_change_l2'] > 0
    assert result['policy_kl'] <= .01
    assert result['teacher_train_kl_after'] <= student.anchor_kl+.1
    current = student.bridge.encoder.model.weights
    np.testing.assert_array_equal(current.indices, weights.indices)
    np.testing.assert_array_equal(current.indptr, weights.indptr)
    np.testing.assert_array_equal(np.sign(current.data), np.sign(weights.data))
    np.testing.assert_array_equal(current[0].toarray(), weights[0].toarray())
    np.testing.assert_array_equal(student.bridge.encoder.model.tau, tau)
    np.testing.assert_array_equal(student.gains[~student.active], 1.)
    assert np.isfinite(student.critic).all()


def test_stale_policy_trajectories_cannot_be_reused_as_on_policy_experience():
    student, episode = learner()
    student.actor.weights[3, -1] += 1
    with pytest.raises(ValueError, match='Stale'):
        student.update([episode])


def test_zero_reward_and_matched_teacher_cannot_invent_a_policy_update():
    student, episode = learner()
    episode['rewards'][:] = 0
    weights = student.actor.weights.copy()
    result = student.update([episode])
    assert not result['accepted']
    np.testing.assert_array_equal(student.actor.weights, weights)
    np.testing.assert_array_equal(student.gains, 1.)


def test_weighted_visual_cache_is_bound_to_the_graph_that_produced_it():
    e = circuit()
    original = Bridge(e, [11])
    gains = original.neutral()
    gains[0] = 1.04
    gains[1] = .96
    gains[2] = 1.02
    drive = np.zeros((3, 4, 2, 2), np.float32)
    drive[:, :, 0] = .1
    drive[:, :, 1] = -.02
    resets = [True, False, False]
    expected = original.replay(drive, resets, gains)
    original.apply(gains)
    taught = Bridge(e, [11])
    # A fresh cache includes the taught incoming weights. Merely loading the
    # taught recurrent matrix cannot make a pre-teaching cache equivalent.
    rebuilt = drive.copy()
    rebuilt[:, :, 0] *= gains[0, original.groups]
    rebuilt[:, :, 1] *= gains[1, original.groups]
    np.testing.assert_allclose(taught.replay(rebuilt, resets, taught.neutral()), expected, rtol=1e-6)
    assert not np.allclose(taught.replay(drive, resets, taught.neutral()), expected, rtol=1e-3)
