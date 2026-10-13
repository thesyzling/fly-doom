import numpy as np
import pytest

from flydoom.adaptive_policy import AdaptiveActor
from flydoom.retention_learning import RetentionStudent, episode_blocks, weighted_kl
from flydoom.reward_student_v2 import distribution
from flydoom.stable_bridge import StableBridge
from test_stable_reward import stable_circuit


def learner_fixture():
    e = stable_circuit(); b = StableBridge(e, [11])
    a = AdaptiveActor([0], [33], [0], [.1]); a.weights[1, 0] = 1
    drive = np.zeros((8, 4, 2, 2), np.float32)
    drive[:, :, 0, :] = np.arange(1, 9)[:, None, None]*.03
    resets = np.array([True, False]*4)
    raw = b.replay(drive, resets, b.neutral()); p = distribution(a, raw)[1]
    rows = [dict(split='train', step=i % 2, seed=i//2, teacher={'probabilities': p[i].tolist()}) for i in range(8)]
    learner = RetentionStudent(b, a, drive, resets, rows, seed=17)
    episode = dict(split='reward_train', seed=1, drive=drive[:2], resets=resets[:2], raw=raw[:2],
                   actions=np.ones(2, int), probabilities=p[:2], rewards=np.array([0., 100.]), kills=1)
    return learner, episode


def test_retention_rejects_test_and_stale_experience():
    learner, episode = learner_fixture()
    with pytest.raises(ValueError, match='reward_train'):
        learner.update([{**episode, 'split': 'test'}])
    assert not learner.memory
    with pytest.raises(ValueError, match='Stale'):
        learner.update([{**episode, 'probabilities': np.full((2, 8), .125)}])
    assert not learner.memory


def test_success_memory_and_full_teacher_checks_preserve_anatomy():
    learner, episode = learner_fixture()
    e = learner.bridge.encoder; before = e.model.weights.copy(); tau = e.model.tau.copy()
    result = learner.update([episode])
    assert result['accepted']
    assert result['full_teacher_guard_frames'] == 8
    assert result['rehearsal_episodes_seen'] == result['rehearsal_total_episodes'] == 4
    assert result['retention_seeds'] == [1] and result['retention_kl_after'] <= .02
    assert result['teacher_train_kl_after'] <= min(result['teacher_train_kl_before']+.03, learner.full_anchor+.1)
    np.testing.assert_array_equal(e.model.weights.indices, before.indices)
    np.testing.assert_array_equal(np.sign(e.model.weights.data), np.sign(before.data))
    np.testing.assert_array_equal(e.model.tau, tau)


def test_causal_teacher_blocks_reject_gaps_and_validation():
    rows = [dict(split='train', step=i % 3, seed=i//3) for i in range(6)]
    assert [b.tolist() for b in episode_blocks(rows)] == [[0, 1, 2], [3, 4, 5]]
    with pytest.raises(ValueError, match='gap'):
        episode_blocks(rows[:1]+rows[2:])
    rows[0]['split'] = 'validation'
    with pytest.raises(ValueError, match='TRAIN'):
        episode_blocks(rows)


def test_weighted_kl_has_no_zero_target_nan():
    target = np.array([[1., 0.], [.25, .75]])
    assert np.isfinite(weighted_kl(target, np.log(np.array([[.5, .5], [.5, .5]])), np.array([.5, .5])))
