import json
from types import SimpleNamespace

import numpy as np
import pytest

from flydoom.data import digest
from flydoom.reward_training_v3 import restore_critic, teaching_checkpoint
from flydoom.reward_student_v2 import RewardStudent
from test_reward_student import learner


def test_more_relay_directions_preserve_reward_credit_and_anatomical_constraints():
    old, episode = learner()
    student = RewardStudent(old.bridge, old.actor, old.demo_drive, old.demo_resets,
                            old.demo_targets, relay_directions=8, seed=17)
    before = student.bridge.encoder.model.weights.copy()
    tau = student.bridge.encoder.model.tau.copy()
    result = student.update([episode])
    assert result['accepted'] and result['relay_change_l2'] > 0
    assert result['relay_gradient_directions'] == 8
    assert result['policy_kl'] <= .01
    after = student.bridge.encoder.model.weights
    np.testing.assert_array_equal(after.indices, before.indices)
    np.testing.assert_array_equal(after.indptr, before.indptr)
    np.testing.assert_array_equal(np.sign(after.data), np.sign(before.data))
    np.testing.assert_array_equal(after[0].toarray(), before[0].toarray())
    np.testing.assert_array_equal(student.bridge.encoder.model.tau, tau)


def test_continuation_restores_identical_critic_for_both_controls_and_rejects_tampering(tmp_path):
    snapshot = tmp_path/'candidate-batch-02.npz'
    np.savez(snapshot, critic=np.array([.2, -.3]))
    report = {'reward_training': {'batches': 2}, 'output_sha256': {snapshot.name: digest(snapshot, 'sha256')}}
    learners = {name: SimpleNamespace(critic=np.zeros(2)) for name in ('candidate', 'decoder_only')}
    assert restore_critic(tmp_path, report, learners)
    for student in learners.values():
        np.testing.assert_array_equal(student.critic, [.2, -.3])
    learners['candidate'].critic[0] = 999
    assert learners['decoder_only'].critic[0] == .2
    with snapshot.open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='changed'):
        restore_critic(tmp_path, report, learners)


def test_teacher_records_come_from_verified_original_teacher_not_rl_manifest(tmp_path, monkeypatch):
    import flydoom.reward_training_v3 as training
    teacher, reward = tmp_path/'teacher', tmp_path/'reward'
    reports = {teacher: {'laya_training': True},
               reward: {'laya_training': True, 'reward_training': {'parent': str(teacher)}}}
    monkeypatch.setattr(training, 'load_any', lambda path: (None, None, reports[path]))
    assert teaching_checkpoint(reward) == teacher
    reports[teacher]['reward_training'] = {'parent': str(reward)}
    with pytest.raises(ValueError, match='cycle'):
        teaching_checkpoint(reward)
    reward.mkdir()
    (reward/'review.json').write_text(json.dumps({'eligible_for_use': False}))
    with pytest.raises(ValueError, match='superseded'):
        teaching_checkpoint(reward)


def test_training_time_budget_does_not_leak_into_standard_test_openings(tmp_path, monkeypatch):
    import flydoom.reward_training_v3 as training
    monkeypatch.chdir(tmp_path)
    counter = iter(range(1, 100))
    monkeypatch.setattr(training.secrets, 'randbelow', lambda _: next(counter))
    opened = []
    def fake_game(row):
        opened.append(row.copy())
        return SimpleNamespace(get_state=lambda: SimpleNamespace(screen_buffer=row['seed']), close=lambda: None)
    monkeypatch.setattr(training, 'open_game', fake_game)
    monkeypatch.setattr(training, 'frame_hash', str)
    plan = training.reserve(2, 8, training_steps=150, training_tics=600, progress=lambda _: None)
    assert len(opened) == 18
    assert {r['timeout_tics'] for r in plan['splits']['reward_train']} == {600}
    for name in ('test', 'gate'):
        assert {r['timeout_tics'] for r in plan['splits'][name]} == {300}
    assert len({r['opening_sha256'] for rows in plan['splits'].values() for r in rows}) == 18
    assert plan['evaluation_steps'] == 75
    assert plan['training_steps'] == 150
