import json
from threading import Condition

import numpy as np
import pytest

from flydoom.recovery_training import training_only_rehearsal, validation_mask


def row(seed, step, split='train', image='image'):
    return {'seed': seed, 'step': step, 'split': split, 'teacher': {'frame_sha256': image}}


def test_validation_rejects_episode_leakage_and_masks_duplicate_images():
    train = [row(1, 0)]
    with pytest.raises(ValueError, match='overlap'):
        validation_mask(train, [row(1, 1, 'validation', 'different')])
    with pytest.raises(ValueError, match='distinct'):
        validation_mask(train, [row(2, 0, 'validation')])
    np.testing.assert_array_equal(validation_mask(train, [row(2, 0, 'validation'), row(2, 1, 'validation', 'new')]), [False, True])


def test_rehearsal_preserves_whole_train_episodes_and_excludes_validation_and_tests(tmp_path, monkeypatch):
    import flydoom.recovery_training as recovery
    rows = [row(1, 0), row(1, 1), row(1, 2), row(2, 0), row(3, 0, 'test'), row(4, 0, 'validation')]
    (tmp_path/'experience.json').write_text(json.dumps(rows))
    monkeypatch.setattr(recovery, 'teaching_checkpoint', lambda _: tmp_path)
    kept, source = training_only_rehearsal(tmp_path, minimum=2)
    assert len(kept) == 3 and source == tmp_path
    assert {r['seed'] for r in kept} == {1}
    assert all(r['split'] == 'train' for r in kept)


def test_recovery_desk_cannot_start_a_different_live_learning_algorithm():
    from flydoom.adaptive_live import AdaptiveSession
    session = AdaptiveSession.__new__(AdaptiveSession)
    session.condition = Condition()
    session.report = {'recovery_training': {'correction_passes': 1}}
    with pytest.raises(ValueError, match='frozen'):
        session.start_learning(2, 'basic')


def test_agreement_baseline_is_chosen_from_train_and_masks_duplicate_validation_images():
    from flydoom.recovery_evidence import teacher_coverage
    def labelled(seed, action, split, image):
        result = row(seed, 0, split, image)
        result['teacher']['probabilities'] = np.eye(8)[action].tolist()
        return result
    rows = [labelled(1, 1, 'train', 'a'), labelled(1, 1, 'train', 'b'),
            labelled(2, 2, 'validation', 'c'), labelled(2, 2, 'validation', 'd'),
            labelled(2, 1, 'validation', 'e'), labelled(2, 1, 'validation', 'a'),
            labelled(3, 2, 'test', 'f')]
    report = teacher_coverage(rows)
    assert report['constant_action_chosen_from_train'] == 'MOVE_LEFT'
    assert report['validation_scored_frames'] == 3
    assert report['constant_action_validation_agreement'] == pytest.approx(1/3)
