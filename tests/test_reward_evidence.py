import json

import numpy as np
import pytest
from threading import Condition

from flydoom.data import digest
from flydoom.reward_evidence import episode_evidence


def test_frozen_reward_desk_cannot_silently_start_the_legacy_learning_algorithm():
    from flydoom.adaptive_live import AdaptiveSession
    session = AdaptiveSession.__new__(AdaptiveSession)
    session.condition = Condition()
    session.report = {'reward_training': {'batches': 8}}
    with pytest.raises(ValueError, match='frozen reward-trained'):
        session.start_learning(2, 'basic')


def test_desk_rejects_a_superseded_experiment_before_opening_a_server(tmp_path, capsys):
    from flydoom.integrated_desk import main
    (tmp_path/'review.json').write_text(json.dumps({'eligible_for_use': False}))
    with pytest.raises(SystemExit) as error:
        main(['--checkpoint', str(tmp_path), '--no-browser'])
    assert error.value.code == 2
    assert 'superseded' in capsys.readouterr().err


def test_reward_evidence_preserves_delayed_credit_and_rejects_changed_records(tmp_path):
    path = tmp_path/'episode.npz'
    np.savez(path, actions=[1, 3], rewards=[-4., 90.], returns=[.851, .9], values=[.2, .3],
             probabilities=np.full((2, 8), .125))
    manifest = tmp_path/'experience.json'
    manifest.write_text(json.dumps([{'path': str(path), 'sha256': digest(path, 'sha256'),
                                    'policy': 'candidate', 'seed': 42, 'kills': 1, 'termination': 'engine_finished'}]))
    (tmp_path/'report.json').write_text(json.dumps({'reward_training': {'batches': 1},
        'output_sha256': {'experience.json': digest(manifest, 'sha256')}}))
    result = episode_evidence(tmp_path, 0)
    assert result['records'][0]['action'] == 'MOVE_LEFT'
    assert result['records'][0]['reward'] == -4
    assert result['records'][0]['advantage_before_scaling'] == pytest.approx(.651)
    with pytest.raises(ValueError, match='Unknown'):
        episode_evidence(tmp_path, -1)
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='identity'):
        episode_evidence(tmp_path, 0)
