import copy
import json

import numpy as np
import pytest

from flydoom.learning_data import ACTIONS, image_state
from flydoom.temporal_data import ObservationHistory, episode_states, timing_baseline


def episode(actions):
    previous = np.asarray([0, *actions[:-1]])
    states = [image_state(np.full((8, 8, 3), i * 20, dtype=np.uint8), previous[i]) for i in range(len(actions))]
    return states, np.asarray(actions), previous


def test_history_excludes_current_and_future_actions_and_future_frames():
    states, actions, previous = episode([0, 1, 3, 2, 0])
    original = episode_states(states, actions, previous)
    changed_states = copy.deepcopy(states)
    changed_actions = actions.copy()
    changed_actions[2:] = [0, 3, 1]
    changed_previous = np.asarray([0, *changed_actions[:-1]])
    for i in range(3, len(states)):
        changed_states[i] = image_state(np.full((8, 8, 3), 255, dtype=np.uint8), changed_previous[i])
    changed = episode_states(changed_states, changed_actions, changed_previous)
    assert changed[:3] == original[:3]
    assert original[2]["recent_actions"] == ["NONE", "WAIT", "MOVE_LEFT"]
    assert original[2]["timing"] == {"completed_decisions": 2, "since_active": 1, "since_shot": "none"}
    assert original[3]["timing"]["since_shot"] == 1


def test_history_resets_per_episode_and_matches_online_observation_order():
    states, actions, previous = episode([1, 0, 3, 0, 2])
    offline = episode_states(states, actions, previous)
    online = ObservationHistory()
    for state, action, expected in zip(states, actions, offline):
        assert online.observe(state) == expected
        online.advance(state, int(action))
    restarted = episode_states(states, actions, previous)
    assert restarted[0]["recent_actions"] == ["NONE"] * 3
    assert restarted[0]["recent_visuals"] == ["unavailable"] * 3
    assert restarted[0]["timing"]["completed_decisions"] == 0
    previous[2] = 2
    with pytest.raises(ValueError, match="causal"):
        episode_states(states, actions, previous)


def test_temporal_image_shuffle_preserves_timing_and_past_buttons():
    from flydoom.laya_teacher import shuffle_images
    states = episode_states(*episode([1, 0, 3, 0, 2]))
    mixed = shuffle_images(states, [4, 3, 2, 1, 0])
    assert mixed[0]["rows"] == states[4]["rows"]
    assert mixed[0]["recent_visuals"] == states[4]["recent_visuals"]
    for key in ("timing", "previous_action", "recent_actions"):
        assert mixed[0][key] == states[0][key]


def test_timing_baseline_does_not_read_validation_actions():
    states, actions, previous = episode([0, 1, 3, 2, 0])
    temporal = episode_states(states, actions, previous)
    train = {"states": temporal, "actions": actions}
    a = timing_baseline(train, {"states": temporal, "actions": actions})
    b = timing_baseline(train, {"states": temporal, "actions": actions[::-1]})
    np.testing.assert_array_equal(a, b)


def test_teacher_rejects_token_overflow_instead_of_silently_truncating(monkeypatch):
    pytest.importorskip("laya")
    from types import SimpleNamespace
    from laya import common
    from flydoom.laya_teacher import encode_states
    monkeypatch.setattr(common, "build_sequence", lambda *args, **kwargs: ([1] * 513, [1, 2, 3, 4]))
    with pytest.raises(ValueError, match="exceeds 512"):
        encode_states(SimpleNamespace(tok=None), [{}])


def test_augmentation_preserves_feature_bytes_labels_and_episode_splits(tmp_path):
    from flydoom.calibration import write_json
    from flydoom.data import digest
    from flydoom.learning_data import QUESTION, read_prepared
    from flydoom.temporal_data import augment
    source = tmp_path / "source"
    source.mkdir()
    manifest = {"schema": "neural_learning_data_v1", "status": "completed", "source": "human_keyboard",
        "actions": ACTIONS, "question": QUESTION, "output_root_ids": ["1", "2"], "episodes": []}
    for index, split in enumerate(("train", "validation", "test")):
        states, actions, previous = episode([0, 1, 2, 3])
        name, state_name = f"episode-{index}.npz", f"episode-{index}.json"
        np.savez_compressed(source / name, features=np.zeros((4, 8), dtype=np.float32),
                            actions=actions, previous_actions=previous)
        write_json(source / state_name, states)
        manifest["episodes"].append({"file": name, "states_file": state_name, "seed": index, "split": split,
            "sha256": digest(source / name, "sha256"), "states_sha256": digest(source / state_name, "sha256")})
    write_json(source / "manifest.json", manifest)
    original_manifest = (source / "manifest.json").read_bytes()
    output = tmp_path / "temporal"
    report = augment(source, output)
    assert not report["neural_features_recomputed"]
    assert (source / "manifest.json").read_bytes() == original_manifest
    for before, after in zip(manifest["episodes"], report["episodes"]):
        assert (source / before["file"]).read_bytes() == (output / after["file"]).read_bytes()
        assert before["split"] == after["split"] and before["seed"] == after["seed"]
        assert before["states_sha256"] != after["states_sha256"]
    _, parts = read_prepared(output)
    for part in parts.values():
        assert part["states"][0]["timing"]["completed_decisions"] == 0
    with pytest.raises(FileExistsError):
        augment(source, output)


def test_warm_start_rejects_changed_dataset_before_model_load(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from flydoom import laya_teacher
    from flydoom.calibration import write_json
    from flydoom.data import digest
    # Use a minimal in-memory dataset; the mismatched manifest hash must reject it.
    data = tmp_path / "data"
    data.mkdir()
    write_json(data / "manifest.json", {"fixture": True})
    labels = np.repeat(np.arange(4), 10)
    monkeypatch.setattr(laya_teacher, "read_prepared", lambda *args: ({}, {
        "train": {"actions": labels}, "validation": {"actions": labels}}))
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "head.safetensors").write_bytes(b"fixture")
    write_json(parent / "report.json", {"schema": "laya_teacher_v1", "status": "completed",
        "dataset_sha256": "different", "head_sha256": digest(parent / "head.safetensors", "sha256")})
    monkeypatch.setattr(laya_teacher, "load_base", lambda *args: pytest.fail("Model must not load"))
    with pytest.raises(ValueError, match="identical dataset"):
        laya_teacher.train(data, tmp_path / "base", tmp_path / "output", initial_teacher=parent)
