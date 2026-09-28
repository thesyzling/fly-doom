import json
from pathlib import Path

import numpy as np
import pytest

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.learning_data import ACTIONS, QUESTION, image_state, read_prepared, read_recording, split_for_episode
from flydoom.laya_teacher import teacher_gate


def make_prepared(directory, *, source="human_keyboard"):
    directory.mkdir()
    episodes = []
    for index, split in enumerate(("train", "validation", "test")):
        name = f"episode-{index}.npz"
        states_file = f"episode-{index}.json"
        labels = np.arange(40, dtype=np.int64) % 4
        features = np.eye(4, dtype=np.float32)[labels]
        np.savez_compressed(directory / name, features=np.concatenate((features, features), axis=1),
                            actions=labels, previous_actions=np.zeros(40, dtype=np.int64))
        write_json(directory / states_file, [{"label_for_test_fixture": int(label), "rows": [str(label)]} for label in labels])
        episodes.append({"file": name, "states_file": states_file, "seed": index, "split": split,
                         "sha256": digest(directory / name, "sha256"),
                         "states_sha256": digest(directory / states_file, "sha256")})
    report = {"schema": "neural_learning_data_v1", "source": source, "status": "completed",
              "actions": ACTIONS, "question": QUESTION, "episodes": episodes,
              "calibration_sha256": "test-calibration", "output_root_ids": ["1", "2"]}
    write_json(directory / "manifest.json", report)
    return report


def test_episode_split_is_separate_and_image_state_has_no_label_or_privileged_game_state():
    assert [split_for_episode(i) for i in range(5)] == ["train", "train", "train", "validation", "test"]
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, 0] = 255
    state = image_state(frame, 2)
    assert set(state) == {"image", "rows", "previous_action"}
    assert state["rows"][0] == "9 0 0 0 0 0 0 0"
    assert state["previous_action"] == "MOVE_RIGHT"


def test_random_recordings_cannot_become_teacher_labels(tmp_path):
    path = tmp_path / "recording"
    path.mkdir()
    write_json(path / "manifest.json", {"schema": "doom_demonstrations_v1", "source": "random_smoke",
                                        "actions": ACTIONS, "episodes": []})
    with pytest.raises(ValueError, match="not eligible"):
        read_recording(path)
    assert read_recording(path, require_human=False)["source"] == "random_smoke"
    make_prepared(tmp_path / "prepared", source="random_smoke")
    with pytest.raises(ValueError, match="Smoke artifacts"):
        read_prepared(tmp_path / "prepared")


def test_prepared_data_checks_hashes_and_rejects_episode_leakage(tmp_path):
    path = tmp_path / "prepared"
    report = make_prepared(path)
    _, parts = read_prepared(path)
    assert parts["train"]["features"].shape == (40, 8)
    report["episodes"][2]["seed"] = report["episodes"][0]["seed"]
    write_json(path / "manifest.json", report)
    with pytest.raises(ValueError, match="Duplicate"):
        read_prepared(path)
    report["episodes"][2]["seed"] = 2
    write_json(path / "manifest.json", report)
    (path / "episode-0.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        read_prepared(path)


def test_teacher_gate_rejects_shortcuts_missing_actions_and_small_validation_sets():
    valid = {"samples": 100, "class_counts": [25, 25, 25, 25], "accuracy": 0.8, "balanced_accuracy": 0.8}
    assert teacher_gate(valid, 0.4, 0.4)
    assert not teacher_gate(valid, 0.8, 0.4)
    assert not teacher_gate(valid, 0.4, 0.8)
    assert not teacher_gate({**valid, "samples": 12}, 0.4, 0.4)
    assert not teacher_gate({**valid, "class_counts": [90, 5, 5, 0]}, 0.4, 0.4)


def test_student_surrogate_gradient_and_safe_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file
    from flydoom.student import SpikingReadout
    torch.manual_seed(5)
    model = SpikingReadout(8, hidden=8)
    inputs = torch.randn(12, 8)
    logits = model(inputs)
    torch.nn.functional.cross_entropy(logits, torch.arange(12) % 4).backward()
    assert model.input.weight.grad.abs().sum() > 0
    assert model.recurrent.weight.grad is not None
    assert torch.isfinite(model.input.weight.grad).all()
    path = tmp_path / "student.safetensors"
    save_file(model.state_dict(), str(path))
    restored = SpikingReadout(8, hidden=8)
    restored.load_state_dict(load_file(str(path)))
    torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)


def test_student_refuses_unaccepted_teacher_before_loading_large_model(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from flydoom import student
    make_prepared(tmp_path / "data")
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    write_json(teacher / "report.json", {"teacher_accepted": False, "status": "completed"})
    monkeypatch.setattr(student, "load_base", lambda *args: pytest.fail("Large model should not load"))
    with pytest.raises(ValueError, match="teacher gate"):
        student.train(tmp_path / "data", tmp_path / "base", teacher, tmp_path / "student")
    assert not (tmp_path / "student").exists()


def test_student_training_changes_weights_and_records_frozen_graph(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file
    from flydoom import student
    torch.set_num_threads(1)
    dataset = tmp_path / "data"
    make_prepared(dataset)
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "head.safetensors").write_bytes(b"fixture")
    write_json(teacher / "report.json", {"teacher_accepted": True, "status": "completed", "base": {"fixture": True},
               "dataset_sha256": digest(dataset / "manifest.json", "sha256"),
               "head_sha256": digest(teacher / "head.safetensors", "sha256")})
    monkeypatch.setattr(student, "load_base", lambda *args: (object(), {"fixture": True}))
    monkeypatch.setattr(student, "restore_head", lambda *args: None)
    monkeypatch.setattr(student, "encode_states", lambda agent, states: states)
    monkeypatch.setattr(student, "probabilities", lambda agent, states: np.eye(4, dtype=np.float32)[[x["label_for_test_fixture"] for x in states]])
    output = tmp_path / "student"
    report = student.train(dataset, tmp_path / "base", teacher, output, epochs=20, hidden=8)
    assert report["student_trained"] and not report["connectome_weights_trained"]
    assert report["status"] == "completed"
    torch.manual_seed(7)
    initial = student.SpikingReadout(8, 8)
    saved = load_file(str(output / "student.safetensors"))
    assert not torch.equal(initial.input.weight, saved["input.weight"])
    assert report["test"]["samples"] == 40


def test_teacher_training_freezes_encoder_and_restores_selected_head(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from flydoom import laya_teacher
    torch.manual_seed(3)
    torch.set_num_threads(1)
    dataset = tmp_path / "data"
    make_prepared(dataset)
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(4, 4)
    model.head = torch.nn.Linear(4, 4)
    model.scorer = torch.nn.Linear(4, 4)
    model.type_emb = torch.nn.Embedding(1, 4)
    agent = SimpleNamespace(model=model, device="cpu")
    encoder_before = model.encoder.weight.detach().clone()
    head_before = model.head.weight.detach().clone()
    monkeypatch.setattr(laya_teacher, "load_base", lambda *args: (agent, {"fixture": True}))
    monkeypatch.setattr(laya_teacher, "encode_states", lambda agent, states: states)

    def logits(agent, states):
        inputs = torch.eye(4)[[int(state["rows"][0]) for state in states]]
        encoded = agent.model.encoder(inputs).detach()
        return agent.model.scorer(agent.model.head(encoded) + agent.model.type_emb.weight)

    monkeypatch.setattr(laya_teacher, "batch_logits", logits)
    output = tmp_path / "teacher"
    report = laya_teacher.train(dataset, tmp_path / "base", output, epochs=2, batch_size=8)
    assert report["status"] == "completed"
    assert report["selected_epoch"] in (1, 2)
    assert torch.equal(encoder_before, model.encoder.weight)
    assert model.encoder.weight.grad is None
    assert not torch.equal(head_before, model.head.weight)
    _, parts = read_prepared(dataset)
    measured = laya_teacher.metrics(laya_teacher.probabilities(agent, parts["test"]["states"]), parts["test"]["actions"])
    assert measured == report["test"]


def test_recording_cancel_keeps_report_and_closes_game(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from flydoom import learning_data
    closed = []
    game = SimpleNamespace(set_seed=lambda seed: None, new_episode=lambda: None,
                           close=lambda: closed.append(True))
    pygame = SimpleNamespace(init=lambda: None, quit=lambda: None, QUIT=1, KEYDOWN=2,
        display=SimpleNamespace(set_mode=lambda size: None, set_caption=lambda title: None),
        font=SimpleNamespace(Font=lambda *args: None),
        event=SimpleNamespace(get=lambda: [SimpleNamespace(type=1)]))
    monkeypatch.setitem(sys.modules, "pygame", pygame)
    monkeypatch.setattr(learning_data, "make_game", lambda seed: (game, list(ACTIONS[1:])))
    output = tmp_path / "recording"
    report = learning_data.record(output, episodes=5)
    assert report["status"] == "interrupted"
    assert report["episodes"] == []
    assert closed == [True]
    assert json.loads((output / "manifest.json").read_text())["status"] == "interrupted"


def test_student_play_uses_saved_weights_without_loading_laya(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from scipy import sparse
    from safetensors.torch import save_file
    from flydoom import student
    from flydoom.bridge import NeuralController, NeuralMapping
    from flydoom.simulation import LIFNetwork
    mapping = NeuralMapping(np.array([0]), np.array([0]),
                            (np.array([1]), np.array([2]), np.array([3])))
    controller = NeuralController(LIFNetwork(sparse.csr_matrix((4, 4))), mapping)
    monkeypatch.setattr(student, "load_calibrated", lambda *args: (np.arange(4), controller, {}))
    monkeypatch.setattr(student, "load_base", lambda *args: pytest.fail("Play must not load Laya"))
    applied, closed = [], []
    game = SimpleNamespace(set_seed=lambda seed: None, new_episode=lambda: None,
        is_episode_finished=lambda: False, get_total_reward=lambda: -len(applied),
        get_state=lambda: SimpleNamespace(screen_buffer=np.zeros((8, 8, 3), dtype=np.uint8)),
        make_action=lambda action, tics: applied.append(action), close=lambda: closed.append(True))
    monkeypatch.setattr(student, "make_game", lambda *args: (game, ["ATTACK", "MOVE_RIGHT", "MOVE_LEFT"]))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = student.SpikingReadout(10, hidden=8)
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.copy_(torch.tensor([0., 0., 0., 5.]))
    save_file(model.state_dict(), str(checkpoint / "student.safetensors"))
    calibration = tmp_path / "calibration.json"
    write_json(calibration, {"fixture": True})
    write_json(checkpoint / "report.json", {"schema": "spiking_student_v1", "status": "completed",
        "student_trained": True, "input_size": 10, "hidden": 8, "output_root_ids": ["1", "2", "3"],
        "student_sha256": digest(checkpoint / "student.safetensors", "sha256"),
        "calibration_sha256": digest(calibration, "sha256"),
        "implementation_sha256": digest(student.__file__, "sha256")})
    result = student.play(checkpoint, calibration, tmp_path / "play", episodes=2, max_decisions=2, visible=False)
    assert result["status"] == "completed" and not result["laya_used_during_play"]
    assert [e["end_reason"] for e in result["episodes"]] == ["decision_limit"] * 2
    assert applied == [[1, 0, 0]] * 16
    assert closed == [True]
