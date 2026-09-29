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
    assert set(state) == {"image", "rows", "previous_action", "blue_region", "vision_format"}
    assert state["rows"][0] == "255 0 0 0 0 0 0 0"
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


@pytest.mark.parametrize("experimental", [False, True])
def test_student_training_changes_weights_and_records_frozen_graph(tmp_path, monkeypatch, experimental):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file
    from flydoom import student
    torch.set_num_threads(1)
    dataset = tmp_path / "data"
    make_prepared(dataset)
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "head.safetensors").write_bytes(b"fixture")
    write_json(teacher / "report.json", {"teacher_accepted": not experimental, "status": "completed", "base": {"fixture": True},
               "dataset_sha256": digest(dataset / "manifest.json", "sha256"),
               "head_sha256": digest(teacher / "head.safetensors", "sha256")})
    monkeypatch.setattr(student, "load_base", lambda *args: (object(), {"fixture": True}))
    monkeypatch.setattr(student, "restore_head", lambda *args: None)
    monkeypatch.setattr(student, "encode_states", lambda agent, states: states)
    monkeypatch.setattr(student, "probabilities", lambda agent, states: np.eye(4, dtype=np.float32)[[x["label_for_test_fixture"] for x in states]])
    output = tmp_path / "student"
    original_teacher_hash = digest(teacher / "report.json", "sha256")
    report = student.train(dataset, tmp_path / "base", teacher, output, epochs=20, hidden=8,
                           experimental_teacher=experimental, evaluate_test=not experimental)
    assert report["student_trained"] and not report["connectome_weights_trained"]
    assert report["status"] == "completed"
    torch.manual_seed(7)
    initial = student.SpikingReadout(8, 8)
    saved = load_file(str(output / "student.safetensors"))
    assert not torch.equal(initial.input.weight, saved["input.weight"])
    assert report["experimental_teacher"] == experimental
    assert report["teacher_accepted"] == (not experimental)
    assert digest(teacher / "report.json", "sha256") == original_teacher_hash
    assert report["validation_zero_neural_features"]["samples"] == 40
    if experimental:
        assert not report["test_evaluated"] and "test" not in report
    else:
        assert report["test"]["samples"] == 40


def test_teacher_training_freezes_encoder_and_restores_selected_head(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from flydoom import laya_teacher
    from flydoom import laya_features
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
    monkeypatch.setattr(laya_features, "cache_features", lambda agent, items, batch_size: items)
    monkeypatch.setattr(laya_features, "cached_logits", logits)
    output = tmp_path / "teacher"
    report = laya_teacher.train(dataset, tmp_path / "base", output, epochs=2, batch_size=8)
    assert report["status"] == "completed"
    assert report["selected_epoch"] in (0, 1, 2)
    assert torch.equal(encoder_before, model.encoder.weight)
    assert model.encoder.weight.grad is None
    assert not torch.equal(head_before, model.head.weight)
    _, parts = read_prepared(dataset)
    measured = laya_teacher.metrics(laya_teacher.probabilities(agent, parts["test"]["states"]), parts["test"]["actions"])
    assert measured == report["test"]
    continued = laya_teacher.train(dataset, tmp_path / "base", tmp_path / "continued", epochs=1,
                                   batch_size=8, initial_teacher=output, evaluate_test=False)
    assert continued["warm_start"]["parent_head_sha256"] == report["head_sha256"]
    assert continued["warm_start"]["parent_selected_epoch"] == report["selected_epoch"]
    assert not continued["test_evaluated"] and "test" not in continued
    assert continued["initial_validation"] == report["validation"]


def test_recording_cancel_keeps_report_and_closes_game(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from flydoom import learning_data
    closed = []
    game = SimpleNamespace(set_seed=lambda seed: None, new_episode=lambda: None,
                           close=lambda: closed.append(True), is_episode_finished=lambda: False,
                           get_state=lambda: SimpleNamespace(screen_buffer=np.zeros((8, 8, 3), dtype=np.uint8)))
    pygame = SimpleNamespace(init=lambda: None, quit=lambda: None, QUIT=1, KEYDOWN=2,
        display=SimpleNamespace(set_mode=lambda size: None, set_caption=lambda title: None),
        font=SimpleNamespace(Font=lambda *args: None),
        event=SimpleNamespace(get=lambda: [SimpleNamespace(type=1)]))
    for index, name in enumerate(("K_LEFT", "K_a", "K_RIGHT", "K_d", "K_SPACE", "K_ESCAPE", "K_RETURN", "KEYUP", "WINDOWFOCUSLOST", "WINDOWFOCUSGAINED"), 10):
        setattr(pygame, name, index)
    monkeypatch.setitem(sys.modules, "pygame", pygame)
    monkeypatch.setattr(learning_data, "make_game", lambda seed: (game, list(ACTIONS[1:])))
    output = tmp_path / "recording"
    report = learning_data.record(output, episodes=5)
    assert report["status"] == "interrupted"
    assert report["episodes"] == []
    assert closed == [True]
    assert json.loads((output / "manifest.json").read_text())["status"] == "interrupted"


def keyboard_fixture():
    from types import SimpleNamespace
    names = ("QUIT", "KEYDOWN", "KEYUP", "WINDOWFOCUSLOST", "WINDOWFOCUSGAINED",
             "K_LEFT", "K_a", "K_RIGHT", "K_d", "K_SPACE", "K_ESCAPE", "K_RETURN")
    pg = SimpleNamespace(**{name: index for index, name in enumerate(names)})
    return pg, lambda kind, key=None: SimpleNamespace(type=kind, key=key)


def test_keyboard_latches_short_taps_and_preserves_holds_and_aliases():
    from flydoom.recording_input import KeyboardCapture
    pg, event = keyboard_fixture()
    capture = KeyboardCapture(pg)
    capture.update([event(pg.KEYDOWN, pg.K_RETURN), event(pg.KEYDOWN, pg.K_SPACE), event(pg.KEYUP, pg.K_SPACE)])
    assert capture.consume() == (3, [0, 0, 1], True)
    assert capture.consume() == (0, [0, 0, 0], False)
    capture.update([event(pg.KEYDOWN, pg.K_LEFT), event(pg.KEYDOWN, pg.K_a), event(pg.KEYUP, pg.K_LEFT)])
    assert capture.consume() == (1, [2, 0, 0], False)
    assert capture.consume() == (1, [0, 0, 0], False)
    capture.update([event(pg.KEYDOWN, pg.K_RIGHT)])
    assert capture.consume()[0] == 0
    capture.update([event(pg.KEYDOWN, pg.K_SPACE)])
    assert capture.consume()[0] == 3


def test_keyboard_focus_loss_pauses_and_discards_stale_input():
    from flydoom.recording_input import KeyboardCapture
    pg, event = keyboard_fixture()
    capture = KeyboardCapture(pg)
    capture.update([event(pg.KEYDOWN, pg.K_SPACE)])
    assert capture.paused and capture.preview() == 3
    capture.update([event(pg.KEYUP, pg.K_SPACE), event(pg.KEYDOWN, pg.K_LEFT)])
    assert capture.preview() == 1
    capture.update([event(pg.KEYDOWN, pg.K_RETURN)])
    assert capture.consume()[0] == 0
    capture.update([event(pg.KEYDOWN, pg.K_LEFT), event(pg.WINDOWFOCUSLOST)])
    assert capture.paused and capture.preview() == 0
    with pytest.raises(ValueError, match="paused"):
        capture.consume()
    capture.update([event(pg.KEYDOWN, pg.K_RETURN)])
    assert capture.paused
    capture.update([event(pg.WINDOWFOCUSGAINED)])
    assert capture.paused
    capture.update([event(pg.KEYDOWN, pg.K_RETURN)])
    assert capture.consume() == (0, [0, 0, 0], False)


def test_coverage_ignores_test_labels_and_rejects_missing_active_examples():
    from flydoom.learning_quality import coverage
    counts = {"train": [100, 10, 10, 10], "validation": [15, 5, 5, 5]}
    assert coverage(counts)["ready"]
    assert coverage({**counts, "test": [0, 0, 0, 0]}) == coverage(counts)
    report = coverage({**counts, "validation": [32, 3, 1, 2]})
    assert not report["ready"]
    assert any("MOVE_RIGHT" in reason for reason in report["reasons"])


def test_class_weights_equalize_total_training_contribution_and_metrics_expose_collapse():
    from flydoom.laya_teacher import class_weights, metrics
    labels = np.repeat(np.arange(4), [80, 10, 5, 5])
    weights = class_weights(labels)
    np.testing.assert_allclose(np.bincount(labels) * weights, [25] * 4)
    probs = np.tile([0.97, 0.01, 0.01, 0.01], (100, 1))
    result = metrics(probs, labels)
    assert result["accuracy"] == 0.8 and result["balanced_accuracy"] == 0.25
    assert result["predicted_counts"] == [100, 0, 0, 0]
    assert result["balanced_nll"] > result["nll"]


def test_visual_summary_uses_pixels_and_shuffle_moves_all_visual_fields():
    from flydoom.laya_teacher import shuffle_images
    left = np.zeros((80, 80, 3), dtype=np.uint8)
    left[20:40, 10:20, 2] = 255
    right = left[:, ::-1].copy()
    a, b = image_state(left, 1), image_state(right, 2)
    assert "left of center" in a["blue_region"]
    assert "right of center" in b["blue_region"]
    shuffled = shuffle_images([a, b], [1, 0])
    assert shuffled[0]["blue_region"] == b["blue_region"]
    assert shuffled[0]["rows"] == b["rows"]
    assert shuffled[0]["previous_action"] == a["previous_action"]
    left[:] = 0
    left[70:, :, 2] = 255
    assert "No substantial" in image_state(left, 0)["blue_region"]


def test_insufficient_teacher_data_stops_before_loading_model(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from flydoom import laya_teacher
    dataset = tmp_path / "data"
    report = make_prepared(dataset)
    name = report["episodes"][0]["file"]
    with np.load(dataset / name) as data:
        contents = dict(data)
    contents["actions"] = np.zeros(40, dtype=np.int64)
    np.savez_compressed(dataset / name, **contents)
    report["episodes"][0]["sha256"] = digest(dataset / name, "sha256")
    write_json(dataset / "manifest.json", report)
    monkeypatch.setattr(laya_teacher, "load_base", lambda *args: pytest.fail("Must not load model"))
    result = laya_teacher.train(dataset, tmp_path / "base", tmp_path / "teacher")
    assert result["status"] == "insufficient_demonstration_coverage"
    assert not result["teacher_accepted"]


def test_recording_applies_and_saves_a_tap_released_before_sampling(tmp_path, monkeypatch):
    import sys
    import itertools
    from types import SimpleNamespace
    from flydoom import learning_data
    pg, event = keyboard_fixture()
    pg.init = lambda: None
    pg.quit = lambda: None
    pg.display = SimpleNamespace(set_mode=lambda size: None, set_caption=lambda title: None)
    pg.font = SimpleNamespace(Font=lambda *args: None)
    pg.event = SimpleNamespace(get=lambda: [event(pg.KEYDOWN, pg.K_RETURN),
        event(pg.KEYDOWN, pg.K_SPACE), event(pg.KEYUP, pg.K_SPACE)])
    game = SimpleNamespace(set_seed=lambda seed: None, new_episode=lambda: None, close=lambda: None,
        is_episode_finished=lambda: False, get_total_reward=lambda: 0,
        get_state=lambda: SimpleNamespace(screen_buffer=np.zeros((8, 8, 3), dtype=np.uint8)))
    applied = []

    def apply(action, tics):
        applied.append(action)
        return 0

    game.make_action = apply
    monkeypatch.setitem(sys.modules, "pygame", pg)
    monkeypatch.setattr(learning_data, "make_game", lambda *args: (game, ["ATTACK", "MOVE_RIGHT", "MOVE_LEFT"]))
    monkeypatch.setattr(learning_data, "draw_recording", lambda *args: None)
    clock = itertools.count()
    monkeypatch.setattr(learning_data, "perf_counter", lambda: next(clock))
    output = tmp_path / "recording"
    report = learning_data.record(output, episodes=5, max_decisions=1)
    assert report["status"] == "completed" and len(report["episodes"]) == 5
    assert applied == [[1, 0, 0]] * 5
    with np.load(output / "episode-0000.npz") as data:
        assert data["actions"].tolist() == [3]
        assert data["latched_taps"].tolist() == [True]
        assert data["key_press_counts"].tolist() == [[0, 0, 1]]


def test_pipeline_checks_data_before_expensive_graph_replay(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from flydoom import learning
    calibration = tmp_path / "calibration.json"
    write_json(calibration, {"engineering_gate_passed": True,
        "simulation_sha256": digest(Path(learning.__file__).with_name("simulation.py"), "sha256")})
    base = tmp_path / "base"
    base.mkdir()
    write_json(base / "download.json", {"sha256": {}})
    monkeypatch.setattr(learning, "record", lambda *args, **kwargs: {"status": "completed"})
    monkeypatch.setattr(learning, "inspect_recording", lambda *args: {
        "ready": False, "counts": {}, "reasons": ["Need more active examples"]})
    monkeypatch.setattr(learning, "prepare", lambda *args: pytest.fail("Must not replay the graph"))
    output = tmp_path / "run"
    learning.run_pipeline(output, episodes=5, max_episodes=5, base=base, calibration=calibration)
    assert json.loads((output / "pipeline.json").read_text())["status"] == "insufficient_demonstration_coverage"


def test_resume_appends_without_changing_saved_episodes_or_split_assignments(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from flydoom import learning_data
    seeds = []
    game = SimpleNamespace(set_seed=lambda seed: seeds.append(seed), new_episode=lambda: None,
        close=lambda: None, is_episode_finished=lambda: False, get_total_reward=lambda: 0,
        get_state=lambda: SimpleNamespace(screen_buffer=np.zeros((8, 8, 3), dtype=np.uint8)),
        make_action=lambda *args: 0)
    monkeypatch.setattr(learning_data, "make_game", lambda *args: (game, list(ACTIONS[1:])))
    output = tmp_path / "recording"
    first = learning_data.record(output, episodes=5, seed=123, max_decisions=1, random_smoke=True)
    original = [ep.copy() for ep in first["episodes"]]
    seeds.clear()
    resumed = learning_data.record(output, episodes=5, max_episodes=10, seed=123,
                                   max_decisions=1, random_smoke=True, resume=True)
    assert seeds == list(range(128, 133))
    assert resumed["episodes"][:5] == original
    for ep in original:
        assert digest(output / ep["file"], "sha256") == ep["sha256"]
    assert len(resumed["episodes"]) == 10
    assert [ep["split"] for ep in resumed["episodes"]] == [split_for_episode(i) for i in range(10)]
    manifest_before = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="seed"):
        learning_data.record(output, episodes=5, max_episodes=15, seed=124,
                             max_decisions=1, random_smoke=True, resume=True)
    assert (output / "manifest.json").read_bytes() == manifest_before
    (output / "episode-0010.npz").write_bytes(b"unlisted partial file")
    with pytest.raises(ValueError, match="overwrite"):
        learning_data.record(output, episodes=5, max_episodes=15, seed=123,
                             max_decisions=1, random_smoke=True, resume=True)
    assert (output / "manifest.json").read_bytes() == manifest_before


def test_resume_pipeline_inherits_original_recording_options(tmp_path, monkeypatch):
    from flydoom import learning
    source = {"minimum_episodes": 30, "maximum_episodes": 100, "episodes": [{"seed": 30000}]}
    monkeypatch.setattr(learning, "read_recording", lambda *args: source)
    captured = {}
    monkeypatch.setattr(learning, "run_pipeline", lambda output, **kwargs: captured.update(kwargs))
    learning.resume_pipeline(tmp_path)
    assert captured["resume"] and captured["seed"] == 30000
    assert captured["episodes"] == 30 and captured["max_episodes"] == 100
    assert captured["max_decisions"] == 75


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
        get_game_variable=lambda variable: 0,
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
    assert [e["kills"] for e in result["episodes"]] == [0, 0]
    assert [e["action_counts"]["ATTACK"] for e in result["episodes"]] == [2, 2]
    assert applied == [[1, 0, 0]] * 16
    assert closed == [True]
