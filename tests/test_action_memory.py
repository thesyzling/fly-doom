import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from flydoom import action_memory, correction_training, correction_data, live_brain
from flydoom.data import digest
from flydoom.learning_data import image_state
from flydoom.live_activity import ConnectionCatalog
from flydoom.student import SpikingReadout
from flydoom.temporal_data import ObservationHistory
from test_student_refine import artifacts
from test_corrections import collected, teacher_fixture
from test_live_brain import catalog, sample


def test_online_memory_equals_recorded_teacher_history_and_resets():
    memory, teacher = action_memory.ActionMemory(), ObservationHistory()
    previous = 0
    for action in [0, 1, 0, 3, 2, 0]:
        image = image_state(np.zeros((8, 8, 3), dtype=np.uint8), previous)
        observed = teacher.observe(image)
        np.testing.assert_array_equal(action_memory.encode(memory.observe()), action_memory.encode(observed))
        unchanged = action_memory.encode(observed)
        observed.update(future_reward=100, current_action=3, blue_region="changed")
        np.testing.assert_array_equal(action_memory.encode(observed), unchanged)
        memory.advance(action)
        teacher.advance(image, action)
        previous = action
    assert memory.observe()["timing"]["since_shot"] == 3
    assert not action_memory.encode(action_memory.ActionMemory().observe()).any()


def test_expansion_preserves_parent_logits_and_feature_mapping():
    torch.manual_seed(3)
    parent = SpikingReadout(10, 8)
    parent.mean.copy_(torch.arange(10) / 10)
    parent.scale.copy_(torch.arange(10) / 10 + 1)
    model = action_memory.expand(parent)
    x = np.random.default_rng(2).normal(size=(10, 10)).astype(np.float32)
    memory = action_memory.ActionMemory()
    memory.advance(3)
    augmented = action_memory.augment(x, [memory.observe()] * 10)
    with torch.no_grad():
        torch.testing.assert_close(parent(torch.tensor(x)), model(torch.tensor(augmented)), rtol=0, atol=0)
    assert not model.input.weight[:, 6:-4].any()
    np.testing.assert_array_equal(augmented[:, :6], x[:, :6])
    np.testing.assert_array_equal(augmented[:, -4:], x[:, -4:])


def test_memory_training_saves_compatible_mapping_without_changing_parent(collected, artifacts, tmp_path, monkeypatch):
    directory, _ = collected
    dataset, teacher, parent = artifacts
    teacher_fixture(monkeypatch)
    labels = tmp_path / "labels"
    correction_training.label(directory, labels, dataset=dataset, checkpoint=parent, teacher=teacher)
    # The tiny human fixture predates temporal observations; add legitimate empty histories.
    original = correction_training.verify_inputs
    def temporal_inputs(*args):
        source, human, parent_report, teacher_report = original(*args)
        for part in human.values():
            part["states"] = [{**s, **action_memory.ActionMemory().observe()} for s in part["states"]]
        return source, human, parent_report, teacher_report
    monkeypatch.setattr(correction_training, "verify_inputs", temporal_inputs)
    before = digest(parent / "student.safetensors", "sha256")
    output = tmp_path / "memory"
    report = correction_training.train(directory, labels, output, dataset=dataset, checkpoint=parent,
        teacher=teacher, memory=True, epochs=30, learning_rate=.003)
    assert report["status"] == "completed" and report["schema"] == action_memory.SCHEMA
    assert report["input_size"] == 8 + len(action_memory.NAMES)
    assert report["selected_epoch"] > 0 and not report["test_evaluated"]
    assert digest(parent / "student.safetensors", "sha256") == before
    weights = load_file(str(output / "student.safetensors"))
    assert weights["input.weight"][:, 4:-4].abs().sum() > 0
    assert report["memory_implementation_sha256"] == digest(action_memory.__file__, "sha256")


def test_memory_and_previous_action_edges_use_separate_columns(catalog):
    model = action_memory.expand(catalog.model)
    expanded = ConnectionCatalog(catalog.ids, catalog.controller, model, catalog.rows)
    s = sample(expanded)
    memory = action_memory.ActionMemory()
    memory.advance(3)
    s["memory"] = action_memory.encode(memory.observe()).tolist()
    s["previous"] = 3
    count = 2 * len(expanded.outputs)
    assert expanded.feature(count) == ("memory:" + action_memory.NAMES[0], "causal action memory")
    assert expanded.feature(model.input.in_features - 1)[0] == "previous:ATTACK"
    assert expanded.inspect("memory:since_shot", s)["value"] == pytest.approx(1 / 75)
    previous = expanded.inspect("previous:ATTACK", s)
    assert previous["value"] == 1
    for edge in previous["edges"]:
        i = int(edge["target"]["id"].split(":")[1])
        assert edge["weight"] == float(model.input.weight[i, -1].detach())


def test_live_memory_uses_actual_buttons_and_resets_between_explicit_seeds(catalog, tmp_path, monkeypatch):
    model = action_memory.expand(catalog.model)
    model.action_memory = True
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.copy_(torch.tensor([0., 0., 0., 5.]))
    expanded = ConnectionCatalog(catalog.ids, catalog.controller, model, catalog.rows)
    current, seeds = {"ticks": 0}, []
    def act(buttons, tics):
        assert buttons == [1, 0, 0]
        current["ticks"] += tics
    game = SimpleNamespace(set_seed=seeds.append, new_episode=lambda: current.update(ticks=0),
        is_episode_finished=lambda: current["ticks"] >= 8, get_total_reward=lambda: -current["ticks"],
        get_game_variable=lambda variable: 0, close=lambda: None,
        get_state=lambda: SimpleNamespace(screen_buffer=np.zeros((8, 8, 3), dtype=np.uint8)), make_action=act)
    monkeypatch.setattr(live_brain, "make_game", lambda *args: (game, ["ATTACK", "MOVE_RIGHT", "MOVE_LEFT"]))
    session = live_brain.LiveSession(expanded, tmp_path / "live", {"student_sha256": "fixture"},
        seeds=[70000, 70002], autoplay=True)
    session.run()
    assert session.phase == "completed" and seeds == [70000, 70002]
    rows = [json.loads(line) for line in (session.output / "decisions.jsonl").read_text().splitlines()]
    memory = action_memory.ActionMemory()
    memory.advance(3)
    for first, second in (rows[:2], rows[2:]):
        assert not any(first["memory"])
        np.testing.assert_array_equal(second["memory"], action_memory.encode(memory.observe()))
    assert session.inspect("memory:has_shot")["value"] == 1.


def test_modified_memory_implementation_is_rejected_before_graph_load(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "student.safetensors").write_bytes(b"fixture")
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    report = {"schema": action_memory.SCHEMA, "status": "completed", "student_trained": True,
              "student_sha256": digest(checkpoint / "student.safetensors", "sha256"),
              "calibration_sha256": digest(calibration, "sha256"),
              "implementation_sha256": digest(live_brain.student.__file__, "sha256"),
              "memory_implementation_sha256": "wrong"}
    (checkpoint / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(live_brain, "load_calibrated", lambda *a: pytest.fail("No graph load allowed"))
    with pytest.raises(ValueError, match="memory implementation"):
        live_brain.load_checkpoint(checkpoint, calibration, tmp_path)


def test_standard_collection_rejects_memory_checkpoint_before_graph_loading(artifacts, tmp_path, monkeypatch):
    dataset, _, parent = artifacts
    report_path = parent / "report.json"
    report = json.loads(report_path.read_text())
    report["schema"] = action_memory.SCHEMA
    report_path.write_text(json.dumps(report))
    monkeypatch.setattr(correction_data, "load_checkpoint", lambda *a: pytest.fail("No graph load allowed"))
    with pytest.raises(ValueError, match="standard student"):
        correction_data.collect(tmp_path / "unsupported", dataset=dataset, checkpoint=parent)
    assert not (tmp_path / "unsupported").exists()
