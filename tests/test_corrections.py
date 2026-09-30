"""Student histories, teacher-only suggestions, split separation, and retention."""

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file

from flydoom import correction_data, correction_training
from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.learning_data import read_prepared
from test_student_refine import artifacts


@pytest.fixture
def collected(artifacts, tmp_path, monkeypatch):
    dataset, teacher, parent = artifacts
    parent_report = json.loads((parent / "report.json").read_text())
    class Policy:
        def __init__(self):
            self.calls = 0
        def __call__(self, x):
            logits = torch.zeros(1, 4)
            logits[0, [1, 3][self.calls % 2]] = 5
            self.calls += 1
            return logits
    controller = SimpleNamespace(reset=lambda: None, decide=lambda frame: {"voltage_min_mv": -52.})
    monkeypatch.setattr(correction_data, "load_checkpoint", lambda *args: (np.array([1, 2]), controller, Policy(), parent_report))
    monkeypatch.setattr(correction_data, "output_features", lambda controller: np.arange(4, dtype=np.float32))
    current = {"ticks": 0, "seed": 56000}
    applied = []
    def advance(action, tics):
        current["ticks"] += tics
        applied.append(action)
    game = SimpleNamespace(set_seed=lambda seed: current.update(seed=seed), new_episode=lambda: current.update(ticks=0),
        is_episode_finished=lambda: current["ticks"] >= 8,
        get_state=lambda: SimpleNamespace(screen_buffer=np.full((8, 8, 3), current["ticks"] + (current["seed"] - 56000) * 10, dtype=np.uint8)),
        make_action=advance, get_total_reward=lambda: -current["ticks"],
        get_game_variable=lambda variable: 0, close=lambda: None)
    monkeypatch.setattr(correction_data, "make_game", lambda *args: (game, ["ATTACK", "MOVE_RIGHT", "MOVE_LEFT"]))
    directory = tmp_path / "collection"
    correction_data.collect(directory, dataset=dataset, checkpoint=parent, train_episodes=2, validation_episodes=1)
    return directory, applied


def teacher_fixture(monkeypatch):
    calls = []
    monkeypatch.setattr(correction_training, "load_base", lambda *a: (object(), {"fixture": True}))
    monkeypatch.setattr(correction_training, "restore_head", lambda *a: None)
    monkeypatch.setattr(correction_training, "encode_states", lambda agent, states: states)
    def probability(agent, states):
        calls.append(len(states))
        return np.stack([np.eye(4, dtype=np.float32)[s["label_for_test_fixture"]]
                         if "label_for_test_fixture" in s else np.array([.05, .05, .85, .05], dtype=np.float32)
                         for s in states])
    monkeypatch.setattr(correction_training, "probabilities", probability)
    return calls


def test_collection_has_student_history_and_cannot_be_used_as_human_data(collected):
    directory, applied = collected
    report, parts = correction_data.read_collection(directory)
    assert report["source"] == "student_policy" and not report["teacher_used_for_actions"]
    assert len(parts["train"]["actions"]) == 4 and len(parts["validation"]["actions"]) == 2
    assert applied == ([[0, 0, 1]] * 4 + [[1, 0, 0]] * 4) * 3
    for part in parts.values():
        assert part["states"][0]["previous_action"] == "WAIT"
        assert part["states"][1]["previous_action"] == "MOVE_LEFT"
        assert part["states"][1]["recent_actions"][-1] == "MOVE_LEFT"
    assert parts["train"]["states"][2]["timing"]["completed_decisions"] == 0
    with pytest.raises(ValueError, match="unsupported"):
        read_prepared(directory)


def test_reader_rejects_privileged_fields_even_with_updated_checksum(collected):
    directory, _ = collected
    report = json.loads((directory / "manifest.json").read_text())
    episode = report["episodes"][0]
    states = json.loads((directory / episode["states_file"]).read_text())
    states[0]["future_reward"] = 100
    write_json(directory / episode["states_file"], states)
    episode["states_sha256"] = digest(directory / episode["states_file"], "sha256")
    write_json(directory / "manifest.json", report)
    with pytest.raises(ValueError, match="causal"):
        correction_data.read_collection(directory)


def test_reserved_seed_overlap_rejected_before_graph_load(artifacts, tmp_path, monkeypatch):
    dataset, _, parent = artifacts
    monkeypatch.setattr(correction_data, "load_checkpoint", lambda *args: pytest.fail("No graph load allowed"))
    with pytest.raises(ValueError, match="disjoint"):
        correction_data.collect(tmp_path / "bad", checkpoint=parent, dataset=dataset, evaluation_seed=56001)


def test_frozen_teacher_labels_keep_the_students_actual_history(collected, artifacts, tmp_path, monkeypatch):
    directory, _ = collected
    dataset, teacher, parent = artifacts
    calls = teacher_fixture(monkeypatch)
    old_hash = digest(directory / "manifest.json", "sha256")
    output = tmp_path / "suggestions"
    result = correction_training.label(directory, output, dataset=dataset, checkpoint=parent, teacher=teacher)
    assert calls == [40, 4, 2]
    assert result["status"] == "completed" and not result["teacher_accepted"]
    assert result["diagnostics"]["train"]["disagreements"] == 4
    assert digest(directory / "manifest.json", "sha256") == old_hash
    _, parts = correction_data.read_collection(directory)
    assert parts["train"]["states"][1]["previous_action"] == "MOVE_LEFT"
    np.testing.assert_array_equal(np.load(output / "train.npy").argmax(1), [2] * 4)


def test_correction_training_updates_readout_without_reloading_teacher_or_changing_parent(collected, artifacts, tmp_path, monkeypatch):
    directory, _ = collected
    dataset, teacher, parent = artifacts
    teacher_fixture(monkeypatch)
    labels = tmp_path / "suggestions"
    correction_training.label(directory, labels, dataset=dataset, checkpoint=parent, teacher=teacher)
    monkeypatch.setattr(correction_training, "load_base", lambda *a: pytest.fail("Training must use saved suggestions"))
    before = load_file(str(parent / "student.safetensors"))
    before_hash = digest(parent / "student.safetensors", "sha256")
    output = tmp_path / "trained"
    report = correction_training.train(directory, labels, output, dataset=dataset, checkpoint=parent,
        teacher=teacher, epochs=30, learning_rate=.003)
    assert report["status"] == "completed" and report["selected_epoch"] > 0
    assert report["validation"]["selection_score"] < report["initial_validation"]["selection_score"]
    assert report["validation"]["human"]["accuracy"] >= report["human_accuracy_floor"]
    assert not report["test_evaluated"] and not report["connectome_weights_trained"]
    assert report["reserved_evaluation_seeds"] == list(range(57000, 57006))
    after = load_file(str(output / "student.safetensors"))
    assert not torch.equal(before["input.weight"], after["input.weight"])
    for key in ("mean", "scale"):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    assert digest(parent / "student.safetensors", "sha256") == before_hash


def test_advice_disagreement_weight_does_not_use_rewards():
    targets = np.eye(4, dtype=np.float32)[[0, 3, 2]]
    np.testing.assert_array_equal(correction_training.correction_weights(targets, np.array([0, 0, 2])), [1, 3, 1])


def test_altered_suggestions_rejected_before_training(collected, artifacts, tmp_path, monkeypatch):
    directory, _ = collected
    dataset, teacher, parent = artifacts
    teacher_fixture(monkeypatch)
    labels = tmp_path / "suggestions"
    correction_training.label(directory, labels, dataset=dataset, checkpoint=parent, teacher=teacher)
    with (labels / "validation.npy").open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ValueError, match="checksum"):
        correction_training.train(directory, labels, tmp_path / "invalid", dataset=dataset, checkpoint=parent, teacher=teacher)
    assert not (tmp_path / "invalid").exists()


def test_merge_preserves_splits_and_file_bytes_and_rejects_duplicate_seeds(collected, tmp_path):
    directory, _ = collected
    other = tmp_path / "other"
    shutil.copytree(directory, other)
    report = json.loads((other / "manifest.json").read_text())
    for key in ("plan", "episodes"):
        for episode in report[key]:
            episode["seed"] += 100
    write_json(other / "manifest.json", report)
    output = tmp_path / "merged"
    result = correction_training.merge_collections([directory, other], output)
    verified, parts = correction_data.read_collection(output)
    assert verified["status"] == "completed" and len(parts["train"]["actions"]) == 8
    assert [e["split"] for e in result["episodes"]] == ["train", "train", "validation"] * 2
    assert result["episodes"][0]["sha256"] == report["episodes"][0]["sha256"]
    assert len(result["source_collections"]) == 2
    with pytest.raises(ValueError, match="Duplicate"):
        correction_training.merge_collections([directory, directory], tmp_path / "bad-merge")


def test_human_validation_regression_retains_parent(collected, artifacts, tmp_path, monkeypatch):
    directory, _ = collected
    dataset, teacher, parent = artifacts
    teacher_fixture(monkeypatch)
    labels = tmp_path / "suggestions"
    correction_training.label(directory, labels, dataset=dataset, checkpoint=parent, teacher=teacher)
    original = correction_training.metrics
    calls = []
    def worsening_validation(probs, actions):
        calls.append(True)
        accuracy = .9 if len(calls) == 1 else .1
        return {**original(probs, actions), "accuracy": accuracy, "balanced_accuracy": accuracy}
    monkeypatch.setattr(correction_training, "metrics", worsening_validation)
    output = tmp_path / "retained"
    report = correction_training.train(directory, labels, output, dataset=dataset, checkpoint=parent, teacher=teacher, epochs=2)
    assert report["selected_epoch"] == 0 and not any(e["eligible"] for e in report["epochs"])
    before, after = (load_file(str(path / "student.safetensors")) for path in (parent, output))
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_validation_deduplication_uses_both_training_sources_and_causal_state():
    def part(values):
        return {"features": np.zeros((len(values), 8), dtype=np.float32), "states": [{"history": v} for v in values]}
    keep = correction_training.novel_validation_rows(part(["human"]), part(["student"]), part(["human", "student", "new"]))
    np.testing.assert_array_equal(keep, [2])
    with pytest.raises(ValueError, match="repeat training"):
        correction_training.novel_validation_rows(part(["human"]), part(["student"]), part(["human", "student"]))
