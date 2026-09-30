"""Balanced sampling, immutable parents, and validation-only model selection."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file

from flydoom import student, student_refine
from flydoom.calibration import write_json
from flydoom.data import digest
from test_learning import make_prepared


def test_balanced_indices_cover_rare_actions_and_are_seeded():
    labels = np.array([0] * 200 + [1, 2, 3])
    first = student_refine.balanced_indices(labels, np.random.default_rng(7))
    second = student_refine.balanced_indices(labels, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.bincount(labels[first]), [8] * 4)
    with pytest.raises(ValueError, match="four"):
        student_refine.balanced_indices(np.array([0, 1, 2]), np.random.default_rng(7))


@pytest.fixture
def artifacts(tmp_path):
    dataset, teacher, parent = (tmp_path / name for name in ("data", "teacher", "parent"))
    source = make_prepared(dataset)
    teacher.mkdir()
    parent.mkdir()
    (teacher / "head.safetensors").write_bytes(b"fixture")
    write_json(teacher / "report.json", {"status": "completed", "teacher_accepted": False,
        "base": {"fixture": True}, "dataset_sha256": digest(dataset / "manifest.json", "sha256"),
        "head_sha256": digest(teacher / "head.safetensors", "sha256")})
    torch.manual_seed(7)
    model = student.SpikingReadout(8, 8)
    save_file(model.state_dict(), str(parent / "student.safetensors"))
    write_json(parent / "report.json", {"schema": "spiking_student_v1", "status": "completed",
        "student_trained": True, "experimental_teacher": True, "hidden": 8, "input_size": 8,
        "student_sha256": digest(parent / "student.safetensors", "sha256"),
        "dataset_sha256": digest(dataset / "manifest.json", "sha256"),
        "calibration_sha256": source["calibration_sha256"], "output_root_ids": source["output_root_ids"],
        "teacher_report_sha256": digest(teacher / "report.json", "sha256"),
        "implementation_sha256": digest(student.__file__, "sha256")})
    return dataset, teacher, parent


def fake_teacher(monkeypatch):
    monkeypatch.setattr(student_refine, "load_base", lambda *a: (object(), {"fixture": True}))
    monkeypatch.setattr(student_refine, "restore_head", lambda *a: None)
    monkeypatch.setattr(student_refine, "encode_states", lambda agent, states: states)
    calls = []
    def probabilities(agent, states):
        calls.append(len(states))
        return np.eye(4, dtype=np.float32)[[s["label_for_test_fixture"] for s in states]]
    monkeypatch.setattr(student_refine, "probabilities", probabilities)
    return calls


def test_refinement_learns_keeps_parent_and_normalization_and_skips_test(artifacts, tmp_path, monkeypatch):
    dataset, teacher, parent = artifacts
    calls = fake_teacher(monkeypatch)
    old = {name: digest(parent / name, "sha256") for name in ("report.json", "student.safetensors")}
    output = tmp_path / "refined"
    report = student_refine.train(dataset, parent, teacher, output, epochs=35, learning_rate=.005)
    assert calls == [40]
    assert report["status"] == "completed" and not report["teacher_accepted"]
    assert report["selected_epoch"] > 0
    assert report["validation"]["balanced_nll"] < report["initial_validation"]["balanced_nll"]
    assert not report["test_evaluated"] and "test" not in report
    assert not report["connectome_weights_trained"]
    before, after = (load_file(str(path / "student.safetensors")) for path in (parent, output))
    assert not torch.equal(before["input.weight"], after["input.weight"])
    for key in ("mean", "scale"):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    for name, value in old.items():
        assert digest(parent / name, "sha256") == value


def test_epoch_zero_is_retained_when_validation_does_not_improve(artifacts, tmp_path, monkeypatch):
    dataset, teacher, parent = artifacts
    fake_teacher(monkeypatch)
    original = student_refine.metrics
    def same_score(probs, labels):
        return {**original(probs, labels), "balanced_nll": 1.0}
    monkeypatch.setattr(student_refine, "metrics", same_score)
    output = tmp_path / "no-improvement"
    report = student_refine.train(dataset, parent, teacher, output, epochs=2)
    assert report["selected_epoch"] == 0
    before, after = (load_file(str(path / "student.safetensors")) for path in (parent, output))
    for name, tensor in before.items():
        torch.testing.assert_close(tensor, after[name], rtol=0, atol=0)


def test_modified_teacher_is_rejected_before_large_model_load(artifacts, tmp_path, monkeypatch):
    dataset, teacher, parent = artifacts
    (teacher / "head.safetensors").write_bytes(b"altered")
    monkeypatch.setattr(student_refine, "load_base", lambda *a: pytest.fail("Must reject before loading Laya"))
    with pytest.raises(ValueError, match="provenance"):
        student_refine.train(dataset, parent, teacher, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
