"""Distillation split integrity and teacher-free student execution."""

import json
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from flydoom import vision_distill
from flydoom.action_memory import NAMES
from flydoom.student import SpikingReadout


def test_validation_excludes_either_rgb_or_input_overlap():
    train = {"features": np.array([[1, 2]]), "rgb_hashes": ["seen"]}
    validation = {"features": np.array([[3, 4], [1, 2], [5, 6]]), "rgb_hashes": ["seen", "new", "novel"]}
    np.testing.assert_array_equal(vision_distill.exclude_training_duplicates(train, validation), [False, False, True])


def test_overlapping_reserved_seeds_rejected_before_reading_runs(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"status": "completed", "train_seeds": [1], "validation_seeds": [2], "evaluation_seeds": [1]}))
    with pytest.raises(ValueError, match="overlap"):
        vision_distill.prepare(path, {})


@pytest.mark.parametrize("damage", [None, "rgb", "npz", "memory", "teacher", "seed"])
def test_preparation_verifies_recording_provenance_and_causal_inputs(tmp_path, damage):
    plan = {"status": "completed", "train_seeds": [1], "validation_seeds": [2], "evaluation_seeds": [3]}
    for split, seed in (("train", 1), ("validation", 2)):
        folder = tmp_path / split
        folder.mkdir()
        image = np.full((8, 8, 3), seed, dtype=np.uint8)
        Image.fromarray(image).save(folder / "state.png")
        vector = np.zeros(23, dtype=np.float32)
        vector[0] = seed
        vector[-4] = 1
        if split == "validation" and damage == "memory":
            vector[-5] = 1
        np.savez(folder / "state.npz", features=vector,
                 teacher_probabilities=[0, 1, 0, 0], student_probabilities=[0, 0, 0, 1])
        row = {"seed": 99 if split == "validation" and damage == "seed" else seed,
               "decision": 1, "observation": "state", "observation_sha256": vision_distill.digest(folder / "state.npz", "sha256"),
               "vision": {"probabilities": [0, 1, 0, 0], "frame_sha256": hashlib.sha256(image.tobytes()).hexdigest()},
               "probabilities": [0, 0, 0, 1], "alpha": 1., "applied_probabilities": [0, 1, 0, 0], "action": "MOVE_LEFT"}
        report = {"schema": "vision_fly_research_v1", "status": "completed", "checkpoint_sha256": "fixture",
                  "vision": dict.fromkeys(("repo", "revision", "source_commit", "manifest_sha256"), "fixture")}
        if split == "validation":
            if damage == "rgb":
                Image.fromarray(image + 1).save(folder / "state.png")
            elif damage == "npz":
                (folder / "state.npz").write_bytes(b"changed")
            elif damage == "teacher":
                report["vision"]["revision"] = "other"
        (folder / "report.json").write_text(json.dumps(report))
        (folder / "decisions.jsonl").write_text(json.dumps(row))
        plan[split + "_runs"] = [str(folder)]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    parent = {"student_sha256": "fixture", "input_size": 23}
    if damage:
        with pytest.raises((ValueError, AssertionError)):
            vision_distill.prepare(path, parent)
    else:
        _, parts, sources, _, removed = vision_distill.prepare(path, parent)
        assert len(parts["validation"]["features"]) == 1 and removed == 0 and len(sources) == 2


def test_distillation_updates_parameters_and_preserves_normalization():
    torch.set_num_threads(2)
    torch.manual_seed(6)
    model = SpikingReadout(3, 8)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    parts = {s: {"features": np.ones((16, 3), dtype=np.float32),
                 "targets": np.tile([0., .1, .1, .8], (16, 1)).astype(np.float32)} for s in ("train", "validation")}
    report = vision_distill.fit(model, parts, epochs=20, learning_rate=.001, seed=7)
    assert report["selected_epoch"] > 0
    assert report["final"]["validation"]["kl"] < report["initial"]["validation"]["kl"]
    torch.testing.assert_close(model.mean, before["mean"], rtol=0, atol=0)
    torch.testing.assert_close(model.scale, before["scale"], rtol=0, atol=0)
    assert not torch.equal(model.readout.weight, before["readout.weight"])


@pytest.mark.parametrize("control", ["connected", "disconnected"])
def test_student_only_evaluation_keeps_causal_history_and_captures_terminal_result(tmp_path, monkeypatch, control):
    applied = []
    game = SimpleNamespace(set_seed=lambda seed: None, new_episode=lambda: None,
        get_state=lambda: None if len(applied) >= 6 else SimpleNamespace(screen_buffer=np.zeros((8, 8, 3), dtype=np.uint8)),
        is_episode_finished=lambda: len(applied) >= 6,
        get_total_reward=lambda: 99 if len(applied) >= 6 else -len(applied),
        get_game_variable=lambda _: int(len(applied) >= 6),
        make_action=lambda action, tics: applied.append(action), close=lambda: None)
    monkeypatch.setattr(vision_distill, "make_game", lambda *args: (game, ["ATTACK", "MOVE_RIGHT", "MOVE_LEFT"]))
    monkeypatch.setattr(vision_distill, "output_features", lambda controller: np.zeros(2, dtype=np.float32))
    controller = SimpleNamespace(control=control, reset=lambda: None, decide=lambda frame: {"voltage_min_mv": -52, "spikes": 3})
    model = SpikingReadout(2 + len(NAMES) + 4, 8).eval()
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.copy_(torch.tensor([0., 0., 0., 10.]))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    report = vision_distill.evaluate_policy(controller, model, {"student_sha256": "fixture"}, [7], tmp_path / "run")
    assert not report["laya_used_during_play"] and report["episodes"][0]["kills"] == 1
    assert report["disconnected"] == (control == "disconnected")
    replay = json.loads((tmp_path / "run/replay.json").read_text())
    assert replay["disconnected"] == (control == "disconnected")
    assert ("Disconnected graph control" in replay["title"]) == (control == "disconnected")
    assert applied == [[1, 0, 0]] * 6
    rows = [json.loads(line) for line in (tmp_path / "run/decisions.jsonl").read_text().splitlines()]
    assert rows[-1]["effect"]["game_tics"] == 2 and rows[-1]["effect"]["episode_finished"]
    assert rows[-1]["vision"]["probabilities"] is None
    with np.load(tmp_path / "run" / (rows[-1]["observation"] + ".npz")) as data:
        assert data["features"][-4:].tolist() == [0, 0, 0, 1]
        assert data["features"][-9] == pytest.approx(1 / 75)
    assert (tmp_path / "run/replay.json").exists()
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
