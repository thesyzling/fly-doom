"""Exact decision labels, collection integrity, and separate candidate training."""

import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from flydoom import action_memory, feedback_training, student
from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.human_feedback import FeedbackStore, read_feedback
from flydoom.live_activity import frame_png
from flydoom.live_brain import LiveSession, make_handler
from test_live_brain import catalog, sample
from test_student_refine import artifacts


def observation(width=10, sequence=2):
    return {"sequence": sequence, "episode": 1, "decision": sequence - 1,
            "features": np.arange(width, dtype=np.float32) / 100,
            "frame": frame_png(np.zeros((8, 8, 3), dtype=np.uint8)),
            "action": "WAIT", "probabilities": [.7, .1, .1, .1]}


def test_exact_features_and_frame_survive_relabel_and_remove(tmp_path):
    store = FeedbackStore(tmp_path / "labels", {"student_sha256": "fixture"}, "train", [123], 10)
    snapshot = observation()
    assert not store.directory.exists()
    assert store.save(snapshot, "ATTACK")["count"] == 1
    original = store.report["records"][0]
    with np.load(store.directory / original["file"]) as saved:
        np.testing.assert_array_equal(saved["features"], snapshot["features"])
        assert saved["frame_png"].tobytes().startswith(b"\x89PNG")
    assert store.save(snapshot, "MOVE_RIGHT")["count"] == 1
    assert store.report["records"][0]["file"] == original["file"]
    assert store.save(snapshot, None)["count"] == 0
    record = json.loads((store.directory / "manifest.json").read_text())["records"][0]
    assert record["revisions"] == ["ATTACK", "MOVE_RIGHT", None]
    assert record["applied_action"] == "WAIT"


def test_feedback_guards_and_read_only_model(catalog, tmp_path):
    session = LiveSession(catalog, tmp_path / "live", {"student_sha256": "fixture"})
    snapshot = {**sample(catalog), **observation()}
    session.samples.append(snapshot)
    before = {k: v.clone() for k, v in session.model.state_dict().items()}
    with pytest.raises(ValueError, match="changed"):
        session.save_feedback(1, "ATTACK")
    with pytest.raises(ValueError, match="changed"):
        session.save_feedback(True, "ATTACK")
    session.busy = True
    with pytest.raises(ValueError, match="Pause"):
        session.save_feedback(2, "ATTACK")
    session.busy = False
    with pytest.raises(ValueError, match="valid"):
        session.save_feedback(2, "JUMP")
    snapshot["decision"] = 0
    with pytest.raises(ValueError, match="Step once"):
        session.save_feedback(2, "ATTACK")
    snapshot["decision"] = 1
    assert session.save_feedback(2, "ATTACK")["count"] == 1
    for key, value in before.items():
        torch.testing.assert_close(value, session.model.state_dict()[key], rtol=0, atol=0)
    assert session.samples[-1]["action"] == "WAIT"
    session.report["disconnected"] = True
    with pytest.raises(ValueError, match="Disconnected"):
        session.save_feedback(2, "ATTACK")


def test_feedback_http_persists_exact_sequence_and_rejects_other_origins(catalog, tmp_path):
    session = LiveSession(catalog, tmp_path / "live", {"student_sha256": "fixture"})
    session.samples.append({**sample(catalog), **observation()})
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    def post(origin, sequence=2):
        return urlopen(Request(base + "/api/feedback", data=json.dumps({"sequence": sequence, "action": "ATTACK"}).encode(),
                               headers={"Content-Type": "application/json", "Origin": origin}))
    try:
        with pytest.raises(HTTPError) as error:
            post("https://other.example")
        assert error.value.code == 403
        with post(base) as response:
            assert json.load(response)["label"] == "ATTACK"
        with pytest.raises(HTTPError) as error:
            post(base, 3)
        assert error.value.code == 400
        assert len(session.feedback.report["records"]) == 1
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_feedback_loader_rejects_overlap_and_tampering(tmp_path):
    parent = {"student_sha256": "fixture", "input_size": 10}
    a = FeedbackStore(tmp_path / "a", parent, "train", [10], 10)
    b = FeedbackStore(tmp_path / "b", parent, "validation", [10], 10)
    a.save(observation(sequence=2), "ATTACK")
    b.save(observation(sequence=3), "WAIT")
    with pytest.raises(ValueError, match="overlap"):
        read_feedback([a.directory, b.directory], parent)
    b.report["seeds"] = [11]
    b.report["records"][0]["seed"] = 11
    write_json(b.directory / "manifest.json", b.report)
    parts, _ = read_feedback([a.directory, b.directory], parent)
    assert parts["train"]["actions"].tolist() == [3]
    (a.directory / a.report["records"][0]["file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        read_feedback([a.directory, b.directory], parent)


@pytest.mark.parametrize("memory", [False, True])
def test_feedback_training_changes_candidate_only_and_preserves_runtime_schema(artifacts, tmp_path, monkeypatch, memory):
    dataset, _, checkpoint = artifacts
    parent = json.loads((checkpoint / "report.json").read_text())
    model = student.SpikingReadout(8, 8)
    if memory:
        model = action_memory.expand(model)
        parent.update(schema=action_memory.SCHEMA, memory_feature_names=list(action_memory.NAMES),
                      memory_implementation_sha256=digest(action_memory.__file__, "sha256"), input_size=25)
    with torch.no_grad():
        model.input.weight.fill_(.05)
        model.input.bias.fill_(1.1)
        model.readout.weight.zero_()
        model.readout.bias.zero_()
    save_file(model.state_dict(), str(checkpoint / "student.safetensors"))
    parent["student_sha256"] = digest(checkpoint / "student.safetensors", "sha256")
    parent["teacher_accepted"] = False
    write_json(checkpoint / "report.json", parent)
    before = {name: digest(checkpoint / name, "sha256") for name in ("report.json", "student.safetensors")}
    source = json.loads((dataset / "manifest.json").read_text())
    rng = np.random.default_rng(42)
    replay = {}
    paths = []
    for split, seed in (("train", 60000), ("validation", 60002)):
        states = [action_memory.ActionMemory().observe() for _ in range(8)]
        features = rng.uniform(.1, .9, (8, 8)).astype(np.float32)
        features[:, -4:] = [1, 0, 0, 0]
        replay[split] = {"features": features, "actions": np.full(8, 3, dtype=np.int64), "states": states}
        store = FeedbackStore(tmp_path / split, parent, split, [seed], parent["input_size"])
        for i in range(8):
            snap = observation(parent["input_size"], i + 2)
            x = features[i].copy()
            x[0] += .123
            snap["features"] = action_memory.augment(x[None, :], [states[i]])[0] if memory else x
            store.save(snap, "ATTACK")
        paths.append(store.directory)
    # This fixture has distinct validation inputs; no test split is provided to training.
    monkeypatch.setattr(feedback_training, "read_prepared", lambda _: (source, replay))
    output = tmp_path / "candidate"
    report = feedback_training.train(paths, checkpoint, output, dataset=dataset, epochs=3, learning_rate=.001)
    assert report["status"] == "completed" and report["selected_epoch"] > 0
    assert report["validation"]["balanced_nll"] < report["initial_feedback_validation"]["balanced_nll"]
    assert not report["teacher_accepted"] and not report["test_evaluated"] and not report["game_skill_validated"]
    assert not report["laya_used_during_training"] and report["schema"] == parent["schema"]
    after = load_file(str(output / "student.safetensors"))
    assert not torch.equal(after["readout.bias"], model.readout.bias)
    for key in ("mean", "scale"):
        torch.testing.assert_close(after[key], model.state_dict()[key], rtol=0, atol=0)
    for name, checksum in before.items():
        assert digest(checkpoint / name, "sha256") == checksum
    # A validation input repeated in training is excluded regardless of its label.
    assert feedback_training.exclude_seen(np.array([[1, 2], [3, 4]]), np.array([[1, 2]])).tolist() == [False, True]


def test_epoch_zero_wins_when_validation_is_unchanged(monkeypatch):
    model = student.SpikingReadout(8, 8)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    part = {"features": np.ones((8, 8), dtype=np.float32), "actions": np.zeros(8, dtype=np.int64)}
    parts = {s: part for s in ("train", "validation")}
    actual = feedback_training.metrics
    monkeypatch.setattr(feedback_training, "metrics", lambda p, y: {**actual(p, y), "balanced_nll": 1.})
    report = feedback_training.fit(model, parts, parts, epochs=2, learning_rate=.001, seed=0)
    assert report["selected_epoch"] == 0
    for k, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[k], rtol=0, atol=0)
