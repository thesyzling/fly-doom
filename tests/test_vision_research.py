"""Fusion semantics, paired observation integrity, controls, and pinned artifacts."""

import hashlib
import json
from http.server import ThreadingHTTPServer
from threading import Event, Thread
from time import monotonic, sleep
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from scipy import sparse
import torch

from flydoom.bridge import NeuralController, NeuralMapping
from flydoom.live_activity import ConnectionCatalog
from flydoom.research import ResearchSession, fuse, research_handler
from flydoom.simulation import LIFNetwork
from flydoom.student import SpikingReadout
from flydoom.vision import verified_manifest


def wait_for(predicate):
    deadline = monotonic() + 4
    while not predicate():
        if monotonic() > deadline:
            pytest.fail("Worker did not reach its expected state")
        sleep(.01)


@pytest.fixture
def catalog():
    ids = np.arange(4, dtype=np.uint64) + 720575940625448968
    mapping = NeuralMapping(np.array([0]), np.array([0]), tuple(np.array([i]) for i in (1, 2, 3)))
    controller = NeuralController(LIFNetwork(sparse.csr_matrix((4, 4), dtype=np.float32)), mapping)
    model = SpikingReadout(10, 8).eval()
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.copy_(torch.tensor([0., 0., 0., 5.]))
    rows = [{"root_id": str(root), "super_class": "descending", "cell_type": "fixture"} for root in ids]
    return ConnectionCatalog(ids, controller, model, rows)


def test_fusion_extremes_and_component_accounting():
    v, s = np.array([0, .8, .1, .1]), np.array([.1, .1, .1, .7])
    np.testing.assert_array_equal(fuse(v, s, 1), v)
    np.testing.assert_array_equal(fuse(v, s, 0), s)
    mixed = fuse(v, s, .8)
    assert mixed.argmax() == 1 and mixed.sum() == pytest.approx(1)
    np.testing.assert_allclose(mixed, .8 * v + .2 * s)
    for bad in (-.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            fuse(v, s, bad)
    with pytest.raises(ValueError):
        fuse([1, 2, 3, 4], s, .5)


def fake_game(monkeypatch):
    from flydoom import research
    applied, closed = [], []
    game = SimpleNamespace(set_seed=lambda seed: None, new_episode=lambda: None,
        is_episode_finished=lambda: len(applied) >= 8, get_total_reward=lambda: -len(applied),
        get_game_variable=lambda variable: 0,
        get_state=lambda: SimpleNamespace(screen_buffer=np.full((8, 8, 3), len(applied), dtype=np.uint8)),
        make_action=lambda action, tics: applied.append(action), close=lambda: closed.append(True))
    monkeypatch.setattr(research, "make_game", lambda *args: (game, ["ATTACK", "MOVE_RIGHT", "MOVE_LEFT"]))
    return applied, closed


def test_paired_run_uses_fused_action_but_retains_student_inspection(catalog, monkeypatch, tmp_path):
    applied, closed = fake_game(monkeypatch)
    observed = []
    def predict(frame):
        observed.append(frame.copy())
        return {"probabilities": [0, 1, 0, 0], "seconds": .01, "head": []}
    vision = SimpleNamespace(metadata={"repo": "fixture"}, predict=predict)
    initial = {k: v.clone() for k, v in catalog.model.state_dict().items()}
    session = ResearchSession(catalog, tmp_path / "run", {"student_sha256": "fixture"}, vision, alpha=.8, episodes=1)
    worker = Thread(target=session.run)
    worker.start()
    try:
        wait_for(lambda: session.state().get("decision") == 0)
        session.control("step")
        wait_for(lambda: session.state().get("decision") == 1)
        first = session.state()
        assert first["action"] == "MOVE_LEFT" and first["student_action"] == "ATTACK"
        assert not first["agreement"] and first["paused"]
        assert first["probabilities"][3] > .9 and first["applied_probabilities"][1] > .8
        assert session.inspect("action:ATTACK", first["sequence"])["probability"] > .9
        inspected = session.inspect("student:0", first["sequence"])
        assert inspected["reconstructed_input_drive"] == pytest.approx(inspected["drive"], abs=1e-6)
        for edge in inspected["edges"]:
            if "input_drive_contribution" in edge:
                assert edge["normalized_value"] == pytest.approx(
                    (edge["raw_value"] - edge["training_mean"]) / edge["training_scale"])
                assert edge["input_drive_contribution"] == pytest.approx(edge["weight"] * edge["normalized_value"])
        session.control("step")
        worker.join(4)
        assert not worker.is_alive() and session.phase == "completed"
        assert applied == [[0, 0, 1]] * 8 and closed == [True]
        assert observed[0].max() == 0 and observed[1].min() == 4
        assert session.snapshot(first["sequence"])["decision"] == 1
        rows = [json.loads(line) for line in (session.output / "decisions.jsonl").read_text().splitlines()]
        assert rows[0]["action"] == "MOVE_LEFT" and rows[0]["student_action"] == "ATTACK"
        assert rows[0]["effect"]["reward_delta"] == -4
        assert rows[1]["effect"]["episode_finished"]
        replay = json.loads((session.output / "replay.json").read_text())
        assert replay["capture"] == "native_run_capture"
        assert len(replay["decisions"]) == 2
        assert len(replay["decisions"][0]["frames"]) == 5
        from PIL import Image
        with Image.open(session.output / replay["decisions"][0]["frames"][-1]) as frame:
            assert np.asarray(frame).min() == 4
        with np.load(session.output / (rows[1]["observation"] + ".npz")) as artifact:
            assert artifact["features"][-4:].tolist() == [0, 1, 0, 0]
            np.testing.assert_allclose(artifact["applied_probabilities"], rows[1]["applied_probabilities"])
        for name, value in catalog.model.state_dict().items():
            torch.testing.assert_close(initial[name], value, rtol=0, atol=0)
    finally:
        with session.condition:
            session.stopped = True
            session.condition.notify_all()
        worker.join(4)


def test_stop_during_vision_prevents_another_game_action(catalog, monkeypatch, tmp_path):
    applied, closed = fake_game(monkeypatch)
    entered, release = Event(), Event()
    def predict(frame):
        entered.set()
        release.wait(3)
        return {"probabilities": [0, 1, 0, 0]}
    session = ResearchSession(catalog, tmp_path / "stop", {"student_sha256": "fixture"},
                              SimpleNamespace(metadata={}, predict=predict), episodes=1)
    worker = Thread(target=session.run)
    worker.start()
    try:
        wait_for(lambda: session.phase == "ready")
        session.control("step")
        assert entered.wait(3)
        session.control("step")
        assert session.steps == 0
        session.control("stop")
        release.set()
        worker.join(4)
        assert not applied and closed == [True] and session.phase == "stopped"
    finally:
        release.set()
        with session.condition:
            session.stopped = True
            session.condition.notify_all()
        worker.join(4)


def test_pinned_worker_rejects_changed_sources_and_path_escape(tmp_path):
    model, source = tmp_path / "model", tmp_path / "source"
    model.mkdir(); source.mkdir()
    (source / "module.py").write_bytes(b"original")
    config = {"schema": "laya_vision_download_v1", "sha256": {},
              "source_sha256": {"module.py": hashlib.sha256(b"original").hexdigest()}}
    manifest = model / "download.json"
    manifest.write_text(json.dumps(config))
    verified_manifest(model, source)
    (source / "module.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verified_manifest(model, source)
    config["source_sha256"] = {"../outside.py": "fake"}
    manifest.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="leaves"):
        verified_manifest(model, source)


def test_new_run_endpoint_rejects_foreign_origin_and_unknown_settings():
    created = []
    def new_run(**settings):
        created.append(settings)
        return {"output": "fixture"}
    server = ThreadingHTTPServer(("127.0.0.1", 0), research_handler(SimpleNamespace(new_run=new_run)))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    def post(body, origin=base):
        return urlopen(Request(base + "/api/new-run", data=json.dumps(body).encode(),
                       headers={"Content-Type": "application/json", "Origin": origin}))
    try:
        with pytest.raises(HTTPError) as foreign:
            post({"seed": 123}, "https://unrelated.example")
        assert foreign.value.code == 403 and not created
        with pytest.raises(HTTPError) as unknown:
            post({"unrecognized": 5})
        assert unknown.value.code == 400 and not created
        with post({"seed": 123, "alpha": .8}) as response:
            assert json.load(response)["output"] == "fixture"
        assert created == [{"seed": 123, "alpha": .8}]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
