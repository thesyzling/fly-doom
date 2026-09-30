"""Observer invariance, exact edges, snapshot timing, and interactive controls."""

from http.server import ThreadingHTTPServer
import json
from threading import Thread
from time import monotonic, sleep
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from scipy import sparse

torch = pytest.importorskip("torch")
from flydoom.bridge import NeuralController, NeuralMapping
from flydoom.live_activity import ConnectionCatalog, ReadoutTelemetry, frame_png
from flydoom.live_brain import LiveSession, load_checkpoint, make_handler
from flydoom.simulation import LIFNetwork
from flydoom.student import SpikingReadout


@pytest.fixture
def catalog():
    ids = np.array([720575940625448968 + i for i in range(4)], dtype=np.uint64)
    weights = sparse.csr_matrix(([2., -3., .5], ([1, 2, 1], [0, 1, 2])), shape=(4, 4))
    mapping = NeuralMapping(np.array([0]), np.array([0]), tuple(np.array([i]) for i in (1, 2, 3)))
    controller = NeuralController(LIFNetwork(weights), mapping)
    model = SpikingReadout(10, hidden=8).eval()
    rows = [{"root_id": str(root), "super_class": "visual_projection" if i == 0 else "descending",
             "cell_type": f"Test cell {i}", "known_nt": "", "top_nt": "fixture"} for i, root in enumerate(ids)]
    return ConnectionCatalog(ids, controller, model, rows)


def sample(catalog):
    return {"sequence": 2, "voltage": np.array([-50., -49., -48., -47.]), "counts": np.array([1, 2, 3, 4]),
            "drive": np.arange(8, dtype=np.float32), "activity": np.arange(8, dtype=np.float32) / 8,
            "probabilities": [.1, .2, .3, .4], "previous": 2}


def test_exact_ids_and_target_row_source_column_direction(catalog):
    key = str(catalog.ids[1])
    data = catalog.inspect(key, sample(catalog))
    assert data["id"] == "720575940625448969"
    assert data["incoming_count"] == 2 and data["outgoing_count"] == 1
    edges = [e for e in data["edges"] if e["channel"] == "fly synaptic weight"]
    triples = {(e["source"]["id"], e["target"]["id"], e["weight"]) for e in edges}
    assert (str(catalog.ids[0]), key, 2.) in triples
    assert (key, str(catalog.ids[2]), -3.) in triples
    assert (str(catalog.ids[2]), key, .5) in triples
    assert data["voltage_mv"] == -49. and data["spikes"] == 2
    assert catalog.feature(0) == (key, "voltage")
    assert catalog.feature(3) == (key, "spike rate")
    assert catalog.feature(8) == ("previous:MOVE_RIGHT", "previous action indicator")
    with pytest.raises(KeyError):
        catalog.inspect("student:-1", sample(catalog))


def test_hooks_observe_the_exact_forward_pass_without_changing_it(catalog):
    model = catalog.model
    x = torch.linspace(-1, 1, 10).unsqueeze(0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        reference = model(x)
        telemetry = ReadoutTelemetry(model)
        observed = model(x)
    torch.testing.assert_close(reference, observed, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    np.testing.assert_allclose(telemetry.activity * 8, np.rint(telemetry.activity * 8), rtol=0, atol=0)
    np.testing.assert_allclose(telemetry.activity @ catalog.action_weights.T + catalog.action_bias,
                               observed[0].numpy(), atol=1e-7)
    telemetry.close()
    assert not model.input._forward_hooks and not model.readout._forward_hooks


def test_action_contributions_are_from_actual_activity(catalog):
    s = sample(catalog)
    result = catalog.inspect("action:ATTACK", s)
    expected = catalog.action_weights[3] @ s["activity"] + catalog.action_bias[3]
    assert result["logit"] == pytest.approx(expected)
    assert sum(e["current_logit_contribution"] for e in result["edges"]) + result["bias"] == pytest.approx(expected)


def wait_until(predicate):
    end = monotonic() + 3
    while not predicate():
        if monotonic() > end:
            pytest.fail("Timed out waiting for the observer worker")
        sleep(.01)


def test_step_pause_and_game_observation_alignment(catalog, tmp_path, monkeypatch):
    from flydoom import live_brain
    applied, closed = [], []
    game = SimpleNamespace(set_seed=lambda seed: None, new_episode=lambda: None,
        is_episode_finished=lambda: len(applied) >= 8, get_total_reward=lambda: -len(applied),
        get_game_variable=lambda variable: 0,
        get_state=lambda: SimpleNamespace(screen_buffer=np.full((8, 8, 3), len(applied), dtype=np.uint8)),
        make_action=lambda action, tics: applied.append(action), close=lambda: closed.append(True))
    monkeypatch.setattr(live_brain, "make_game", lambda *args: (game, ["ATTACK", "MOVE_RIGHT", "MOVE_LEFT"]))
    monkeypatch.setattr(live_brain.student, "load_base", lambda *args: pytest.fail("Observer must not load Laya"))
    with torch.no_grad():
        catalog.model.readout.weight.zero_()
        catalog.model.readout.bias.copy_(torch.tensor([0., 0., 0., 5.]))
    session = LiveSession(catalog, tmp_path / "live", {"student_sha256": "fixture"}, episodes=1)
    worker = Thread(target=session.run)
    worker.start()
    try:
        wait_until(lambda: session.state().get("decision") == 0)
        assert not session.samples[-1]["counts"].any()
        session.control("step")
        wait_until(lambda: session.state().get("decision") == 1)
        sleep(.03)
        first = session.state()
        assert first["paused"] and not first["busy"] and first["decision"] == 1
        assert len(applied) == 4 and first["return"] == -4
        assert first["frame"] == frame_png(np.zeros((8, 8, 3), dtype=np.uint8))
        session.control("step")
        worker.join(3)
        assert not worker.is_alive()
        assert session.state()["phase"] == "completed"
        assert session.samples[-1]["frame"] == frame_png(np.full((8, 8, 3), 4, dtype=np.uint8))
        assert applied == [[1, 0, 0]] * 8 and closed == [True]
        assert session.inspect(str(catalog.ids[1]), first["sequence"])["sequence"] == first["sequence"]
        report = json.loads((session.output / "report.json").read_text())
        assert report["episodes"][0]["decisions"] == 2
        assert not report["laya_used_during_play"]
        with pytest.raises(ValueError, match="ended"):
            session.control("run")
    finally:
        with session.condition:
            session.stopped = True
            session.condition.notify_all()
        worker.join(3)


def test_local_http_controls_and_route_limits(catalog, tmp_path):
    session = LiveSession(catalog, tmp_path / "http", {"student_sha256": "fixture"})
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/") as response:
            assert b"DIRECTED CONNECTION EXPLORER" in response.read()
        with urlopen(base + "/api/meta") as response:
            assert json.load(response)["neurons"] == 4
        def control(origin):
            return urlopen(Request(base + "/api/control", data=b'{"command":"step"}',
                headers={"Content-Type": "application/json", "Origin": origin}))
        with pytest.raises(HTTPError) as error:
            control("https://unrelated.example")
        assert error.value.code == 403 and session.steps == 0
        with control(base) as response:
            assert json.load(response)["ok"] and session.steps == 1
        for path in ("/../student.py", "/api/unknown"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + path)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_observer_rejects_incomplete_checkpoint_before_loading_graph(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"schema": "spiking_student_v1", "status": "running"}))
    with pytest.raises(ValueError, match="not complete"):
        load_checkpoint(tmp_path, tmp_path / "calibration", tmp_path / "graph")
