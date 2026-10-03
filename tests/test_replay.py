"""Replay integrity, engine alignment, and the local artifact HTTP boundary."""

import hashlib
from http.server import ThreadingHTTPServer
import json
from threading import Thread
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest

from flydoom.replay import ReplayLibrary, build_replay, safe_path
from flydoom.research import research_handler


def test_library_checks_hashes_bounds_paths_and_serves_bytes(tmp_path):
    folder = tmp_path / "replays" / "fixture"
    folder.mkdir(parents=True)
    body = b"fixture image bytes"
    (folder / "frame.png").write_bytes(body)
    value = {"schema": "vision_replay_v1", "title": "fixture", "capture": "native_run_capture",
             "episodes": [{"kills": 1}], "decisions": [{"frames": ["frame.png"]}],
             "frame_sha256": {"frame.png": hashlib.sha256(body).hexdigest()}}
    manifest = folder / "replay.json"
    manifest.write_text(json.dumps(value))
    library = ReplayLibrary(tmp_path)
    key = library.listing()[0]["id"]
    assert library.listing()[0]["kills"] == 1
    server = ThreadingHTTPServer(("127.0.0.1", 0), research_handler(SimpleNamespace(replays=library)))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/replay?id=" + key) as response:
            frame_url = json.load(response)["decisions"][0]["frames"][0]
        with urlopen(base + frame_url) as response:
            assert response.headers["Content-Type"] == "image/png"
            assert response.read() == body
        for suffix in ("/api/replay?id=unknown", f"/api/replay-frame?id={key}&frame=-1",
                       f"/api/replay-frame?id={key}&frame=1"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + suffix)
            assert error.value.code == 400
        (folder / "frame.png").write_bytes(b"changed")
        with pytest.raises(ValueError, match="checksum"):
            library.frame(key, 0)
        value["frame_sha256"] = {"../outside.png": "fake"}
        manifest.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="leaves"):
            library.frame(key, 0)
        with pytest.raises(ValueError, match="leaves"):
            safe_path(folder, tmp_path / "outside.png")
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("wrong_hash,wrong_reward", [(False, False), (True, False), (False, True)])
def test_reconstruction_requires_original_observation_and_outcome(monkeypatch, tmp_path, wrong_hash, wrong_reward):
    from flydoom import replay
    run = tmp_path / "original"
    run.mkdir()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    probs = [0, 0, 0, 1]
    np.savez(run / "state.npz", student_probabilities=probs, teacher_probabilities=probs,
             student_activity=np.array([.5]))
    row = dict(episode=1, decision=1, seed=7, action="ATTACK", student_action="ATTACK",
               probabilities=probs, applied_probabilities=probs, alpha=.8, agreement=True,
               brain_spikes=3, observation="state",
               observation_sha256=hashlib.sha256((run / "state.npz").read_bytes()).hexdigest(),
               vision={"probabilities": probs, "frame_sha256": "wrong" if wrong_hash else hashlib.sha256(image.tobytes()).hexdigest()})
    row["return"] = 0 if wrong_reward else 99
    (run / "decisions.jsonl").write_text(json.dumps(row))
    (run / "report.json").write_text(json.dumps(dict(schema="vision_fly_research_v1", status="completed",
        alpha=.8, checkpoint_sha256="fixture", vision={}, episodes=[dict(seed=7, decisions=1, kills=1)])))
    applied, closed = [], []
    game = SimpleNamespace(set_seed=lambda seed: None, new_episode=lambda: None,
        get_state=lambda: None if applied else SimpleNamespace(screen_buffer=image),
        get_total_reward=lambda: 99 if applied else 0, get_game_variable=lambda _: int(bool(applied)),
        is_episode_finished=lambda: bool(applied), make_action=lambda action, tics: applied.append(action),
        close=lambda: closed.append(True))
    monkeypatch.setattr(replay, "make_game", lambda *args: (game, ["ATTACK", "MOVE_RIGHT", "MOVE_LEFT"]))
    if wrong_hash or wrong_reward:
        with pytest.raises(ValueError, match="differs"):
            build_replay(run, tmp_path / "output")
    else:
        result = build_replay(run, tmp_path / "output")
        effect = result["decisions"][0]["effect"]
        assert effect["kill_delta"] == 1 and effect["episode_finished"]
        assert not effect["terminal_image_available"]
        assert len(result["decisions"][0]["frames"]) == 1
        assert applied == [[1, 0, 0]]
    assert closed == [True]
