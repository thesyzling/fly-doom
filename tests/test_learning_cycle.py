"""Learning gates, data identity, temporal inputs and persistent neural replay."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from flydoom import learning_cycle as cycle
from flydoom.data import digest
from flydoom.graph_paths import directed_path
from flydoom.learning_curriculum import learning_target
from flydoom.learning_features import ContrastMotionController
from flydoom.neural_archive import NeuralArchive
from flydoom.named_anatomy import read_ply


def episodes():
    return [{"task": "basic", "seed": 1, "return": -20., "kills": 0, "range_progress": 0.},
            {"task": "distance", "seed": 2, "return": -20., "kills": 0, "range_progress": 0.}]


def test_gate_requires_real_improvement():
    before = episodes()
    assert not cycle.paired_gate(before, before, .3, .1)["accepted"]
    after = episodes(); after[0]["kills"] = 1; after[0]["return"] = 20.
    assert cycle.paired_gate(before, after, .3, .2)["accepted"]
    assert not cycle.paired_gate(before, after, .3, .4)["accepted"]


@pytest.mark.parametrize("metric,value", [("return", -21.), ("kills", -1), ("range_progress", -.1), ("return", float("nan"))])
def test_one_task_cannot_hide_another_task_regression(metric, value):
    before, after = episodes(), episodes()
    after[0]["return"] = 100.; after[1][metric] = value
    assert not cycle.paired_gate(before, after, .2, .1)["accepted"]


def test_gate_rejects_unpaired_or_duplicate_seeds():
    with pytest.raises(ValueError): cycle.paired_gate(episodes(), episodes()[::-1], .2, .1)
    repeated = [episodes()[0], episodes()[0]]
    with pytest.raises(ValueError): cycle.paired_gate(repeated, repeated, .2, .1)


def checkpoint(folder, value):
    folder.mkdir(); (folder / "synapses.npz").write_bytes(value)
    cycle.atomic_json(folder / "report.json", {"status": "completed", "synapses_sha256": digest(folder / "synapses.npz", "sha256")})


def test_promotion_is_atomic_identity_checked_and_reversible(tmp_path, monkeypatch):
    first, second, root = tmp_path / "first", tmp_path / "second", tmp_path / "registry"
    checkpoint(first, b"first"); checkpoint(second, b"second")
    monkeypatch.setattr(cycle, "INITIAL", first)
    previous = cycle.champion(root)
    assert not cycle.promote(root, second, previous, {"accepted": False})
    assert not (root / "registry.json").exists()
    assert cycle.promote(root, second, previous, {"accepted": True})
    assert cycle.champion(root)["path"] == str(second.resolve())
    cycle.rollback(root)
    assert cycle.champion(root) == previous
    (first / "synapses.npz").write_bytes(b"changed")
    with pytest.raises(ValueError): cycle.champion(root)


def test_cycle_lock_prevents_parallel_writers_and_releases_on_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with cycle.CycleLock(tmp_path):
            with pytest.raises(FileExistsError):
                with cycle.CycleLock(tmp_path): pass
            raise RuntimeError("cancelled")
    assert not (tmp_path / "cycle.lock").exists()


def sample(sequence=1):
    return {"sequence": sequence, "decision": sequence-1, "frame": "image", "features": None,
            "voltage": np.array([-52., -48.], np.float32), "counts": np.array([0, 7], np.uint16),
            "current": np.array([.1, .2], np.float32), "refractory": np.array([0, 2], np.int32),
            "activity": np.array([.25], np.float32), "drive": np.array([.6], np.float32)}


def test_archive_restores_every_neural_array_after_restart(tmp_path):
    archive = NeuralArchive(tmp_path / "record", {"graph_sha256": "exact"})
    for seq in range(1, 13): archive.save(sample(seq))
    reopened = NeuralArchive(tmp_path / "record")
    assert reopened.sequences() == list(range(1, 13))
    restored = reopened.load(1)
    for key in ("voltage", "counts", "current", "refractory", "activity", "drive"):
        np.testing.assert_array_equal(restored[key], sample()[key])
    assert restored["features"] is None
    assert len(list((tmp_path / "record").glob("*.tmp"))) == 0


def test_archive_detects_corruption_and_sequence_replacement(tmp_path):
    archive = NeuralArchive(tmp_path / "record", {})
    archive.save(sample())
    with pytest.raises(ValueError): archive.save(sample())
    (tmp_path / "record/0001.npz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"): archive.load(1)
    with pytest.raises(ValueError): archive.load(0)


def test_archive_disk_budget_fails_without_committing_record(tmp_path, monkeypatch):
    monkeypatch.setattr("flydoom.neural_archive.MAX_BYTES", 1)
    archive = NeuralArchive(tmp_path / "record", {})
    with pytest.raises(ValueError, match="budget"): archive.save(sample())
    assert archive.sequences() == []


def test_encoder_carries_motion_only_within_episode():
    controller = ContrastMotionController(SimpleNamespace(network=SimpleNamespace(reset=lambda: None)), .2)
    dark = np.zeros((16, 16, 3), np.uint8); bright = np.full_like(dark, 180)
    first = controller.transform(dark)
    moving = controller.transform(bright)
    stationary = controller.transform(bright)
    assert not np.array_equal(moving, stationary)
    controller.reset()
    np.testing.assert_array_equal(controller.transform(dark), first)
    assert first.dtype == np.uint8 and moving.max() <= 255


def test_reward_target_only_reinforces_measured_positive_progress():
    teacher = np.ones(6) / 6
    unchanged, mix = learning_target(teacher, 5, {"shaping_reward": -.3})
    np.testing.assert_allclose(unchanged, teacher); assert mix == 0
    target, mix = learning_target(teacher, 5, {"shaping_reward": 1.})
    assert mix == .35 and np.argmax(target) == 5
    assert np.isclose(sum(target), 1.)


def test_directed_path_follows_presynaptic_to_postsynaptic_columns():
    # CSR rows are postsynaptic: 0 -> 1 -> 2, never 2 -> 0.
    weights = sparse.csr_matrix([[0., 0., 0.], [2., 0., 0.], [0., -3., 0.]])
    catalog = SimpleNamespace(ids=np.array([10, 20, 30]), lookup={"10": 0, "20": 1, "30": 2},
                              outgoing=weights.tocsc(), edge_details=lambda a, b: {"source": a, "target": b})
    assert directed_path(catalog, "10", "30")["ids"] == ["10", "20", "30"]
    assert not directed_path(catalog, "30", "10")["found"]
    assert not directed_path(catalog, "10", "30", hops=1)["found"]
    assert directed_path(catalog, "10", "20", breadth=0)["ids"] == ["10", "20"]


def test_duplicate_validation_frames_are_replayed_but_not_scored(monkeypatch):
    result = {"probabilities": [[.8, .2], [.1, .9]], "samples": 2, "kl": 7.}
    monkeypatch.setattr(cycle, "full_replay", lambda *args: dict(result))
    rows = [[{"teacher": {"probabilities": [.8, .2]}, "score": True},
             {"teacher": {"probabilities": [.9, .1]}, "score": False}]]
    measured = cycle.replay(None, None, rows)
    assert measured["kl"] == 0 and measured["samples"] == 1 and measured["replayed_frames"] == 2


def test_ply_reader_rejects_nontriangular_geometry():
    header = b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list int int vertex_indices\nend_header\n"
    points = np.zeros((3, 3), "<f4").tobytes()
    vertices, faces = read_ply(header + points + np.array([3, 0, 1, 2], "<i4").tobytes())
    assert vertices.shape == (3, 3) and faces.shape == (1, 3)
    with pytest.raises(ValueError): read_ply(header + points + np.array([4, 0, 1, 2], "<i4").tobytes())


def test_cancel_marker_interrupts_before_new_work(tmp_path):
    cycle.check_cancel(tmp_path)
    (tmp_path / "cancel.request").write_text("cancel")
    with pytest.raises(InterruptedError): cycle.check_cancel(tmp_path)


def test_game_initialization_matches_reused_reservation_engine():
    from flydoom.learning_curriculum import make_game, act
    game = make_game(1)
    try:
        for seed, task in ((194731, "basic"), (194732, "distance")):
            game.set_seed(seed); game.new_episode()
            if task == "distance":
                for _ in range(10): act(game, 4, task)
            expected = cycle.array_hash(game.get_state().screen_buffer)
            fresh = make_game(seed, task)
            try: assert cycle.array_hash(fresh.get_state().screen_buffer) == expected
            finally: fresh.close()
    finally: game.close()


def test_descriptive_benchmark_statistics_preserve_pairs():
    from flydoom.learning_audit import summarize
    before, after = episodes(), episodes()
    for row in before+after: row["terminal"] = False
    after[0]["return"] += 5
    rows = summarize({"gate": {"champion": before, "candidate": after}})
    assert rows[0]["return"]["delta"] == 5
    assert rows[0]["return"]["paired_95_interval"] == [5., 5.]
    assert rows[0]["pairs"] == 1 and rows[0]["truncated_episodes"] == 2


def test_ui_worker_launch_is_bounded_and_cancel_targets_only_its_cycle(tmp_path, monkeypatch):
    from flydoom import learning_job
    monkeypatch.setattr(learning_job, "ROOT", tmp_path)
    monkeypatch.setattr(learning_job, "champion", lambda: {"path": "unchanged"})
    calls = []
    process = SimpleNamespace(pid=123, returncode=None, poll=lambda: None)
    monkeypatch.setattr(learning_job.subprocess, "Popen", lambda argv, **kwargs: calls.append(argv) or process)
    job = learning_job.LearningJob()
    assert job.start()["started"]
    assert calls[0][-2:] == ["--cycles", "1"]
    with pytest.raises(ValueError): job.start()
    folder = tmp_path / "cycle-1"; folder.mkdir()
    cycle.atomic_json(tmp_path / "latest.json", {"status": "running", "output": str(folder)})
    assert job.cancel()["cancellation_requested"]
    assert (folder / "cancel.request").exists()
    assert job.state()["selected_checkpoint"] == {"path": "unchanged"}
