"""Live model selection must preserve benchmark identity and active decisions."""

from types import SimpleNamespace
from threading import RLock

import pytest

from flydoom import experiment


@pytest.fixture
def benchmark(monkeypatch):
    episodes = [{"seed": 1, "kills": 1, "return": 20., "decisions": 3,
                 "action_counts": {"WAIT": 0, "MOVE_LEFT": 1, "MOVE_RIGHT": 0, "ATTACK": 2}}]
    plan = {"conditions": [{"name": "parent", "checkpoint": "parent"}],
            "openings": [{"seed": 1, "rgb_sha256": "image"}]}
    report = {"schema": "vision_recovery_gameplay_v1", "status": "completed", "plan": plan,
              "plan_sha256": "plan-hash", "training_during_run": False, "laya_used_during_play": False,
              "results": {"parent": {"episodes": episodes, "opening_hashes": ["image"], "checkpoint_sha256": "weights",
                                      "scores": experiment.score(episodes)}}, "paired": {}}
    monkeypatch.setattr(experiment, "verify_plan", lambda _: plan)
    monkeypatch.setattr(experiment, "read_json", lambda _: report)
    monkeypatch.setattr(experiment, "digest", lambda *_: "plan-hash")
    monkeypatch.setattr(experiment, "verified_parent", lambda _: {"student_sha256": "weights"})
    return report


@pytest.mark.parametrize("damage", [None, "status", "plan_sha256", "training_during_run",
                                    "laya_used_during_play", "score", "seed", "opening", "checkpoint"])
def test_catalog_rejects_incomplete_or_mismatched_benchmark(benchmark, damage):
    if damage in {"status", "plan_sha256"}:
        benchmark[damage] = "changed"
    elif damage in {"training_during_run", "laya_used_during_play"}:
        benchmark[damage] = True
    elif damage == "score":
        benchmark["results"]["parent"]["scores"]["hits"] = 0
    elif damage == "seed":
        benchmark["results"]["parent"]["episodes"][0]["seed"] = 2
    elif damage == "opening":
        benchmark["results"]["parent"]["opening_hashes"] = ["other"]
    elif damage == "checkpoint":
        benchmark["results"]["parent"]["checkpoint_sha256"] = "other"
    if damage:
        with pytest.raises(ValueError):
            experiment.benchmark_catalog("report", "plan")
    else:
        assert experiment.benchmark_catalog("report", "plan")[0]["sha256"] == "weights"


def workbench():
    work = object.__new__(experiment.ExperimentWorkbench)
    work.lock = RLock()
    work.protected_seeds = {100, 200}
    work.entries = {"parent": {"checkpoint": "parent", "sha256": "weights", "report": {"student_sha256": "weights"}}}
    work.checkpoint_report = {"student_sha256": "weights"}
    work.session = SimpleNamespace(busy=False, paused=True)
    work.worker = SimpleNamespace(is_alive=lambda: True)
    return work


@pytest.mark.parametrize("busy,paused", [(True, True), (False, False)])
def test_switch_rejects_inflight_or_running_policy(monkeypatch, busy, paused):
    work = workbench()
    work.session.busy, work.session.paused = busy, paused
    monkeypatch.setattr(experiment, "compatible_model", lambda *_: pytest.fail("Must reject before loading"))
    with pytest.raises(ValueError, match="Pause"):
        work.choose("parent")


@pytest.mark.parametrize("model_id,seed", [("unknown", 1), ("parent", -1), ("parent", True), ("parent", 2**32)])
def test_switch_rejects_unregistered_models_and_invalid_seed(model_id, seed):
    with pytest.raises(ValueError):
        workbench().choose(model_id, seed)


def test_switch_rejects_changed_checkpoint_before_stopping(monkeypatch):
    work = workbench()
    monkeypatch.setattr(experiment, "compatible_model", lambda *_: (None, {"student_sha256": "changed"}))
    with pytest.raises(ValueError, match="changed"):
        work.choose("parent")
    assert work.checkpoint_report["student_sha256"] == "weights"


def test_feature_mapping_mismatch_rejected_before_loading_weights(monkeypatch):
    monkeypatch.setattr(experiment, "verified_parent", lambda _: {"input_size": 20})
    with pytest.raises(ValueError, match="input_size"):
        experiment.compatible_model("unused", {"input_size": 21})


@pytest.mark.parametrize("seed,episodes", [(100, 1), (98, 3), (199, 2)])
def test_reserved_test_seed_cannot_be_used_in_live_episode_range(seed, episodes):
    with pytest.raises(ValueError, match="reserved"):
        workbench().new_run(seed=seed, episodes=episodes)


def test_reserved_seed_rejected_before_loading_selected_model(monkeypatch):
    monkeypatch.setattr(experiment, "compatible_model", lambda *_: pytest.fail("Must not load reserved start"))
    with pytest.raises(ValueError, match="reserved"):
        workbench().choose("parent", seed=100)


def test_reservation_hash_changed(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text("{}")
    path.with_suffix(".sha256").write_text("wrong")
    with pytest.raises(ValueError, match="changed"):
        experiment.reserved_seeds(path)


def test_rgb_inventory_does_not_confuse_file_hashes_with_pixels():
    from flydoom.reserve_recovery import image_hashes
    assert image_hashes({"frame_sha256": "a" * 64, "student_sha256": "b" * 64,
                         "opening_hashes": ["c" * 64], "sha256": "d" * 64}) == {"a" * 64, "c" * 64}
