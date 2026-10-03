"""Protocol locks, paired comparisons and scene leakage checks."""

import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from flydoom import vision_validation as validation
from flydoom.data import digest
from flydoom.vision_distill import evaluate_policy


def test_seed_history_includes_nested_plans_and_episodes():
    value = {"seed": 3, "evaluation_seeds": [7, 8], "episodes": [{"seed": 9}],
             "settings": {"train_seeds": [1, 2]}, "hidden": 64}
    assert validation.collect_seeds(value) == {1, 2, 3, 7, 8, 9}


@pytest.mark.parametrize("damage", [None, "plan", "artifact", "duplicate_seed", "condition"])
def test_locked_plan_rejects_changes(tmp_path, damage):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"original")
    plan = {"schema": "vision_validation_plan_v1", "conditions": list(validation.CONDITIONS),
            "evaluation_seeds": list(range(20)), "max_decisions": 75,
            "artifacts": validation.fingerprint([artifact])}
    if damage == "duplicate_seed":
        plan["evaluation_seeds"][-1] = 0
    if damage == "condition":
        plan["conditions"] = ["candidate"]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    path.with_suffix(".sha256").write_text(digest(path, "sha256"))
    if damage == "plan":
        path.write_text(json.dumps(plan) + " ")
    if damage == "artifact":
        artifact.write_bytes(b"changed")
    if damage:
        with pytest.raises(ValueError):
            validation.verify_plan(path)
    else:
        assert validation.verify_plan(path) == plan


def test_paired_statistics_group_duplicate_openings_and_preserve_direction():
    parent = [{"seed": i, "kills": 0, "return": -20} for i in range(3)]
    candidate = [{"seed": i, "kills": 1, "return": 80} for i in range(3)]
    result = validation.paired_summary(parent, candidate, ["same", "same", "different"])
    assert result["unique_opening_groups"] == 2
    assert result["hit_rate_difference"] == 1
    assert result["return_interval95"] == [100, 100]
    with pytest.raises(ValueError, match="identities"):
        validation.paired_summary(parent, candidate[::-1], ["a", "b", "c"])


def test_overlap_audit_counts_rgb_and_input_matches_separately(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    image = np.ones((8, 8, 3), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "frame.png")
    rgb_hash = hashlib.sha256(image.tobytes()).hexdigest()
    np.savez(checkpoint / "train.npz", rgb_hashes=[rgb_hash], features=[[1., 2.]])
    np.savez(checkpoint / "validation.npz", rgb_hashes=["other"], features=[[3., 4.]])
    np.savez(tmp_path / "frame.npz", features=[3., 4.])
    row = {"seed": 77, "observation": "frame", "playback_frames": ["frame.png"],
           "observation_sha256": digest(tmp_path / "frame.npz", "sha256")}
    (tmp_path / "decisions.jsonl").write_text(json.dumps(row))
    result = validation.overlap_audit(tmp_path, checkpoint, {"episodes": [{"seed": 77}], "opening_hashes": [rgb_hash]})
    assert result["train"] == {"opening_seeds": [77], "rgb_decisions": 1, "input_decisions": 0, "trajectory_seeds": [77]}
    assert result["validation"] == {"opening_seeds": [], "rgb_decisions": 0, "input_decisions": 1, "trajectory_seeds": [77]}


@pytest.mark.parametrize("seeds,limit", [([], 75), ([1, 1], 75), (list(range(21)), 75), ([1], 0)])
def test_evaluation_bounds_fail_before_creating_output(tmp_path, seeds, limit):
    with pytest.raises(ValueError, match="distinct"):
        evaluate_policy(None, None, {}, seeds, tmp_path / "run", limit)
    assert not (tmp_path / "run").exists()
