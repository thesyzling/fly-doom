"""Locked repetition evaluation must retain every reserved condition and scene."""

import json

import pytest

from flydoom import vision_recovery_eval as evaluation
from flydoom.data import digest


@pytest.mark.parametrize("damage", [None, "plan", "artifact", "seed", "scene", "condition"])
def test_gameplay_lock_and_reserved_identity(tmp_path, damage):
    weights = tmp_path / "weights"
    weights.write_bytes(b"fixed")
    openings = [{"seed": i, "rgb_sha256": str(i)} for i in range(20)]
    if damage in {"seed", "scene"}:
        key = "seed" if damage == "seed" else "rgb_sha256"
        openings[-1][key] = openings[0][key]
    conditions = [{"name": name} for name in evaluation.CONDITIONS]
    if damage == "condition":
        conditions.pop()
    plan = {"schema": "vision_recovery_gameplan_v1", "openings": openings,
            "conditions": conditions, "max_decisions": 75,
            "artifact_sha256": {str(weights): digest(weights, "sha256")}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    path.with_suffix(".sha256").write_text(digest(path, "sha256"))
    if damage == "plan":
        path.write_text(json.dumps(plan) + " ")
    if damage == "artifact":
        weights.write_bytes(b"changed")
    if damage:
        with pytest.raises(ValueError):
            evaluation.verify_plan(path)
    else:
        assert evaluation.verify_plan(path) == plan


def test_scores_retain_failures_and_applied_action_counts():
    episodes = [{"kills": 1, "return": 80, "decisions": 5,
                 "action_counts": {"WAIT": 0, "MOVE_LEFT": 4, "MOVE_RIGHT": 0, "ATTACK": 1}},
                {"kills": 0, "return": -300, "decisions": 75,
                 "action_counts": {"WAIT": 75, "MOVE_LEFT": 0, "MOVE_RIGHT": 0, "ATTACK": 0}}]
    result = evaluation.score(episodes)
    assert result["hits"] == 1 and result["episodes"] == 2
    assert result["mean_return"] == -110 and result["decisions"] == 80
    assert result["action_counts"]["WAIT"] == 75
