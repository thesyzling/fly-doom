"""Recovery collection must keep scene identity and teaching splits separate."""

import json

import pytest

from flydoom import vision_recovery as recovery
from flydoom.data import digest


def test_opening_allocation_excludes_history_and_duplicates_across_splits():
    rows = [{"seed": i, "rgb_sha256": h} for i, h in enumerate(["old", "a", "a", "b", "c", "d", "e"])]
    result = recovery.select_openings(rows, {"old"}, {5}, {"train": 2, "validation": 1, "evaluation": 1})
    assert result == {"train": [rows[1], rows[3]], "validation": [rows[4]], "evaluation": [rows[6]]}


def test_insufficient_unique_scenes_are_not_filled_with_duplicates():
    rows = [{"seed": i, "rgb_sha256": "same"} for i in range(20)]
    result = recovery.select_openings(rows, set(), set(), {"train": 1, "validation": 1})
    assert len(result["train"]) == 1 and result["validation"] == []


@pytest.mark.parametrize("damage", [None, "plan", "artifact", "scene", "seed"])
def test_recovery_lock_rejects_changes_and_split_leakage(tmp_path, damage):
    artifact = tmp_path / "parent.bin"
    artifact.write_bytes(b"fixed")
    rows = [{"seed": i, "rgb_sha256": str(i)} for i in range(38)]
    plan = {"schema": "vision_recovery_plan_v1", "training_seeds": list(recovery.TRAINING_SEEDS),
            "alpha": 0., "collection_decisions": 24,
            "splits": {"train": rows[:12], "validation": rows[12:18], "evaluation": rows[18:]},
            "artifact_sha256": {str(artifact): digest(artifact, "sha256")}}
    if damage in {"scene", "seed"}:
        key = "rgb_sha256" if damage == "scene" else "seed"
        plan["splits"]["validation"][0][key] = plan["splits"]["train"][0][key]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    (tmp_path / "plan.sha256").write_text(digest(path, "sha256"))
    if damage == "plan":
        path.write_text(json.dumps(plan) + " ")
    if damage == "artifact":
        artifact.write_bytes(b"modified")
    if damage:
        with pytest.raises(ValueError):
            recovery.verify_study(tmp_path)
    else:
        assert recovery.verify_study(tmp_path) == plan


def test_actual_collection_must_reproduce_locked_openings(tmp_path):
    rows = [{"seed": 7, "decision": 1, "vision": {"frame_sha256": "opening"}},
            {"seed": 7, "decision": 2, "vision": {"frame_sha256": "later"}}]
    (tmp_path / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    recovery.verify_collected_openings(tmp_path, [{"seed": 7, "rgb_sha256": "opening"}])
    with pytest.raises(ValueError, match="differ"):
        recovery.verify_collected_openings(tmp_path, [{"seed": 7, "rgb_sha256": "different"}])
