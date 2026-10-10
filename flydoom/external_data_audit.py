"""Offline action-compatibility audit of pinned Doom dataset samples.

This module never imports external samples into training. Action-name agreement
alone does not establish matched observation timing, dynamics or label quality.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from flydoom.data import digest
from flydoom.learning_cycle import atomic_json
from flydoom.movement_core import ACTIONS

GAMEWAM_ACTIONS = ["shoot", "move_forward", "move_backward", "strafe_left", "strafe_right",
                   "speed", "turn_left", "turn_right", "turn_delta"]
GAMEWAM_TO_LOCAL = ["ATTACK", "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT"]


def map_gamewam(action):
    """Return an exact single-button semantic match or an explicit rejection."""
    values = np.asarray(action, dtype=float)
    if values.shape != (9,) or not np.isfinite(values).all(): return None, "invalid_vector"
    if not np.isin(values[:8], [0., 1.]).all() or abs(values[8]) > 1: return None, "invalid_vector"
    if np.any(values[6:] != 0): return None, "rotation_not_supported"
    if values[5] != 0: return None, "speed_not_supported"
    pressed = np.flatnonzero(values[:5])
    if len(pressed) > 1: return None, "simultaneous_buttons"
    return (GAMEWAM_TO_LOCAL[int(pressed[0])] if len(pressed) else "WAIT"), None


def audit(root):
    import pyarrow.parquet as pq
    root = Path(root)
    manifest = json.loads((root / "download-manifest.json").read_text(encoding="utf-8"))
    for item in manifest:
        path = root / item["dataset"].replace("/", "--") / item["file"]
        if digest(path, "sha256") != item["sha256"]: raise ValueError("Downloaded source changed")
    action_map = json.loads((root / "brahmandam--DoomFrameDataset/action_map.json").read_text())
    exact = {str(row["id"]): row["name"] for row in action_map if row["name"] in ACTIONS}
    result = {"scope": "Metadata and two deterministic first-episode samples; not representative sampling or training data admission.",
              "external_training_frames_admitted": 0, "downloaded_bytes": sum(row["bytes"] for row in manifest),
              "doom_frame_dataset": {"action_categories": len(action_map), "exact_name_matches": exact,
                                     "missing_local_actions": sorted(set(ACTIONS) - set(exact.values())),
                                     "license": "No license declared in the inspected dataset card"}, "gamewam": []}
    pdoom = json.loads((root / "p-doom--doom-dataset/metadata.json").read_text())
    result["pdoom_metadata"] = {"environment_field": pdoom.get("env"), "num_actions": pdoom.get("num_actions"),
                                "warning": "The metadata environment says coinrun while the card describes Doom; provenance needs clarification before import."}
    for path in sorted((root / "Yunncheng--gamewam-vizdoom").glob("*/data/chunk-000/episode_000000.parquet")):
        scenario = path.parents[2].name
        info = json.loads((path.parents[2] / "meta/info.json").read_text())
        if info["features"]["action"]["names"] != GAMEWAM_ACTIONS: raise ValueError("Unknown source action order")
        table = pq.read_table(path)
        indices = np.asarray(table["frame_index"].to_pylist())
        timestamps = np.asarray(table["timestamp"].to_pylist())
        if not np.array_equal(indices, np.arange(len(indices))): raise ValueError("Non-contiguous episode")
        if not np.allclose(timestamps, indices / info["fps"], atol=1e-5): raise ValueError("Timestamp mismatch")
        if len(set(table["episode_index"].to_pylist())) != 1: raise ValueError("Mixed episodes")
        matches, rejected, prefix = Counter(), Counter(), 0
        prefix_open = True
        for action in table["action"].to_pylist():
            mapped, reason = map_gamewam(action)
            if mapped is None: rejected[reason] += 1; prefix_open = False
            else:
                matches[mapped] += 1
                if prefix_open: prefix += 1
        result["gamewam"].append({"scenario": scenario, "episode": 0, "rows": table.num_rows,
                                  "source_fps": info["fps"], "local_action_duration_tics": 4,
                                  "matching_single_button_rows": sum(matches.values()), "action_counts": dict(matches),
                                  "rejected_rows": dict(rejected), "matching_causal_prefix_frames": prefix,
                                  "admission": "Not admitted: images absent, action timing differs, and unsupported steps cannot be spliced out of recurrent trajectories."})
    atomic_json(root / "compatibility.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(json.dumps(audit(parser.parse_args().folder), indent=2))
