"""Predeclared demonstration coverage checks and transparent data diagnostics."""

import json
from pathlib import Path

import numpy as np


NAMES = ("WAIT", "MOVE_LEFT", "MOVE_RIGHT", "ATTACK")


def coverage(counts):
    """Use training and validation coverage only; never inspect test labels here."""
    reasons = []
    for split, minimum in (("train", 10), ("validation", 5)):
        values = counts.get(split, [0] * 4)
        for action in range(1, 4):
            if values[action] < minimum:
                reasons.append(f"{split}: {NAMES[action]} has {values[action]}; need at least {minimum}")
        if split == "train" and values[0] == 0:
            reasons.append("train: need at least one naturally occurring WAIT example")
    if sum(counts.get("validation", [0] * 4)) < 30:
        reasons.append("validation: need at least 30 decisions")
    return {"ready": not reasons, "reasons": reasons,
            "counts": {key: list(map(int, counts.get(key, [0] * 4))) for key in ("train", "validation")},
            "note": "Coverage is a prerequisite, not evidence of useful imitation or game skill"}


def inspect_recording(directory):
    from flydoom.learning_data import read_recording, image_state
    directory = Path(directory)
    manifest = read_recording(directory)
    counts = {key: np.zeros(4, dtype=np.int64) for key in ("train", "validation")}
    observations = {key: set() for key in counts}
    old_grids = {key: set() for key in counts}
    from flydoom.bridge import encode_frame
    taps = 0
    for ep in manifest["episodes"]:
        if ep["split"] not in counts:
            continue
        with np.load(directory / ep["file"], allow_pickle=False) as data:
            labels = data["actions"]
            if labels.ndim != 1 or labels.dtype.kind not in "iu" or not np.isin(labels, range(4)).all():
                raise ValueError("Invalid recorded action labels")
            if len(data["frames"]) != len(labels):
                raise ValueError("Frame/action count mismatch")
            counts[ep["split"]] += np.bincount(labels, minlength=4)
            if "latched_taps" in data:
                taps += int(data["latched_taps"].sum())
            for frame in data["frames"]:
                old_grids[ep["split"]].add(tuple(np.rint(encode_frame(frame) * 9).astype(int)))
                # Constant previous action isolates visual representation diversity.
                observations[ep["split"]].add(json.dumps(image_state(frame, 0), sort_keys=True))
    report = coverage(counts)
    report["schema"] = "demonstration_quality_v1"
    report["recording"] = str(directory)
    report["recovered_taps_train_validation"] = taps
    report["input_event_audit_available"] = manifest.get("input_capture") == "events_with_tap_latch_v2"
    report["visual_diversity"] = {key: {"samples": int(counts[key].sum()),
        "old_10_level_grids": len(old_grids[key]), "new_observations": len(observations[key])} for key in counts}
    report["visual_diversity_note"] = "More distinct observations do not establish target recognition or better decisions"
    return report


def print_quality(report):
    for split, counts in report["counts"].items():
        print(f"{split}: " + " | ".join(f"{name}={count}" for name, count in zip(NAMES, counts)), flush=True)
    print(f"Demonstration coverage ready: {report['ready']}", flush=True)
    for reason in report["reasons"]:
        print(f"  {reason}", flush=True)
