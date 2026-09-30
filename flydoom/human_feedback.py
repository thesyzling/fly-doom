"""Save explicit operator corrections to exact, already observed decisions."""

import base64
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from flydoom.data import digest
from flydoom.learning_data import ACTIONS


SCHEMA = "live_human_feedback_v1"


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class FeedbackStore:
    """Called under the live session lock; saving never changes applied actions."""

    def __init__(self, directory, parent, split, seeds, input_size):
        if split not in {"train", "validation"}:
            raise ValueError("Feedback split must be train or validation")
        self.directory = Path(directory)
        self.report = {"schema": SCHEMA, "source": "explicit_operator_labels",
                       "split": split, "seeds": list(seeds), "input_size": input_size,
                       "parent_student_sha256": parent["student_sha256"],
                       "calibration_sha256": parent.get("calibration_sha256"),
                       "implementation_sha256": parent.get("implementation_sha256"),
                       "memory_implementation_sha256": parent.get("memory_implementation_sha256"),
                       "output_root_ids": parent.get("output_root_ids"),
                       "actions": list(ACTIONS), "training_during_play": False, "records": []}

    def status(self, sequence=None):
        records = self.report["records"]
        record = next((r for r in records if r["sequence"] == sequence), None)
        return {"count": sum(r["label"] is not None for r in records),
                "label": record["label"] if record else None, "split": self.report["split"],
                "directory": str(self.directory)}

    def save(self, sample, action):
        if action is not None and action not in ACTIONS:
            raise ValueError("Choose a valid correction action")
        if sample.get("decision", 0) < 1 or sample.get("features") is None:
            raise ValueError("Step once before labeling a model decision")
        sequence = sample["sequence"]
        old = next((r for r in self.report["records"] if r["sequence"] == sequence), None)
        if old is None and action is None:
            return self.status(sequence)
        self.directory.mkdir(parents=True, exist_ok=True)
        if old is None:
            path = self.directory / f"decision-{sequence:06d}-{uuid4().hex}.npz"
            features = np.asarray(sample["features"], dtype=np.float32)
            if features.shape != (self.report["input_size"],) or not np.isfinite(features).all():
                raise ValueError("Invalid correction features")
            # The displayed image precedes the action; memory contains only past actions.
            png = base64.b64decode(sample["frame"].split(",", 1)[1], validate=True)
            with path.open("xb") as stream:
                np.savez_compressed(stream, features=features,
                                    probabilities=np.asarray(sample["probabilities"], dtype=np.float32),
                                    frame_png=np.frombuffer(png, dtype=np.uint8))
            old = {"sequence": sequence, "episode": sample["episode"], "decision": sample["decision"],
                   "seed": self.report["seeds"][sample["episode"] - 1],
                   "applied_action": sample["action"], "file": path.name,
                   "sha256": digest(path, "sha256"), "revisions": []}
        record = {**old, "label": action, "revisions": [*old["revisions"], action]}
        records = [r for r in self.report["records"] if r["sequence"] != sequence] + [record]
        updated = {**self.report, "records": records}
        atomic_json(self.directory / "manifest.json", updated)
        self.report = updated
        return self.status(sequence)


def read_feedback(directories, parent):
    """Verify immutable observations and reject train/validation episode leakage."""
    parts = {split: {"features": [], "actions": [], "seeds": []} for split in ("train", "validation")}
    sources, seen = [], set()
    for directory in map(Path, directories):
        manifest = directory / "manifest.json"
        manifest_bytes = manifest.read_bytes()
        report = json.loads(manifest_bytes)
        if (report.get("schema") != SCHEMA or report.get("source") != "explicit_operator_labels"
                or report.get("actions") != list(ACTIONS) or report.get("split") not in parts
                or report.get("training_during_play") is not False):
            raise ValueError("Unsupported feedback collection")
        for key, expected in (("parent_student_sha256", parent["student_sha256"]),
                              ("input_size", parent["input_size"]),
                              *[(k, parent.get(k)) for k in ("calibration_sha256", "implementation_sha256",
                                  "memory_implementation_sha256", "output_root_ids")]):
            if report.get(key) != expected:
                raise ValueError(f"Feedback provenance mismatch: {key}")
        part = parts[report["split"]]
        for row in report["records"]:
            if row["label"] is None:
                continue
            if row["label"] not in ACTIONS or row["seed"] not in report["seeds"] or row["decision"] < 1:
                raise ValueError("Invalid feedback label or episode")
            key = (row["seed"], row["decision"])
            if key in seen:
                raise ValueError("Duplicate feedback decision; do not reuse episode seeds")
            seen.add(key)
            path = directory / row["file"]
            if Path(row["file"]).name != row["file"] or digest(path, "sha256") != row["sha256"]:
                raise ValueError("Feedback observation checksum or path mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                x, probs = arrays["features"], arrays["probabilities"]
                if (x.shape != (parent["input_size"],) or not np.isfinite(x).all()
                        or probs.shape != (4,) or not np.isfinite(probs).all() or np.any(probs < 0)
                        or not np.isclose(probs.sum(), 1) or ACTIONS[int(probs.argmax())] != row["applied_action"]):
                    raise ValueError("Invalid saved feedback arrays")
                part["features"].append(x.copy())
            part["actions"].append(ACTIONS.index(row["label"]))
            part["seeds"].append(row["seed"])
        sources.append({"directory": str(directory), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()})
    if set(parts["train"]["seeds"]) & set(parts["validation"]["seeds"]):
        raise ValueError("Training and validation episode seeds overlap")
    for part in parts.values():
        if not part["features"]:
            raise ValueError("Need labeled decisions in separate training and validation collections")
        part["features"] = np.stack(part["features"]).astype(np.float32)
        part["actions"] = np.asarray(part["actions"], dtype=np.int64)
    return parts, sources
