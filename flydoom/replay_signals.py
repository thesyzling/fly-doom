"""Verify archived inputs and expose exact readout arithmetic for synchronized inspection."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch

from flydoom import action_memory, student
from flydoom.bridge import encode_frame
from flydoom.data import digest
from flydoom.live_activity import ReadoutTelemetry
from flydoom.replay import safe_path


def checkpoint_for(root, checksum):
    for report_path in Path(root).glob("*/report.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("student_sha256") != checksum:
            continue
        weights = report_path.parent / "student.safetensors"
        if digest(weights, "sha256") != checksum or digest(student.__file__, "sha256") != report["implementation_sha256"]:
            raise ValueError("Archived student checkpoint or implementation changed")
        if report.get("action_memory") and digest(action_memory.__file__, "sha256") != report["memory_implementation_sha256"]:
            raise ValueError("Archived action-memory implementation changed")
        return weights, report
    raise ValueError("The exact checkpoint for this recording is unavailable")


def inspect_archive(library, key):
    manifest_path, manifest = library.manifest(key)
    source = Path(manifest["source_run"]).resolve()
    if not source.is_relative_to(library.root.resolve()):
        raise ValueError("Replay source leaves the run library")
    trace = source / "decisions.jsonl"
    if digest(trace, "sha256") != manifest["source_trace_sha256"]:
        raise ValueError("Original decision trace checksum changed")
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    if not 1 <= len(rows) <= 1500 or len(rows) != len(manifest["decisions"]):
        raise ValueError("Replay and original trace length differ")
    weights, report = checkpoint_for(library.root, manifest["checkpoint_sha256"])
    state = load_file(str(weights))
    model = student.SpikingReadout(report["input_size"], report["hidden"]).eval()
    model.load_state_dict(state)
    roots = report["output_root_ids"]
    count = len(roots)
    names = [{"id": root, "channel": channel} for channel in ("voltage feature", "spike-rate feature") for root in roots]
    names += [{"id": name, "channel": "action memory"} for name in report.get("memory_feature_names", [])]
    names += [{"id": name, "channel": "previous action"} for name in ("WAIT", "MOVE_LEFT", "MOVE_RIGHT", "ATTACK")]
    if len(names) != report["input_size"]:
        raise ValueError("Checkpoint input mapping is inconsistent")
    result = []
    telemetry = ReadoutTelemetry(model)
    try:
        for row, archived in zip(rows, manifest["decisions"]):
            if any(row[k] != archived[k] for k in ("seed", "episode", "decision", "action", "student_action",
                    "alpha", "brain_spikes", "probabilities", "applied_probabilities")) or row["vision"]["probabilities"] != archived["vision_probabilities"]:
                raise ValueError("Replay decision identity differs from source")
            artifact = safe_path(source, row["observation"] + ".npz")
            if digest(artifact, "sha256") != row["observation_sha256"]:
                raise ValueError("Recorded neural observation checksum changed")
            with np.load(artifact, allow_pickle=False) as stored:
                features = stored["features"].astype(np.float32)
                activity = stored["student_activity"].astype(np.float32)
            if features.shape != (report["input_size"],) or not np.isfinite(features).all():
                raise ValueError("Invalid archived neural features")
            with torch.inference_mode():
                logits = model(torch.tensor(features)[None])[0]
                probabilities = logits.softmax(-1).numpy()
            if not np.allclose(probabilities, row["probabilities"], atol=1e-6) or not np.allclose(activity, telemetry.activity, atol=1e-7):
                raise ValueError("Archived student forward pass does not reproduce")
            if not np.array_equal(np.rint(activity * 8).astype(int), archived["student_spikes"]):
                raise ValueError("Replay activity differs from recorded neural observation")
            normalized = (features - state["mean"].numpy()) / state["scale"].numpy()
            products = state["input.weight"].numpy() * normalized[None]
            drive = products.sum(axis=1) + state["input.bias"].numpy()
            if not np.allclose(drive, telemetry.drive, atol=1e-4):
                raise ValueError("Input contribution reconstruction differs")
            frame_name = archived["frames"][0]
            frame_path = safe_path(manifest_path.parent, frame_name)
            if digest(frame_path, "sha256") != manifest["frame_sha256"][frame_name]:
                raise ValueError("Replay observation image checksum changed")
            with Image.open(frame_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            expected_rgb = row["vision"].get("frame_sha256")
            if expected_rgb and hashlib.sha256(rgb.tobytes()).hexdigest() != expected_rgb:
                raise ValueError("Replay image differs from Vision observation")
            top = np.argsort(-np.abs(products), axis=1, kind="stable")[:, :4]
            cells = []
            for i, indices in enumerate(top):
                cells.append({"drive": float(telemetry.drive[i]), "bias": float(state["input.bias"][i]),
                    "fly_sum": float(products[i, :2 * count].sum()),
                    "memory_sum": float(products[i, 2 * count:-4].sum()),
                    "previous_sum": float(products[i, -4:].sum()),
                    "top_inputs": [{"feature": int(j), "raw": float(features[j]), "normalized": float(normalized[j]),
                                    "weight": float(state["input.weight"][i, j]), "product": float(products[i, j])} for j in indices]})
            descending = np.argsort(-np.abs(features[:count]), kind="stable")[:8]
            result.append({"seed": row["seed"], "episode": row["episode"], "decision": row["decision"],
                "brightness": encode_frame(rgb).tolist(), "logits": logits.tolist(), "activity": activity.tolist(),
                "descending_voltage": features[:count].tolist(), "descending_rate": features[count:2 * count].tolist(),
                "memory": features[2 * count:-4].tolist(), "previous": features[-4:].tolist(),
                "cells": cells, "descending": [{"id": roots[i], "voltage_feature": float(features[i]),
                    "rate_feature": float(features[count + i])} for i in descending]})
    finally:
        telemetry.close()
    return {"schema": "replay_signals_v1", "checkpoint_sha256": manifest["checkpoint_sha256"],
            "distilled_from_vision": bool(report.get("vision_distillation")),
            "feature_names": names, "descending_count": count,
            "output_root_ids": roots, "memory_feature_names": report.get("memory_feature_names", []),
            "readout_weight": state["readout.weight"].tolist(), "readout_bias": state["readout.bias"].tolist(),
            "decisions": result,
            "scope": "Stored descending features and student spikes; input drives and logits verified by replaying the fixed student. Full biological graph state was not saved. No Vision-to-fly synaptic matrix exists."}
