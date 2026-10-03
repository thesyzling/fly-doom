"""Recorded neural provenance and exact contribution accounting."""

import hashlib
import json

import numpy as np
from PIL import Image
import pytest
from safetensors.torch import save_file
import torch

from flydoom import student
from flydoom.data import digest
from flydoom.live_activity import ReadoutTelemetry
from flydoom.replay import ReplayLibrary
from flydoom.replay_signals import inspect_archive


@pytest.fixture
def archive(tmp_path):
    torch.manual_seed(4)
    model = student.SpikingReadout(8, 4).eval()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(model.state_dict(), str(checkpoint / "student.safetensors"))
    checksum = digest(checkpoint / "student.safetensors", "sha256")
    report = {"student_sha256": checksum, "implementation_sha256": digest(student.__file__, "sha256"),
              "input_size": 8, "hidden": 4, "output_root_ids": ["101", "202"]}
    (checkpoint / "report.json").write_text(json.dumps(report))
    run = tmp_path / "experiment" / "run-test"
    run.mkdir(parents=True)
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    Image.fromarray(image).save(run / "image.png")
    features = np.arange(8, dtype=np.float32) / 4
    telemetry = ReadoutTelemetry(model)
    with torch.no_grad():
        probabilities = model(torch.tensor(features)[None]).softmax(-1)[0].tolist()
    telemetry.close()
    np.savez(run / "state.npz", features=features, student_activity=telemetry.activity)
    row = {"seed": 1, "episode": 1, "decision": 1, "action": "WAIT", "student_action": "WAIT",
           "alpha": 0, "brain_spikes": 9, "probabilities": probabilities, "applied_probabilities": probabilities,
           "vision": {"probabilities": None}, "observation": "state",
           "observation_sha256": digest(run / "state.npz", "sha256")}
    (run / "decisions.jsonl").write_text(json.dumps(row))
    saved = {**row, "frames": ["image.png"], "vision_probabilities": None,
             "student_spikes": np.rint(telemetry.activity * 8).astype(int).tolist()}
    manifest = {"schema": "vision_replay_v1", "checkpoint_sha256": checksum,
                "source_run": str(run), "source_trace_sha256": digest(run / "decisions.jsonl", "sha256"),
                "decisions": [saved], "frame_sha256": {"image.png": digest(run / "image.png", "sha256")}}
    (run / "replay.json").write_text(json.dumps(manifest))
    library = ReplayLibrary(tmp_path)
    return library, next(iter(library.entries())), run, checkpoint, model, features


def test_signal_inputs_and_output_products_reconstruct_forward_pass(archive):
    library, key, _, _, model, features = archive
    result = inspect_archive(library, key)
    data = result["decisions"][0]
    assert result["output_root_ids"] == ["101", "202"]
    np.testing.assert_array_equal(data["descending_voltage"], features[:2])
    np.testing.assert_array_equal(data["descending_rate"], features[2:4])
    np.testing.assert_array_equal(data["previous"], features[-4:])
    assert result["feature_names"][0] == {"id": "101", "channel": "voltage feature"}
    assert result["feature_names"][2]["channel"] == "spike-rate feature"
    assert np.allclose(data["brightness"], 100 / 255)
    for i, cell in enumerate(data["cells"]):
        assert cell["drive"] == pytest.approx(cell["fly_sum"] + cell["memory_sum"] + cell["previous_sum"] + cell["bias"], abs=1e-6)
        for term in cell["top_inputs"]:
            assert term["raw"] == features[term["feature"]]
            assert term["product"] == pytest.approx(term["normalized"] * term["weight"], abs=1e-6)
    logits = np.asarray(result["readout_weight"]) @ data["activity"] + result["readout_bias"]
    np.testing.assert_allclose(logits, data["logits"], atol=1e-6)


@pytest.mark.parametrize("damage", ["weights", "npz", "trace", "frame", "activity", "decision", "probabilities", "outside"])
def test_signal_inspector_rejects_wrong_or_modified_recording(archive, damage):
    library, key, run, checkpoint, _, _ = archive
    if damage in ("weights", "npz", "trace", "frame"):
        path = {"weights": checkpoint / "student.safetensors", "npz": run / "state.npz",
                "trace": run / "decisions.jsonl", "frame": run / "image.png"}[damage]
        path.write_bytes(b"changed")
    else:
        manifest = json.loads((run / "replay.json").read_text())
        if damage == "outside":
            manifest["source_run"] = str(library.root.parent / "outside")
        elif damage == "activity":
            manifest["decisions"][0]["student_spikes"][0] += 1
        elif damage == "decision":
            manifest["decisions"][0]["decision"] = 2
        else:
            manifest["decisions"][0]["probabilities"] = [1, 0, 0, 0]
        (run / "replay.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        inspect_archive(library, key)
