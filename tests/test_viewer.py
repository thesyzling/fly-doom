import csv
from http.server import ThreadingHTTPServer
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest

from flydoom.data import digest
from flydoom.viewer import RecordedExperiment, make_handler


@pytest.fixture
def recorded(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    run.mkdir()
    ids = np.array([720575940625448968, 720575940625448969], dtype=np.uint64)
    np.save(data / "root_ids.npy", ids)
    fields = ["root_id", "pos_x", "pos_y", "pos_z", "super_class", "cell_type", "side", "top_nt", "known_nt"]
    with (data / "neuron_annotations.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(fields)
        writer.writerow([str(ids[0]), 0, 0, 0, "sensory", "R1-6", "left", "acetylcholine", ""])
        writer.writerow([str(ids[1]), 10, 10, 10, "descending", "", "right", "gaba", "gaba"])
    np.savez(run / "pulse.npz", root_ids=ids, spike_counts=np.array([5, 2]), stimulated_root_ids=ids[:1])
    report = {"created_utc": "2026-09-27T00:00:00Z", "gates": {}, "limitations": [],
              "provenance": {"prepared_sha256": {name: digest(data / name, "sha256")
                             for name in ("root_ids.npy", "neuron_annotations.tsv")}},
              "conditions": {"pulse": {"spikes": 7}},
              "output_sha256": {"pulse.npz": digest(run / "pulse.npz", "sha256")}}
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return run, data


def test_viewer_uses_anisotropic_source_coordinates_and_exact_ids(recorded):
    experiment = RecordedExperiment(*recorded)
    np.testing.assert_allclose(experiment.positions_um[1], [0.04, 0.04, 0.4])
    point = experiment.neuron("pulse", 1)
    assert point["root_id"] == "720575940625448969"
    assert point["spikes"] == 2
    assert not point["directly_stimulated"]
    vertices = np.frombuffer(experiment.buffers["pulse"], dtype="<f4").reshape(-1, 6)
    np.testing.assert_array_equal(vertices[:, 3], [5, 2])
    np.testing.assert_array_equal(vertices[:, 4], [1, 0])
    assert experiment.metadata()["conditions"]["pulse"]["group_totals"] == {"descending": 2, "sensory": 5}


def test_viewer_rejects_modified_recording(recorded):
    run, _ = recorded
    with (run / "pulse.npz").open("ab") as f:
        f.write(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        RecordedExperiment(*recorded)


def test_viewer_requires_completed_report(tmp_path):
    with pytest.raises(ValueError, match="completed"):
        RecordedExperiment(tmp_path, tmp_path)


def test_viewer_serves_only_known_routes(recorded):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(RecordedExperiment(*recorded)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/meta") as r:
            assert json.load(r)["neurons"] == 2
        with urlopen(base + "/api/points/pulse") as r:
            assert len(r.read()) == 2 * 6 * 4
        with urlopen(base + "/") as r:
            assert b"RECORDED EXPERIMENT" in r.read()
        for path in ("/api/neuron/pulse/-1", "/api/neuron/missing/0", "/../pyproject.toml"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + path)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def add_temporal_recording(recorded, corrupt=False):
    run, _ = recorded
    path = run / "pulse.npz"
    with np.load(path) as archive:
        data = {name: archive[name] for name in archive.files}
    data.update(sample_times_ms=np.array([0., 10., 20.]),
                spike_bins=np.array([[0, 0], [3, 0], [2, 2 if not corrupt else 3]], dtype=np.uint32),
                voltage_mv=np.array([[-52, -52], [-50, -55], [-51, -53]], dtype=np.float32))
    np.savez(path, **data)
    report_path = run / "report.json"
    report = json.loads(report_path.read_text())
    report["conditions"]["pulse"]["duration_ms"] = 20
    report["output_sha256"][path.name] = digest(path, "sha256")
    report_path.write_text(json.dumps(report), encoding="utf-8")


def test_temporal_binary_order_and_legacy_handling(recorded):
    legacy = RecordedExperiment(*recorded)
    assert legacy.metadata()["conditions"]["pulse"]["sample_times_ms"] == []
    with pytest.raises(KeyError):
        legacy.replay("pulse")
    add_temporal_recording(recorded)
    experiment = RecordedExperiment(*recorded)
    binary = np.frombuffer(experiment.replay("pulse"), dtype="<f4").reshape(3, 2, 2)
    np.testing.assert_array_equal(binary[1, :, 0], [3, 0])
    np.testing.assert_array_equal(binary[1, :, 1], [-50, -55])
    assert experiment.metadata()["conditions"]["pulse"]["sample_times_ms"] == [0, 10, 20]


def test_reject_temporal_counts_that_disagree_with_total(recorded):
    add_temporal_recording(recorded, corrupt=True)
    with pytest.raises(ValueError, match="temporal counts"):
        RecordedExperiment(*recorded)
