import base64
import numpy as np
import pytest

from flydoom.action_memory import expand
from flydoom.live_activity import ConnectionCatalog
from flydoom.live_brain import LiveSession
from flydoom.live_map import build_map, activity_snapshot
from test_live_brain import catalog, sample


def coordinates(catalog):
    for i, row in enumerate(catalog.rows):
        row.update(pos_x=str(100+i*5), pos_y=str(200+i*3), pos_z=str(30+i*2))


def test_map_preserves_exact_ids_anatomical_units_and_real_weights(catalog):
    coordinates(catalog)
    report = build_map(catalog)
    assert report["ids"] == list(map(str, catalog.ids))
    positions = np.frombuffer(base64.b64decode(report["positions_f32"]), dtype="<f4").reshape(-1,3)
    restored = positions * (report["extent_um"] / 1.7) + report["center_um"]
    expected = np.array([[float(row[f"pos_{axis}"]) for axis in "xyz"] for row in catalog.rows]) * [.004,.004,.040]
    np.testing.assert_allclose(restored, expected, atol=1e-7)
    assert list(base64.b64decode(report["roles_u8"])) == [1,2,2,2]
    for edge in report["edges"]:
        if edge["target"].startswith("student:"):
            target = int(edge["target"].split(":")[1])
            matching = [j for j in range(6) if catalog.feature(j) == (edge["source"], edge["channel"])]
            assert len(matching) == 1
            assert edge["weight"] == float(catalog.input_weights[target, matching[0]])
        else:
            source = int(edge["source"].split(":")[1])
            action = ["WAIT","MOVE_LEFT","MOVE_RIGHT","ATTACK"].index(edge["target"].split(":")[1])
            assert edge["weight"] == float(catalog.action_weights[action, source])


def test_memory_overview_edges_come_from_actual_memory_columns(catalog):
    coordinates(catalog)
    model = expand(catalog.model)
    model.input.weight.data[:,6:-4] = .125
    expanded = ConnectionCatalog(catalog.ids,catalog.controller,model,catalog.rows)
    memory = [e for e in build_map(expanded)["edges"] if e["source"].startswith("memory:")]
    assert len(memory) == 17 and all(e["weight"] == .125 for e in memory)


def test_map_activity_requires_exact_snapshot_and_does_not_change_counts(catalog,tmp_path):
    session = LiveSession(catalog,tmp_path/"map",{"student_sha256":"fixture"})
    snapshot = sample(catalog)
    session.samples.append(snapshot)
    result = session.map_activity(2)
    assert result["sequence"] == 2
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(result["counts_u16"]),dtype="<u2"), snapshot["counts"])
    with pytest.raises(ValueError,match="expired"):
        session.map_activity(1)
    with pytest.raises(ValueError,match="bounds"):
        activity_snapshot({"counts":np.array([65536]),"sequence":3})
    np.testing.assert_array_equal(snapshot["counts"],[1,2,3,4])


def test_invalid_coordinates_do_not_get_invented(catalog):
    coordinates(catalog)
    catalog.rows[0]["pos_x"] = "nan"
    with pytest.raises(ValueError,match="finite"):
        build_map(catalog)
