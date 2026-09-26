import numpy as np
import pytest

from flydoom.data import align_annotations, build_graph, digest, verify


def test_direction_aggregation_and_isolated_neurons():
    # IDs exceed float64's exact integer range; do not round through floats.
    a, b, isolated = 720575940625448968, 720575940640978048, 720575940640978049
    ids, graph = build_graph(np.array([isolated, b, a]), np.array([a, a, b]),
                             np.array([b, b, b]), np.array([2, 3, 1]))
    np.testing.assert_array_equal(ids, [a, b, isolated])
    np.testing.assert_array_equal(graph @ np.array([1, 0, 0]), [0, 5, 0])
    assert graph.nnz == 2
    assert graph.sum() == 6
    assert graph[1, 1] == 1
    assert graph.shape == (3, 3)


@pytest.mark.parametrize("roots,pre,post,counts", [
    ([1, 1], [1], [1], [1]),
    ([1, 2], [3], [1], [1]),
    ([1, 3], [2], [1], [1]),
    ([1, 2], [1], [2], [0]),
    ([1, 2], [1], [2], [-1]),
    ([1, 2], [1.0], [2], [1]),
    ([1, 2], [1, 2], [2], [1]),
])
def test_reject_invalid_graph(roots, pre, post, counts):
    with pytest.raises(ValueError):
        build_graph(*(np.asarray(x) for x in (roots, pre, post, counts)))


def test_corrupt_archive_rejected(tmp_path):
    path = tmp_path / "proofread_root_ids_783.npy"
    path.write_bytes(b"corrupt")
    assert len(digest(path, "sha256")) == 64
    with pytest.raises(ValueError, match="checksum"):
        verify(path)


def test_mixed_signed_unsigned_ids_do_not_round():
    a = 720575940625448968
    b = a + 1
    ids, graph = build_graph(np.array([a, b], dtype=np.uint64),
                             np.array([b], dtype=np.int64),
                             np.array([a], dtype=np.int64), np.array([7]))
    np.testing.assert_array_equal(ids, np.array([a, b], dtype=np.uint64))
    assert graph[0, 1] == 7
    assert graph[0, 0] == 0


def test_negative_neuron_ids_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        build_graph(np.array([-1, 2]), np.array([-1]), np.array([2]), np.array([1]))


def test_annotations_match_identity_not_row_order():
    a = 720575940625448968
    rows = [{"root_id": str(a + 1), "cell_type": "B"},
            {"root_id": str(a), "cell_type": "A"},
            {"root_id": "42", "cell_type": "extra"}]
    aligned, extra = align_annotations(np.array([a, a + 1], dtype=np.uint64), rows)
    assert [row["cell_type"] for row in aligned] == ["A", "B"]
    assert extra == 1


def test_annotations_reject_missing_or_duplicate_ids():
    with pytest.raises(ValueError, match="Missing"):
        align_annotations([1, 2], [{"root_id": "1"}])
    with pytest.raises(ValueError, match="Duplicate"):
        align_annotations([1], [{"root_id": "1"}, {"root_id": "1"}])
