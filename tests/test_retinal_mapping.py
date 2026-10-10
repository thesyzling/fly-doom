import numpy as np
from scipy import sparse

from flydoom.retinal_mapping import dominant, heldout_points, infer_paths


def test_holdout_excludes_coordinates_even_when_source_has_duplicate_nodes():
    xyz = np.column_stack((np.arange(200), np.zeros(200), np.zeros(200)))
    xyz = np.repeat(xyz, 2, axis=0)
    old = set(map(tuple, xyz[np.linspace(0, len(xyz)-1, 32).astype(int)]))
    new = heldout_points(xyz)
    assert len(set(map(tuple, new))) == 32
    assert not old.intersection(map(tuple, new))


def test_dominance_rejects_zero_and_ambiguous_counts():
    assert not dominant([0, 0])[-1]
    assert not dominant([6, 5])[-1]
    assert dominant([20, 2])[-1]


def test_receptor_dominance_includes_unmapped_l1_targets():
    # PR -> mapped L1 has 10 contacts but -> unmapped L1 has 90: reject PR.
    ids = np.array([720575940604737708 + k for k in range(4)], dtype=np.uint64)
    rows = [{'cell_type': kind, 'side': 'left'} for kind in ['R1-6', 'L1', 'L1', 'Mi1']]
    counts = sparse.csr_matrix(([10, 90, 30], ([1, 2, 3], [0, 0, 1])), shape=(4, 4))
    supported = [{'candidate_root_id': str(ids[3]), 'side': 'left', 'column_index': 0, 'optical_direction': [1, 0, 0]}]
    columns, receptors = infer_paths(ids, rows, counts, supported)
    assert columns[0]['l1_supported']
    assert receptors == []
    counts[2, 0] = 1
    _, receptors = infer_paths(ids, rows, counts, supported)
    assert receptors[0]['root_id'] == str(ids[0])
    assert receptors[0]['receptor_to_l1_contacts'] == 10
