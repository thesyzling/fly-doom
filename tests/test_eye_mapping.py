import numpy as np
import pytest

from flydoom.eye_mapping import rank_candidates, source_columns


def test_geometric_candidates_preserve_exact_ids_and_reject_duplicate_matches():
    roots = ['720575940604737708', '720575940604737709']
    skeletons = [(roots[0], np.array([[0., 0, 0], [0, 1, 0]])),
                 (roots[1], np.array([[10000., 0, 0], [10000, 1, 0]]))]
    samples = np.array([np.tile([0., 0, 0], (32, 1)), np.tile([10000., 0, 0], (32, 1))])
    rows = rank_candidates(samples, skeletons)
    assert [r['candidate_root_id'] for r in rows] == roots
    assert all(r['geometry_supported'] for r in rows)
    assert not any(r['segmentation_verified'] for r in rows)
    duplicates = rank_candidates(samples[[0, 0]], skeletons)
    assert not any(r['geometry_supported'] for r in duplicates)


def test_distant_or_ambiguous_morphology_is_not_supported():
    skeletons = [('1', np.array([[0., 0, 0]])), ('2', np.array([[10000., 0, 0]]))]
    samples = np.array([[[0., 0, 0]] * 16 + [[10000., 0, 0]] * 16])
    assert not rank_candidates(samples, skeletons)[0]['geometry_supported']
    samples = np.array([[[0., 2000, 0]] * 32])
    assert not rank_candidates(samples, skeletons)[0]['geometry_supported']


def test_source_index_chain_uses_both_r_index_arrays():
    pd = pytest.importorskip('pandas')
    # Eye row 2 -> Mi1-neuron row 1 -> annotation skid 11.
    eye = {'eyemap': np.array([[2, 7]]), 'ucl_rot_sm': np.array([[1., 0, 0]]),
           'med_xyz': np.array([[40., 50, 60]])}
    cells = {'Mi1_neu_ind': np.array([2, 1]),
             'anno_Mi1': pd.DataFrame({'skid': [11, 22]}),
             'Mi1_M10_xyz': np.array([[10., 20, 30], [40, 50, 60]]),
             'Mi1': {'11': {'d': pd.DataFrame({'X': [1, 2], 'Y': [3, 4], 'Z': [5, 6]})}}}
    _, skids, _, _, samples = source_columns(eye, cells)
    assert skids.tolist() == [11]
    assert samples.shape == (1, 32, 3)
    eye['med_xyz'][0, 0] += 1
    with pytest.raises(ValueError, match='index chains'):
        source_columns(eye, cells)
