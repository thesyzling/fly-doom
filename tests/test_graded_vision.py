import numpy as np
import pytest
from scipy import sparse

from flydoom.graded_vision import GradedVision, type_scaled_weights


def test_no_input_is_stationary_and_single_cell_matches_analytic_leak():
    net = GradedVision(sparse.csr_matrix((1, 1)), [20])
    for _ in range(20): net.step([0])
    assert net.delta[0] == 0
    for _ in range(20): net.step([.5])
    assert net.delta[0] == pytest.approx(.5 * (1 - np.exp(-1)), abs=1e-6)


def test_inhibitory_receptor_and_l1_path_inverts_twice_without_spikes():
    w = sparse.csr_matrix(([ -.5, -.5 ], ([1, 2], [0, 1])), shape=(3, 3))
    net = GradedVision(w, [20, 20, 20])
    for _ in range(200): net.step([.5, 0, 0])
    assert net.delta[0] > 0 and net.delta[1] < 0 and net.delta[2] > 0
    net.reset()
    for _ in range(200): net.step([.5, 0, 0], connected=False)
    np.testing.assert_array_equal(net.delta[1:], [0, 0])
    assert net.baseline_residual < 1e-5
    net.reset()
    net.blocked_release[1] = True
    for _ in range(200): net.step([.5, 0, 0])
    assert net.delta[1] < 0  # Blocking release does not clamp the cell state.
    assert net.delta[2] == 0


def test_type_scaling_preserves_signs_zeros_and_contact_ratios():
    c = sparse.csr_matrix([[0, 0, 0], [10, 0, 0], [20, 0, 0]])
    w, gains = type_scaled_weights(c, [-1, 1, 1], ['R', 'L', 'L'])
    assert w.nnz == c.nnz
    assert w[2, 0] / w[1, 0] == pytest.approx(2)
    assert w[1, 0] < 0
    assert np.asarray(abs(w).sum(axis=1)).max() <= .600001
    with pytest.raises(ValueError, match='contraction'):
        GradedVision(sparse.eye(2), [20, 20])
