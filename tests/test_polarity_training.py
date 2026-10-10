import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize

from flydoom.polarity_training import PairGains, fixed_point, prior_loss, response


def test_fixed_point_matches_analytic_linear_response_and_rejects_instability():
    weights = sparse.csr_matrix([[0, 0, 0], [-.2, 0, 0], [0, -.3, 0]])
    actual = response(weights, [.5, 0, 0])
    expected = np.linalg.solve(np.eye(3) - weights.toarray(), [.5, 0, 0])
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    with pytest.raises(ValueError, match='contractive'):
        fixed_point(sparse.eye(2), [0, 0])


def test_shared_gains_preserve_structure_signs_ratios_and_other_target_cells():
    weights = sparse.csr_matrix([[0, 0, 0, 0], [-.1, 0, 0, 0],
                                 [.2, -.1, 0, 0], [.4, -.2, 0, 0]])
    pairs = PairGains(weights, ['R', 'L', 'Mi4', 'Mi4'])
    fitted = pairs.weights(np.log([.5, 1.2]))
    np.testing.assert_array_equal(fitted.indices, weights.indices)
    np.testing.assert_array_equal(fitted.indptr, weights.indptr)
    np.testing.assert_array_equal(np.sign(fitted.data), np.sign(weights.data))
    assert fitted[1, 0] == weights[1, 0]
    assert fitted[3, 0] / fitted[2, 0] == pytest.approx(2)
    assert fitted[3, 1] / fitted[2, 1] == pytest.approx(2)
    with pytest.raises(ValueError, match='bounds'):
        pairs.weights([np.log(2), 0])


def test_qualitative_optimizer_can_correct_conflicting_paths_without_changing_signs():
    # Two positive input cells excite/inhibit Mi4. Only existing Mi4 inputs vary.
    weights = sparse.csr_matrix([[0, 0, 0], [0, 0, 0], [.1, -.2, 0]])
    pairs = PairGains(weights, ['exc', 'inh', 'Mi4'])
    drive = np.array([.5, .5, 0])
    def objective(parameters):
        value = response(pairs.weights(parameters), drive)[2]
        return prior_loss(np.array([value]), np.array([1]), np.array([.05]))
    result = minimize(objective, [0., 0.], method='L-BFGS-B',
                      bounds=[(np.log(.05), np.log(1.5))] * 2)
    assert response(weights, drive)[2] < 0
    assert response(pairs.weights(result.x), drive)[2] > 0
    assert result.fun < 1e-8
    np.testing.assert_array_equal(np.sign(pairs.weights(result.x).data), np.sign(weights.data))


def test_margin_is_not_an_amplitude_target_and_inactive_labels_do_not_drive_loss():
    assert prior_loss(np.array([10., -.3]), np.array([1, -1]), np.ones(2)) == 0
    assert prior_loss(np.array([0.]), np.array([1]), np.ones(1)) > 0
