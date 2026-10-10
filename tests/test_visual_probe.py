import numpy as np
from scipy import sparse

from flydoom.simulation import LIFNetwork, LIFParameters
from flydoom.visual_probe import stimulus, run_condition


def test_sweeps_reverse_spatial_order_and_preserve_pulse_boundaries():
    directions = np.array([[-1., 0, 0], [0, 1, 0], [1, 0, 0]])
    assert stimulus('Mi1_x_positive', 52, directions).tolist() == [1, 0, 0]
    assert stimulus('Mi1_x_negative', 52, directions).tolist() == [0, 0, 1]
    assert not stimulus('Mi1_flash', 39.5, directions).any()
    assert stimulus('Mi1_flash', 40, directions).all()
    assert not stimulus('Mi1_flash', 160, directions).any()


def test_disconnected_probe_prevents_downstream_transmission():
    # Keep the test below the experiment's population-wide spike guard.
    matrix = sparse.csr_matrix(([10.], ([1], [0])), shape=(100, 100))
    net = LIFNetwork(matrix, LIFParameters(dt_ms=1))
    populations = {'input': np.array([0]), 'target': np.array([1])}
    args = (net, np.array([0]), np.array([[1., 0, 0]]), populations)
    result, counts = run_condition(*args, 'Mi1_disconnected')
    assert result['completed']
    assert counts[0] > 0 and counts[1] == 0
    assert all(t['populations']['target']['mean_delta_mv'] == 0 for t in result['trace'])
    result, counts = run_condition(*args, 'no_input')
    assert not counts.any()
    assert result['duration_ms'] == 200
