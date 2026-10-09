"""Plasticity must change real simulator edges while preserving graph constraints."""

import numpy as np
import pytest
from scipy import sparse

from flydoom.simulation import LIFNetwork, LIFParameters
from flydoom.synaptic import PlasticEdges


def test_only_masked_magnitudes_change_and_restore_exactly():
    weights = sparse.csr_matrix(np.array([[0., -2., 3.], [4., 0., 1.], [-5., 0., 0.]], dtype=np.float32))
    before, indices, indptr = weights.data.copy(), weights.indices.copy(), weights.indptr.copy()
    patch = PlasticEdges(weights, [0, 3], [0, 1], ['a', 'b'])
    patch.apply([.75, 1.25])
    np.testing.assert_array_equal(weights.indices, indices)
    np.testing.assert_array_equal(weights.indptr, indptr)
    np.testing.assert_array_equal(weights.data[[1, 2, 4]], before[[1, 2, 4]])
    np.testing.assert_array_equal(np.sign(weights.data), np.sign(before))
    np.testing.assert_allclose(weights.data[[0, 3]], before[[0, 3]] * [.75, 1.25])
    patch.apply([1, 1])
    np.testing.assert_array_equal(weights.data, before)


@pytest.mark.parametrize('gains', [[0.], [-1.], [1.3], [float('nan')], [1., 1.]])
def test_out_of_bound_gain_rejected_without_mutation(gains):
    weights = sparse.csr_matrix([[1.]])
    patch = PlasticEdges(weights, [0], [0], ['a'])
    with pytest.raises(ValueError): patch.apply(gains)
    assert weights.data[0] == 1


def test_synaptic_change_reaches_lif_dynamics_not_just_metadata():
    params = LIFParameters()
    network = LIFNetwork(sparse.csr_matrix([[0., 0.], [2., 0.]]), params)
    patch = PlasticEdges(network.weights, [0], [0], ['a'])
    def response(gain):
        patch.apply([gain]); network.reset()
        for _ in range(18): network.step(np.array([100., 0.], np.float32))
        return network.voltage[1]
    low, high = response(.75), response(1.25)
    assert high > low > params.rest_mv


def test_duplicate_or_zero_edge_mask_rejected():
    weights = sparse.csr_matrix(([0., 2.], [0, 1], [0, 2, 2]), shape=(2, 2))
    with pytest.raises(ValueError): PlasticEdges(weights, [1, 1], [0, 0], ['a'])
    with pytest.raises(ValueError): PlasticEdges(weights, [0], [0], ['a'])


def test_direct_eligibility_matches_finite_difference_with_fixed_spike_schedule():
    from flydoom.synaptic_eligibility import Eligibility
    def run(weight, observe=False):
        network = LIFNetwork(sparse.csr_matrix([[0., 0.], [weight, 0.]]))
        trace = Eligibility(network, np.array([0]))
        for _ in range(24):
            arrived = network.pending[network.tick % len(network.pending)].copy()
            active = network.refractory == 0
            spikes = network.step(np.array([100., 0.], np.float32))
            if observe: trace.advance(arrived, active, spikes)
        return float(network.voltage[1]), float(trace.voltage[0])
    base, derivative = run(2., True)
    plain, _ = run(2.)
    assert plain == base
    plus, _ = run(2.02); minus, _ = run(1.98)
    assert derivative > 0
    assert derivative == pytest.approx((plus-minus)/.04, rel=.004, abs=.001)


def test_individual_edges_in_same_group_can_learn_different_gains():
    from flydoom.synaptic_eligibility import EdgePatch
    weights = sparse.csr_matrix([[0., 2.], [-3., 0.]])
    patch = EdgePatch(weights, [0,1], [0,0], ['same_annotation'])
    patch.apply([.99,1.01])
    np.testing.assert_allclose(weights.data,[1.98,-3.03])
    assert patch.gains[0] == pytest.approx(1.)
    np.testing.assert_array_equal(np.sign(weights.data),[1,-1])
