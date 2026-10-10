import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat
from scipy.optimize import least_squares

from flydoom.timing_sources import decode_measurements, verify
from flydoom.timing_training import TimingCircuit, fit_scales, metrics, observe


def toy_circuit(dt_ms=1000/120/20):
    return TimingCircuit(sparse.csr_matrix([[0, 0, 0], [-.2, 0, 0], [-.3, 0, 0]]),
                         ['R1-6', 'L1', 'L2'], [0], dt_ms=dt_ms)


def test_source_adapter_preserves_dark_light_and_voltage_sign(tmp_path):
    time = np.arange(63)/120
    raw = np.zeros((2, 63))
    raw[0, 3], raw[1, 3] = -.02, .03
    path = tmp_path / 'source.mat'
    savemat(path, {'t': time, 'meanResp': raw})
    t, actual = decode_measurements(path)
    np.testing.assert_allclose(t, time*1000)
    assert actual[0, 3] == pytest.approx(.02)
    assert actual[1, 3] == pytest.approx(-.03)
    savemat(path, {'t': time*2, 'meanResp': raw})
    with pytest.raises(ValueError, match='timing axis'):
        decode_measurements(path)
    with pytest.raises(ValueError, match='Git blob'):
        verify(b'altered measurements', '0'*40)


def test_observation_bin_is_causal_and_has_correct_ramp_average():
    t = np.arange(0, 31, .1)
    values = t[:, None]
    measured = observe(t, values, np.array([0, 5, 10, 20]), 0, bin_ms=10)
    np.testing.assert_allclose(measured[:, 0], [0, 1.25, 5, 15], atol=1e-10)
    slow = observe(t, values, np.array([0, 5, 10, 20]), 5, bin_ms=10)
    assert slow[0, 0] == 0 and np.all(slow[1:, 0] < measured[1:, 0])


def test_real_graph_kernel_keeps_polarity_and_disconnected_control_silent():
    circuit = toy_circuit()
    t = np.arange(63)*1000/120
    prediction = circuit.simulate([10, 20, 30], t)
    assert prediction.shape == (2, 2, 63)
    np.testing.assert_array_equal(prediction[:, 0], -prediction[:, 1])
    assert prediction[0, 0].max() > 0 and prediction[0, 1].min() < 0
    np.testing.assert_array_equal(circuit.simulate([10, 20, 30], t, connected=False), 0)
    with pytest.raises(ValueError, match='constants'):
        circuit.simulate([0, 20, 30], t)


def test_recovers_one_identifiable_tau_when_other_dynamics_are_known():
    circuit = toy_circuit()
    t = np.arange(25)*1000/120
    target = circuit.simulate([7, 14, 28], t)
    def residual(log_tau):
        prediction = circuit.simulate([7, np.exp(log_tau[0]), 28], t)
        _, fitted = fit_scales(prediction, target)
        return (1000*(fitted-target)).ravel()
    result = least_squares(residual, [np.log(40)], bounds=(np.log(2), np.log(100)))
    assert np.exp(result.x[0]) == pytest.approx(14, rel=.002)
    assert np.linalg.norm(result.fun) < 1e-4


def test_shared_observation_gain_cannot_hide_dark_light_asymmetry():
    prediction = np.array([[[1., 2.], [-1., -2.]], [[2., 4.], [-2., -4.]]])
    target = prediction * np.array([2., 3.])[:, None, None]
    gain, fitted = fit_scales(prediction, target)
    np.testing.assert_allclose(gain, [2, 3])
    np.testing.assert_allclose(fitted, target)
    asymmetric = target.copy()
    asymmetric[0, 0] *= 4
    _, fitted = fit_scales(prediction, asymmetric)
    assert not np.allclose(fitted, asymmetric)


def test_peak_metric_does_not_force_a_slow_prediction_into_the_early_window():
    t = np.arange(63)*1000/120
    target = np.zeros((2, 2, 63))
    prediction = target.copy()
    target[:, 0, 3], target[:, 1, 3] = 1, -1
    prediction[:, 0, 7], prediction[:, 1, 7] = 1, -1
    result = metrics(prediction, target, t, np.ones(2))
    assert result['traces'][0]['first_peak_error_ms'] == pytest.approx(1000/30)
