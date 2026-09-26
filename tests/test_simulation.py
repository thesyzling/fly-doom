import numpy as np
import pytest
from scipy import sparse

from flydoom.simulation import LIFNetwork, LIFParameters, signed_weights, transmitter_signs
from flydoom.brain_probe import run_probe


def test_rest_and_analytic_constant_drive():
    p = LIFParameters(dt_ms=1)
    net = LIFNetwork(sparse.csr_matrix((1, 1)), p)
    for _ in range(20):
        assert not net.step([0]).any()
    assert net.voltage[0] == p.rest_mv
    for _ in range(10):
        assert not net.step([2]).any()
    expected = p.rest_mv + 2 * (1 - np.exp(-10 / p.membrane_tau_ms))
    assert net.voltage[0] == pytest.approx(expected, abs=2e-5)


@pytest.mark.parametrize("weight", [10, -10])
def test_direction_and_exact_delay(weight):
    # A spike at 1 ms reaches its target at 3 ms; integration follows arrival.
    net = LIFNetwork(sparse.csr_matrix([[0, 0], [weight, 0]]),
                     LIFParameters(dt_ms=1, delay_ms=2))
    assert net.step([1000, 0]).tolist() == [True, False]
    assert net.voltage[1] == -52
    net.step([0, 0])
    net.step([0, 0])
    assert net.voltage[1] == -52
    net.step([0, 0])
    assert (net.voltage[1] > -52) == (weight > 0)
    assert net.current[0] == 0


def test_refractory_counts_complete_intervals():
    net = LIFNetwork(sparse.csr_matrix((1, 1)), LIFParameters(dt_ms=1, refractory_ms=2))
    spikes = [bool(net.step([1000])[0]) for _ in range(7)]
    assert spikes == [True, False, False, True, False, False, True]


def test_reset_clears_current_refractory_and_delayed_spikes():
    net = LIFNetwork(sparse.csr_matrix([[0, 0], [10, 0]]))
    net.step([1000, 0])
    net.reset()
    for _ in range(20):
        assert not net.step([0, 0]).any()
    np.testing.assert_array_equal(net.voltage, [-52, -52])
    assert not net.current.any()


def test_current_decays_during_refractory_and_equal_time_constants():
    p = LIFParameters(dt_ms=1, membrane_tau_ms=5, synapse_tau_ms=5)
    net = LIFNetwork(sparse.csr_matrix((1, 1)), p)
    net.current[0] = 3
    net.step([0])
    assert net.voltage[0] == pytest.approx(-52 + 3 * (1 / 5) * np.exp(-1 / 5), abs=1e-5)
    net.refractory[0] = 2
    before = net.current[0]
    net.step([1000])
    assert net.voltage[0] == -52
    assert net.current[0] == pytest.approx(before * np.exp(-1 / 5))


def test_sign_assignment_and_presynaptic_columns():
    def row(kind="", known="", predicted=""):
        return {"cell_type": kind, "known_nt": known, "top_nt": predicted}
    signs, rules = transmitter_signs([
        row("R1-6", predicted="acetylcholine"),
        row(known="acetylcholine, sNPF", predicted="gaba"),
        row(known="acetylcholine, gaba"), row(predicted="dopamine"),
        row(predicted="gaba"), row(),
    ])
    np.testing.assert_array_equal(signs, [-1, 1, 0, 0, -1, 0])
    assert sum(rules.values()) == 6
    matrix = sparse.csr_matrix([[0, 2, 3], [4, 0, 0], [0, 1, 0]])
    weights = signed_weights(matrix, signs[:3], 0.5)
    np.testing.assert_array_equal(weights.toarray(), [[0, 1, 0], [-2, 0, 0], [0, 0.5, 0]])
    assert weights.nnz == matrix.nnz


def test_disconnected_control_prevents_transmission():
    net = LIFNetwork(sparse.csr_matrix([[0, 0], [100, 0]]))
    for _ in range(40):
        net.step([100, 0], connected=False)
    assert net.voltage[1] == -52
    assert net.current[1] == 0


@pytest.mark.parametrize("kwargs", [{"dt_ms": 0}, {"delay_ms": 0},
    {"membrane_tau_ms": -1}, {"threshold_mv": -60}, {"dt_ms": float("nan")}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        LIFParameters(**kwargs)


def test_invalid_drive_does_not_advance_time():
    net = LIFNetwork(sparse.csr_matrix((1, 1)))
    with pytest.raises(ValueError):
        net.step([float("nan")])
    assert net.tick == 0


def test_probe_compares_connected_and_disconnected_with_clean_reset():
    net = LIFNetwork(sparse.csr_matrix([[0, 0], [100, 0]]), LIFParameters(dt_ms=1))
    kwargs = dict(duration_ms=30, pulse_start_ms=0, pulse_end_ms=10, amplitude_mv=1000)
    connected, counts = run_probe(net, [0], [1], **kwargs)
    disconnected, _ = run_probe(net, [0], [1], connected=False, **kwargs)
    assert connected["outside_stimulus_spikes"] > 0
    assert connected["descending_spikes"] == int(counts[1])
    assert disconnected["outside_stimulus_spikes"] == 0
    assert disconnected["outside_stimulus_max_deviation_mv"] == 0
    assert connected["trace"][-1]["cumulative_spikes"] == int(counts.sum())


def test_subthreshold_time_step_refinement_matches_analytic_solution():
    values = []
    for dt in (1, 0.5, 0.25):
        net = LIFNetwork(sparse.csr_matrix((1, 1)), LIFParameters(dt_ms=dt))
        for _ in range(round(10 / dt)):
            net.step([2])
        values.append(float(net.voltage[0]))
    np.testing.assert_allclose(values, -52 + 2 * (1 - np.exp(-0.5)), atol=5e-5, rtol=0)


def test_temporal_recording_preserves_totals_and_does_not_change_dynamics():
    net = LIFNetwork(sparse.csr_matrix([[0, 0], [100, 0]]), LIFParameters(dt_ms=1))
    kwargs = dict(duration_ms=25, pulse_start_ms=0, pulse_end_ms=10, amplitude_mv=1000)
    _, reference = run_probe(net, [0], [1], **kwargs)
    recording = {}
    _, counts = run_probe(net, [0], [1], recording=recording, **kwargs)
    np.testing.assert_array_equal(counts, reference)
    np.testing.assert_array_equal(recording["sample_times_ms"], [0, 10, 20, 25])
    np.testing.assert_array_equal(recording["spike_bins"].sum(axis=0), counts)
    np.testing.assert_array_equal(recording["voltage_mv"][0], [-52, -52])
    np.testing.assert_array_equal(recording["voltage_mv"][-1], net.voltage)
    assert not recording["spike_bins"][0].any()
