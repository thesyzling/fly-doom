import numpy as np
import pytest
from scipy import sparse

from flydoom.graded_vision import GradedVision
from flydoom.visual_observer import VisualObserver, camera_projection, sample_frame


def test_camera_retains_offscreen_mask_and_axis_orientation():
    rays = np.array([[0, -1, 0], [-.2, -.98, 0], [.2, -.98, 0], [0, 1, 0.]])
    uv, visible, camera = camera_projection(rays, 320, 240)
    np.testing.assert_allclose(uv[0], [.5, .5])
    assert visible.tolist() == [True, True, True, False]
    assert uv[1, 0] > uv[2, 0]
    assert camera['horizontal_fov_degrees'] == 90
    with pytest.raises(ValueError):
        camera_projection([[0, 1, 0], [0, -1, 0]], 320, 240)


def test_sampling_is_bilinear_and_invisible_receptors_are_neutral():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[1, 1] = 255
    value = sample_frame(frame, np.array([[.5, .5], [1, 1], [-100, 100]]), np.array([True, True, False]))
    np.testing.assert_allclose(value, [-.05, .1, 0])
    with pytest.raises(ValueError):
        sample_frame(frame.astype(float), np.array([[0, 0]]), np.array([True]))


def make_observer():
    observer = VisualObserver.__new__(VisualObserver)
    observer.ids = np.array([720575940600010668, 720575940600010669], dtype=np.uint64)
    observer.model = GradedVision(sparse.csr_matrix([[0, 0], [-.5, 0]]), [5, 5], dt_ms=1)
    observer.types = np.array(['R1-6', 'L1'])
    observer.inputs = np.array([0])
    observer.groups = {'R1-6': np.array([0]), 'L1': np.array([1])}
    observer.directions = np.array([[0, -1, 0.]])
    observer.identity = {'controls_action': False}
    observer.reset()
    return observer


def test_stateful_observer_preserves_exact_ids_and_signed_contributions(tmp_path):
    observer = make_observer()
    frame = np.full((4, 4, 3), 255, np.uint8)
    initial = observer.observe(frame, 0, 1, tmp_path)
    assert initial['trace'] == [] and initial['simulated_ms'] == 0
    assert all(c['delta'] == 0 for c in initial['cells'])
    result = observer.observe(frame, 4, 2, tmp_path)
    assert result['interval_ms'] == 4000/35
    assert result['trace'][-1]['time_ms'] == pytest.approx(4000/35)
    cell = result['cells'][1]
    assert cell['delta'] < 0
    edge = cell['edges'][0]
    assert edge['source'] == '720575940600010668'
    assert edge['contribution'] == pytest.approx(edge['weight'] * edge['presynaptic_release'])
    assert cell['total_synaptic_drive'] == edge['contribution']
    with np.load(tmp_path/'0002.npz') as saved:
        assert saved['root_ids'].dtype == np.uint64
        np.testing.assert_array_equal(saved['frame'], frame)
    observer.observe(frame, 1, 3, tmp_path)
    assert observer.elapsed_ms == pytest.approx(5000/35)
    observer.reset()
    assert observer.elapsed_ms == 0 and not observer.model.delta.any()


def test_observer_rejects_invalid_game_interval(tmp_path):
    with pytest.raises(ValueError):
        make_observer().observe(np.zeros((4,4,3), np.uint8), 5, 1, tmp_path)


@pytest.mark.parametrize('phase', ['completed', 'stopped', 'error'])
def test_laboratory_stop_is_idempotent_after_terminal_state(phase):
    from threading import Condition
    from flydoom.laboratory import LabSession
    session = LabSession.__new__(LabSession)
    session.condition, session.phase, session.paused = Condition(), phase, False
    session.control('stop')
    assert session.phase == phase and session.paused
    session.control('pause')
