import io
import numpy as np
import pytest

from flydoom.independent_physiology import RestrictedReader, shape_metrics


def test_external_serialized_code_global_is_rejected():
    with pytest.raises(ValueError, match='Unsupported serialized global'):
        RestrictedReader(io.BytesIO(b'cbuiltins\neval\n.')).load()


def test_timecourse_metric_uses_stimulus_window_and_physical_time():
    time = np.array([-.1, 0., .1, .2, .5, 1., 2.])
    observed = np.array([0., 0., .5, 1., .5, .2, 9.])
    prediction = np.array([0., 0., 1., .5, .5, .2, 0.])
    result = shape_metrics(time, observed, prediction)
    assert result['measured_peak_ms'] == 200
    assert result['predicted_peak_ms'] == 100
    assert result['peak_error_ms'] == 100
    assert result['normalized_rmse_during_step'] == pytest.approx(np.sqrt(.1))
