import numpy as np
import pytest
from scipy import sparse

from flydoom.graded_vision import GradedVision
from flydoom.stable_vision import StableGradedVision, evoked_release


def test_release_preserves_sub_ulp_signals_and_rectification():
    baseline = np.array([1, 1, 1, -1, -1, 0], dtype=np.float32)
    delta = np.array([1e-9, -1e-9, -2, .5, 2, -.5], dtype=np.float32)
    expected = np.maximum(baseline.astype(float)+delta, 0)-np.maximum(baseline.astype(float), 0)
    np.testing.assert_allclose(evoked_release(baseline, delta), expected, rtol=1e-7, atol=1e-16)
    assert (np.maximum(baseline+delta, 0)-np.maximum(baseline, 0))[0] == 0
    with pytest.raises(ValueError):
        evoked_release([1], [1, 2])


def test_deep_weak_path_matches_float64_and_preserves_disconnection():
    n = 8
    weights = sparse.diags([np.full(n-1, .05)], [-1], shape=(n, n), format='csr', dtype=np.float32)
    stable = StableGradedVision(weights, np.ones(n))
    legacy = GradedVision(weights, np.ones(n))
    reference = np.zeros(n)
    baseline = stable.baseline.astype(float)
    drive = np.r_[.01, np.zeros(n-1)].astype(np.float32)
    for _ in range(200):
        reference = stable.decay.astype(float)*reference + (1-stable.decay).astype(float)*(drive + weights.astype(float) @ (np.maximum(baseline+reference, 0)-np.maximum(baseline, 0)))
        stable.step(drive)
        legacy.step(drive)
    assert legacy.delta[-1] == 0
    assert stable.delta[-1] > 0
    np.testing.assert_allclose(stable.delta, reference, rtol=1e-3, atol=1e-16)
    stable.reset()
    for _ in range(200):
        stable.step(drive, connected=False)
    np.testing.assert_array_equal(stable.delta[1:], 0)
    stable.reset()
    stable.blocked_release[2] = True
    for _ in range(200):
        stable.step(drive)
    assert stable.delta[2] > 0
    np.testing.assert_array_equal(stable.delta[3:], 0)


def test_corrected_checkpoint_cannot_silently_load_with_legacy_arithmetic(tmp_path):
    import json
    from flydoom.adaptive_policy import load_any as legacy_load
    from flydoom.stable_vision import load_any, SCHEMA, MODE
    (tmp_path/'report.json').write_text(json.dumps({'schema': SCHEMA, 'status': 'completed',
        'neural_numerics': MODE, 'source_sha256': {}}))
    with pytest.raises(ValueError):
        legacy_load(tmp_path)
    with pytest.raises(ValueError, match='source-pinned'):
        load_any(tmp_path)
