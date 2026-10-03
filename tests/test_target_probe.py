from types import SimpleNamespace

import numpy as np
import pytest

from flydoom.bridge import encode_frame
from flydoom.target_probe import (analysis_indices, classification_metrics, ridge_predict,
                                 rgb_bins, target_label)


def test_engine_labels_are_target_specific_and_center_is_not_a_side():
    player = SimpleNamespace(object_name="DoomPlayer", x=0, width=30, y=0, height=30)
    target = SimpleNamespace(object_name="Cacodemon", x=150, width=20, y=90, height=30)
    assert target_label([player, target], 320)["side"] == -1
    target.x = 30
    assert target_label([player, target], 320)["side"] == 0
    target.x = 250
    assert target_label([player, target], 320)["side"] == 1
    assert target_label([player], 320)["reason"] == "missing_or_ambiguous_target"
    assert target_label([target, target], 320)["side"] == -1


def test_rgb_preserves_color_lost_by_brightness():
    red = np.zeros((16, 16, 3), dtype=np.uint8)
    red[:, :8, 0] = 255
    blue = red[:, :, ::-1].copy()
    np.testing.assert_array_equal(encode_frame(red), encode_frame(blue))
    assert not np.array_equal(rgb_bins(red), rgb_bins(blue))
    np.testing.assert_allclose(rgb_bins(red).reshape(64, 3).mean(axis=1), encode_frame(red))


def test_split_deduplication_keeps_training_priority_and_excludes_center():
    rows = [{"split": s, "frame_sha256": key, "target": {"side": label}}
            for s, key, label in [("validation", "shared", 1), ("train", "shared", 1),
                                  ("train", "shared", 1), ("validation", "unique", 0),
                                  ("validation", "center", -1)]]
    indices = analysis_indices(rows)
    assert indices["train"].tolist() == [1]
    assert indices["validation"].tolist() == [3]


def test_ridge_recovers_signal_without_fitting_query_statistics():
    x = np.array([[-2, 5], [-1, 5], [1, 5], [2, 5]], dtype=float)
    y = np.array([0, 0, 1, 1])
    query = np.array([[-3, 5], [3, 5]], dtype=float)
    scores = ridge_predict(x, y, query)
    assert (scores >= 0).tolist() == [False, True]
    extra = ridge_predict(x, y, np.concatenate([query, [[10000, -50000]]]))
    np.testing.assert_allclose(scores, extra[:2], atol=1e-10)
    with pytest.raises(ValueError):
        ridge_predict(x, np.zeros(4, dtype=int), query)


def test_balanced_metrics_expose_majority_only_predictor():
    metrics = classification_metrics([0, 0, 0, 1], [-1, -1, -1, -1])
    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["confusion_actual_rows_predicted_columns"] == [[3, 0], [1, 0]]
