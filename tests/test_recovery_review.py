import numpy as np

from flydoom.recovery_review import summarize


def test_review_streaks_and_disagreements_keep_applied_buttons_separate():
    result = summarize([0, 0, 1, 0, 0, 0, 3], np.eye(4)[[0, 2, 1, 0, 2, 2, 3]])
    assert result["longest_wait"] == 3 and result["longest_wait_start"] == 4
    assert result["teacher_disagreements"] == result["teacher_active_when_student_waited"] == 3
    assert result["student_action_counts"] == [5, 1, 0, 1]
    assert result["teacher_action_counts"] == [2, 1, 3, 1]


def test_review_without_waits_has_no_wait_start():
    result = summarize([1, 3], np.eye(4)[[1, 3]])
    assert result["longest_wait"] == 0 and result["longest_wait_start"] is None
