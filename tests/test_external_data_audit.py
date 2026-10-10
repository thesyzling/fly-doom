import pytest

from flydoom.external_data_audit import map_gamewam


@pytest.mark.parametrize("index, expected", [(0, "ATTACK"), (1, "MOVE_FORWARD"), (2, "MOVE_BACKWARD"), (3, "MOVE_LEFT"), (4, "MOVE_RIGHT")])
def test_exact_button_mapping(index, expected):
    action = [0.] * 9; action[index] = 1
    assert map_gamewam(action) == (expected, None)


def test_rotation_is_never_relabeled_as_strafing():
    assert map_gamewam([0, 0, 0, 0, 0, 0, 1, 0, 0]) == (None, "rotation_not_supported")
    assert map_gamewam([0, 0, 0, 0, 0, 0, 0, 0, .0001]) == (None, "rotation_not_supported")


def test_compound_speed_and_invalid_actions_are_rejected():
    assert map_gamewam([1, 1, 0, 0, 0, 0, 0, 0, 0])[1] == "simultaneous_buttons"
    assert map_gamewam([0, 1, 0, 0, 0, 1, 0, 0, 0])[1] == "speed_not_supported"
    assert map_gamewam([0, .5, 0, 0, 0, 0, 0, 0, 0])[1] == "invalid_vector"
    assert map_gamewam([float("nan")] * 9)[1] == "invalid_vector"
    assert map_gamewam([0] * 8)[1] == "invalid_vector"
    assert map_gamewam([0] * 9) == ("WAIT", None)
