import io
from types import SimpleNamespace

import numpy as np

from flydoom.teacher_play import run_episode, timing_table


def test_autonomous_history_uses_own_actions_and_resets_per_episode():
    class Game:
        def set_seed(self, seed): self.seed = seed
        def new_episode(self): self.ticks = 0
        def is_episode_finished(self): return self.ticks >= 5
        def get_state(self): return SimpleNamespace(screen_buffer=np.zeros((8, 8, 3), dtype=np.uint8))
        def make_action(self, buttons, ticks):
            expected = [1, 0, 0] if self.ticks < 4 else [0, 0, 1]
            assert buttons == expected
            self.ticks += ticks
        def get_total_reward(self): return 100 - self.ticks
        def get_game_variable(self, variable):
            assert self.is_episode_finished()
            return 1

    states = []
    def choose(state):
        states.append(state)
        assert "reward" not in state and "kills" not in state
        return 3 if state["timing"]["completed_decisions"] == 0 else 1

    game = Game()
    for seed in (10, 11):
        summary = run_episode(game, ["ATTACK", "MOVE_RIGHT", "MOVE_LEFT"], choose, seed,
                              max_decisions=3, trace=io.StringIO())
        assert summary["decisions"] == 2 and summary["kills"] == 1
        assert summary["end_reason"] == "game_finished"
    assert states[1]["previous_action"] == "ATTACK"
    assert states[1]["recent_actions"] == ["NONE", "NONE", "ATTACK"]
    assert states[2]["timing"]["completed_decisions"] == 0
    assert states[2]["previous_action"] == "WAIT"


def test_timing_control_uses_training_counts_only():
    states = [{"timing": {"completed_decisions": 0}, "previous_action": "WAIT"}] * 3
    table, fallback = timing_table({"states": states, "actions": np.array([0, 1, 1])})
    assert table[(0, "WAIT")] == 1
    assert fallback == 1


def test_single_frame_comparison_receives_exact_original_observation():
    from flydoom.learning_data import image_state
    from flydoom.teacher_play import policy_observation
    from flydoom.temporal_data import ObservationHistory, TEMPORAL_FORMAT
    current = image_state(np.zeros((8, 8, 3), dtype=np.uint8), 0)
    temporal = ObservationHistory().observe(current)
    assert policy_observation(temporal, "single_frame") == current
    assert list(policy_observation(temporal, "single_frame")) == list(current)
    assert policy_observation(temporal, TEMPORAL_FORMAT) == temporal
