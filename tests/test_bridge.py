import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from flydoom.bridge import NeuralController, NeuralMapping, decode_spikes, encode_frame
from flydoom.simulation import LIFNetwork
from flydoom import brain_doom
from flydoom.data import digest


def small_controller(**kwargs):
    # A single visual input can reach only the LEFT output neuron.
    weights = sparse.csr_matrix(([100.0], ([1], [0])), shape=(4, 4))
    mapping = NeuralMapping(np.array([0]), np.array([0]),
                            (np.array([1]), np.array([2]), np.array([3])))
    return NeuralController(LIFNetwork(weights), mapping, input_gain_mv=100, **kwargs)


def test_image_bins_preserve_space_and_normalization():
    frame = np.zeros((16, 24, 3), dtype=np.uint8)
    frame[:2, :3] = 255
    frame[-2:, -3:, 0] = 255
    features = encode_frame(frame)
    assert features[0] == 1
    assert features[-1] == pytest.approx(1 / 3)
    assert np.count_nonzero(features) == 2
    assert encode_frame(np.full((17, 25, 3), 255, dtype=np.uint8)).tolist() == [1] * 64
    with pytest.raises(ValueError):
        encode_frame(np.zeros((3, 16, 24), dtype=np.uint8))


def test_readout_normalizes_population_size_and_waits_for_silence_or_tie():
    groups = (np.array([0, 1]), np.array([2]), np.array([3]))
    action, rates = decode_spikes([2, 2, 3, 0], groups, 100)
    assert action == "MOVE_RIGHT"
    assert rates == {"MOVE_LEFT": 20, "MOVE_RIGHT": 30, "ATTACK": 0}
    assert decode_spikes([2, 2, 2, 0], groups, 100)[0] == "WAIT"
    assert decode_spikes([0, 0, 0, 0], groups, 100)[0] == "WAIT"


def test_pixels_need_synaptic_transmission_to_produce_action():
    white = np.full((8, 8, 3), 255, dtype=np.uint8)
    result = small_controller().decide(white)
    assert result["action"] == "MOVE_LEFT"
    assert result["descending_spikes"]["MOVE_LEFT"] > 0
    disconnected = small_controller(control="disconnected").decide(white)
    assert disconnected["input_spikes"] > 0
    assert disconnected["action"] == "WAIT"
    assert sum(disconnected["descending_spikes"].values()) == 0
    zero = small_controller(control="zero-input").decide(white)
    assert zero["features"] == [1] * 64
    assert zero["spikes"] == 0
    assert zero["action"] == "WAIT"
    assert small_controller().decide(np.zeros_like(white))["action"] == "WAIT"


def test_controller_retains_state_between_frames_and_resets_between_episodes():
    frame = np.full((8, 8, 3), 255, dtype=np.uint8)
    controller = small_controller(brain_ms=25)
    first = controller.decide(frame)
    second = controller.decide(frame)
    assert first["brain_time_ms"] == 25
    assert second["brain_time_ms"] == 50
    controller.reset()
    repeated = controller.decide(frame)
    for key in first.keys() - {"compute_seconds"}:
        assert repeated[key] == first[key]


def test_mapping_uses_exact_ids_and_is_independent_of_annotation_row_order():
    ids = np.arange(2**60, 2**60 + 70, dtype=np.uint64)
    rows = [{"super_class": "visual_projection" if i < 64 else "descending"} for i in range(70)]
    signs = np.ones(70)
    mapping = NeuralMapping.from_annotations(ids, rows, signs)
    permutation = np.random.default_rng(7).permutation(70)
    shuffled = NeuralMapping.from_annotations(ids[permutation], [rows[i] for i in permutation], signs[permutation])
    assert mapping.describe(ids) == shuffled.describe(ids[permutation])
    assert mapping.describe(ids)["input_root_ids"][:2] == [str(2**60), str(2**60 + 1)]


def test_output_cells_cannot_receive_direct_pixel_drive():
    mapping = NeuralMapping(np.array([0]), np.array([0]),
                            (np.array([0]), np.array([2]), np.array([3])))
    with pytest.raises(ValueError, match="disjoint"):
        NeuralController(LIFNetwork(sparse.csr_matrix((4, 4))), mapping)


@pytest.fixture
def small_game_bridge(monkeypatch):
    """Use real neural dynamics with a tiny graph and a bounded game double."""
    ids = np.arange(67, dtype=np.uint64)
    rows = [{"super_class": "visual_projection" if i < 64 else "descending"} for i in range(67)]
    weights = sparse.csr_matrix(([100.0], ([64], [0])), shape=(67, 67))
    monkeypatch.setattr(brain_doom, "load_connectome", lambda *args: (ids, rows, weights, np.ones(67), {}))

    class Game:
        def __init__(self):
            self.closed = False
            self.interrupt = False

        def load_config(self, value): pass
        def set_vizdoom_path(self, value): pass
        def set_doom_game_path(self, value): pass
        def set_window_visible(self, value): pass
        def set_mode(self, value): pass
        def set_screen_resolution(self, value): pass
        def set_screen_format(self, value): pass
        def set_render_hud(self, value): pass
        def set_sound_enabled(self, value): pass
        def set_seed(self, value): pass
        def init(self): pass

        def get_available_buttons(self):
            # Deliberately differ from the readout's group order.
            return [brain_doom.vzd.Button.ATTACK, brain_doom.vzd.Button.MOVE_RIGHT,
                    brain_doom.vzd.Button.MOVE_LEFT]

        def new_episode(self):
            self.tick = 14
            self.reward = 0

        def get_state(self):
            if self.interrupt and self.tick > 14:
                raise KeyboardInterrupt
            return SimpleNamespace(screen_buffer=np.full((8, 8, 3), 255, dtype=np.uint8))

        def is_episode_finished(self): return False
        def get_episode_time(self): return self.tick
        def get_total_reward(self): return self.reward

        def make_action(self, action, tics):
            assert action == [0, 0, 1]
            self.tick += tics
            self.reward -= tics
            return -tics

        def close(self):
            self.closed = True

    game = Game()
    monkeypatch.setattr(brain_doom.vzd, "DoomGame", lambda: game)
    return game


def test_game_loop_records_applied_actions_resets_neurons_and_preserves_prior_runs(tmp_path, small_game_bridge):
    output = tmp_path / "run"
    report = brain_doom.run(data_dir=tmp_path, output=output, episodes=2,
                            max_decisions=1, input_gain_mv=100)
    trace = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
    assert [row["brain_time_ms"] for row in trace] == [50, 50]
    assert [row["action_buttons"] for row in trace] == [[0, 0, 1], [0, 0, 1]]
    assert [row["total_reward"] for row in trace] == [-4, -4]
    assert all(episode["end_reason"] == "decision_limit" for episode in report["episodes"])
    assert small_game_bridge.closed and report["status"] == "completed"
    assert report["output_sha256"]["decisions.jsonl"] == digest(output / "decisions.jsonl", "sha256")
    previous_report = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        brain_doom.run(data_dir=tmp_path, output=output)
    assert (output / "report.json").read_bytes() == previous_report


def test_interruption_closes_game_and_saves_completed_decisions(tmp_path, small_game_bridge):
    small_game_bridge.interrupt = True
    output = tmp_path / "interrupted"
    report = brain_doom.run(data_dir=tmp_path, output=output, max_decisions=3, input_gain_mv=100)
    assert report["status"] == "interrupted"
    assert report["episodes"][0]["end_reason"] == "interrupted"
    assert report["episodes"][0]["decisions"] == 1
    assert len((output / "decisions.jsonl").read_text().splitlines()) == 1
    assert json.loads((output / "report.json").read_text())["status"] == "interrupted"
    assert small_game_bridge.closed
