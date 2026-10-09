"""Versioned six-action controller; historical four-action artifacts stay intact."""

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
import vizdoom as vzd

from flydoom import action_memory
from flydoom.calibration import output_features
from flydoom.student import SpikingReadout

ACTIONS = ("WAIT", "MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "MOVE_FORWARD", "MOVE_BACKWARD")
MEMORY = tuple(f"past_{lag}_{action}" for lag in (3, 2, 1) for action in ACTIONS) + action_memory.NAMES[-5:]
SCHEMA = "movement_student_v1"


def make_game(seed):
    package = Path(os.path.relpath(Path(vzd.__file__).parent))
    game = vzd.DoomGame()
    try:
        game.load_config(str(package / "scenarios/basic.cfg"))
        game.set_vizdoom_path(str(package / ("vizdoom.exe" if os.name == "nt" else "vizdoom")))
        game.set_doom_game_path(str(package / "freedoom2.wad"))
        game.set_available_buttons([getattr(vzd.Button, name) for name in ACTIONS[1:]])
        game.set_available_game_variables([vzd.GameVariable.KILLCOUNT, vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y])
        game.set_mode(vzd.Mode.PLAYER)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.set_render_hud(False)
        game.set_sound_enabled(False)
        game.set_window_visible(False)
        game.set_seed(seed)
        game.init()
        assert [b.name for b in game.get_available_buttons()] == list(ACTIONS[1:])
        return game
    except BaseException:
        game.close()
        raise


def apply_action(game, action):
    if type(action) is not int or action not in range(6):
        raise ValueError("Require one of six canonical actions")
    before = game.get_total_reward()
    position = lambda: [game.get_game_variable(v) for v in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y)]
    start = position()
    old_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
    tics = 0
    for _ in range(4):
        if game.is_episode_finished():
            break
        game.make_action([int(i == action) for i in range(1, 6)], 1)
        tics += 1
    kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
    return {"reward_delta": game.get_total_reward() - before, "kill_delta": kills - old_kills,
            "kills": kills, "game_tics": tics, "episode_finished": game.is_episode_finished(),
            "position_before": start, "position_after": position(),
            "note": "Engine position is diagnostics only; it is never a policy or teacher input."}


class Memory:
    def __init__(self):
        self.actions = []

    def encode(self):
        n = len(self.actions)
        recent = [-1] * max(0, 3 - n) + self.actions[-3:]
        active = next((i for i in range(n - 1, -1, -1) if self.actions[i] != 0), None)
        shot = next((i for i in range(n - 1, -1, -1) if self.actions[i] == 3), None)
        return np.asarray([float(a == i) for a in recent for i in range(6)] +
                          [n / 75, 0 if active is None else (n - active) / 75,
                           0 if shot is None else (n - shot) / 75, float(active is not None), float(shot is not None)], dtype=np.float32)

    def advance(self, action):
        if type(action) is not int or action not in range(6) or len(self.actions) >= 75:
            raise ValueError("Invalid six-action history")
        self.actions.append(action)


def vector(controller, memory):
    previous = memory.actions[-1] if memory.actions else 0
    return np.concatenate((output_features(controller), memory.encode(), np.eye(6, dtype=np.float32)[previous]))


class MovementReadout(SpikingReadout):
    def __init__(self, input_size, hidden=64):
        super().__init__(input_size, hidden)
        self.readout = nn.Linear(hidden, 6)


def expand(parent, output_count):
    """Copy matching features by name; append genuine forward/backward history channels."""
    size = output_count * 2
    old_names = [f"neural:{i}" for i in range(size)] + list(action_memory.NAMES) + list(ACTIONS[:4])
    names = [f"neural:{i}" for i in range(size)] + list(MEMORY) + list(ACTIONS)
    torch.manual_seed(1909)
    model = MovementReadout(len(names), parent.input.out_features)
    with torch.no_grad():
        model.input.weight.zero_()
        for old_index, name in enumerate(old_names):
            index = names.index(name)
            model.input.weight[:, index].copy_(parent.input.weight[:, old_index])
            model.mean[index].copy_(parent.mean[old_index])
            model.scale[index].copy_(parent.scale[old_index])
        model.input.bias.copy_(parent.input.bias)
        model.recurrent.weight.copy_(parent.recurrent.weight)
        model.readout.weight[:4].copy_(parent.readout.weight)
        model.readout.bias[:4].copy_(parent.readout.bias)
        model.readout.weight[4:].zero_()
        model.readout.bias[4:].fill_(-2.)
    model.eval()
    return model
