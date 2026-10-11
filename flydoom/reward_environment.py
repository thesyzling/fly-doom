"""Explicit training episode budgets; the standard test environment stays fixed."""
import os
from pathlib import Path

import numpy as np
import vizdoom as vzd

from flydoom.adaptive_policy import ACTIONS, action


def open_game(opening):
    timeout = opening.get('timeout_tics', 300)
    if type(timeout) is not int or not 300 <= timeout <= 1200:
        raise ValueError('Training timeout must be 300 through 1200 tics')
    package = Path(os.path.relpath(Path(vzd.__file__).parent))
    game = vzd.DoomGame()
    try:
        game.load_config(str(package/'scenarios/basic.cfg'))
        game.set_vizdoom_path(str(package/('vizdoom.exe' if os.name == 'nt' else 'vizdoom')))
        game.set_doom_game_path(str(package/'freedoom2.wad'))
        game.set_available_buttons([getattr(vzd.Button, name) for name in ACTIONS[1:]])
        game.set_available_game_variables([vzd.GameVariable.KILLCOUNT, vzd.GameVariable.POSITION_X,
                                          vzd.GameVariable.POSITION_Y])
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_window_visible(False)
        game.set_render_hud(False)
        game.set_labels_buffer_enabled(False)
        game.set_sound_enabled(False)
        game.set_console_enabled(False)
        game.set_episode_timeout(timeout)
        game.set_seed(opening['seed'])
        game.init()
        game.new_episode()
        rng = np.random.default_rng(opening['seed'])
        for _ in range(6):
            action(game, int(rng.choice([1, 2])))
        if game.is_episode_finished():
            raise ValueError('Warmup ended episode')
        return game
    except BaseException:
        game.close()
        raise
