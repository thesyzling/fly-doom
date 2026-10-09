"""Reproducible shooting and distance-control experience with explicit reward provenance."""

import os
from pathlib import Path

import numpy as np
import vizdoom as vzd

from flydoom.movement_core import ACTIONS, apply_action

TASKS = ("basic", "distance")


def make_game(seed, task="basic"):
    if task not in TASKS:
        raise ValueError("Unknown curriculum task")
    package = Path(os.path.relpath(Path(vzd.__file__).parent))
    game = vzd.DoomGame()
    try:
        game.load_config(str(package / "scenarios/basic.cfg"))
        game.set_vizdoom_path(str(package / ("vizdoom.exe" if os.name == "nt" else "vizdoom")))
        game.set_doom_game_path(str(package / "freedoom2.wad"))
        game.set_available_buttons([getattr(vzd.Button, name) for name in ACTIONS[1:]])
        game.set_available_game_variables([vzd.GameVariable.KILLCOUNT, vzd.GameVariable.POSITION_X,
                                          vzd.GameVariable.POSITION_Y, vzd.GameVariable.HEALTH])
        game.set_objects_info_enabled(True)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_render_hud(False)
        game.set_seed(int(seed))
        game.init()
        # init() starts an episode itself; reseed before the recorded episode so
        # a reused reservation engine and a fresh collection engine agree.
        game.set_seed(int(seed))
        game.new_episode()
        # The same range objective has near and far starting conditions. No task ID
        # or engine geometry enters the policy; geometry is reward diagnostics only.
        if task == "distance" and seed % 2 == 0:
            for _ in range(10):
                if game.is_episode_finished(): break
                apply_action(game, 4)
        return game
    except BaseException:
        game.close()
        raise


def range_state(game):
    state = game.get_state()
    if state is None:
        return {"distance": None, "potential": 0.}
    x = game.get_game_variable(vzd.GameVariable.POSITION_X)
    y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
    monsters = [o for o in state.objects if o.name not in {"DoomPlayer", "BulletPuff", "Blood"}
                and any(s in o.name.lower() for s in ("zombie", "shotgun", "imp", "demon", "cacodemon", "baron", "hellknight"))]
    if not monsters:
        return {"distance": None, "potential": 0.}
    distance = min(float(np.hypot(o.position_x - x, o.position_y - y)) for o in monsters)
    # A research shaping objective, not the native Doom score or a biological law.
    potential = -max(0., abs(distance - 180.) - 35.) / 180.
    return {"distance": distance, "potential": potential}


def act(game, action, task):
    before = range_state(game)
    health = game.get_game_variable(vzd.GameVariable.HEALTH)
    effect = apply_action(game, int(action))
    after = range_state(game)
    damage = max(0., health - game.get_game_variable(vzd.GameVariable.HEALTH))
    progress = (after["potential"] - before["potential"]) if before["distance"] is not None and after["distance"] is not None else 0.
    effect.update(range_before=before["distance"], range_after=after["distance"], range_progress=progress,
                  damage=damage, task=task, shaping_reward=progress if task == "distance" else 0.)
    return effect


def learning_target(teacher, action, effect):
    """Positive measured progress supplies a bounded auxiliary self-imitation target."""
    probabilities = np.asarray(teacher, np.float32)
    alpha = min(.35, max(0., effect["shaping_reward"]) * 3)
    result = (1 - alpha) * probabilities
    result[action] += alpha
    return result.tolist(), float(alpha)
