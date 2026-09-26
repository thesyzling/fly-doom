"""Reproducible random-policy integration check, not a fly-brain agent."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random

import vizdoom as vzd


def run(episodes, seed, frame_skip):
    if episodes < 1 or frame_skip < 1 or not 0 <= seed <= 2**32 - episodes:
        raise ValueError("Require positive episodes/frame_skip and uint32 episode seeds")
    # Native ViZDoom on Windows may misdecode absolute non-ASCII paths.
    # Our local environment is inside the project, so relative paths avoid it.
    package = Path(os.path.relpath(Path(vzd.__file__).parent))
    config = package / "scenarios" / "basic.cfg"
    game = vzd.DoomGame()
    results = []
    try:
        game.load_config(str(config))
        game.set_vizdoom_path(str(package / ("vizdoom.exe" if os.name == "nt" else "vizdoom")))
        game.set_doom_game_path(str(package / "freedoom2.wad"))
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_seed(seed)
        game.init()
        buttons = [str(button) for button in game.get_available_buttons()]
        # No-op plus each individual button; avoid contradictory combinations.
        actions = [[0] * len(buttons)]
        actions += [[int(i == j) for i in range(len(buttons))] for j in range(len(buttons))]
        for episode in range(episodes):
            episode_seed = seed + episode
            rng = random.Random(episode_seed)
            game.set_seed(episode_seed)
            game.new_episode()
            decisions = 0
            first_frame_shape = None
            while not game.is_episode_finished():
                state = game.get_state()
                if state is None or state.screen_buffer is None:
                    raise RuntimeError("Running episode has no visual observation")
                if first_frame_shape is None:
                    first_frame_shape = list(state.screen_buffer.shape)
                game.make_action(rng.choice(actions), frame_skip)
                decisions += 1
            results.append({"seed": episode_seed, "return": game.get_total_reward(),
                            "decisions": decisions,
                            "frame_shape": first_frame_shape})
        return {"policy": "random_baseline", "trained": False, "connectome_loaded": False,
                "vizdoom_version": vzd.__version__, "scenario": "basic",
                "scenario_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "wad_sha256": hashlib.sha256(config.with_suffix(".wad").read_bytes()).hexdigest(),
                "frame_skip": frame_skip, "buttons": buttons, "actions": actions,
                "episodes": results, "created_utc": datetime.now(timezone.utc).isoformat()}
    finally:
        game.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("runs/doom-smoke.json"))
    args = parser.parse_args()
    report = run(args.episodes, args.seed, args.frame_skip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
