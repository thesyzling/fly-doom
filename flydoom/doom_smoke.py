"""Reproducible random-policy integration check, not a fly-brain agent."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter, sleep

import vizdoom as vzd


def run(episodes, seed, frame_skip, *, visible=False, speed=1.0, log_actions=False):
    if episodes < 1 or frame_skip < 1 or not 0 <= seed <= 2**32 - episodes:
        raise ValueError("Require positive episodes/frame_skip and uint32 episode seeds")
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Speed must be finite and positive")
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
        game.set_window_visible(visible)
        game.set_mode(vzd.Mode.PLAYER)
        if visible:
            game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
            game.set_render_hud(True)
        game.set_sound_enabled(False)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_seed(seed)
        game.init()
        buttons = [str(button) for button in game.get_available_buttons()]
        # No-op plus each individual button; avoid contradictory combinations.
        actions = [[0] * len(buttons)]
        actions += [[int(i == j) for i in range(len(buttons))] for j in range(len(buttons))]
        action_names = ["WAIT"] + [button.removeprefix("Button.") for button in buttons]
        if visible or log_actions:
            print("RANDOM CONTROL DEMO | No brain model or training is connected.", flush=True)
            print("Basic task: move sideways and shoot the target. Stop with Ctrl+C in this terminal.", flush=True)
        for episode in range(episodes):
            episode_seed = seed + episode
            rng = random.Random(episode_seed)
            game.set_seed(episode_seed)
            game.new_episode()
            if visible or log_actions:
                print(f"Episode {episode + 1}/{episodes} | seed={episode_seed}", flush=True)
            if visible:
                # Let the viewer see the initial state before the first action.
                sleep(0.75)
            decisions = 0
            first_frame_shape = None
            while not game.is_episode_finished():
                state = game.get_state()
                if state is None or state.screen_buffer is None:
                    raise RuntimeError("Running episode has no visual observation")
                if first_frame_shape is None:
                    first_frame_shape = list(state.screen_buffer.shape)
                action_index = rng.randrange(len(actions))
                action = actions[action_index]
                if log_actions:
                    print(f"  Decision {decisions + 1:03d}: {action_names[action_index]}", flush=True)
                if visible:
                    # Advance the same action one tic at a time for smooth viewing.
                    # Synchronous PLAYER mode keeps wall-clock pacing out of game logic.
                    for _ in range(frame_skip):
                        if game.is_episode_finished():
                            break
                        started = perf_counter()
                        game.make_action(action, 1)
                        sleep(max(0.0, 1.0 / (35.0 * speed) - (perf_counter() - started)))
                else:
                    game.make_action(action, frame_skip)
                decisions += 1
            results.append({"seed": episode_seed, "return": game.get_total_reward(),
                            "decisions": decisions,
                            "frame_shape": first_frame_shape})
            if visible or log_actions:
                print(f"  Episode finished | return={game.get_total_reward():.1f} | decisions={decisions}", flush=True)
            if visible:
                sleep(0.75)
        return {"policy": "random_baseline", "trained": False, "connectome_loaded": False,
                "vizdoom_version": vzd.__version__, "scenario": "basic",
                "scenario_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "wad_sha256": hashlib.sha256(config.with_suffix(".wad").read_bytes()).hexdigest(),
                "frame_skip": frame_skip, "buttons": buttons, "actions": actions,
                "presentation": {"visible": visible, "mode": "PLAYER", "sound": False,
                                 "requested_speed": speed if visible else None,
                                 "hud": visible, "log_actions": log_actions},
                "episodes": results, "created_utc": datetime.now(timezone.utc).isoformat()}
    finally:
        game.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--visible", action="store_true", help="Open the game window and pace it for viewing")
    parser.add_argument("--speed", type=float, default=1.0, help="Viewing speed with --visible (0.5 = half speed)")
    parser.add_argument("--log-actions", action="store_true", help="Print each randomly selected action")
    parser.add_argument("--output", type=Path, default=Path("runs/doom-smoke.json"))
    args = parser.parse_args()
    try:
        report = run(args.episodes, args.seed, args.frame_skip,
                     visible=args.visible, speed=args.speed, log_actions=args.log_actions)
    except ValueError as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("\nDemo stopped. No completed report was written.")
        return
    except (vzd.ViZDoomIsNotRunningException, vzd.ViZDoomUnexpectedExitException) as error:
        parser.exit(1, f"The game window closed or the engine exited early: {error}\nNo completed report was written.\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
