"""Bounded experimental Doom control through the real connectome; not trained."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from time import perf_counter, sleep

import numpy as np
import scipy
import vizdoom as vzd

from flydoom.bridge import ACTION_NAMES, NeuralController, NeuralMapping
from flydoom.data import digest
from flydoom.simulation import LIFNetwork, LIFParameters, load_connectome


def run(*, data_dir, output, episodes=1, seed=42, max_decisions=24, frame_skip=4,
        visible=False, brain_ms=50.0, input_gain_mv=40.0, control="connected"):
    if not 1 <= episodes <= 20 or not 1 <= max_decisions <= 1000 or not 1 <= frame_skip <= 35:
        raise ValueError("Require 1..20 episodes, 1..1000 decisions, and 1..35 tics per action")
    if not 0 <= seed <= 2**32 - episodes:
        raise ValueError("Episode seeds must fit uint32")
    output = Path(output)
    # Exclusive creation preserves earlier runs, including interrupted ones.
    output.mkdir(parents=True, exist_ok=False)
    print("Loading and verifying the real connectome...", flush=True)
    params = LIFParameters()
    ids, rows, weights, signs, provenance = load_connectome(data_dir, params)
    mapping = NeuralMapping.from_annotations(ids, rows, signs)
    network = LIFNetwork(weights, params)
    controller = NeuralController(network, mapping, brain_ms=brain_ms,
                                  input_gain_mv=input_gain_mv, control=control)
    (output / "mapping.json").write_text(json.dumps(mapping.describe(ids), indent=2) + "\n", encoding="utf-8")
    del rows, signs
    print(f"EXPERIMENTAL NEURAL BRIDGE | {len(ids):,} neurons | control={control}", flush=True)
    print("Retina bypass; artificial input/output assignments; no training or random fallback.", flush=True)
    print("The game pauses while the CPU computes each neural decision. Ctrl+C stops and saves partial results.", flush=True)

    package = Path(os.path.relpath(Path(vzd.__file__).parent))
    config = package / "scenarios" / "basic.cfg"
    game = vzd.DoomGame()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "experimental_connectome_fixed_readout_v1", "trained": False,
        "connectome_loaded": True, "pixels_used": True, "retina_bypassed": True,
        "biologically_validated": False, "control": control, "neurons": len(ids),
        "parameters": asdict(params), "requested_brain_ms": brain_ms,
        "input_gain_mv": input_gain_mv, "frame_skip": frame_skip, "max_decisions": max_decisions,
        "input_neurons": len(mapping.input_indices),
        "output_group_sizes": dict(zip(ACTION_NAMES, map(len, mapping.output_groups))),
        "readout": "Largest mean spike rate per output cell; silence or tied maxima produce WAIT",
        "scenario": "basic", "scenario_sha256": digest(config, "sha256"),
        "wad_sha256": digest(config.with_suffix(".wad"), "sha256"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "vizdoom": vzd.__version__},
        "implementation_sha256": {name: digest(Path(__file__).with_name(name), "sha256")
                                  for name in ("brain_doom.py", "bridge.py", "simulation.py")},
        "presentation": {"visible": visible, "resolution": [320, 240], "hud": False, "sound": False},
        "provenance": provenance, "episodes": [], "status": "running",
    }
    # The trace is flushed per decision so partial work survives interruption.
    try:
        with (output / "decisions.jsonl").open("w", encoding="utf-8") as trace:
            game.load_config(str(config))
            game.set_vizdoom_path(str(package / ("vizdoom.exe" if os.name == "nt" else "vizdoom")))
            game.set_doom_game_path(str(package / "freedoom2.wad"))
            game.set_window_visible(visible)
            game.set_mode(vzd.Mode.PLAYER)
            # Keep observations identical between visible and headless runs.
            game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
            game.set_screen_format(vzd.ScreenFormat.RGB24)
            game.set_render_hud(False)
            game.set_sound_enabled(False)
            game.set_seed(seed)
            game.init()
            buttons = [str(b).removeprefix("Button.") for b in game.get_available_buttons()]
            if set(buttons) != set(ACTION_NAMES):
                raise ValueError("The bridge requires the basic scenario's three action buttons")
            report["buttons"] = buttons
            for episode in range(episodes):
                game.set_seed(seed + episode)
                game.new_episode()
                controller.reset()
                summary = {"episode": episode + 1, "seed": seed + episode,
                           "decisions": 0, "return": 0.0, "end_reason": "interrupted"}
                report["episodes"].append(summary)
                print(f"Episode {episode + 1}/{episodes}: starting from neural rest", flush=True)
                while not game.is_episode_finished() and summary["decisions"] < max_decisions:
                    state = game.get_state()
                    if state is None or state.screen_buffer is None:
                        raise RuntimeError("Running game has no RGB observation")
                    decision = controller.decide(state.screen_buffer)
                    action = [int(button == decision["action"]) for button in buttons]
                    before = game.get_episode_time()
                    reward = 0.0
                    for _ in range(frame_skip):
                        if game.is_episode_finished():
                            break
                        started = perf_counter()
                        reward += game.make_action(action, 1)
                        if visible:
                            sleep(max(0.0, 1 / 35 - (perf_counter() - started)))
                    summary["decisions"] += 1
                    summary["return"] = game.get_total_reward()
                    decision.update(episode=episode + 1, decision=summary["decisions"],
                                    game_tic_before=before, game_tic_after=game.get_episode_time(),
                                    action_buttons=action, reward=reward, total_reward=summary["return"])
                    trace.write(json.dumps(decision, allow_nan=False) + "\n")
                    trace.flush()
                    rates = decision["rates_hz_per_neuron"]
                    print(f"  {summary['decisions']:03d}/{max_decisions} | {decision['action']:<10} | "
                          f"L={rates['MOVE_LEFT']:.2f} R={rates['MOVE_RIGHT']:.2f} "
                          f"A={rates['ATTACK']:.2f} Hz/cell | "
                          f"{decision['spikes']} spikes | compute={decision['compute_seconds']:.2f}s", flush=True)
                summary["end_reason"] = "game_finished" if game.is_episode_finished() else "decision_limit"
                print(f"  Stopped: {summary['end_reason']} | return={summary['return']:.1f}", flush=True)
            report["status"] = "completed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        print("\nStopped by user; saving partial results.", flush=True)
    except Exception as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        if report["episodes"] and report["episodes"][-1]["end_reason"] == "interrupted":
            report["episodes"][-1]["end_reason"] = "error"
        raise
    finally:
        try:
            game.close()
        finally:
            report["output_sha256"] = {name: digest(output / name, "sha256")
                                       for name in ("mapping.json", "decisions.jsonl") if (output / name).exists()}
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"Results saved to {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/fafb783"))
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are never overwritten")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-decisions", type=int, default=24)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--brain-ms", type=float, default=50.0)
    parser.add_argument("--input-gain-mv", type=float, default=40.0)
    parser.add_argument("--control", choices=("connected", "disconnected", "zero-input"), default="connected")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("runs") / datetime.now().strftime("brain-doom-%Y%m%d-%H%M%S-%f")
    try:
        run(**vars(args))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    except (vzd.ViZDoomIsNotRunningException, vzd.ViZDoomUnexpectedExitException) as error:
        parser.exit(1, f"The game window closed or the engine exited early: {error}\n")


if __name__ == "__main__":
    main()
