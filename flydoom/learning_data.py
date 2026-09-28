"""Human demonstration recording and episode-separated learning artifacts."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import vizdoom as vzd

from flydoom.bridge import encode_frame
from flydoom.calibration import load_calibrated, output_features, write_json
from flydoom.data import digest


ACTIONS = ("WAIT", "MOVE_LEFT", "MOVE_RIGHT", "ATTACK")
QUESTION = {"action": {"type": "choice",
    "instructions": "Predict the human player's next button in the basic Doom target shooting task from the coarse image and previous action.",
    "criteria": {"WAIT": "Wait without pressing a button", "MOVE_LEFT": "Strafe left",
                 "MOVE_RIGHT": "Strafe right", "ATTACK": "Fire the weapon"}}}


def make_game(seed=42, visible=False):
    package = Path(os.path.relpath(Path(vzd.__file__).parent))
    game = vzd.DoomGame()
    try:
        game.load_config(str(package / "scenarios/basic.cfg"))
        game.set_vizdoom_path(str(package / ("vizdoom.exe" if os.name == "nt" else "vizdoom")))
        game.set_doom_game_path(str(package / "freedoom2.wad"))
        game.set_mode(vzd.Mode.PLAYER)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.set_render_hud(False)
        game.set_sound_enabled(False)
        game.set_window_visible(visible)
        game.set_seed(seed)
        game.init()
        buttons = [str(b).removeprefix("Button.") for b in game.get_available_buttons()]
        if set(buttons) != set(ACTIONS[1:]):
            raise ValueError("Unexpected basic scenario buttons")
        return game, buttons
    except BaseException:
        game.close()
        raise


def split_for_episode(index):
    return ("train", "train", "train", "validation", "test")[index % 5]


def image_state(frame, previous_action):
    grid = np.rint(encode_frame(frame).reshape(8, 8) * 9).astype(int)
    return {"image": "8 by 8 brightness grid, top to bottom, left to right; 0 black, 9 white",
            "rows": [" ".join(map(str, row)) for row in grid],
            "previous_action": ACTIONS[int(previous_action)]}


def record(output, *, episodes=10, seed=1000, max_decisions=75, random_smoke=False):
    if not 5 <= episodes <= 100 or not 1 <= max_decisions <= 75 or not 0 <= seed <= 2**32 - episodes:
        raise ValueError("Require 5..100 episodes, 1..75 decisions and uint32 seeds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "doom_demonstrations_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source": "random_smoke" if random_smoke else "human_keyboard",
              "actions": ACTIONS, "frame_skip": 4, "episodes": [], "status": "running",
              "label_note": "Recorded actions are demonstrations, not guaranteed optimal labels"}
    game = None
    pygame = None
    try:
        game, buttons = make_game(seed)
        if not random_smoke:
            import pygame as pg
            pygame = pg
            pygame.init()
            screen = pygame.display.set_mode((640, 560))
            pygame.display.set_caption("Fly Doom - record your demonstrations")
            font = pygame.font.Font(None, 25)
        for episode in range(episodes):
            episode_seed = seed + episode
            game.set_seed(episode_seed)
            game.new_episode()
            frames, actions, rewards, previous = [], [], [], []
            last_action = 0
            rng = np.random.default_rng(episode_seed)
            print(f"Record episode {episode + 1}/{episodes} | {split_for_episode(episode)}", flush=True)
            if pygame:
                # Show the starting frame; begin only after ENTER so startup waits do not become labels.
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            raise KeyboardInterrupt
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                            waiting = False
                    draw_recording(pygame, screen, font, game.get_state().screen_buffer,
                                   f"Episode {episode + 1}/{episodes}: press ENTER to begin")
                    sleep(0.02)
            while not game.is_episode_finished() and len(actions) < max_decisions:
                started = perf_counter()
                frame = game.get_state().screen_buffer.copy()
                if pygame:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            raise KeyboardInterrupt
                    keys = pygame.key.get_pressed()
                    # One button per decision; shooting takes precedence, contradictory movement waits.
                    left, right = keys[pygame.K_LEFT] or keys[pygame.K_a], keys[pygame.K_RIGHT] or keys[pygame.K_d]
                    action = 3 if keys[pygame.K_SPACE] else 1 if left and not right else 2 if right and not left else 0
                    draw_recording(pygame, screen, font, frame,
                                   f"Episode {episode + 1}/{episodes} | {ACTIONS[action]} | return {game.get_total_reward():.0f}")
                else:
                    action = int(rng.integers(4))
                frames.append(frame)
                actions.append(action)
                previous.append(last_action)
                rewards.append(game.make_action([int(b == ACTIONS[action]) for b in buttons], 4))
                last_action = action
                if pygame:
                    sleep(max(0, 4 / 35 - (perf_counter() - started)))
            name = f"episode-{episode:04d}.npz"
            np.savez_compressed(output / name, frames=np.stack(frames), actions=np.asarray(actions, dtype=np.int64),
                                previous_actions=np.asarray(previous, dtype=np.int64), rewards=np.asarray(rewards))
            report["episodes"].append({"file": name, "sha256": digest(output / name, "sha256"),
                "seed": episode_seed, "split": split_for_episode(episode), "decisions": len(actions),
                "return": game.get_total_reward(),
                "end_reason": "game_finished" if game.is_episode_finished() else "decision_limit"})
            write_json(output / "manifest.json", report)
        report["status"] = "completed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        print("Stopped. Completed episodes were kept; the unfinished episode was discarded.", flush=True)
    except Exception:
        report["status"] = "error"
        raise
    finally:
        if game:
            game.close()
        if pygame:
            pygame.quit()
        write_json(output / "manifest.json", report)
    return report


def draw_recording(pygame, screen, font, frame, status):
    surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
    screen.fill((18, 20, 28))
    screen.blit(pygame.transform.scale(surface, (640, 480)), (0, 0))
    for i, line in enumerate((status, "LEFT/RIGHT or A/D: move | SPACE: fire | ESC: save and stop")):
        screen.blit(font.render(line, True, (235, 235, 245)), (12, 490 + i * 28))
    pygame.display.flip()


def read_recording(directory, *, require_human=True):
    directory = Path(directory)
    report = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if report.get("schema") != "doom_demonstrations_v1" or list(report["actions"]) != list(ACTIONS):
        raise ValueError("Unsupported demonstration format")
    if require_human and report["source"] != "human_keyboard":
        raise ValueError("Random smoke data is not eligible for teacher training")
    seeds = [ep["seed"] for ep in report["episodes"]]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate episode seeds would leak between splits")
    for ep in report["episodes"]:
        if Path(ep["file"]).name != ep["file"] or ep["split"] not in {"train", "validation", "test"}:
            raise ValueError("Invalid episode path or split")
        if digest(directory / ep["file"], "sha256") != ep["sha256"]:
            raise ValueError("Demonstration checksum mismatch")
    return report


def prepare(recording, calibration, output, *, data_dir="data/processed/fafb783", allow_smoke=False):
    recording, output = Path(recording), Path(output)
    source = read_recording(recording, require_human=not allow_smoke)
    if {e["split"] for e in source["episodes"]} != {"train", "validation", "test"}:
        raise ValueError("Complete at least five episodes for independent train/validation/test splits")
    output.mkdir(parents=True, exist_ok=False)
    ids, controller, calibration_report = load_calibrated(data_dir, calibration)
    report = {"schema": "neural_learning_data_v1", "source": source["source"], "status": "running",
              "recording_manifest_sha256": digest(recording / "manifest.json", "sha256"),
              "calibration_sha256": digest(calibration, "sha256"), "calibration": calibration_report,
              "actions": ACTIONS, "question": QUESTION, "episodes": [],
              "feature_rule": "Descending voltage/rest delta divided by 20 mV, spike rate divided by 100 Hz, previous action one-hot",
              "output_root_ids": [str(ids[i]) for i in np.sort(np.concatenate(controller.mapping.output_groups))]}
    try:
        for ep in source["episodes"]:
            controller.reset()
            features, states = [], []
            with np.load(recording / ep["file"], allow_pickle=False) as data:
                frames, actions, previous = data["frames"], data["actions"], data["previous_actions"]
                if (len(frames) != len(actions) or len(previous) != len(actions)
                        or not np.isin(actions, range(4)).all() or not np.isin(previous, range(4)).all()):
                    raise ValueError("Invalid demonstration arrays")
                for i, frame in enumerate(frames):
                    decision = controller.decide(frame)
                    if decision["voltage_min_mv"] < -90:
                        raise ValueError("Recorded frames failed the engineering voltage gate; recalibration required")
                    features.append(np.concatenate((output_features(controller), np.eye(4, dtype=np.float32)[previous[i]])))
                    states.append(image_state(frame, previous[i]))
                    if (i + 1) % 10 == 0 or i + 1 == len(frames):
                        print(f"{ep['file']}: {i + 1}/{len(frames)} neural decisions", flush=True)
                name = ep["file"]
                np.savez_compressed(output / name, features=np.stack(features), actions=actions,
                                    previous_actions=previous)
            state_name = name.replace(".npz", ".json")
            write_json(output / state_name, states)
            report["episodes"].append({**ep, "sha256": digest(output / name, "sha256"),
                                       "states_file": state_name, "states_sha256": digest(output / state_name, "sha256")})
            write_json(output / "manifest.json", report)
        report["status"] = "completed"
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        write_json(output / "manifest.json", report)
    return report


def read_prepared(directory, *, require_human=True):
    directory = Path(directory)
    report = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if report.get("schema") != "neural_learning_data_v1" or report.get("status") != "completed":
        raise ValueError("Prepared dataset is incomplete or unsupported")
    if report.get("actions") != list(ACTIONS) or report.get("question") != QUESTION:
        raise ValueError("Prepared action or question format changed")
    if require_human and report["source"] != "human_keyboard":
        raise ValueError("Smoke artifacts cannot be used as demonstration labels")
    result = {split: {"features": [], "actions": [], "previous": [], "states": []}
              for split in ("train", "validation", "test")}
    seen = set()
    for ep in report["episodes"]:
        if ep["seed"] in seen:
            raise ValueError("Duplicate episode seed")
        seen.add(ep["seed"])
        if ep["split"] not in result:
            raise ValueError("Invalid split")
        for key, checksum in (("file", "sha256"), ("states_file", "states_sha256")):
            if Path(ep[key]).name != ep[key] or digest(directory / ep[key], "sha256") != ep[checksum]:
                raise ValueError("Prepared data checksum or path mismatch")
        part = result[ep["split"]]
        with np.load(directory / ep["file"], allow_pickle=False) as data:
            features, actions, previous = data["features"], data["actions"], data["previous_actions"]
            if (features.ndim != 2 or features.shape[1] != 2 * len(report["output_root_ids"]) + 4
                    or not np.isfinite(features).all() or len(features) != len(actions)
                    or actions.ndim != 1 or previous.ndim != 1
                    or actions.dtype.kind not in "iu" or previous.dtype.kind not in "iu"
                    or len(previous) != len(actions) or not np.isin(actions, range(4)).all()
                    or not np.isin(previous, range(4)).all()):
                raise ValueError("Invalid prepared numerical arrays")
            part["features"].extend(features)
            part["actions"].extend(actions)
            part["previous"].extend(previous)
        states = json.loads((directory / ep["states_file"]).read_text(encoding="utf-8"))
        if len(states) != len(actions):
            raise ValueError("State/label count mismatch")
        part["states"].extend(states)
    for part in result.values():
        if not part["actions"]:
            raise ValueError("All three splits must be nonempty")
        part["features"] = np.asarray(part["features"], dtype=np.float32)
        part["actions"] = np.asarray(part["actions"], dtype=np.int64)
        part["previous"] = np.asarray(part["previous"], dtype=np.int64)
    return report, result
