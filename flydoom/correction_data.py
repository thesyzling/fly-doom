"""Record student-controlled trajectories for later, separate teacher queries.

These records are policy-generated observations, never human demonstrations.
The teacher does not choose buttons during collection. Episode splits and
reserved evaluation seeds are declared before the game runs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flydoom.calibration import output_features, write_json
from flydoom.data import digest
from flydoom.learning_data import ACTIONS, image_state, make_game, read_prepared
from flydoom.live_brain import load_checkpoint
from flydoom.temporal_data import ObservationHistory, TEMPORAL_FORMAT


def collect(output, *, checkpoint="runs/fly-student-experimental-v2",
            dataset="runs/temporal-prepared-v1", calibration="runs/calibration-training-v1/report.json",
            data_dir="data/processed/fafb783", train_episodes=4, validation_episodes=2,
            seed=56000, evaluation_seed=57000, evaluation_episodes=6, max_decisions=75):
    total = train_episodes + validation_episodes
    if (not 1 <= train_episodes <= 20 or not 1 <= validation_episodes <= 10
            or not 1 <= evaluation_episodes <= 20 or not 1 <= max_decisions <= 75
            or not 0 <= seed <= 2**32 - total or not 0 <= evaluation_seed <= 2**32 - evaluation_episodes):
        raise ValueError("Invalid collection limits or seeds")
    source, _ = read_prepared(dataset)
    old_seeds = {e["seed"] for e in source["episodes"]}
    collected = set(range(seed, seed + total))
    reserved = set(range(evaluation_seed, evaluation_seed + evaluation_episodes))
    if collected & reserved or (collected | reserved) & old_seeds:
        raise ValueError("Collection, human demonstrations, and evaluation seeds must be disjoint")
    header = json.loads((Path(checkpoint) / "report.json").read_text(encoding="utf-8"))
    if header.get("schema") != "spiking_student_v1":
        raise ValueError("Correction collection currently requires a standard student; memory checkpoints are supported by live_brain playback")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    ids, controller, model, parent = load_checkpoint(checkpoint, calibration, data_dir)
    if (parent["dataset_sha256"] != digest(Path(dataset) / "manifest.json", "sha256")
            or parent["calibration_sha256"] != source["calibration_sha256"]):
        raise ValueError("Collection parent/data provenance mismatch")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "student_correction_observations_v1", "source": "student_policy",
              "status": "running", "actions": list(ACTIONS), "observation_format": TEMPORAL_FORMAT,
              "teacher_used_for_actions": False, "training_during_collection": False,
              "parent_student_sha256": parent["student_sha256"],
              "parent_report_sha256": digest(Path(checkpoint) / "report.json", "sha256"),
              "human_dataset_sha256": parent["dataset_sha256"],
              "calibration_sha256": parent["calibration_sha256"], "output_root_ids": parent["output_root_ids"],
              "collector_sha256": digest(__file__, "sha256"),
              "reserved_evaluation_seeds": sorted(reserved), "max_decisions": max_decisions,
              "plan": [{"seed": seed + i, "split": "train" if i < train_episodes else "validation"} for i in range(total)],
              "episodes": []}
    write_json(output / "manifest.json", report)
    game = None
    try:
        import vizdoom as vzd
        game, buttons = make_game(seed, False)
        for index, planned in enumerate(report["plan"]):
            game.set_seed(planned["seed"])
            game.new_episode()
            controller.reset()
            history = ObservationHistory()
            previous = 0
            frames, features, states, actions, probabilities, previous_actions, returns = [], [], [], [], [], [], []
            for decision_number in range(max_decisions):
                if game.is_episode_finished():
                    break
                frame = game.get_state().screen_buffer.copy()
                observed = image_state(frame, previous)
                state = history.observe(observed)
                decision = controller.decide(frame)
                if decision["voltage_min_mv"] < -90:
                    raise ValueError("Collection failed the engineering voltage gate")
                vector = np.concatenate((output_features(controller), np.eye(4, dtype=np.float32)[previous]))
                with torch.no_grad():
                    probs = model(torch.tensor(vector).unsqueeze(0)).softmax(-1)[0].numpy()
                action = int(probs.argmax())
                frames.append(frame)
                features.append(vector)
                states.append(state)
                actions.append(action)
                probabilities.append(probs)
                previous_actions.append(previous)
                for _ in range(4):
                    if game.is_episode_finished():
                        break
                    game.make_action([int(b == ACTIONS[action]) for b in buttons], 1)
                returns.append(float(game.get_total_reward()))
                history.advance(observed, action)
                previous = action
                if (decision_number + 1) % 15 == 0:
                    print(f"Collect {index + 1}/{total} ({planned['split']}): {decision_number + 1} decisions", flush=True)
            if not actions:
                raise ValueError("Cannot save an empty episode")
            name = f"episode-{index:03d}"
            np.savez_compressed(output / f"{name}.npz", frames=np.stack(frames), features=np.stack(features),
                student_probabilities=np.stack(probabilities), actions=np.asarray(actions, dtype=np.int64),
                previous_actions=np.asarray(previous_actions, dtype=np.int64), returns=np.asarray(returns, dtype=np.float32))
            write_json(output / f"{name}.json", states)
            episode = {**planned, "file": f"{name}.npz", "states_file": f"{name}.json",
                       "sha256": digest(output / f"{name}.npz", "sha256"),
                       "states_sha256": digest(output / f"{name}.json", "sha256"),
                       "decisions": len(actions), "return": float(game.get_total_reward()),
                       "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
                       "end_reason": "game_finished" if game.is_episode_finished() else "decision_limit"}
            report["episodes"].append(episode)
            write_json(output / "manifest.json", report)
            print(f"Saved {name}: seed={planned['seed']}, return={episode['return']:.0f}, kills={episode['kills']}", flush=True)
        report["status"] = "completed"
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        if game is not None:
            game.close()
        write_json(output / "manifest.json", report)
    return report


def read_collection(directory):
    directory = Path(directory)
    report = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (report.get("schema") != "student_correction_observations_v1" or report.get("status") != "completed"
            or report.get("source") != "student_policy" or report.get("actions") != list(ACTIONS)
            or report.get("observation_format") != TEMPORAL_FORMAT or report.get("teacher_used_for_actions") is not False):
        raise ValueError("Incomplete or unsupported student collection")
    if [{"seed": e["seed"], "split": e["split"]} for e in report["episodes"]] != report["plan"]:
        raise ValueError("Episode splits differ from the declared plan")
    seeds = [e["seed"] for e in report["episodes"]]
    if len(set(seeds)) != len(seeds) or set(seeds) & set(report["reserved_evaluation_seeds"]):
        raise ValueError("Collection seed overlap")
    parts = {s: {"features": [], "actions": [], "states": [], "student_probabilities": []} for s in ("train", "validation")}
    for episode in report["episodes"]:
        if episode["split"] not in parts:
            raise ValueError("Invalid correction split")
        for key, checksum in (("file", "sha256"), ("states_file", "states_sha256")):
            if Path(episode[key]).name != episode[key] or digest(directory / episode[key], "sha256") != episode[checksum]:
                raise ValueError("Correction observation checksum or path mismatch")
        states = json.loads((directory / episode["states_file"]).read_text(encoding="utf-8"))
        with np.load(directory / episode["file"], allow_pickle=False) as data:
            x, actions, previous, frames, probs = (data[k] for k in ("features", "actions", "previous_actions", "frames", "student_probabilities"))
            count = episode["decisions"]
            if (not 1 <= count <= report["max_decisions"] or x.shape != (count, 2 * len(report["output_root_ids"]) + 4)
                    or not np.isfinite(x).all() or actions.shape != (count,) or previous.shape != (count,)
                    or actions.dtype.kind not in "iu" or previous.dtype.kind not in "iu"
                    or not np.isin(actions, range(4)).all() or not np.isin(previous, range(4)).all()
                    or frames.ndim != 4 or frames.shape[0] != count or frames.shape[-1] != 3 or frames.dtype != np.uint8
                    or probs.shape != (count, 4) or not np.isfinite(probs).all() or np.any(probs < 0)
                    or not np.allclose(probs.sum(1), 1, atol=1e-5) or not np.array_equal(probs.argmax(1), actions)
                    or len(states) != count):
                raise ValueError("Invalid correction observation arrays")
            expected_previous = np.concatenate(([0], actions[:-1]))
            if not np.array_equal(previous, expected_previous) or not np.array_equal(x[:, -4:], np.eye(4, dtype=np.float32)[previous]):
                raise ValueError("Previous actions do not match the student's actual history")
            history = ObservationHistory()
            for frame, prev, action, state in zip(frames, previous, actions, states):
                observed = image_state(frame, int(prev))
                if state != history.observe(observed):
                    raise ValueError("Teacher observation differs from the causal student trajectory")
                history.advance(observed, int(action))
            part = parts[episode["split"]]
            part["features"].append(x.copy())
            part["actions"].append(actions.copy())
            part["student_probabilities"].append(probs.copy())
            part["states"].extend(states)
    for part in parts.values():
        if not part["states"]:
            raise ValueError("Require nonempty training and validation episodes")
        for key in ("features", "actions", "student_probabilities"):
            part[key] = np.concatenate(part[key])
    return report, parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/fly-student-experimental-v2"))
    parser.add_argument("--dataset", type=Path, default=Path("runs/temporal-prepared-v1"))
    parser.add_argument("--calibration", type=Path, default=Path("runs/calibration-training-v1/report.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/fafb783"))
    parser.add_argument("--train-episodes", type=int, default=4)
    parser.add_argument("--validation-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=56000)
    parser.add_argument("--evaluation-seed", type=int, default=57000)
    parser.add_argument("--evaluation-episodes", type=int, default=6)
    parser.add_argument("--max-decisions", type=int, default=75)
    try:
        collect(**vars(parser.parse_args()))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
