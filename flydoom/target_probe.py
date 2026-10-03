"""Diagnose target-side information without training or loading a game policy."""

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import vizdoom as vzd

from flydoom.bridge import encode_frame
from flydoom.calibration import load_calibrated, output_features, write_json
from flydoom.data import digest
from flydoom.live_activity import frame_png


def target_label(labels, width):
    """Use the visible Cacodemon bounding box only as a diagnostic target."""
    targets = [label for label in labels if label.object_name == "Cacodemon"]
    if len(targets) != 1:
        return {"side": -1, "reason": "missing_or_ambiguous_target"}
    label = targets[0]
    fraction = (label.x + label.width / 2) / width
    side = 0 if fraction < 0.475 else 1 if fraction > 0.525 else -1
    return {"side": side, "reason": "visible" if side >= 0 else "near_center",
            "x_fraction": fraction, "box": [label.x, label.y, label.width, label.height]}


def rgb_bins(frame):
    """Keep color in the same spatial bins as the existing brightness encoder."""
    encode_frame(frame)  # Validate the shared image contract.
    return np.asarray([cell.mean(axis=(0, 1)) / 255 for band in np.array_split(frame, 8)
                       for cell in np.array_split(band, 8, axis=1)], dtype=np.float32).ravel()


def make_diagnostic_game(seed):
    """Match the basic RGB game; enable privileged labels for scoring only."""
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
        game.set_window_visible(False)
        game.set_labels_buffer_enabled(True)
        game.set_seed(seed)
        game.init()
        return game
    except BaseException:
        game.close()
        raise


def analysis_indices(records):
    """Hold out whole episodes and remove exact RGB overlap and repetition."""
    chosen, seen = {"train": [], "validation": []}, set()
    for split in chosen:
        for index, record in enumerate(records):
            if record["split"] != split or record["target"]["side"] < 0:
                continue
            key = record["frame_sha256"]
            if key not in seen:
                chosen[split].append(index)
                seen.add(key)
    return {key: np.asarray(value, dtype=np.int64) for key, value in chosen.items()}


def ridge_predict(train, labels, query, alpha=0.1):
    """Balanced dual ridge classifier; all normalization is training-only.

    The kernel is divided by feature count so the fixed penalty is comparable
    across representations. No regularization search uses validation results.
    """
    train, query = np.asarray(train, dtype=np.float64), np.asarray(query, dtype=np.float64)
    labels = np.asarray(labels)
    if (train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1]
            or len(train) != len(labels) or not np.isfinite(train).all()
            or not np.isfinite(query).all() or not np.isfinite(alpha) or alpha <= 0
            or set(labels.tolist()) != {0, 1}):
        raise ValueError("Require finite aligned features, both binary classes, and positive alpha")
    mean, scale = train.mean(axis=0), np.maximum(train.std(axis=0), 1e-3)
    x, q = (train - mean) / scale, (query - mean) / scale
    weights = len(labels) / (2 * np.bincount(labels)[labels])
    root = np.sqrt(weights)
    # A constant feature supplies an intercept; its penalty is also fixed.
    kernel = x @ x.T / train.shape[1] + 1
    cross = q @ x.T / train.shape[1] + 1
    solution = np.linalg.solve(kernel * root[:, None] * root[None, :]
                               + alpha * np.eye(len(train)), (2 * labels - 1) * root)
    return cross @ (root * solution)


def classification_metrics(labels, scores):
    labels, prediction = np.asarray(labels), (np.asarray(scores) >= 0).astype(int)
    confusion = np.zeros((2, 2), dtype=int)
    np.add.at(confusion, (labels, prediction), 1)
    counts = confusion.sum(axis=1)
    recalls = [float(confusion[i, i] / counts[i]) if counts[i] else None for i in range(2)]
    return {"count": len(labels), "class_counts_left_right": counts.tolist(),
            "accuracy": float(np.mean(labels == prediction)) if len(labels) else None,
            "balanced_accuracy": float(np.mean(recalls)) if all(x is not None for x in recalls) else None,
            "recall_left_right": recalls, "confusion_actual_rows_predicted_columns": confusion.tolist()}


def analyze(representations, records):
    indices = analysis_indices(records)
    labels = np.asarray([r["target"]["side"] for r in records])
    train, val = indices["train"], indices["validation"]
    if any(set(labels[part].tolist()) != {0, 1} for part in (train, val)):
        raise ValueError("Diagnostic needs both sides in each separated split")
    result = {"indices": {k: v.tolist() for k, v in indices.items()}, "representations": {}}
    for name, features in representations.items():
        scores = ridge_predict(features[train], labels[train], features)
        rng = np.random.default_rng(71)
        null_scores = [classification_metrics(labels[val], ridge_predict(
            features[train], rng.permutation(labels[train]), features[val]))["balanced_accuracy"]
                       for _ in range(20)]
        train_keys = {hashlib.sha256(row.tobytes()).hexdigest() for row in features[train]}
        measured = {"dimensions": features.shape[1],
                    "train": classification_metrics(labels[train], scores[train]),
                    "validation": classification_metrics(labels[val], scores[val]),
                    "shuffled_training_label_control": {"repetitions": 20,
                        "mean_balanced_accuracy": float(np.mean(null_scores)),
                        "min": min(null_scores), "max": max(null_scores),
                        "statistical_significance_test": False},
                    "validation_exact_representation_matches_train": sum(
                        hashlib.sha256(row.tobytes()).hexdigest() in train_keys for row in features[val]),
                    "scores": scores.tolist()}
        result["representations"][name] = measured
    return result


def protected_hashes():
    paths = [Path(__file__).with_name(name) for name in
             ("student.py", "simulation.py", "action_memory.py", "bridge.py")]
    paths += [Path("runs") / name / "student.safetensors" for name in
              ("fly-student-memory-v1", "fly-student-human-feedback-v1")]
    return {str(p): digest(p, "sha256") for p in paths if p.exists()}


def run(output, data_dir, calibration):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    before = protected_hashes()
    plan = {"schema": "target_side_probe_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "collecting", "diagnostic_only": True, "game_policy_trained": False,
            "seeds": list(range(71000, 71024)), "training_episodes": 16,
            "observations_per_episode": 4, "tics_between_observations": [8, 16, 8],
            "action_schedule": "LEFT RIGHT LEFT; mirrored on odd episode indices; no shooting",
            "labels": "Visible Cacodemon bounding-box center: LEFT < .475, RIGHT > .525; otherwise excluded",
            "labels_used_as_model_input": False, "episode_split_before_collection": True,
            "policy_training_or_test_datasets_used": False,
            "decoder": {"type": "balanced linear dual ridge", "alpha": 0.1,
                        "normalization": "training-only mean/std with .001 floor",
                        "hyperparameter_search": False},
            "protected_sha256_before": before,
            "calibration_sha256": digest(calibration, "sha256"),
            "source_sha256": digest(Path(__file__), "sha256"),
            "vizdoom_version": vzd.__version__,
            "label_documentation": "https://vizdoom.farama.org/main/api/python/gameState/",
            "limitations": ["Small development diagnostic on one basic map, not a gameplay evaluation",
                "Four prescribed observations per episode; different from autonomous policy trajectories",
                "Exact RGB duplicates removed, but related scenes and backgrounds can remain",
                "Decodability does not establish use by the current policy or biological topology superiority",
                "A weak linear decoder does not prove information is absent",
                "Probe classifiers are diagnostic artifacts, never exported as gameplay checkpoints"]}
    write_json(output / "plan.json", plan)
    frames, records = [], []
    for episode, seed in enumerate(plan["seeds"]):
        game = make_diagnostic_game(seed)
        try:
            buttons = [str(b).removeprefix("Button.") for b in game.get_available_buttons()]
            moves = ["MOVE_LEFT", "MOVE_RIGHT", "MOVE_LEFT"]
            if episode % 2:
                moves = ["MOVE_RIGHT", "MOVE_LEFT", "MOVE_RIGHT"]
            for step in range(4):
                state = game.get_state()
                if state is None:
                    raise ValueError("Prescribed diagnostic episode ended early")
                frame = state.screen_buffer.copy()
                index = len(frames)
                name = f"frames/{index:03d}.png"
                (output / name).write_bytes(base64.b64decode(frame_png(frame).split(",", 1)[1]))
                frames.append(frame)
                records.append({"seed": seed, "step": step, "tic": state.tic,
                    "split": "train" if episode < 16 else "validation", "image": name,
                    "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                    "target": target_label(state.labels, frame.shape[1])})
                if step < 3:
                    game.make_action([int(b == moves[step]) for b in buttons],
                                     plan["tics_between_observations"][step])
        finally:
            game.close()
    write_json(output / "observations.json", records)
    print(f"Collected {len(frames)} frames; loading the frozen graph", flush=True)
    ids, controller, calibration_report = load_calibrated(data_dir, calibration)
    neural, traces = [], []
    for index, (frame, record) in enumerate(zip(frames, records)):
        if record["step"] == 0:
            controller.reset()
        trace = controller.decide(frame)
        neural.append(output_features(controller))
        traces.append({key: trace[key] for key in
                       ("spikes", "descending_spikes", "brain_time_ms", "compute_seconds")})
        print(f"Graph observation {index + 1}/{len(frames)} | seed {record['seed']} | "
              f"{trace['compute_seconds']:.2f}s", flush=True)
    representations = {"rgb_8x8": np.stack([rgb_bins(f) for f in frames]),
                       "brightness_8x8": np.stack([encode_frame(f) for f in frames]),
                       "descending_outputs": np.stack(neural)}
    if not all(np.isfinite(value).all() for value in representations.values()):
        raise FloatingPointError("Non-finite diagnostic features")
    np.savez_compressed(output / "features.npz", **representations)
    report = {**plan, "status": "completed", "neurons": len(ids),
              "graph_provenance": calibration_report["provenance"],
              "observation_counts": dict(Counter(r["split"] for r in records)),
              "target_counts": dict(Counter(r["target"]["reason"] for r in records)),
              "analysis": analyze(representations, records), "traces": traces,
              "protected_sha256_after": protected_hashes()}
    if report["protected_sha256_after"] != before:
        raise ValueError("Protected runtime or policy changed during diagnostic")
    report["protected_files_unchanged"] = True
    report["artifact_sha256"] = {name: digest(output / name, "sha256") for name in
                                 ("plan.json", "observations.json", "features.npz")}
    write_json(output / "report.json", report)
    from flydoom.target_probe_view import write_view
    write_view(output, report, records, representations["brightness_8x8"])
    for name, result in report["analysis"]["representations"].items():
        print(f"{name}: validation balanced accuracy={result['validation']['balanced_accuracy']:.3f}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/fafb783"))
    parser.add_argument("--calibration", type=Path, default=Path("runs/calibration-training-v1/report.json"))
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
