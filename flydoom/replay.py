"""Verified game-action replays and a bounded local replay artifact library."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.learning_data import ACTIONS, make_game
from flydoom.live_activity import frame_png


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Replay artifact leaves its run directory")
    return path


def summarize(row, effect, frames, activity):
    return {**{k: row[k] for k in ("episode", "decision", "seed", "action", "student_action",
             "probabilities", "applied_probabilities", "alpha", "agreement", "return", "brain_spikes")},
            "vision_probabilities": row["vision"]["probabilities"],
            "effect": effect, "frames": frames, "student_spikes": np.rint(activity * 8).astype(int).tolist()}


def build_replay(run, output):
    """Replay saved actions in the engine; require every original frame and return to match.

    This is a deterministic reconstruction of gameplay, not another model run.
    No terminal image is fabricated when ViZDoom ends the scenario immediately.
    """
    import vizdoom as vzd
    run, output = Path(run), Path(output)
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if report.get("schema") != "vision_fly_research_v1" or report["status"] != "completed":
        raise ValueError("Require a completed paired research run")
    rows = [json.loads(line) for line in (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    if not rows or len(rows) > 450:
        raise ValueError("Require 1..450 recorded decisions")
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    game = None
    previous_episode = None
    decisions, files = [], {}
    try:
        game, buttons = make_game(rows[0]["seed"], False)
        for index, row in enumerate(rows):
            if row["episode"] != previous_episode:
                game.set_seed(row["seed"])
                game.new_episode()
                previous_episode = row["episode"]
            state = game.get_state()
            if state is None or hashlib.sha256(state.screen_buffer.tobytes()).hexdigest() != row["vision"]["frame_sha256"]:
                raise ValueError(f"Engine reconstruction differs at decision {index + 1}")
            npz = safe_path(run, row["observation"] + ".npz")
            if digest(npz, "sha256") != row["observation_sha256"]:
                raise ValueError("Recorded observation checksum changed")
            with np.load(npz, allow_pickle=False) as stored:
                np.testing.assert_allclose(stored["student_probabilities"], row["probabilities"], atol=1e-7)
                np.testing.assert_allclose(stored["teacher_probabilities"], row["vision"]["probabilities"], atol=1e-7)
                activity = stored["student_activity"].copy()
            before_return = game.get_total_reward()
            before_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
            images = [state.screen_buffer.copy()]
            tics = 0
            for _ in range(4):
                if game.is_episode_finished():
                    break
                game.make_action([int(b == row["action"]) for b in buttons], 1)
                tics += 1
                state = game.get_state()
                if state is not None:
                    images.append(state.screen_buffer.copy())
            reward, kills = game.get_total_reward(), int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
            if not np.isclose(reward, row["return"], atol=1e-7):
                raise ValueError("Reconstructed reward differs from the original run")
            paths = []
            for tic, frame in enumerate(images):
                name = f"frames/{index:04d}-{tic}.png"
                (output / name).write_bytes(base64.b64decode(frame_png(frame).split(",", 1)[1]))
                files[name] = digest(output / name, "sha256")
                paths.append(name)
            effect = {"reward_delta": reward - before_return, "kill_delta": kills - before_kills,
                      "kills": kills, "episode_finished": game.is_episode_finished(), "game_tics": tics,
                      "terminal_image_available": not game.is_episode_finished()}
            decisions.append(summarize(row, effect, paths, activity))
        for summary in report["episodes"]:
            group = [row for row in decisions if row["seed"] == summary["seed"]]
            if len(group) != summary["decisions"] or group[-1]["effect"]["kills"] != summary["kills"]:
                raise ValueError("Episode summary differs from the original run")
        manifest = {"schema": "vision_replay_v1", "title": f"Paired run / alpha {report['alpha']} / {len(rows)} decisions",
                    "source_run": str(run), "capture": "verified_engine_reconstruction",
                    "source_trace_sha256": digest(run / "decisions.jsonl", "sha256"),
                    "source_report_sha256": digest(run / "report.json", "sha256"),
                    "checkpoint_sha256": report["checkpoint_sha256"], "vision": report["vision"],
                    "episodes": report["episodes"], "decisions": decisions, "frame_sha256": files,
                    "note": "Every pre-action RGB hash and post-action return matched the original trace. Terminal events use engine counters; the last available frame is held when no terminal image exists."}
        write_json(output / "replay.json", manifest)
        print(f"Verified {len(decisions)} decisions and {len(files)} game frames: {output}", flush=True)
        return manifest
    finally:
        if game:
            game.close()


def finish_live_replay(run, report):
    """Index frames captured during the real run, including stopped prefixes."""
    run = Path(run)
    trace = run / "decisions.jsonl"
    if not trace.exists():
        return
    decisions, files = [], {}
    for line in trace.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "playback_frames" not in row:
            return
        paths = row["playback_frames"]
        for name in paths:
            files[name] = digest(safe_path(run, name), "sha256")
        with np.load(safe_path(run, row["observation"] + ".npz"), allow_pickle=False) as stored:
            decisions.append(summarize(row, row["effect"], paths, stored["student_activity"]))
    if decisions:
        label = "Student only" if report.get("laya_used_during_play") is False else f"alpha {report['alpha']}"
        if report.get("disconnected"):
            label += " / Disconnected graph control"
        write_json(run / "replay.json", {"schema": "vision_replay_v1", "title": f"{run.parent.name} / {run.name} / {label} / {len(decisions)} decisions",
            "source_run": str(run), "capture": "native_run_capture", "episodes": report["episodes"],
            "checkpoint_sha256": report["checkpoint_sha256"], "decisions": decisions, "frame_sha256": files,
            "disconnected": bool(report.get("disconnected")),
            "source_trace_sha256": digest(trace, "sha256"),
            "note": "Frames were captured around each applied action. Terminal events are engine counters, not inferred from the final image."})


class ReplayLibrary:
    def __init__(self, root="runs"):
        self.root = Path(root)

    def entries(self):
        paths = list(self.root.glob("replays/*/replay.json")) + list(self.root.glob("*/run-*/replay.json"))
        result = {}
        for path in paths:
            key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
            result[key] = path
        return result

    def manifest(self, key):
        path = self.entries().get(key)
        if path is None:
            raise ValueError("Unknown replay")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != "vision_replay_v1":
            raise ValueError("Unsupported replay format")
        return path, value

    def listing(self):
        result = []
        for key in self.entries():
            _, item = self.manifest(key)
            result.append({"id": key, "title": item["title"], "capture": item["capture"],
                           "checkpoint_sha256": item.get("checkpoint_sha256"),
                           "student_only": bool(item["decisions"] and item["decisions"][0].get("vision_probabilities") is None),
                           "decisions": len(item["decisions"]), "episodes": len(item["episodes"]),
                           "kills": sum(e["kills"] for e in item["episodes"])})
        return result

    def data(self, key):
        _, item = self.manifest(key)
        for row in item["decisions"]:
            row["frames"] = [f"/api/replay-frame?id={key}&frame={list(item['frame_sha256']).index(name)}" for name in row["frames"]]
        return {k: v for k, v in item.items() if k not in {"frame_sha256", "vision"}}

    def frame(self, key, index):
        path, item = self.manifest(key)
        names = list(item["frame_sha256"])
        if not 0 <= index < len(names):
            raise ValueError("Frame index out of range")
        name = names[index]
        frame = safe_path(path.parent, name)
        body = frame.read_bytes()
        if hashlib.sha256(body).hexdigest() != item["frame_sha256"][name]:
            raise ValueError("Replay frame checksum changed")
        return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    build_replay(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
