"""Bounded autonomous Doom evaluation of a local teacher and diagnostic controls.

An unaccepted teacher may be evaluated experimentally, but this command never
approves distillation or changes the saved teacher gate. No fly graph is loaded.
"""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.learning_data import ACTIONS, image_state, make_game, read_prepared
from flydoom.temporal_data import ObservationHistory, TEMPORAL_FORMAT


def timing_table(part):
    table = {}
    for state, action in zip(part["states"], part["actions"]):
        key = (state["timing"]["completed_decisions"], state["previous_action"])
        table.setdefault(key, Counter())[int(action)] += 1
    return {key: counts.most_common(1)[0][0] for key, counts in table.items()}, int(np.bincount(part["actions"], minlength=4).argmax())


def policy_observation(state, observation_format):
    if observation_format == "single_frame":
        return {key: state[key] for key in ("vision_format", "blue_region", "image", "rows", "previous_action")}
    if observation_format != TEMPORAL_FORMAT:
        raise ValueError("Unsupported observation format")
    return state


def run_episode(game, buttons, choose, seed, *, max_decisions=75, visible=False, trace=None, policy="laya"):
    game.set_seed(seed)
    game.new_episode()
    history = ObservationHistory()
    previous = 0
    summary = {"seed": seed, "decisions": 0, "return": 0.0, "action_counts": [0] * 4,
               "end_reason": "interrupted", "policy": policy}
    for index in range(max_decisions):
        if game.is_episode_finished():
            break
        current = image_state(game.get_state().screen_buffer, previous)
        state = history.observe(current)
        action = int(choose(state))
        if action not in range(4):
            raise ValueError("Policy returned an invalid button")
        for _ in range(4):
            if game.is_episode_finished():
                break
            started = perf_counter()
            game.make_action([int(button == ACTIONS[action]) for button in buttons], 1)
            if visible:
                sleep(max(0, 1 / 35 - (perf_counter() - started)))
        history.advance(current, action)
        previous = action
        summary["decisions"] = index + 1
        summary["return"] = float(game.get_total_reward())
        summary["action_counts"][action] += 1
        if trace is not None:
            trace.write(json.dumps({"policy": policy, "seed": seed, "decision": index + 1,
                "action": ACTIONS[action], "return": summary["return"]}) + "\n")
            trace.flush()
    summary["end_reason"] = "game_finished" if game.is_episode_finished() else "decision_limit"
    # This diagnostic is read after play only and never enters the policy state.
    import vizdoom as vzd
    summary["kills"] = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
    return summary


def evaluate(teacher, output, *, base="models/laya-base", dataset=None, episodes=6, seed=50000,
             max_decisions=75, visible=False, policies=("laya", "timing", "random")):
    from flydoom.laya_teacher import load_base, restore_head, encode_states, probabilities
    import torch
    if (not 1 <= episodes <= 20 or not 1 <= max_decisions <= 75 or not 0 <= seed <= 2**32 - episodes
            or not policies or len(set(policies)) != len(policies) or not set(policies) <= {"laya", "timing", "random"}):
        raise ValueError("Invalid evaluation policies, limits or seed")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    teacher, output = Path(teacher), Path(output)
    teacher_report = json.loads((teacher / "report.json").read_text(encoding="utf-8"))
    observation_format = teacher_report.get("observation_format", "single_frame")
    if teacher_report.get("status") != "completed" or observation_format not in {TEMPORAL_FORMAT, "single_frame"}:
        raise ValueError("This evaluator requires a completed supported teacher")
    if observation_format == "single_frame" and dataset is None:
        raise ValueError("A single-frame teacher requires its matching dataset to verify the image format")
    if digest(teacher / "head.safetensors", "sha256") != teacher_report["head_sha256"]:
        raise ValueError("Teacher weights changed")
    table, fallback = {}, 0
    if dataset is not None:
        source, splits = read_prepared(dataset)
        if digest(Path(dataset) / "manifest.json", "sha256") != teacher_report["dataset_sha256"]:
            raise ValueError("Evaluation dataset does not match the teacher")
        used_seeds = {ep["seed"] for ep in source["episodes"]}
        if used_seeds.intersection(range(seed, seed + episodes)):
            raise ValueError("Evaluation seeds overlap the demonstration episodes")
        if observation_format == "single_frame" and any(s.get("vision_format") != "brightness_color_v2" for s in splits["train"]["states"]):
            raise ValueError("The older ten-level image format is not supported by this evaluator")
        if "timing" in policies:
            if source.get("observation_format") != TEMPORAL_FORMAT:
                raise ValueError("The timing control requires temporal training observations")
            table, fallback = timing_table(splits["train"])
    elif "timing" in policies:
        raise ValueError("The timing control requires the matching prepared dataset")
    output.mkdir(parents=True, exist_ok=False)
    agent = None
    if "laya" in policies:
        agent, provenance = load_base(base)
        if provenance != teacher_report["base"]:
            raise ValueError("Teacher base mismatch")
        restore_head(agent, teacher / "head.safetensors")
    report = {"schema": "temporal_teacher_game_evaluation_v1", "status": "running",
        "teacher_accepted_for_distillation": teacher_report["teacher_accepted"],
        "teacher_report_sha256": digest(teacher / "report.json", "sha256"),
        "connectome_used": False, "training_during_play": False, "episodes": [],
        "observation_format": observation_format,
        "seed_overlap_checked": dataset is not None,
        "note": "Small bounded gameplay experiment, not evidence of fly-brain learning; teacher gate unchanged"}
    game = None
    try:
        with (output / "decisions.jsonl").open("w", encoding="utf-8") as trace:
            for policy in policies:
                game, buttons = make_game(seed, visible)
                for episode in range(episodes):
                    rng = np.random.default_rng(seed + episode)
                    if policy == "laya":
                        def choose(state):
                            observed = policy_observation(state, observation_format)
                            return probabilities(agent, encode_states(agent, [observed]))[0].argmax()
                    elif policy == "timing":
                        def choose(state):
                            return table.get((state["timing"]["completed_decisions"], state["previous_action"]), fallback)
                    else:
                        def choose(state):
                            return rng.integers(4)
                    result = run_episode(game, buttons, choose, seed + episode, max_decisions=max_decisions,
                                         visible=visible, trace=trace, policy=policy)
                    report["episodes"].append(result)
                    write_json(output / "report.json", report)
                    print(f"{policy} | seed {result['seed']} | return {result['return']:.0f} | kills {result['kills']} | {result['end_reason']}", flush=True)
                game.close()
                game = None
        report["summary"] = {}
        for policy in policies:
            rows = [ep for ep in report["episodes"] if ep["policy"] == policy]
            report["summary"][policy] = {"episodes": len(rows), "kills": sum(ep["kills"] for ep in rows),
                "mean_return": float(np.mean([ep["return"] for ep in rows])),
                "decision_limit_episodes": sum(ep["end_reason"] == "decision_limit" for ep in rows)}
        report["status"] = "completed"
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        if game is not None:
            game.close()
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("models/laya-base"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--max-decisions", type=int, default=75)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--policies", nargs="+", choices=("laya", "timing", "random"), default=["laya"])
    args = vars(parser.parse_args())
    if args["output"] is None:
        args["output"] = Path("runs") / datetime.now().strftime("teacher-play-%Y%m%d-%H%M%S-%f")
    evaluate(**args)


if __name__ == "__main__":
    main()
