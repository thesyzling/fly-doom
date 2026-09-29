"""Add strictly causal, episode-local history to existing teacher observations."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.learning_data import ACTIONS, read_prepared


TEMPORAL_FORMAT = "three_past_observations_v1"
NONVISUAL_KEYS = ("previous_action", "recent_actions", "timing")


def visual_summary(state):
    """Compact past-frame evidence; never includes its demonstrated next action."""
    grid = np.asarray([[int(x) for x in row.split()] for row in state["rows"]])
    if grid.shape != (8, 8):
        raise ValueError("Temporal history needs an 8 by 8 brightness grid")
    bands = [int(np.rint(grid[:, columns].mean())) for columns in (slice(0, 3), slice(3, 5), slice(5, 8))]
    return f"{state.get('blue_region', 'Color not available')}; brightness left/center/right {bands[0]}/{bands[1]}/{bands[2]}"


class ObservationHistory:
    """Use observe(frame), apply an action, then advance(action); reset per episode."""

    def __init__(self):
        self.visuals = []
        self.actions = []
        self.last_active = None
        self.last_shot = None

    def observe(self, state):
        index = len(self.actions)
        expected_previous = ACTIONS[self.actions[-1]] if self.actions else ACTIONS[0]
        if state["previous_action"] != expected_previous:
            raise ValueError("Observation previous action does not match the episode history")
        return {"temporal_format": TEMPORAL_FORMAT,
                "timing": {"completed_decisions": index,
                           "since_active": "none" if self.last_active is None else index - self.last_active,
                           "since_shot": "none" if self.last_shot is None else index - self.last_shot},
                "recent_actions": ["NONE"] * max(0, 3 - index) + [ACTIONS[a] for a in self.actions[-3:]],
                "recent_visuals": ["unavailable"] * max(0, 3 - index) + self.visuals[-3:],
                **state}

    def advance(self, state, action):
        if action not in range(4):
            raise ValueError("Invalid applied action")
        index = len(self.actions)
        self.visuals.append(visual_summary(state))
        self.actions.append(int(action))
        if action:
            self.last_active = index
        if action == 3:
            self.last_shot = index


def episode_states(states, actions, previous):
    if len(states) != len(actions) or len(previous) != len(actions):
        raise ValueError("Episode arrays have different lengths")
    history = ObservationHistory()
    output = []
    for i, state in enumerate(states):
        expected = int(actions[i - 1]) if i else 0
        if int(previous[i]) != expected:
            raise ValueError("Stored previous actions are not causal")
        if "temporal_format" in state:
            raise ValueError("This dataset already contains temporal history")
        output.append(history.observe(state))
        history.advance(state, int(actions[i]))
    return output


def augment(dataset, output):
    source, _ = read_prepared(dataset)
    dataset, output = Path(dataset), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {**source, "status": "running", "episodes": [],
              "observation_format": TEMPORAL_FORMAT,
              "parent_prepared_sha256": digest(dataset / "manifest.json", "sha256"),
              "temporal_implementation_sha256": digest(__file__, "sha256"),
              "history_note": "Three past visual summaries and actions; current action and future frames excluded; reset each episode",
              "neural_features_recomputed": False}
    try:
        for episode in source["episodes"]:
            with np.load(dataset / episode["file"], allow_pickle=False) as arrays:
                states = json.loads((dataset / episode["states_file"]).read_text(encoding="utf-8"))
                temporal = episode_states(states, arrays["actions"], arrays["previous_actions"])
            shutil.copyfile(dataset / episode["file"], output / episode["file"])
            if digest(output / episode["file"], "sha256") != episode["sha256"]:
                raise ValueError("Copied neural feature checksum differs")
            write_json(output / episode["states_file"], temporal)
            report["episodes"].append({**episode, "states_sha256": digest(output / episode["states_file"], "sha256")})
        report["status"] = "completed"
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        write_json(output / "manifest.json", report)
    print(f"Added causal history to {len(report['episodes'])} episodes without rerunning the graph: {output}", flush=True)
    return report


def timing_baseline(train, validation):
    """Diagnostic: a clock and previous-action lookup, with no visual input."""
    tables = {}
    global_counts = Counter(map(int, train["actions"]))
    majority = global_counts.most_common(1)[0][0]
    for state, label in zip(train["states"], train["actions"]):
        key = (state["timing"]["completed_decisions"], state["previous_action"])
        tables.setdefault(key, Counter())[int(label)] += 1
    result = []
    for state in validation["states"]:
        key = (state["timing"]["completed_decisions"], state["previous_action"])
        result.append(tables[key].most_common(1)[0][0] if key in tables else majority)
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    augment(**vars(args))


if __name__ == "__main__":
    main()
