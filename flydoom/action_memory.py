"""Causal engineering memory for the added cells; no pixels, rewards, or labels."""

import numpy as np
import torch

from flydoom.learning_data import ACTIONS
from flydoom.student import SpikingReadout


NAMES = tuple(f"past_{lag}_{action}" for lag in (3, 2, 1) for action in ACTIONS) + (
    "completed_decisions", "since_active", "since_shot", "has_active", "has_shot")
SCHEMA = "action_memory_student_v1"


def encode(state):
    """Read only three past buttons and timing fields, scaled for <=75 decisions."""
    recent, timing = state["recent_actions"], state["timing"]
    if len(recent) != 3 or any(a not in (*ACTIONS, "NONE") for a in recent):
        raise ValueError("Invalid past action history")
    completed = timing["completed_decisions"]
    if not isinstance(completed, int) or not 0 <= completed <= 75:
        raise ValueError("Invalid completed decision count")
    result = [float(a == action) for a in recent for action in ACTIONS]
    result.append(completed / 75)
    for key in ("since_active", "since_shot"):
        age = timing[key]
        if age != "none" and (not isinstance(age, int) or not 1 <= age <= completed):
            raise ValueError("Invalid past action age")
        result.append(0. if age == "none" else age / 75)
    result.extend(float(timing[key] != "none") for key in ("since_active", "since_shot"))
    return np.asarray(result, dtype=np.float32)


class ActionMemory:
    def __init__(self):
        self.actions = []
        self.last_active = self.last_shot = None

    def observe(self):
        n = len(self.actions)
        return {"recent_actions": ["NONE"] * max(0, 3 - n) + [ACTIONS[a] for a in self.actions[-3:]],
                "timing": {"completed_decisions": n,
                           "since_active": "none" if self.last_active is None else n - self.last_active,
                           "since_shot": "none" if self.last_shot is None else n - self.last_shot}}

    def advance(self, action):
        if action not in range(4):
            raise ValueError("Invalid applied action")
        if action:
            self.last_active = len(self.actions)
        if action == 3:
            self.last_shot = len(self.actions)
        self.actions.append(int(action))


def augment(features, states):
    """Keep neural channels first and previous-action indicators last."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] < 4 or len(features) != len(states):
        raise ValueError("Memory inputs and observations must align")
    return np.concatenate((features[:, :-4], np.stack([encode(s) for s in states]), features[:, -4:]), axis=1)


def expand(parent):
    """Add zero-weight memory inputs so initial logits preserve the parent policy."""
    old = parent.state_dict()
    model = SpikingReadout(parent.input.in_features + len(NAMES), parent.input.out_features)
    new = model.state_dict()
    for key, value in old.items():
        if key not in {"input.weight", "mean", "scale"}:
            new[key].copy_(value)
    with torch.no_grad():
        new["input.weight"].zero_()
        for key in ("input.weight", "mean", "scale"):
            new[key][..., :-4 - len(NAMES)].copy_(old[key][..., :-4])
            new[key][..., -4:].copy_(old[key][..., -4:])
    model.load_state_dict(new)
    return model
