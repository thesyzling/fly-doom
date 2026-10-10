"""Diagnose recorded before/after predictions without retraining or game replay."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from flydoom.data import digest
from flydoom.learning_cycle import atomic_json, load_experience
from flydoom.movement_core import ACTIONS
from flydoom.synaptic_consensus import episode_losses


def diagnose(folder):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text())
    plan = json.loads((folder / "plan.json").read_text())
    if report.get("status") != "completed" or digest(folder / "plan.json", "sha256") != report["plan_sha256"]:
        raise ValueError("Incomplete cycle or changed plan")
    for name, expected in plan.get("recovered_experience", {}).get("sha256", {}).items():
        if digest(name, "sha256") != expected: raise ValueError("Recovered teaching labels changed")
    source = Path(plan.get("recovered_experience", {}).get("path", folder))
    train, _ = load_experience(source, "train")
    validation, _ = load_experience(source, "validation")
    hashes = {r["teacher"]["frame_sha256"] for ep in train for r in ep}
    # Rehearsal is training data too; include its verified label hashes.
    for name, expected in plan.get("rehearsal", {}).get("sha256", {}).items():
        if digest(name, "sha256") != expected: raise ValueError("Rehearsal source changed")
        if name.endswith("trace.json"):
            rows = json.loads(Path(name).read_text())
            seeds = list(dict.fromkeys(r["seed"] for r in rows))[:plan["rehearsal"]["episodes"]]
            hashes.update(r["teacher"]["frame_sha256"] for r in rows if r["seed"] in seeds and r["decision"] <= 2)
    for ep in validation:
        for row in ep: row["score"] = row["teacher"]["frame_sha256"] not in hashes
    result = {"scope": "Descriptive diagnosis of saved predictions; episode concentration is not a causal explanation.",
              "source_report_sha256": digest(folder / "report.json", "sha256"),
              "source_plan_sha256": digest(folder / "plan.json", "sha256"),
              "changed_from_champion": report.get("changed_from_champion"), "splits": {}}
    for split, episodes in (("train", train), ("validation", validation)):
        before = episode_losses(episodes, report["initial"][split])
        after = episode_losses(episodes, report["final"][split])
        targets = Counter(ACTIONS[int(np.argmax(r["teacher"]["probabilities"]))] for ep in episodes for r in ep)
        result["splits"][split] = {"teacher_top_action_counts_all_frames": dict(targets),
            "initial_frame_mean_kl": report["initial"][split]["kl"], "final_frame_mean_kl": report["final"][split]["kl"],
            "episodes": [{"seed": a["seed"], "task": ep[0].get("task"), "scored_frames": a["samples"],
                          "initial_kl": a["kl"], "final_kl": b["kl"], "delta_kl": b["kl"] - a["kl"]}
                         for ep, a, b in zip(episodes, before, after)]}
    atomic_json(folder / "diagnosis.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("folder", type=Path)
    print(json.dumps(diagnose(parser.parse_args().folder), indent=2))
