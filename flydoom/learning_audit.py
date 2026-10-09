"""Independent artifact checks and descriptive paired benchmark uncertainty."""

import argparse
import json
from pathlib import Path

import numpy as np

from flydoom.data import digest
from flydoom.learning_cycle import atomic_json, load_experience
from flydoom.movement_core import ACTIONS
from flydoom.synaptic import load_system, array_hash
from flydoom.synaptic_eligibility import restore


def summarize(benchmarks):
    result = []
    rng = np.random.default_rng(919)
    for split, pair in benchmarks.items():
        if not all(key in pair for key in ("champion", "candidate")): continue
        before, after = pair["champion"], pair["candidate"]
        if [(r["seed"], r["task"]) for r in before] != [(r["seed"], r["task"]) for r in after]: raise ValueError("Unpaired benchmark")
        for task in sorted({r["task"] for r in before}):
            a = [r for r in before if r["task"] == task]; b = [r for r in after if r["task"] == task]
            indices = rng.integers(0, len(a), (10000, len(a)))
            row = {"split": split, "task": task, "pairs": len(a), "truncated_episodes": sum(not r["terminal"] for r in a+b),
                   "uncertainty_scope": "Descriptive paired bootstrap, not an acceptance criterion; few starts cannot establish generalization."}
            for metric in ("return", "kills", "range_progress"):
                x = np.array([r[metric] for r in a]); y = np.array([r[metric] for r in b]); difference = y-x
                row[metric] = {"champion": float(x.mean()), "candidate": float(y.mean()), "delta": float(difference.mean()),
                               "paired_95_interval": np.quantile(difference[indices].mean(1), [.025, .975]).tolist()}
            result.append(row)
    return result


def audit(folder):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text()); plan = json.loads((folder / "plan.json").read_text())
    if report["status"] != "completed": raise ValueError("Cycle is incomplete")
    if digest(folder / "plan.json", "sha256") != report["plan_sha256"]: raise ValueError("Plan changed")
    for name, expected in plan["source_sha256"].items():
        if digest(Path(name), "sha256") != expected: raise ValueError("Locked source changed: " + name)
    source = Path(plan.get("recovered_experience", {}).get("path", folder))
    for name, expected in plan.get("recovered_experience", {}).get("sha256", {}).items():
        if digest(Path(name), "sha256") != expected: raise ValueError("Recovered labels changed")
    train, train_info = load_experience(source, "train"); val, val_info = load_experience(source, "validation")
    for episodes in (train, val):
        for ep in episodes:
            for row in ep:
                if row["action"] not in ACTIONS: raise ValueError("Unknown experience action")
                p = np.asarray(row["teacher"]["probabilities"])
                if p.shape != (6,) or np.any(p < 0) or not np.isclose(p.sum(), 1): raise ValueError("Invalid learning target")
    ids, controller, _, model, annotations, identity = load_system()
    before = controller.network.weights.data.copy()
    patch, restored = restore(controller, ids, folder, identity)
    weights = controller.network.weights
    mask = np.ones(weights.nnz, bool); mask[patch.offsets] = False
    if not np.array_equal(before[mask], weights.data[mask]): raise ValueError("Unselected edges changed")
    if not np.array_equal(np.sign(before), np.sign(weights.data)): raise ValueError("Graph signs changed")
    changed = int(np.count_nonzero(before != weights.data))
    if changed != report["changed_edges"]: raise ValueError("Incorrect changed-edge count")
    benchmarks = json.loads((folder / "benchmarks.json").read_text())
    for pair in benchmarks.values():
        for episodes in pair.values():
            for episode in episodes:
                if episode["trace"][0]["rgb_sha256"] != episode["rgb_sha256"]: raise ValueError("Opening identity changed")
                for step in episode["trace"]:
                    if step["action"] != ACTIONS[int(np.argmax(step["probabilities"]))]: raise ValueError("Benchmark is not autonomous argmax")
    result = {"status": "passed", "changed_edges": changed, "trained_weights_sha256": array_hash(weights.data),
              "train_frames": train_info["frames"], "validation_frames_replayed": val_info["frames"],
              "outside_mask_unchanged": True, "signs_unchanged": True, "benchmark": summarize(benchmarks),
              "scope": "Independent reload, frame/label/source/checkpoint hashes and recorded action audit. Full neural loss replay and game engine are not executed again."}
    atomic_json(folder / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("folder", type=Path)
    print(json.dumps(audit(parser.parse_args().folder), indent=2))
