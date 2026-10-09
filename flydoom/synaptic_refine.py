"""Small antithetic gain search after the coarse plasticity pilot was rejected."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.synaptic import (BOUND, CALIBRATION, SCHEMA, STUDENT, PlasticEdges,
                             array_hash, examples, gameplay, load_system, replay)


def train(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    ids, controller, _, model, rows, identity = load_system()
    patch = PlasticEdges.select(controller, rows)
    train_data, validation = examples("train", 8, 2), examples("validation", 4, 2)
    earlier = json.loads(Path("runs/synaptic-pilot-v1/report.json").read_text())
    if earlier["status"] != "completed" or earlier["changed_edges"]:
        raise ValueError("This follow-up assumes the preserved coarse search selected the base graph")
    inputs = [Path(__file__), Path("flydoom/synaptic.py"), Path("flydoom/simulation.py"), Path("flydoom/bridge.py"),
              Path("flydoom/movement_core.py"), Path("flydoom/student.py"), CALIBRATION,
              STUDENT / "student.safetensors", STUDENT / "report.json", STUDENT / "train/trace.json", STUDENT / "validation/trace.json"]
    rng = np.random.default_rng(2210)
    directions = [rng.choice([-1., 1.], len(patch.labels)).tolist() for _ in range(4)]
    plan = {"schema": SCHEMA, "source_pilot": "runs/synaptic-pilot-v1",
            "reason": "All coarse training perturbations worsened or preserved training KL; reduce trust region, jointly vary gains",
            "optimizer": "Antithetic pattern search with fixed directions; greedy training-only acceptance",
            "seed": 2210, "directions": directions, "steps": [.01, .0025], "evaluations": 16,
            "training_scenes": 8, "validation_scenes": 4, "prefix_decisions": 2,
            "regularization": .002, "validation_role": "Only report before and after; never select or tune on it",
            "development_game_seeds": [74000, 74001], "development_decision_limit": 12,
            "scope": "Second bounded engineering pilot on previously seen prefix scenes; no generalization claim.",
            "artifact_sha256": {str(p): digest(p, "sha256") for p in inputs}}
    write_json(output / "plan.json", plan)
    weights = controller.network.weights
    outside = np.ones(weights.nnz, bool); outside[patch.offsets] = False
    outside_sha = array_hash(weights.data[outside]); topology = (array_hash(weights.indices), array_hash(weights.indptr))
    report = {"schema": SCHEMA, "status": "running", "student_sha256": identity["student_sha256"],
              "calibration_sha256": digest(CALIBRATION, "sha256"), "plan_sha256": digest(output / "plan.json", "sha256"),
              "base_weights_sha256": array_hash(weights.data), "group_labels": patch.labels,
              "plastic_edges": len(patch.offsets), "total_edges": weights.nnz,
              "group_edge_counts": np.bincount(patch.groups).tolist(), "history": [], "scope": plan["scope"],
              "readout_trained": False, "teacher_trained": False, "biology_validated": False}
    write_json(output / "report.json", report)
    try:
        report["initial"] = {"train": replay(controller, model, train_data), "validation": replay(controller, model, validation)}
        gains = np.ones(len(patch.labels), np.float32); best = report["initial"]["train"]["kl"]
        report["history"].append({"evaluation": 0, "gains": gains.tolist(), "kl": best, "objective": best, "accepted": True})
        print(f"Fine search initial KL {best:.7f}", flush=True)
        for step in plan["steps"]:
            for direction in directions:
                origin = gains.copy(); choices = []
                for sign in (-1, 1):
                    candidate = np.clip(origin + sign * step * np.array(direction, np.float32), 1-BOUND, 1+BOUND)
                    patch.apply(candidate)
                    try:
                        result = replay(controller, model, train_data)
                        score = result["kl"] + .002 * float(np.mean((candidate-1)**2))
                        choices.append((score, candidate.copy()))
                        event = {"evaluation": len(report["history"]), "gains": candidate.tolist(), "kl": result["kl"], "objective": score, "accepted": score < best}
                    except ValueError as error:
                        event = {"evaluation": len(report["history"]), "gains": candidate.tolist(), "rejected": str(error), "accepted": False}
                    report["history"].append(event); write_json(output / "report.json", report)
                    print(f"Fine candidate {event['evaluation']}/16: {event.get('kl', event.get('rejected'))}", flush=True)
                for score, candidate in choices:
                    if score < best: best, gains = score, candidate
                patch.apply(gains)
        report["final"] = {"train": replay(controller, model, train_data), "validation": replay(controller, model, validation)}
        report["gains"] = gains.tolist(); report["changed_edges"] = int(np.count_nonzero(weights.data[patch.offsets] != patch.base))
        report["connectome_weights_trained"] = bool(report["changed_edges"])
        report["trained_weights_sha256"] = array_hash(weights.data)
        patch.save(output / "synapses.npz", ids); report["synapses_sha256"] = digest(output / "synapses.npz", "sha256")
        report["development_gameplay"] = {}
        for name, values in (("base_graph", np.ones(len(gains))), ("trained_graph", gains)):
            patch.apply(values); report["development_gameplay"][name] = gameplay(controller, model, plan["development_game_seeds"])
            print(f"Fine search development run: {name}", flush=True)
        patch.apply(gains)
        if (array_hash(weights.data[outside]) != outside_sha or topology != (array_hash(weights.indices), array_hash(weights.indptr))
                or not np.array_equal(np.sign(patch.base), np.sign(weights.data[patch.offsets]))): raise ValueError("Graph constraints failed")
        if any(digest(p, "sha256") != h for p, h in plan["artifact_sha256"].items()): raise ValueError("Locked input changed")
        report.update(status="completed", topology_unchanged=True, signs_unchanged=True, outside_mask_unchanged=True, source_student_unchanged=True)
    except BaseException:
        report["status"] = "interrupted_or_failed"; raise
    finally:
        write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    result = train(parser.parse_args().output)
    print({k: result[k] for k in ("status", "changed_edges", "gains")})
