"""Bounded task optimization of existing fly synapses with a frozen readout.

This first plasticity experiment uses tied magnitude gains on presynaptic
annotation groups projecting to descending cells. It replays the entire exact
LIF graph for every objective evaluation; no detached feature cache is trained.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.live_brain import load_checkpoint
from flydoom.movement_core import ACTIONS, Memory, MovementReadout, apply_action, make_game, vector
from flydoom.movement_study import CALIBRATION, DATA, PARENT

STUDENT = Path("runs/movement-pilot-v1")
SCHEMA = "fly_synaptic_gains_v1"
BOUND = 0.25


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_system():
    identity = json.loads((STUDENT / "report.json").read_text())
    if identity["status"] != "completed" or digest(STUDENT / "student.safetensors", "sha256") != identity["student_sha256"]:
        raise ValueError("The completed six-action student is required")
    ids, controller, parent, _ = load_checkpoint(PARENT, CALIBRATION, DATA)
    model = MovementReadout(identity["input_size"], identity["hidden"])
    model.load_state_dict(load_file(str(STUDENT / "student.safetensors")))
    model.eval().requires_grad_(False)
    with (DATA / "neuron_annotations.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return ids, controller, parent, model, rows, identity


class PlasticEdges:
    """Only selected existing nonzero CSR entries may change, by positive gain."""

    def __init__(self, weights, offsets, groups, labels):
        offsets = np.asarray(offsets, dtype=np.int64)
        groups = np.asarray(groups, dtype=np.int64)
        if (offsets.ndim != 1 or not len(offsets) or groups.shape != offsets.shape
                or len(np.unique(offsets)) != len(offsets) or offsets.min() < 0
                or offsets.max() >= len(weights.data) or groups.min() < 0
                or groups.max() >= len(labels) or np.any(weights.data[offsets] == 0)):
            raise ValueError("Invalid plastic edge mask")
        self.weights, self.offsets, self.groups, self.labels = weights, offsets, groups, list(labels)
        self.base = weights.data[offsets].copy()
        self.gains = np.ones(len(labels), dtype=np.float32)

    @classmethod
    def select(cls, controller, rows, limit=4):
        weights = controller.network.weights
        outputs = np.sort(np.concatenate(controller.mapping.output_groups))
        offsets = np.concatenate([np.arange(weights.indptr[i], weights.indptr[i+1]) for i in outputs])
        offsets = offsets[weights.data[offsets] != 0]
        names = np.array([r["super_class"] or "unclassified" for r in rows])
        edge_names = names[weights.indices[offsets]]
        mass = {name: float(np.abs(weights.data[offsets[edge_names == name]]).sum()) for name in np.unique(edge_names)}
        labels = sorted(mass, key=lambda name: (-mass[name], name))[:limit]
        keep = np.isin(edge_names, labels)
        return cls(weights, offsets[keep], [labels.index(n) for n in edge_names[keep]], labels)

    def apply(self, gains):
        gains = np.asarray(gains, dtype=np.float32)
        if gains.shape != self.gains.shape or not np.isfinite(gains).all() or np.any(np.abs(gains - 1) > BOUND + 1e-7):
            raise ValueError("Synaptic gains must remain within [0.75, 1.25]")
        self.weights.data[self.offsets] = self.base * gains[self.groups]
        self.gains = gains.copy()

    def save(self, path, ids):
        targets = np.searchsorted(self.weights.indptr, self.offsets, side="right") - 1
        np.savez_compressed(path, offsets=self.offsets, groups=self.groups, gains=self.gains,
                            base=self.base, trained=self.weights.data[self.offsets],
                            source_ids=ids[self.weights.indices[self.offsets]], target_ids=ids[targets])


def restore(controller, ids, folder, identity):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text())
    if (report.get("schema") != SCHEMA or report.get("status") != "completed"
            or report["student_sha256"] != identity["student_sha256"]
            or digest(folder / "synapses.npz", "sha256") != report["synapses_sha256"]
            or digest(CALIBRATION, "sha256") != report["calibration_sha256"]):
        raise ValueError("Synaptic checkpoint identity mismatch")
    weights = controller.network.weights
    if array_hash(weights.data) != report["base_weights_sha256"]:
        raise ValueError("Synaptic checkpoint belongs to different base weights")
    with np.load(folder / "synapses.npz", allow_pickle=False) as d:
        patch = PlasticEdges(weights, d["offsets"], d["groups"], report["group_labels"])
        targets = np.searchsorted(weights.indptr, patch.offsets, side="right") - 1
        if (not np.array_equal(patch.base, d["base"])
                or not np.array_equal(ids[weights.indices[patch.offsets]], d["source_ids"])
                or not np.array_equal(ids[targets], d["target_ids"])):
            raise ValueError("Synaptic endpoint or baseline mismatch")
        patch.apply(d["gains"])
        if not np.array_equal(weights.data[patch.offsets], d["trained"]):
            raise ValueError("Synaptic gain reconstruction mismatch")
    if array_hash(weights.data) != report["trained_weights_sha256"]:
        raise ValueError("Trained graph checksum mismatch")
    return patch, report


def examples(split, scenes, steps):
    folder = STUDENT / split
    rows = json.loads((folder / "trace.json").read_text())
    seeds = list(dict.fromkeys(r["seed"] for r in rows))[:scenes]
    episodes = []
    for seed in seeds:
        trajectory = [r for r in rows if r["seed"] == seed][:steps]
        if [r["decision"] for r in trajectory] != list(range(1, len(trajectory) + 1)):
            raise ValueError("Replay must start at the episode boundary")
        for row in trajectory:
            path = folder / row["image"]
            if digest(path, "sha256") != row["png_sha256"]:
                raise ValueError("Changed source frame")
            row["rgb"] = np.array(Image.open(path).convert("RGB"))
            if array_hash(row["rgb"]) != row["teacher"]["frame_sha256"]:
                raise ValueError("Teacher/image mismatch")
        episodes.append(trajectory)
    return episodes


def replay(controller, model, episodes):
    logits, targets, minimum, maximum = [], [], 0., 0.
    for episode in episodes:
        controller.reset()
        memory = Memory()
        for row in episode:
            decision = controller.decide(row["rgb"])
            minimum = min(minimum, decision["voltage_min_mv"])
            maximum = max(maximum, decision["voltage_max_mv_after_reset"])
            if minimum < -90 or not np.isfinite(controller.network.voltage).all():
                raise ValueError("Synaptic candidate failed the engineering voltage bound")
            with torch.inference_mode():
                logits.append(model(torch.tensor(vector(controller, memory))[None])[0])
            targets.append(row["teacher"]["probabilities"])
            memory.advance(ACTIONS.index(row["action"]))
    logits, targets = torch.stack(logits), torch.tensor(targets, dtype=torch.float32)
    return {"kl": float(F.kl_div(logits.log_softmax(1), targets, reduction="batchmean")),
            "teacher_agreement": float((logits.argmax(1) == targets.argmax(1)).float().mean()),
            "samples": len(logits), "minimum_voltage_mv": minimum,
            "probabilities": logits.softmax(1).tolist()}


def gameplay(controller, model, seeds, limit=12):
    result = []
    game = make_game(seeds[0])
    try:
        for seed in seeds:
            game.set_seed(seed); game.new_episode(); controller.reset(); memory = Memory(); trace = []
            for _ in range(limit):
                if game.is_episode_finished(): break
                frame = game.get_state().screen_buffer.copy()
                decision = controller.decide(frame)
                if decision["voltage_min_mv"] < -90: raise ValueError("Live voltage bound failed")
                with torch.inference_mode(): p = model(torch.tensor(vector(controller, memory))[None]).softmax(1)[0].numpy()
                action = int(p.argmax()); effect = apply_action(game, action); memory.advance(action)
                trace.append({"action": ACTIONS[action], "probabilities": p.tolist(), "effect": effect})
            result.append({"seed": seed, "return": game.get_total_reward(), "kills": trace[-1]["effect"]["kills"], "trace": trace})
    finally:
        game.close()
    return result


def train(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    ids, controller, _, model, rows, identity = load_system()
    patch = PlasticEdges.select(controller, rows)
    train_data, val_data = examples("train", 8, 2), examples("validation", 4, 2)
    hashes = {r["teacher"]["frame_sha256"] for ep in train_data for r in ep}
    if any(r["teacher"]["frame_sha256"] in hashes for ep in val_data for r in ep):
        raise ValueError("Exact teaching split overlap")
    locked = [Path(__file__), Path("flydoom/simulation.py"), Path("flydoom/bridge.py"),
              Path("flydoom/movement_core.py"), Path("flydoom/student.py"), CALIBRATION,
              STUDENT / "student.safetensors", STUDENT / "report.json",
              STUDENT / "train/trace.json", STUDENT / "validation/trace.json"]
    plan = {"schema": SCHEMA, "training_scenes": 8, "validation_scenes": 4, "prefix_decisions": 2,
            "optimizer": "Deterministic bounded coordinate search; frozen engineered readout",
            "steps": [0.15, 0.075], "regularization": 0.002, "gain_bounds": [0.75, 1.25],
            "group_rule": "Four largest presynaptic super-classes by absolute base weight into descending neurons",
            "selection": "Training KL plus gain penalty; validation reported without selecting or tuning",
            "development_game_seeds": [74000, 74001], "development_decision_limit": 12,
            "scope": "First synaptic optimization pilot. Prefix teaching scenes were already seen by the frozen student. No generalization claim.",
            "artifact_sha256": {str(p): digest(p, "sha256") for p in locked}}
    write_json(output / "plan.json", plan)
    report = {"schema": SCHEMA, "status": "running", "student_sha256": identity["student_sha256"],
              "calibration_sha256": digest(CALIBRATION, "sha256"), "plan_sha256": digest(output / "plan.json", "sha256"),
              "base_weights_sha256": array_hash(controller.network.weights.data), "group_labels": patch.labels,
              "plastic_edges": len(patch.offsets), "total_edges": controller.network.weights.nnz,
              "group_edge_counts": np.bincount(patch.groups).tolist(), "history": [], "scope": plan["scope"],
              "readout_trained": False, "teacher_trained": False, "biology_validated": False}
    write_json(output / "report.json", report)
    topology = (array_hash(controller.network.weights.indices), array_hash(controller.network.weights.indptr))
    outside = np.ones(controller.network.weights.nnz, dtype=bool)
    outside[patch.offsets] = False
    untouched_sha = array_hash(controller.network.weights.data[outside])
    try:
        report["initial"] = {"train": replay(controller, model, train_data), "validation": replay(controller, model, val_data)}
        gains = np.ones(len(patch.labels), np.float32); best = report["initial"]["train"]["kl"]
        report["history"].append({"evaluation": 0, "gains": gains.tolist(), "kl": best, "objective": best, "accepted": True})
        print(f"Plastic mask: {len(patch.offsets)} existing edges / {patch.labels}; initial train KL {best:.6f}", flush=True)
        for step in plan["steps"]:
            for group in range(len(gains)):
                origin, candidates = gains.copy(), []
                for direction in (-1, 1):
                    candidate = origin.copy(); candidate[group] = np.clip(candidate[group] + direction * step, .75, 1.25)
                    patch.apply(candidate)
                    try:
                        measured = replay(controller, model, train_data)
                        score = measured["kl"] + plan["regularization"] * float(np.mean((candidate - 1)**2))
                        candidates.append((score, candidate.copy()))
                        event = {"evaluation": len(report["history"]), "gains": candidate.tolist(), "kl": measured["kl"],
                                 "objective": score, "accepted": score < best}
                    except ValueError as error:
                        event = {"evaluation": len(report["history"]), "gains": candidate.tolist(), "rejected": str(error), "accepted": False}
                    report["history"].append(event); write_json(output / "report.json", report)
                    print(f"Synaptic search {event['evaluation']}/16: {event.get('kl', event.get('rejected'))}", flush=True)
                for score, candidate in candidates:
                    if score < best: best, gains = score, candidate
                patch.apply(gains)
        report["final"] = {"train": replay(controller, model, train_data), "validation": replay(controller, model, val_data)}
        report["gains"] = gains.tolist()
        report["changed_edges"] = int(np.count_nonzero(controller.network.weights.data[patch.offsets] != patch.base))
        report["connectome_weights_trained"] = report["changed_edges"] > 0
        report["trained_weights_sha256"] = array_hash(controller.network.weights.data)
        patch.save(output / "synapses.npz", ids)
        report["synapses_sha256"] = digest(output / "synapses.npz", "sha256")
        report["development_gameplay"] = {}
        for name, values in (("base_graph", np.ones(len(gains))), ("trained_graph", gains)):
            patch.apply(values)
            report["development_gameplay"][name] = gameplay(controller, model, plan["development_game_seeds"])
            print(f"Development gameplay completed: {name}", flush=True)
        patch.apply(gains)
        if topology != (array_hash(controller.network.weights.indices), array_hash(controller.network.weights.indptr)):
            raise ValueError("Topology changed")
        if (array_hash(controller.network.weights.data[outside]) != untouched_sha
                or not np.array_equal(np.sign(patch.base), np.sign(controller.network.weights.data[patch.offsets]))):
            raise ValueError("A non-plastic edge or transmitter sign changed")
        if any(digest(path, "sha256") != sha for path, sha in plan["artifact_sha256"].items()):
            raise ValueError("Locked source or input changed")
        report.update(status="completed", topology_unchanged=True, signs_unchanged=True,
                      outside_mask_unchanged=True, source_student_unchanged=True)
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = train(parser.parse_args().output)
    print({k: result[k] for k in ("status", "plastic_edges", "changed_edges", "gains")})
