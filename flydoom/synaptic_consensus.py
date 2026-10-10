"""Sparse, episode-consistent synaptic learning with training-only rehearsal.

Run one bounded experiment with ``python -m flydoom.synaptic_consensus``.
The frozen teacher and decoder are unchanged; only existing synapses are trained.
"""

import json
import secrets
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from flydoom import learning_cycle as cycle
from flydoom.data import digest
from flydoom.learning_features import ContrastMotionController
from flydoom.movement_teacher import MovementTeacher
from flydoom.synaptic import CALIBRATION, STUDENT, array_hash, examples, load_system
from flydoom.synaptic_eligibility import SCHEMA, gradient, restore


def episode_losses(episodes, metric):
    """Keep causal prefixes intact and score independent frames per episode."""
    probabilities = np.asarray(metric["probabilities"], dtype=np.float64)
    if len(probabilities) != sum(map(len, episodes)):
        raise ValueError("Prediction and trajectory lengths differ")
    result, offset = [], 0
    for episode in episodes:
        p = probabilities[offset:offset + len(episode)]; offset += len(episode)
        q = np.asarray([r["teacher"]["probabilities"] for r in episode])
        keep = np.array([r.get("score", True) for r in episode], bool)
        if not keep.any():
            raise ValueError("An episode has no independent scored frames")
        losses = np.sum(q * (np.log(np.maximum(q, 1e-30)) - np.log(np.maximum(p, 1e-30))), axis=1)
        result.append({"seed": episode[0]["seed"], "samples": int(keep.sum()), "kl": float(losses[keep].mean())})
    return result


def objective(fresh, rehearsal):
    """Equal episode weighting prevents long trajectories dominating selection."""
    return .75 * float(np.mean([r["kl"] for r in fresh])) + .25 * float(np.mean([r["kl"] for r in rehearsal]))


def consensus_direction(gradients, limit=4096, support=3, agreement=.75):
    values = np.asarray(gradients, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or limit < 1 or support < 1 or not .5 < agreement <= 1:
        raise ValueError("Invalid consensus inputs")
    normalized = np.zeros_like(values)
    for i, row in enumerate(values):
        nonzero = np.abs(row[row != 0])
        if len(nonzero):
            normalized[i] = np.clip(row / max(float(np.quantile(nonzero, .95)), 1e-12), -1, 1)
    positive, negative = (values > 0).sum(0), (values < 0).sum(0)
    active = positive + negative
    eligible = (active >= support) & (np.maximum(positive, negative) >= agreement * np.maximum(active, 1))
    # Minority episodes cannot reverse the consensus direction through magnitude.
    direction = -np.sign(positive - negative) * np.mean(np.abs(normalized), axis=0)
    selected = np.flatnonzero(eligible & (direction != 0))
    selected = selected[np.argsort(-np.abs(direction[selected]), kind="stable")[:limit]]
    sparse = np.zeros(values.shape[1], np.float32); sparse[selected] = direction[selected]
    return sparse, {"supported_edges": int(eligible.sum()), "selected_edges": len(selected),
                    "minimum_episode_support": support, "minimum_sign_agreement": agreement, "edge_limit": limit}


def reserve_splits(rng):
    """Exclude previous trajectory frames as well as all recorded openings."""
    prior_hashes = set()
    for path in cycle.ROOT.glob("*/**/trace.json"):
        prior_hashes.update(cycle.image_hashes(json.loads(path.read_text(encoding="utf-8"))))
    for _ in range(100):
        splits = cycle.reserve(dict(train=6, validation=4, gate=6, test=6), rng)
        if not any(r["rgb_sha256"] in prior_hashes for rows in splits.values() for r in rows):
            return splits
    raise ValueError("Could not reserve independent openings")


def run():
    torch.set_num_threads(2)
    root = cycle.ROOT
    with cycle.CycleLock(root):
        folder = root / datetime.now().strftime("consensus-%Y%m%d-%H%M%S-%f"); folder.mkdir()
        previous = cycle.champion(root)
        status = {"status": "running", "output": str(folder), "champion": previous}
        def phase(name, **values):
            status.update(phase=name, **values)
            cycle.atomic_json(folder / "status.json", status); cycle.atomic_json(root / "latest.json", status)
            print(f"Consensus phase: {name}", flush=True)
        try:
            phase("reserve")
            seed = secrets.randbits(32); rng = np.random.default_rng(seed); splits = reserve_splits(rng)
            import vizdoom
            assets = Path(vizdoom.__file__).parent
            sources = [Path(__file__), Path(cycle.__file__), *[Path("flydoom") / name for name in
                       ("learning_features.py", "learning_curriculum.py", "simulation.py", "bridge.py", "synaptic.py",
                        "synaptic_eligibility.py", "movement_core.py", "movement_teacher.py", "student.py")],
                       assets / "scenarios/basic.cfg", assets / "scenarios/basic.wad", assets / "freedoom2.wad"]
            rehearsal_data = examples("train", 8, 2)
            rehearsal_files = [STUDENT / "train/trace.json"] + [STUDENT / "train" / r["image"] for ep in rehearsal_data for r in ep]
            plan = {"schema": "synaptic_consensus_cycle_v1", "random_seed": seed, "splits": splits,
                    "champion": previous, "decisions": 16, "benchmark_decisions": 32, "steps": [.000025, .00005],
                    "optimizer": "Episode-consistent direct voltage eligibility; maximum 4096 existing edges; no random dense perturbation",
                    "derivative_scope": "Presynaptic spikes/reset schedule fixed locally; no exact BPTT. Selection replays the actual hard-spike recurrent network.",
                    "consensus": {"support": 3, "agreement": .75, "edge_limit": 4096},
                    "selection": "0.75 fresh + 0.25 rehearsal equal-episode KL; each training episode regression <= 0.005; validation is veto only; test never selects",
                    "rehearsal": {"split": "train", "episodes": len(rehearsal_data), "frames": sum(map(len, rehearsal_data)),
                                  "sha256": {str(p): digest(p, "sha256") for p in rehearsal_files}},
                    "source_sha256": {str(p): digest(p, "sha256") for p in sources}}
            cycle.atomic_json(folder / "plan.json", plan)
            ids, raw, _, model, annotations, identity = load_system()
            patch, parent = restore(raw, ids, previous["path"], identity)
            controller = ContrastMotionController(raw, previous["encoder_mix"])
            decide = controller.decide
            def checked_decide(rgb):
                cycle.check_cancel(folder)
                return decide(rgb)
            controller.decide = checked_decide
            original = patch.edge_gains.copy(); weights = patch.weights
            topology = (array_hash(weights.indices), array_hash(weights.indptr))
            mask = np.ones(weights.nnz, bool); mask[patch.offsets] = False; outside = array_hash(weights.data[mask])
            phase("collect")
            teacher = MovementTeacher(folder / "teacher.log")
            try:
                train, train_info = cycle.collect(controller, model, splits["train"], folder / "train", 16, rng, teacher, folder)
                validation, val_info = cycle.collect(controller, model, splits["validation"], folder / "validation", 16, rng, teacher, folder)
                cycle.atomic_json(folder / "teacher.json", teacher.metadata)
            finally:
                teacher.close()
            training_hashes = {r["teacher"]["frame_sha256"] for ep in train + rehearsal_data for r in ep}
            for ep in validation:
                for row in ep: row["score"] = row["teacher"]["frame_sha256"] not in training_hashes
            val_info["excluded_training_duplicate_frames"] = sum(not r["score"] for ep in validation for r in ep)
            phase("baseline", experience={"train": train_info, "validation": val_info, "rehearsal": plan["rehearsal"]["frames"]})
            report = {"schema": SCHEMA, "status": "running", "student_sha256": identity["student_sha256"],
                      "calibration_sha256": digest(CALIBRATION, "sha256"), "base_weights_sha256": parent["base_weights_sha256"],
                      "group_labels": patch.labels, "group_edge_counts": np.bincount(patch.groups).tolist(),
                      "plastic_edges": len(patch.offsets), "total_edges": weights.nnz, "readout_trained": False,
                      "teacher_trained": False, "biology_validated": False, "history": [],
                      "parameterization": "Sparse episode-consistent individual edge gains; fixed image adapter",
                      "scope": "Bounded consensus/rehearsal pilot; failed candidates are retained and never deployed.",
                      "plan_sha256": digest(folder / "plan.json", "sha256")}
            def measure(episodes):
                metric = cycle.replay(controller, model, episodes)
                metric["episodes"] = episode_losses(episodes, metric)
                return metric
            initial = {"train": measure(train), "validation": measure(validation), "rehearsal": measure(rehearsal_data)}
            report["initial"] = initial
            best = objective(initial["train"]["episodes"], initial["rehearsal"]["episodes"])
            best_gains = original.copy(); best_metrics = initial
            report["history"].append({"evaluation": 0, "method": "champion", "accepted": True, "objective": best, "kl": initial["train"]["kl"]})
            phase("eligibility")
            gradients = []
            for episode in train:
                gradients.append(gradient(controller, model, patch, [episode]))
                phase("eligibility", episodes_complete=len(gradients))
            direction, report["consensus"] = consensus_direction(gradients)
            np.savez_compressed(folder / "eligibility.npz", gradients=gradients, direction=direction)
            for step in plan["steps"]:
                patch.apply(np.clip(original + step * direction, .75, 1.25))
                phase("optimize", step=step)
                fresh, rehearsal = measure(train), measure(rehearsal_data)
                score = objective(fresh["episodes"], rehearsal["episodes"])
                regressions = [b["kl"] - a["kl"] for key, rows in (("train", fresh), ("rehearsal", rehearsal))
                               for a, b in zip(initial[key]["episodes"], rows["episodes"])]
                accepted = score < best and max(regressions) <= .005
                if accepted:
                    best, best_gains = score, patch.edge_gains.copy()
                    best_metrics = {"train": fresh, "rehearsal": rehearsal}
                report["history"].append({"evaluation": len(report["history"]), "method": "sparse_consensus", "step": step,
                                          "kl": fresh["kl"], "objective": score, "accepted": accepted,
                                          "maximum_episode_regression": max(regressions), "train": fresh, "rehearsal": rehearsal})
                cycle.atomic_json(folder / "report.json", report)
            patch.apply(best_gains); phase("validation")
            report["final"] = {"train": best_metrics["train"], "rehearsal": best_metrics["rehearsal"], "validation": measure(validation)}
            report.update(encoder_mix=controller.encoder_mix, gains=patch.gains.tolist(),
                          group_gain_ranges=[[float(best_gains[patch.groups == i].min()), float(best_gains[patch.groups == i].max())] for i in range(len(patch.labels))],
                          changed_edges=int(np.count_nonzero(weights.data[patch.offsets] != patch.base)),
                          changed_from_champion=int(np.count_nonzero(best_gains != original)), trained_weights_sha256=array_hash(weights.data))
            report["connectome_weights_trained"] = bool(report["changed_edges"])
            patch.save(folder / "synapses.npz", ids); report["synapses_sha256"] = digest(folder / "synapses.npz", "sha256")
            evaluations = {}
            for split in ("gate", "test"):
                phase(split); evaluations[split] = {}
                for name, gains in (("champion", original), ("candidate", best_gains)):
                    patch.apply(gains)
                    evaluations[split][name] = cycle.benchmark(controller, model, splits[split], 32, folder)
                    cycle.atomic_json(folder / "benchmarks.json", evaluations)
                if split == "gate":
                    gate = cycle.paired_gate(evaluations[split]["champion"], evaluations[split]["candidate"], initial["validation"]["kl"], report["final"]["validation"]["kl"])
                    cycle.atomic_json(folder / "gate.json", gate)
            patch.apply(best_gains)
            if topology != (array_hash(weights.indices), array_hash(weights.indptr)) or outside != array_hash(weights.data[mask]):
                raise ValueError("Topology or unselected weights changed")
            if not np.array_equal(np.sign(patch.base), np.sign(weights.data[patch.offsets])):
                raise ValueError("Synaptic signs changed")
            for paths in (plan["source_sha256"], plan["rehearsal"]["sha256"]):
                if any(digest(p, "sha256") != h for p, h in paths.items()): raise ValueError("Locked source or rehearsal data changed")
            observed = {r["teacher"]["frame_sha256"] for ep in validation for r in ep} | training_hashes
            benchmark_hashes = {t["rgb_sha256"] for pair in evaluations.values() for eps in pair.values() for ep in eps for t in ep["trace"]}
            if observed & benchmark_hashes:
                gate.update(accepted=False, trajectory_overlap_veto=len(observed & benchmark_hashes))
            report.update(status="completed", gate=gate, experience=status["experience"], topology_unchanged=True,
                          signs_unchanged=True, outside_mask_unchanged=True,
                          development_gameplay={"base_graph": evaluations["gate"]["champion"], "trained_graph": evaluations["gate"]["candidate"]})
            cycle.atomic_json(folder / "report.json", report); cycle.atomic_json(folder / "gate.json", gate)
            promoted = cycle.promote(root, folder, previous, gate)
            phase("completed", status="completed", promoted=promoted, gate=gate, candidate=str(folder))
            return status
        except BaseException as error:
            cancelled = isinstance(error, (KeyboardInterrupt, InterruptedError))
            phase("cancelled" if cancelled else "failed", status="cancelled" if cancelled else "failed", error=str(error))
            raise


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
