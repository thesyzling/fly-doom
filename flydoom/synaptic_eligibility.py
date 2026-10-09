"""Per-edge local eligibility gradients, verified by exact full-graph replay.

The proposal differentiates direct subthreshold synaptic effects and the frozen
surrogate-spike decoder. Presynaptic spike times, reset events and indirect
recurrent feedback are held fixed for this local derivative. This is not full
BPTT. Acceptance always measures the actual unmodified hard-spike simulator.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.movement_core import Memory, ACTIONS, vector
from flydoom.synaptic import (PlasticEdges, STUDENT, CALIBRATION, array_hash,
                             examples, gameplay, load_system, replay, restore as restore_grouped)

SCHEMA = "fly_synaptic_edges_v1"


class EdgePatch(PlasticEdges):
    def __init__(self, weights, offsets, groups, labels):
        super().__init__(weights, offsets, groups, labels)
        self.edge_gains = np.ones(len(self.offsets), np.float32)

    def apply(self, gains):
        gains = np.asarray(gains, np.float32)
        if gains.shape != self.base.shape or not np.isfinite(gains).all() or np.any(np.abs(gains-1) > .2500001):
            raise ValueError("Per-edge gains must be finite and within [0.75, 1.25]")
        self.edge_gains = gains.copy(); self.weights.data[self.offsets] = self.base * gains
        self.gains = np.array([gains[self.groups==i].mean() for i in range(len(self.labels))], np.float32)

    def save(self, path, ids):
        targets = np.searchsorted(self.weights.indptr, self.offsets, side="right") - 1
        np.savez_compressed(path, offsets=self.offsets, groups=self.groups, edge_gains=self.edge_gains,
                            base=self.base, trained=self.weights.data[self.offsets],
                            source_ids=ids[self.weights.indices[self.offsets]], target_ids=ids[targets])


def restore(controller, ids, folder, identity):
    folder = Path(folder); report = json.loads((folder / "report.json").read_text())
    if report.get("schema") != SCHEMA: return restore_grouped(controller, ids, folder, identity)
    weights = controller.network.weights
    if (report.get("status") != "completed" or report["student_sha256"] != identity["student_sha256"]
            or report["calibration_sha256"] != digest(CALIBRATION, "sha256")
            or report["synapses_sha256"] != digest(folder / "synapses.npz", "sha256")
            or report["base_weights_sha256"] != array_hash(weights.data)):
        raise ValueError("Per-edge checkpoint identity mismatch")
    with np.load(folder / "synapses.npz", allow_pickle=False) as d:
        patch = EdgePatch(weights, d["offsets"], d["groups"], report["group_labels"])
        targets = np.searchsorted(weights.indptr, patch.offsets, side="right") - 1
        if (not np.array_equal(patch.base, d["base"])
                or not np.array_equal(ids[weights.indices[patch.offsets]], d["source_ids"])
                or not np.array_equal(ids[targets], d["target_ids"])): raise ValueError("Edge identity mismatch")
        patch.apply(d["edge_gains"])
        if not np.array_equal(weights.data[patch.offsets], d["trained"]): raise ValueError("Changed edge reconstruction")
    if array_hash(weights.data) != report["trained_weights_sha256"]: raise ValueError("Changed graph identity")
    return patch, report


class Eligibility:
    """Direct dI/dW and dV/dW for each selected existing edge."""
    def __init__(self, network, offsets):
        self.network = network
        self.source = network.weights.indices[offsets]
        self.target = np.searchsorted(network.weights.indptr, offsets, side="right") - 1
        self.current = np.zeros(len(offsets), np.float32)
        self.voltage = self.current.copy()

    def advance(self, arrived, active, spikes):
        net = self.network
        self.current += arrived[self.source]
        self.voltage = np.where(active[self.target], self.voltage * net.membrane_decay + self.current * net.synapse_integral, 0).astype(np.float32)
        self.voltage[spikes[self.target]] = 0
        self.current *= net.synapse_decay


def gradient(controller, model, patch, episodes):
    net = controller.network; original = net.step
    outputs = np.sort(np.concatenate(controller.mapping.output_groups))
    post = np.searchsorted(net.weights.indptr, patch.offsets, side="right") - 1
    output_position = np.searchsorted(outputs, post)
    total = np.zeros(len(patch.offsets), np.float64); samples = 0
    try:
        for episode in episodes:
            controller.reset(); memory = Memory(); trace = Eligibility(net, patch.offsets)
            def step(drive, *, connected=True):
                arrived = net.pending[net.tick % len(net.pending)].copy()
                active = net.refractory == 0
                spikes = original(drive, connected=connected)
                trace.advance(arrived, active, spikes)
                return spikes
            net.step = step
            for row in episode:
                decision = controller.decide(row["rgb"])
                if decision["voltage_min_mv"] < -90: raise ValueError("Eligibility rollout left the voltage bound")
                features = torch.tensor(vector(controller, memory), requires_grad=True)
                target = torch.tensor([row["teacher"]["probabilities"]], dtype=torch.float32)
                loss = F.kl_div(model(features[None]).log_softmax(1), target, reduction="batchmean")
                derivative = torch.autograd.grad(loss, features)[0].detach().numpy()[:len(outputs)] / 20
                total += derivative[output_position] * trace.voltage * patch.base
                samples += 1; memory.advance(ACTIONS.index(row["action"]))
    finally:
        net.step = original
    total /= samples
    if not np.isfinite(total).all(): raise ValueError("Nonfinite eligibility gradient")
    return total.astype(np.float32)


def train(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(2)
    ids, controller, _, model, rows, identity = load_system()
    selected = PlasticEdges.select(controller, rows)
    patch = EdgePatch(controller.network.weights, selected.offsets, selected.groups, selected.labels)
    train_data, val_data = examples("train", 8, 2), examples("validation", 4, 2)
    files = [Path(__file__), Path("flydoom/synaptic.py"), Path("flydoom/simulation.py"), Path("flydoom/bridge.py"),
             Path("flydoom/movement_core.py"), Path("flydoom/student.py"), CALIBRATION,
             STUDENT / "student.safetensors", STUDENT / "report.json", STUDENT / "train/trace.json", STUDENT / "validation/trace.json"]
    plan = {"schema": SCHEMA, "optimizer": "One direct eligibility proposal, exact hard-spike replay line search",
            "derivative_scope": "Direct voltage path only; frozen presynaptic spikes/reset schedule; indirect feedback and spike-rate derivatives omitted",
            "steps": [-.01, -.003, -.001, -.0001, .0001, .001, .003, .01],
            "normalization": "Divide by 95th percentile nonzero absolute gradient and clip to [-1,1]",
            "regularization": .002, "selection": "Penalized training KL only; validation not used to select",
            "training_scenes": 8, "validation_scenes": 4, "prefix_decisions": 2,
            "development_game_seeds": [74000,74001], "development_decision_limit": 12,
            "scope": "Small per-edge engineering experiment on already-seen scene prefixes; not full BPTT or a generalization benchmark.",
            "artifact_sha256": {str(p): digest(p, "sha256") for p in files}}
    write_json(output / "plan.json", plan)
    weights = controller.network.weights
    outside = np.ones(weights.nnz, bool); outside[patch.offsets] = False
    untouched = array_hash(weights.data[outside]); topology = (array_hash(weights.indices), array_hash(weights.indptr))
    report = {"schema": SCHEMA, "status": "running", "student_sha256": identity["student_sha256"],
              "calibration_sha256": digest(CALIBRATION, "sha256"), "plan_sha256": digest(output / "plan.json", "sha256"),
              "base_weights_sha256": array_hash(weights.data), "group_labels": patch.labels,
              "plastic_edges": len(patch.offsets), "total_edges": weights.nnz, "group_edge_counts": np.bincount(patch.groups).tolist(),
              "history": [], "scope": plan["scope"], "readout_trained": False, "teacher_trained": False,
              "biology_validated": False, "parameterization": "Individual edge gains; displayed group gains are means"}
    write_json(output / "report.json", report)
    try:
        report["initial"] = {"train": replay(controller, model, train_data), "validation": replay(controller, model, val_data)}
        derivative = gradient(controller, model, patch, train_data)
        nonzero = np.abs(derivative[derivative!=0])
        if not len(nonzero): raise ValueError("No nonzero biological-edge learning signal")
        direction = np.clip(-derivative / max(np.quantile(nonzero,.95),1e-12), -1, 1)
        np.savez_compressed(output / "eligibility.npz", gradient=derivative, direction=direction)
        report["gradient_nonzero_edges"] = len(nonzero)
        best = report["initial"]["train"]["kl"]; gains = np.ones(len(patch.offsets), np.float32)
        report["history"].append({"evaluation":0, "kl":best, "objective":best, "accepted":True, "gains":[1.]*len(patch.labels), "step":0.})
        print(f"Eligibility: {len(nonzero)} edges have a nonzero learning signal; initial KL {best:.7f}", flush=True)
        for step_size in plan["steps"]:
            candidate = 1 + step_size * direction; patch.apply(candidate)
            try:
                measured = replay(controller, model, train_data)
                score = measured["kl"] + .002 * float(np.mean((candidate-1)**2))
                accepted = score < best
                if accepted: best, gains = score, candidate.copy()
                event = {"evaluation":len(report["history"]), "kl":measured["kl"], "objective":score,
                         "accepted":accepted, "step":step_size, "gains":patch.gains.tolist()}
            except ValueError as error:
                event = {"evaluation":len(report["history"]), "rejected":str(error), "accepted":False, "step":step_size, "gains":patch.gains.tolist()}
            report["history"].append(event); write_json(output / "report.json", report)
            print(f"Eligibility candidate {event['evaluation']}/8: {event.get('kl',event.get('rejected'))}, accepted {event['accepted']}", flush=True)
        patch.apply(gains)
        report["final"] = {"train":replay(controller, model, train_data), "validation":replay(controller, model, val_data)}
        report["gains"] = patch.gains.tolist()
        report["group_gain_ranges"] = [[float(gains[patch.groups==i].min()),float(gains[patch.groups==i].max())] for i in range(len(patch.labels))]
        report["changed_edges"] = int(np.count_nonzero(weights.data[patch.offsets]!=patch.base))
        report["connectome_weights_trained"] = bool(report["changed_edges"])
        report["trained_weights_sha256"] = array_hash(weights.data)
        patch.save(output / "synapses.npz",ids); report["synapses_sha256"] = digest(output / "synapses.npz","sha256")
        report["development_gameplay"] = {}
        for name, values in (("base_graph",np.ones(len(gains))), ("trained_graph",gains)):
            patch.apply(values); report["development_gameplay"][name] = gameplay(controller,model,plan["development_game_seeds"])
            print(f"Eligibility development gameplay: {name}",flush=True)
        patch.apply(gains)
        if (untouched!=array_hash(weights.data[outside]) or topology!=(array_hash(weights.indices),array_hash(weights.indptr))
                or not np.array_equal(np.sign(patch.base),np.sign(weights.data[patch.offsets]))): raise ValueError("Graph constraint violation")
        if any(digest(p,"sha256")!=h for p,h in plan["artifact_sha256"].items()): raise ValueError("Locked source changed")
        report.update(status="completed", topology_unchanged=True, signs_unchanged=True, outside_mask_unchanged=True, source_student_unchanged=True)
    except BaseException:
        report["status"]="interrupted_or_failed";raise
    finally:
        write_json(output / "report.json",report)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    result=train(parser.parse_args().output);print({k:result[k] for k in ("status","changed_edges","gains")})
