"""Balance recorded human actions while refining an existing spiking readout.

This remains supervised distillation, not game-reward learning. The original
student implementation, normalization, fly graph, and teacher remain unchanged.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import nn
import torch.nn.functional as F

from flydoom import student
from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.learning_data import ACTIONS, read_prepared
from flydoom.laya_teacher import encode_states, load_base, metrics, probabilities, restore_head


def balanced_indices(actions, rng, per_class=8):
    """Sample each demonstrated action equally, using training labels only."""
    groups = [np.flatnonzero(actions == i) for i in range(len(ACTIONS))]
    if any(not len(group) for group in groups):
        raise ValueError("All four human actions must be present in training")
    indices = np.concatenate([rng.choice(group, per_class, replace=len(group) < per_class) for group in groups])
    return rng.permutation(indices)


def verify_inputs(dataset, checkpoint, teacher):
    source, parts = read_prepared(dataset)
    parent = json.loads((Path(checkpoint) / "report.json").read_text(encoding="utf-8"))
    teacher_report = json.loads((Path(teacher) / "report.json").read_text(encoding="utf-8"))
    if (parent.get("schema") != "spiking_student_v1" or parent.get("status") != "completed"
            or not parent.get("student_trained") or teacher_report.get("status") != "completed"):
        raise ValueError("Require completed teacher and student checkpoints")
    dataset_hash = digest(Path(dataset) / "manifest.json", "sha256")
    checks = [parent["dataset_sha256"] == dataset_hash, teacher_report["dataset_sha256"] == dataset_hash,
              parent["teacher_report_sha256"] == digest(Path(teacher) / "report.json", "sha256"),
              parent["implementation_sha256"] == digest(student.__file__, "sha256"),
              parent["student_sha256"] == digest(Path(checkpoint) / "student.safetensors", "sha256"),
              teacher_report["head_sha256"] == digest(Path(teacher) / "head.safetensors", "sha256"),
              parent["calibration_sha256"] == source["calibration_sha256"],
              parent["output_root_ids"] == source["output_root_ids"],
              parent["input_size"] == parts["train"]["features"].shape[1]]
    if not all(checks):
        raise ValueError("Parent, teacher, data, or implementation provenance mismatch")
    if not teacher_report.get("teacher_accepted") and not parent.get("experimental_teacher"):
        raise ValueError("A rejected teacher requires an already explicit experimental parent")
    balanced_indices(parts["train"]["actions"], np.random.default_rng(0))
    return source, parts, parent, teacher_report


def train(dataset, checkpoint, teacher, output, *, base="models/laya-base", epochs=100, seed=17, learning_rate=3e-4):
    if not 1 <= epochs <= 1000 or not 0 < learning_rate <= .01 or not 0 <= seed < 2**32:
        raise ValueError("Invalid refinement settings")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source, parts, parent, teacher_report = verify_inputs(dataset, checkpoint, teacher)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = student.SpikingReadout(parent["input_size"], parent["hidden"])
    model.load_state_dict(load_file(str(Path(checkpoint) / "student.safetensors")))
    model.eval()
    x = {split: torch.tensor(parts[split]["features"]) for split in ("train", "validation")}
    y = torch.tensor(parts["train"]["actions"])
    with torch.no_grad():
        initial = metrics(model(x["validation"]).softmax(-1).numpy(), parts["validation"]["actions"])
    print("Obtaining teacher probabilities on training observations only...", flush=True)
    agent, provenance = load_base(base)
    if provenance != teacher_report["base"]:
        raise ValueError("Teacher base provenance mismatch")
    restore_head(agent, Path(teacher) / "head.safetensors")
    targets = probabilities(agent, encode_states(agent, parts["train"]["states"]))
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "training_teacher_probabilities.npy", targets, allow_pickle=False)
    targets = torch.tensor(targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    updates = int(np.ceil(len(y) / 32))
    report = {"schema": "spiking_student_v1", "status": "running", "student_trained": False,
              "experimental_teacher": parent.get("experimental_teacher", False),
              "teacher_accepted": bool(teacher_report.get("teacher_accepted")),
              "teacher_rejection_reasons": teacher_report.get("rejection_reasons", []),
              "connectome_weights_trained": False, "laya_needed_at_inference": False,
              "game_skill_validated": False, "test_evaluated": False,
              "training_method": "Warm-start distillation: 8 human examples per action per update; KL plus 0.25 human cross-entropy",
              "selection_metric": "validation balanced_nll; unchanged parent eligible as epoch zero",
              "normalization": "Frozen parent mean and scale", "optimizer_reset": True,
              "learning_rate": learning_rate, "updates_per_epoch": updates, "batch_size": 32, "seed": seed,
              "input_size": parent["input_size"], "hidden": parent["hidden"],
              "dataset_sha256": parent["dataset_sha256"],
              "teacher_report_sha256": parent["teacher_report_sha256"],
              "calibration_sha256": source["calibration_sha256"], "output_root_ids": source["output_root_ids"],
              "parent_report_sha256": digest(Path(checkpoint) / "report.json", "sha256"),
              "parent_student_sha256": parent["student_sha256"],
              "teacher_targets_sha256": digest(output / "training_teacher_probabilities.npy", "sha256"),
              "implementation_sha256": digest(student.__file__, "sha256"),
              "training_implementation_sha256": digest(__file__, "sha256"),
              "initial_validation": initial, "selected_epoch": 0, "epochs": []}
    best = initial["balanced_nll"]
    save_file(model.state_dict(), str(output / "student.safetensors"))
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for _ in range(updates):
                indices = balanced_indices(parts["train"]["actions"], rng)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x["train"][indices])
                loss = F.kl_div(logits.log_softmax(-1), targets[indices], reduction="batchmean") + .25 * F.cross_entropy(logits, y[indices])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite refinement loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                validation = metrics(model(x["validation"]).softmax(-1).numpy(), parts["validation"]["actions"])
            report["epochs"].append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation})
            if validation["balanced_nll"] < best:
                best = validation["balanced_nll"]
                report["selected_epoch"] = epoch
                save_file(model.state_dict(), str(output / "student.safetensors"))
            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch}: accuracy={validation['accuracy']:.3f}, balanced={validation['balanced_accuracy']:.3f}", flush=True)
                write_json(output / "report.json", report)
        model.load_state_dict(load_file(str(output / "student.safetensors")))
        model.eval()
        with torch.no_grad():
            validation_x = x["validation"]
            report["validation"] = metrics(model(validation_x).softmax(-1).numpy(), parts["validation"]["actions"])
            zero = validation_x.clone()
            zero[:, :-4] = 0
            report["validation_zero_neural_features"] = metrics(model(zero).softmax(-1).numpy(), parts["validation"]["actions"])
            shuffled = validation_x.clone()
            order = np.random.default_rng(seed + 1).permutation(len(shuffled))
            shuffled[:, :-4] = shuffled[order, :-4].clone()
            report["validation_shuffled_neural_features"] = metrics(model(shuffled).softmax(-1).numpy(), parts["validation"]["actions"])
        report.update(status="completed", student_trained=True,
                      student_sha256=digest(output / "student.safetensors", "sha256"))
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("runs/temporal-prepared-v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/fly-student-experimental-v2"))
    parser.add_argument("--teacher", type=Path, default=Path("runs/laya-temporal-teacher-v2"))
    parser.add_argument("--base", type=Path, default=Path("models/laya-base"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = vars(parser.parse_args())
    try:
        train(**args)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
