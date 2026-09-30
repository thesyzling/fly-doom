"""Train a separate spiking student from reviewed decisions; never train during play."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from flydoom import action_memory, student
from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.human_feedback import read_feedback
from flydoom.laya_teacher import metrics
from flydoom.learning_data import read_prepared


def exclude_seen(features, seen):
    """Remove exact model-input duplicates, even if their labels disagree."""
    keys = {np.asarray(row, dtype=np.float32).tobytes() for row in seen}
    return np.array([np.asarray(row, dtype=np.float32).tobytes() not in keys for row in features])


def fit(model, feedback, replay, *, epochs, learning_rate, seed):
    """Keep epoch zero eligible and constrain regression on earlier demonstrations."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = {s: torch.tensor(feedback[s]["features"]) for s in ("train", "validation")}
    y = {s: torch.tensor(feedback[s]["actions"]) for s in ("train", "validation")}
    rx = {s: torch.tensor(replay[s]["features"]) for s in ("train", "validation")}
    ry = {s: torch.tensor(replay[s]["actions"]) for s in ("train", "validation")}
    def evaluate():
        model.eval()
        with torch.no_grad():
            return (metrics(model(x["validation"]).softmax(-1).numpy(), y["validation"].numpy()),
                    metrics(model(rx["validation"]).softmax(-1).numpy(), ry["validation"].numpy()))
    initial, original = evaluate()
    with torch.no_grad():
        anchor = model(rx["train"]).softmax(-1)
    best = {k: v.clone() for k, v in model.state_dict().items()}
    best_loss, selected, history = initial["balanced_nll"], 0, []
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    groups = [np.flatnonzero(feedback["train"]["actions"] == a) for a in range(4)]
    groups = [g for g in groups if len(g)]
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(max(1, int(np.ceil(len(x["train"]) / 32)))):
            batch = np.concatenate([rng.choice(g, max(1, 32 // len(groups)), replace=True) for g in groups])
            old = rng.integers(len(rx["train"]), size=32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(rx["train"][old])
            loss = (F.cross_entropy(model(x["train"][batch]), y["train"][batch])
                    + .5 * F.kl_div(logits.log_softmax(-1), anchor[old], reduction="batchmean")
                    + .25 * F.cross_entropy(logits, ry["train"][old]))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite feedback training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation, retained = evaluate()
        eligible = all(retained[k] >= original[k] - .05 for k in ("accuracy", "balanced_accuracy"))
        if eligible and validation["balanced_nll"] < best_loss:
            best_loss, selected = validation["balanced_nll"], epoch
            best = {k: v.clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation": validation,
                        "earlier_human_validation": retained, "eligible": eligible})
    model.load_state_dict(best)
    validation, retained = evaluate()
    return {"initial_feedback_validation": initial, "initial_earlier_human_validation": original,
            "validation": validation, "earlier_human_validation": retained,
            "selected_epoch": selected, "epochs": history}


def train(collections, checkpoint, output, *, dataset="runs/temporal-prepared-v1",
          epochs=50, learning_rate=1e-5, seed=41):
    if not 1 <= epochs <= 500 or not 0 < learning_rate <= .001 or not 0 <= seed < 2**32:
        raise ValueError("Invalid training settings")
    output, checkpoint = Path(output), Path(checkpoint)
    if output.exists():
        raise FileExistsError(output)
    parent = json.loads((checkpoint / "report.json").read_text(encoding="utf-8"))
    memory = parent.get("schema") == action_memory.SCHEMA
    if (parent.get("schema") not in {"spiking_student_v1", action_memory.SCHEMA}
            or parent.get("status") != "completed" or not parent.get("student_trained")
            or parent["student_sha256"] != digest(checkpoint / "student.safetensors", "sha256")
            or parent["implementation_sha256"] != digest(student.__file__, "sha256")):
        raise ValueError("Require a verified completed parent student")
    if memory and (parent["memory_implementation_sha256"] != digest(action_memory.__file__, "sha256")
                   or parent["memory_feature_names"] != list(action_memory.NAMES)):
        raise ValueError("Memory implementation or feature mapping changed")
    feedback, sources = read_feedback(collections, parent)
    source, human = read_prepared(dataset)
    if (digest(Path(dataset) / "manifest.json", "sha256") != parent["dataset_sha256"]
            or source["calibration_sha256"] != parent["calibration_sha256"]
            or source["output_root_ids"] != parent["output_root_ids"]):
        raise ValueError("Replay data does not match the parent")
    for split, other in (("train", "validation"), ("validation", "train")):
        if set(feedback[split]["seeds"]) & {e["seed"] for e in source["episodes"] if e["split"] in {other, "test"}}:
            raise ValueError("Feedback seeds overlap the opposite earlier demonstration split")
    replay = {s: {"features": action_memory.augment(human[s]["features"], human[s]["states"]) if memory
                  else human[s]["features"], "actions": human[s]["actions"]} for s in ("train", "validation")}
    training_inputs = np.concatenate((replay["train"]["features"], feedback["train"]["features"]))
    removed = {}
    for name, parts in (("feedback", feedback), ("earlier_human", replay)):
        keep = exclude_seen(parts["validation"]["features"], training_inputs)
        removed[name] = int((~keep).sum())
        for key in ("features", "actions"):
            parts["validation"][key] = parts["validation"][key][keep]
        if not keep.any():
            raise ValueError("No independent validation inputs remain; record different starts")
    torch.set_num_threads(4)
    model = student.SpikingReadout(parent["input_size"], parent["hidden"])
    model.load_state_dict(load_file(str(checkpoint / "student.safetensors")))
    mean, scale = model.mean.clone(), model.scale.clone()
    report = {k: parent[k] for k in ("schema", "input_size", "hidden", "dataset_sha256",
              "calibration_sha256", "output_root_ids", "implementation_sha256")}
    for key in ("memory_feature_names", "memory_implementation_sha256", "teacher_report_sha256",
                "teacher_accepted", "teacher_rejection_reasons", "experimental_teacher"):
        if key in parent:
            report[key] = parent[key]
    report.update(status="running", student_trained=False, action_memory=memory,
                  training_method="Human correction cross-entropy with previous-policy replay and human replay",
                  parent_student_sha256=parent["student_sha256"],
                  parent_report_sha256=digest(checkpoint / "report.json", "sha256"),
                  feedback_sources=sources, validation_duplicates_removed=removed,
                  feedback_train_samples=len(feedback["train"]["actions"]),
                  training_implementation_sha256=digest(__file__, "sha256"),
                  connectome_weights_trained=False, laya_needed_at_inference=False,
                  laya_used_during_training=False, game_skill_validated=False, test_evaluated=False,
                  learning_rate=learning_rate, seed=seed, normalization="Frozen parent mean and scale",
                  selection_metric="Feedback validation balanced NLL, with <=5pp earlier human accuracy regression; epoch zero eligible",
                  default_checkpoint_changed=False)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    try:
        report.update(fit(model, feedback, replay, epochs=epochs, learning_rate=learning_rate, seed=seed))
        if not torch.equal(mean, model.mean) or not torch.equal(scale, model.scale):
            raise ValueError("Parent normalization changed")
        save_file(model.state_dict(), str(output / "student.safetensors"))
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
    parser.add_argument("--collections", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/fly-student-memory-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("runs/temporal-prepared-v1"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=41)
    try:
        result = train(**vars(parser.parse_args()))
        print(f"Saved separate candidate; selected epoch {result['selected_epoch']}. Gameplay evaluation is still required.")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
