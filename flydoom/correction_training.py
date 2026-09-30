"""Query a frozen teacher on student trajectories, then train a separate readout.

Teacher disagreements are suggestions, not verified errors or human labels.
Training mixes original demonstrations with student-visited observations.
No game rewards or reserved evaluation episodes select or update weights.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import nn
import torch.nn.functional as F

from flydoom import student, action_memory
from flydoom.calibration import write_json
from flydoom.correction_data import read_collection
from flydoom.data import digest
from flydoom.laya_teacher import encode_states, load_base, metrics, probabilities, restore_head
from flydoom.student_refine import verify_inputs


def merge_collections(collections, output):
    """Combine immutable recordings without changing any episode's split."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if len(collections) < 2:
        raise ValueError("Choose at least two collections")
    sources = [(Path(path), read_collection(path)[0]) for path in collections]
    first = sources[0][1]
    keys = ("parent_student_sha256", "parent_report_sha256", "human_dataset_sha256",
            "calibration_sha256", "output_root_ids", "reserved_evaluation_seeds", "max_decisions")
    seen = set()
    for _, report in sources:
        if any(report[key] != first[key] for key in keys):
            raise ValueError("Collection provenance or evaluation reservations differ")
        seeds = {e["seed"] for e in report["episodes"]}
        if seen & seeds:
            raise ValueError("Duplicate seeds across collections")
        seen.update(seeds)
    output.mkdir(parents=True, exist_ok=False)
    merged = {**first, "status": "running", "episodes": [],
              "plan": [planned for _, report in sources for planned in report["plan"]],
              "merger_sha256": digest(__file__, "sha256"),
              "source_collections": [{"directory": str(path), "manifest_sha256": digest(path / "manifest.json", "sha256")} for path, _ in sources]}
    try:
        for directory, report in sources:
            for episode in report["episodes"]:
                copied = dict(episode)
                for key, checksum in (("file", "sha256"), ("states_file", "states_sha256")):
                    copied[key] = f"seed-{episode['seed']}" + Path(episode[key]).suffix
                    shutil.copyfile(directory / episode[key], output / copied[key])
                    if digest(output / copied[key], "sha256") != episode[checksum]:
                        raise ValueError("Copied observation checksum changed")
                merged["episodes"].append(copied)
        merged["status"] = "completed"
    except BaseException:
        merged["status"] = "interrupted_or_failed"
        raise
    finally:
        write_json(output / "manifest.json", merged)
    return merged


def validate_probabilities(array, count):
    if (array.shape != (count, 4) or not np.isfinite(array).all() or np.any(array < 0)
            or not np.allclose(array.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid teacher probabilities")
    return array.astype(np.float32)


def collection_matches(collection, source, parent, checkpoint):
    if (collection["parent_student_sha256"] != parent["student_sha256"]
            or collection["parent_report_sha256"] != digest(Path(checkpoint) / "report.json", "sha256")
            or collection["human_dataset_sha256"] != parent["dataset_sha256"]
            or collection["calibration_sha256"] != source["calibration_sha256"]
            or collection["output_root_ids"] != source["output_root_ids"]):
        raise ValueError("Collection does not belong to this parent, data, or calibration")
    human_seeds = {e["seed"] for e in source["episodes"]}
    if human_seeds & ({e["seed"] for e in collection["episodes"]} | set(collection["reserved_evaluation_seeds"])):
        raise ValueError("Human and correction/evaluation seeds overlap")


def label(collection, output, *, dataset="runs/temporal-prepared-v1",
          checkpoint="runs/fly-student-experimental-v2", teacher="runs/laya-temporal-teacher-v2",
          base="models/laya-base", human_cache=None):
    output, collection = Path(output), Path(collection)
    if output.exists():
        raise FileExistsError(output)
    source, human, parent, teacher_report = verify_inputs(dataset, checkpoint, teacher)
    recording, corrections = read_collection(collection)
    collection_matches(recording, source, parent, checkpoint)
    cached = None
    cache_hash = None
    if human_cache is not None:
        cache = Path(human_cache)
        cache_report = json.loads((cache / "report.json").read_text(encoding="utf-8"))
        cache_path = cache / "training_teacher_probabilities.npy"
        if (cache_report.get("status") != "completed" or cache_report["dataset_sha256"] != parent["dataset_sha256"]
                or cache_report["teacher_report_sha256"] != parent["teacher_report_sha256"]
                or digest(cache_path, "sha256") != cache_report["teacher_targets_sha256"]):
            raise ValueError("Human probability cache provenance mismatch")
        cached = validate_probabilities(np.load(cache_path, allow_pickle=False), len(human["train"]["actions"]))
        cache_hash = digest(cache_path, "sha256")
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "student_teacher_suggestions_v1", "status": "running",
              "teacher_accepted": bool(teacher_report.get("teacher_accepted")),
              "human_dataset_sha256": parent["dataset_sha256"],
              "teacher_report_sha256": parent["teacher_report_sha256"],
              "collection_manifest_sha256": digest(collection / "manifest.json", "sha256"),
              "human_cache_sha256": cache_hash,
              "labeler_sha256": digest(__file__, "sha256"), "files": {}, "diagnostics": {},
              "note": "Frozen teacher probabilities on the student's own causal history; disagreements are not guaranteed mistakes"}
    write_json(output / "manifest.json", report)
    try:
        agent, provenance = load_base(base)
        if provenance != teacher_report["base"]:
            raise ValueError("Teacher base provenance mismatch")
        restore_head(agent, Path(teacher) / "head.safetensors")
        for split in ("human_train", "train", "validation"):
            states = human["train"]["states"] if split == "human_train" else corrections[split]["states"]
            print(f"Teacher query: {split}, {len(states)} recorded observations", flush=True)
            targets = cached if split == "human_train" and cached is not None else probabilities(agent, encode_states(agent, states))
            targets = validate_probabilities(targets, len(states))
            filename = f"{split}.npy"
            np.save(output / filename, targets, allow_pickle=False)
            report["files"][split] = {"file": filename, "sha256": digest(output / filename, "sha256"), "samples": len(states)}
            if split != "human_train":
                applied = corrections[split]["actions"]
                suggested = targets.argmax(1)
                report["diagnostics"][split] = {"samples": len(states),
                    "disagreements": int(np.count_nonzero(applied != suggested)),
                    "student_action_counts": np.bincount(applied, minlength=4).tolist(),
                    "teacher_action_counts": np.bincount(suggested, minlength=4).tolist(),
                    "teacher_active_when_student_waited": int(np.count_nonzero((applied == 0) & (suggested != 0)))}
            write_json(output / "manifest.json", report)
        report["status"] = "completed"
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        write_json(output / "manifest.json", report)
    return report


def read_labels(labels, collection, parent, sizes):
    labels = Path(labels)
    report = json.loads((labels / "manifest.json").read_text(encoding="utf-8"))
    if (report.get("schema") != "student_teacher_suggestions_v1" or report.get("status") != "completed"
            or report["human_dataset_sha256"] != parent["dataset_sha256"]
            or report["teacher_report_sha256"] != parent["teacher_report_sha256"]
            or report["collection_manifest_sha256"] != digest(Path(collection) / "manifest.json", "sha256")):
        raise ValueError("Teacher suggestion provenance mismatch")
    targets = {}
    for split, count in sizes.items():
        artifact = report["files"][split]
        if (Path(artifact["file"]).name != artifact["file"] or artifact["samples"] != count
                or digest(labels / artifact["file"], "sha256") != artifact["sha256"]):
            raise ValueError("Teacher suggestion checksum or size mismatch")
        targets[split] = validate_probabilities(np.load(labels / artifact["file"], allow_pickle=False), count)
    return report, targets


def correction_weights(targets, applied):
    """Prioritize advice that differs from the recorded behavior, without calling it truth."""
    return np.where(targets.argmax(1) != applied, 3., 1.).astype(np.float32)


def novel_validation_rows(human_train, correction_train, correction_validation):
    """Distinct seeds can replay identical states; exclude exact training inputs."""
    def signatures(part):
        for vector, state in zip(part["features"], part["states"]):
            encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
            yield hashlib.sha256(np.asarray(vector, dtype="<f4").tobytes() + encoded).digest()
    known = set(signatures(human_train)) | set(signatures(correction_train))
    keep = np.array([i for i, signature in enumerate(signatures(correction_validation)) if signature not in known], dtype=np.int64)
    if not len(keep):
        raise ValueError("All correction validation inputs repeat training; collect different validation episodes")
    return keep


def train(collection, labels, output, *, dataset="runs/temporal-prepared-v1",
          checkpoint="runs/fly-student-experimental-v2", teacher="runs/laya-temporal-teacher-v2",
          epochs=100, seed=29, learning_rate=3e-4, memory=False):
    if not 1 <= epochs <= 1000 or not 0 <= seed < 2**32 or not 0 < learning_rate <= .01:
        raise ValueError("Invalid correction-training settings")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source, human, parent, teacher_report = verify_inputs(dataset, checkpoint, teacher)
    recording, corrections = read_collection(collection)
    collection_matches(recording, source, parent, checkpoint)
    sizes = {"human_train": len(human["train"]["actions"]), **{s: len(p["actions"]) for s, p in corrections.items()}}
    _, targets_np = read_labels(labels, collection, parent, sizes)
    validation_rows = novel_validation_rows(human["train"], corrections["train"], corrections["validation"])
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = student.SpikingReadout(parent["input_size"], parent["hidden"])
    model.load_state_dict(load_file(str(Path(checkpoint) / "student.safetensors")))
    if memory:
        model = action_memory.expand(model)
    model.eval()
    def inputs(part):
        return torch.tensor(action_memory.augment(part["features"], part["states"]) if memory else part["features"])
    human_x = {s: inputs(human[s]) for s in ("train", "validation")}
    human_y = torch.tensor(human["train"]["actions"])
    correction_x = {s: inputs(p) for s, p in corrections.items()}
    targets = {s: torch.tensor(t) for s, t in targets_np.items()}
    weights = torch.tensor(correction_weights(targets_np["train"], corrections["train"]["actions"]))

    def validation():
        with torch.no_grad():
            original = metrics(model(human_x["validation"]).softmax(-1).numpy(), human["validation"]["actions"])
            logits = model(correction_x["validation"][validation_rows])
            kl = float(F.kl_div(logits.log_softmax(-1), targets["validation"][validation_rows], reduction="batchmean"))
            agreement = float(np.mean(logits.argmax(-1).numpy() == targets_np["validation"][validation_rows].argmax(1)))
        return {"human": original, "correction_kl": kl, "teacher_agreement": agreement,
                "selection_score": kl + .25 * original["nll"]}

    initial = validation()
    floor_accuracy = initial["human"]["accuracy"] - .05
    floor_balanced = initial["human"]["balanced_accuracy"] - .05
    output.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    updates = int(np.ceil((len(human_y) + len(weights)) / 64))
    report = {"schema": action_memory.SCHEMA if memory else "spiking_student_v1", "status": "running", "student_trained": False,
              "action_memory": memory,
              "memory_feature_names": list(action_memory.NAMES) if memory else [],
              "memory_implementation_sha256": digest(action_memory.__file__, "sha256") if memory else None,
              "experimental_teacher": parent.get("experimental_teacher", False),
              "teacher_accepted": bool(teacher_report.get("teacher_accepted")),
              "teacher_rejection_reasons": teacher_report.get("rejection_reasons", []),
              "connectome_weights_trained": False, "laya_needed_at_inference": False,
              "game_skill_validated": False, "test_evaluated": False,
              "training_method": "Half human distillation (KL + 0.25 human CE), half correction KL; disagreement weight 3, agreement weight 1",
              "normalization": "Frozen parent mean and scale", "optimizer_reset": True,
              "learning_rate": learning_rate, "seed": seed, "updates_per_epoch": updates,
              "human_batch_size": 32, "correction_batch_size": 32,
              "input_size": model.input.in_features, "hidden": parent["hidden"],
              "dataset_sha256": parent["dataset_sha256"], "teacher_report_sha256": parent["teacher_report_sha256"],
              "calibration_sha256": source["calibration_sha256"], "output_root_ids": source["output_root_ids"],
              "parent_student_sha256": parent["student_sha256"],
              "parent_report_sha256": digest(Path(checkpoint) / "report.json", "sha256"),
              "collection_manifest_sha256": digest(Path(collection) / "manifest.json", "sha256"),
              "suggestions_manifest_sha256": digest(Path(labels) / "manifest.json", "sha256"),
              "reserved_evaluation_seeds": recording["reserved_evaluation_seeds"],
              "correction_validation_deduplication": {
                  "original_samples": len(corrections["validation"]["actions"]),
                  "retained_rows": validation_rows.tolist(),
                  "retained_samples": len(validation_rows),
                  "rule": "Exclude exact feature-vector plus causal teacher-state duplicates of either training source"},
              "implementation_sha256": digest(student.__file__, "sha256"),
              "training_implementation_sha256": digest(__file__, "sha256"),
              "selection_rule": "Minimize correction-validation KL + 0.25 human-validation NLL; parent eligible at epoch zero; both human accuracies must stay within five points of parent",
              "human_accuracy_floor": floor_accuracy, "human_balanced_accuracy_floor": floor_balanced,
              "initial_validation": initial, "selected_epoch": 0, "epochs": []}
    best = initial["selection_score"]
    save_file(model.state_dict(), str(output / "student.safetensors"))
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for _ in range(updates):
                hi = rng.integers(len(human_y), size=32)
                ci = rng.integers(len(weights), size=32)
                optimizer.zero_grad(set_to_none=True)
                human_logits = model(human_x["train"][hi])
                human_loss = F.kl_div(human_logits.log_softmax(-1), targets["human_train"][hi], reduction="batchmean") + .25 * F.cross_entropy(human_logits, human_y[hi])
                correction_logits = model(correction_x["train"][ci])
                per_row = F.kl_div(correction_logits.log_softmax(-1), targets["train"][ci], reduction="none").sum(-1)
                correction_loss = (per_row * weights[ci]).sum() / weights[ci].sum()
                loss = .5 * human_loss + .5 * correction_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite correction loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            measured = validation()
            eligible = (measured["human"]["accuracy"] >= floor_accuracy
                        and measured["human"]["balanced_accuracy"] >= floor_balanced)
            report["epochs"].append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": measured, "eligible": eligible})
            if eligible and measured["selection_score"] < best:
                best = measured["selection_score"]
                report["selected_epoch"] = epoch
                save_file(model.state_dict(), str(output / "student.safetensors"))
            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch}: correction KL={measured['correction_kl']:.3f}, human accuracy={measured['human']['accuracy']:.3f}, eligible={eligible}", flush=True)
                write_json(output / "report.json", report)
        model.load_state_dict(load_file(str(output / "student.safetensors")))
        model.eval()
        report["validation"] = validation()
        with torch.no_grad():
            zero = human_x["validation"].clone()
            zero[:, :parent["input_size"] - 4] = 0
            report["human_validation_zero_neural_features"] = metrics(model(zero).softmax(-1).numpy(), human["validation"]["actions"])
            if memory:
                zero_memory = human_x["validation"].clone()
                zero_memory[:, parent["input_size"] - 4:-4] = 0
                report["human_validation_zero_memory"] = metrics(model(zero_memory).softmax(-1).numpy(), human["validation"]["actions"])
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
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("merge", help="Combine compatible collections while preserving each episode split")
    p.add_argument("--collections", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ("label", "train"):
        p = sub.add_parser(name)
        p.add_argument("--collection", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--dataset", type=Path, default=Path("runs/temporal-prepared-v1"))
        p.add_argument("--checkpoint", type=Path, default=Path("runs/fly-student-experimental-v2"))
        p.add_argument("--teacher", type=Path, default=Path("runs/laya-temporal-teacher-v2"))
        if name == "label":
            p.add_argument("--base", type=Path, default=Path("models/laya-base"))
            p.add_argument("--human-cache", type=Path)
        else:
            p.add_argument("--labels", type=Path, required=True)
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--seed", type=int, default=29)
            p.add_argument("--learning-rate", type=float, default=3e-4)
            p.add_argument("--memory", action="store_true", help="Add causal past-action and timing inputs to the same 64 cells")
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        {"label": label, "train": train, "merge": merge_collections}[command](**args)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
