"""Local Laya download, supervised decision-head adaptation, and teacher gates.

This adapter uses labelled imitation with cross-entropy, not upstream RLCD or
game-reward reinforcement learning. The pretrained text encoder stays frozen.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.learning_data import ACTIONS, QUESTION, read_prepared
from flydoom.learning_quality import coverage, print_quality


def download(output):
    from huggingface_hub import HfApi, snapshot_download
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    repo = "convaiinnovations/laya"
    revision = HfApi().model_info(repo).sha
    print(f"Downloading Laya English checkpoint at {revision}", flush=True)
    snapshot_download(repo, revision=revision, local_dir=output,
                      allow_patterns=["rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*"])
    hashes = {str(p.relative_to(output)).replace("\\", "/"): digest(p, "sha256")
              for p in output.rglob("*") if p.is_file() and ".cache" not in p.parts}
    write_json(output / "download.json", {"repo": repo, "revision": revision, "sha256": hashes})
    print(f"Pinned model saved to {output}", flush=True)


def load_base(path, device="auto"):
    import laya
    import torch
    path = Path(path)
    provenance = json.loads((path / "download.json").read_text(encoding="utf-8"))
    for name, expected in provenance["sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or digest(path / name, "sha256") != expected:
            raise ValueError("Laya model checksum mismatch")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = laya.load(str(path.resolve()), device=device)
    if str(agent.device) != device:
        raise RuntimeError(f"Requested {device}, but Laya loaded on {agent.device}")
    agent.model.eval()
    # Keep training and inference arithmetic consistent and explicit.
    agent.model.float()
    return agent, provenance


def encode_states(agent, states):
    from laya.common import build_sequence
    question = {"t": "choice", "ins": QUESTION["action"]["instructions"],
                "crit": QUESTION["action"]["criteria"]}
    items = []
    for state in states:
        # One extra token detects overflow instead of silently discarding history.
        ids, markers = build_sequence(agent.tok, state, question, max_len=513, head_max_len=160)
        if len(ids) > 512:
            raise ValueError("Teacher observation exceeds 512 tokens; shorten the representation rather than truncate it")
        if len(markers) != len(ACTIONS):
            raise ValueError("Teacher input lost an action option during tokenization")
        items.append({"ids": ids, "markers": markers, "qtype": 0})
    return items


def batch_logits(agent, items):
    from laya.common import collate_items
    batch = collate_items([[item] for item in items], agent.tok.pad_token_id)
    keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    logits, _ = agent.model(**{key: batch[key].to(agent.device) for key in keys}, detach_encoder=True)
    return logits


def probabilities(agent, items, batch_size=2, temperature=1.0):
    import torch
    agent.model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            output.append(torch.softmax(batch_logits(agent, items[start:start + batch_size]) / temperature, -1).cpu().numpy())
    return np.concatenate(output)


def metrics(probs, labels):
    predicted = probs.argmax(axis=1)
    recalls = [float(np.mean(predicted[labels == c] == c)) for c in range(4) if np.any(labels == c)]
    losses = -np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-8))
    confusion = np.zeros((4, 4), dtype=int)
    np.add.at(confusion, (labels, predicted), 1)
    return {"samples": len(labels), "accuracy": float(np.mean(predicted == labels)),
            "balanced_accuracy": float(np.mean(recalls)),
            "nll": float(losses.mean()),
            "balanced_nll": float(np.mean([losses[labels == c].mean() for c in range(4) if np.any(labels == c)])),
            "class_counts": np.bincount(labels, minlength=4).tolist(),
            "predicted_counts": np.bincount(predicted, minlength=4).tolist(),
            "confusion_true_rows_predicted_columns": confusion.tolist()}


def class_weights(labels):
    counts = np.bincount(labels, minlength=4)
    if np.any(counts == 0):
        raise ValueError("Teacher training needs examples of every action")
    return (len(labels) / (4 * counts)).astype(np.float32)


def rejection_reasons(validation, baseline_accuracy, shuffled_accuracy):
    reasons = []
    if validation["samples"] < 30:
        reasons.append("Need at least 30 validation decisions")
    for i in range(1, 4):
        if validation["class_counts"][i] < 5:
            reasons.append(f"Need at least five validation examples of {ACTIONS[i]}")
    if validation["accuracy"] < baseline_accuracy + 0.05:
        reasons.append(f"Accuracy {validation['accuracy']:.3f} must exceed baseline {baseline_accuracy:.3f} by 0.05")
    if validation["balanced_accuracy"] < 0.5:
        reasons.append(f"Balanced accuracy {validation['balanced_accuracy']:.3f} is below 0.500")
    if validation["accuracy"] < shuffled_accuracy + 0.05:
        reasons.append(f"Image advantage {validation['accuracy'] - shuffled_accuracy:.3f} is below 0.050")
    return reasons


def teacher_gate(validation, baseline_accuracy, shuffled_accuracy):
    # This gate is a pilot screen, not a statistical significance claim.
    return not rejection_reasons(validation, baseline_accuracy, shuffled_accuracy)


def shuffle_images(states, permutation):
    # Replace every visual field, including color descriptions, while preserving
    # the recipient's previous action. Never shuffle only the brightness grid.
    from flydoom.temporal_data import NONVISUAL_KEYS
    return [{**states[j], **{key: states[i][key] for key in NONVISUAL_KEYS if key in states[i]}}
            for i, j in enumerate(permutation)]


def balanced_batches(labels, rng, update_batch_size=16):
    """Every optimizer update sees all classes before gradients are clipped."""
    if update_batch_size < 4 or update_batch_size % 4:
        raise ValueError("Optimizer batch size must be a positive multiple of four")
    per_class = update_batch_size // 4
    steps = int(np.ceil(len(labels) / update_batch_size))
    pools = []
    for action in range(4):
        pool = np.flatnonzero(labels == action)
        if len(pool) == 0:
            raise ValueError("Balanced training requires every action")
        count = steps * per_class
        pools.append(np.concatenate([rng.permutation(pool) for _ in range(int(np.ceil(count / len(pool))))])[:count])
    for step in range(steps):
        yield rng.permutation(np.concatenate([pool[step * per_class:(step + 1) * per_class] for pool in pools]))


def train(dataset, base, output, *, epochs=15, batch_size=2, seed=7, device="auto",
          update_batch_size=16, learning_rate=1e-4, evaluate_test=True, initial_teacher=None):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from flydoom.laya_features import cache_features, cached_logits, cached_probabilities
    if (not 1 <= epochs <= 100 or not 1 <= batch_size <= 16 or not 4 <= update_batch_size <= 64
            or update_batch_size % 4 or not 0 < learning_rate <= 1e-3):
        raise ValueError("Require 1..100 epochs and 1..16 examples per batch")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    source, splits = read_prepared(dataset)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    quality = coverage({key: np.bincount(splits[key]["actions"], minlength=4) for key in ("train", "validation")})
    if not quality["ready"]:
        report = {"schema": "laya_teacher_v1", "status": "insufficient_demonstration_coverage",
                  "teacher_accepted": False, "quality": quality, "rejection_reasons": quality["reasons"]}
        write_json(output / "report.json", report)
        print_quality(quality)
        return report
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    parent = None
    if initial_teacher is not None:
        initial_teacher = Path(initial_teacher)
        parent = json.loads((initial_teacher / "report.json").read_text(encoding="utf-8"))
        if (parent.get("schema") != "laya_teacher_v1" or parent.get("status") != "completed"
                or parent["dataset_sha256"] != digest(Path(dataset) / "manifest.json", "sha256")
                or parent["head_sha256"] != digest(initial_teacher / "head.safetensors", "sha256")):
            raise ValueError("Warm start requires a complete, verified teacher on the identical dataset")
    agent, provenance = load_base(base, device)
    if parent is not None:
        if parent["base"] != provenance:
            raise ValueError("Warm-start teacher belongs to a different Laya base")
        restore_head(agent, initial_teacher / "head.safetensors")
    for name, parameter in agent.model.named_parameters():
        parameter.requires_grad_(name.startswith(("head.", "scorer.", "type_emb.")))
    parameters = [p for p in agent.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    items = {key: encode_states(agent, part["states"]) for key, part in splits.items() if key != "test"}
    features = {key: cache_features(agent, encoded, batch_size) for key, encoded in items.items()}
    train_part, validation = splits["train"], splits["validation"]
    majority = int(np.bincount(train_part["actions"], minlength=4).argmax())
    baseline = max(float(np.mean(validation["actions"] == majority)),
                   float(np.mean(validation["actions"] == validation["previous"])))
    timing_accuracy = None
    if source.get("observation_format") == "three_past_observations_v1":
        from flydoom.temporal_data import timing_baseline
        predictions = timing_baseline(train_part, validation)
        timing_accuracy = float(np.mean(predictions == validation["actions"]))
        baseline = max(baseline, timing_accuracy)
    initial = cached_probabilities(agent, features["validation"], batch_size)
    direct = probabilities(agent, items["validation"][:batch_size], batch_size)
    cache_error = float(np.max(np.abs(direct - initial[:batch_size])))
    if not np.allclose(direct, initial[:batch_size], atol=1e-5, rtol=1e-4):
        raise ValueError("Cached encoder path differs from the original Laya model")
    report = {"schema": "laya_teacher_v1", "status": "running", "teacher_accepted": False,
              "training_method": "balanced optimizer batches; microbatch gradient accumulation before clipping; frozen encoder cache",
              "sampling_uses_train_only": True, "selection_metric": "validation balanced_nll",
              "update_batch_size": update_batch_size, "microbatch_size": batch_size, "learning_rate": learning_rate,
              "cached_probability_max_abs_error": cache_error, "encoder_cached": True,
              "base": provenance, "dataset_sha256": digest(Path(dataset) / "manifest.json", "sha256"),
              "actions": ACTIONS, "question": QUESTION, "seed": seed,
              "initial_validation": metrics(initial, validation["actions"]),
              "validation_baseline_accuracy": baseline, "epochs": [], "temperature": 1.0,
              "probabilities_calibrated": False, "calibration_sha256": source["calibration_sha256"],
              "observation_format": source.get("observation_format", "single_frame"),
              "test_note": "Test never selects checkpoints; repeated experiments on this dataset require fresh final evaluation"}
    if timing_accuracy is not None:
        report["timing_only_validation_accuracy"] = timing_accuracy
    if parent is not None:
        report["warm_start"] = {"parent_report_sha256": digest(initial_teacher / "report.json", "sha256"),
            "parent_head_sha256": parent["head_sha256"], "parent_selected_epoch": parent["selected_epoch"],
            "method": "Selected head weights retained; AdamW optimizer state reset"}
    # Retain epoch zero so a training run cannot silently replace a better base.
    best = report["initial_validation"]["balanced_nll"]
    report["selected_epoch"] = 0
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in agent.model.state_dict().items()
               if name.startswith(("head.", "scorer.", "type_emb."))}, str(output / "head.safetensors"))
    try:
        for epoch in range(epochs):
            agent.model.train()
            agent.model.encoder.eval()
            losses = []
            gradient_norms = []
            updates = list(balanced_batches(train_part["actions"], rng, update_batch_size))
            for update, chosen in enumerate(updates):
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for offset in range(0, len(chosen), batch_size):
                    indices = chosen[offset:offset + batch_size]
                    logits = cached_logits(agent, [features["train"][i] for i in indices])
                    labels = torch.tensor(train_part["actions"][indices], device=agent.device)
                    loss = F.cross_entropy(logits, labels, reduction="sum") / len(chosen)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite teacher loss")
                    loss.backward()
                    total_loss += float(loss.detach())
                gradient_norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, 1.0)))
                optimizer.step()
                losses.append(total_loss)
                if (update + 1) % 16 == 0:
                    print(f"Teacher epoch {epoch + 1}: {update + 1}/{len(updates)} balanced updates", flush=True)
            val = metrics(cached_probabilities(agent, features["validation"], batch_size), validation["actions"])
            report["epochs"].append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                "mean_gradient_norm_before_clip": float(np.mean(gradient_norms)), "validation": val})
            print(f"Teacher epoch {epoch + 1}: accuracy={val['accuracy']:.3f}, balanced={val['balanced_accuracy']:.3f}", flush=True)
            if val["balanced_nll"] < best:
                best = val["balanced_nll"]
                report["selected_epoch"] = epoch + 1
                save_file({name: tensor.detach().cpu().contiguous() for name, tensor in agent.model.state_dict().items()
                           if name.startswith(("head.", "scorer.", "type_emb."))}, str(output / "head.safetensors"))
            write_json(output / "report.json", report)
        restore_head(agent, output / "head.safetensors")
        val_probs = cached_probabilities(agent, features["validation"], batch_size)
        val = metrics(val_probs, validation["actions"])
        # Keep previous-action hints fixed; shuffle only image content to expose shortcut policies.
        permutation = rng.permutation(len(validation["states"]))
        shuffled_states = shuffle_images(validation["states"], permutation)
        shuffled = metrics(probabilities(agent, encode_states(agent, shuffled_states), batch_size), validation["actions"])
        report["validation"] = val
        report["train"] = metrics(cached_probabilities(agent, features["train"], batch_size), train_part["actions"])
        report["shuffled_image_validation"] = shuffled
        report["teacher_accepted"] = teacher_gate(val, baseline, shuffled["accuracy"])
        report["rejection_reasons"] = rejection_reasons(val, baseline, shuffled["accuracy"])
        # Test is evaluated once after selection; its labels never select a checkpoint or a gate.
        report["test_evaluated"] = evaluate_test
        if evaluate_test:
            report["test"] = metrics(probabilities(agent, encode_states(agent, splits["test"]["states"]), batch_size), splits["test"]["actions"])
        report["head_sha256"] = digest(output / "head.safetensors", "sha256")
        report["status"] = "completed"
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        write_json(output / "report.json", report)
    print(f"Teacher accepted for distillation: {report['teacher_accepted']} | {output}", flush=True)
    return report


def restore_head(agent, path):
    from safetensors.torch import load_file
    tensors = load_file(str(path), device=str(agent.device))
    allowed = {name for name in agent.model.state_dict() if name.startswith(("head.", "scorer.", "type_emb."))}
    if set(tensors) != allowed:
        raise ValueError("Teacher head does not match the pinned base architecture")
    agent.model.load_state_dict(tensors, strict=False)
    agent.model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download_parser = sub.add_parser("download")
    download_parser.add_argument("--output", type=Path, default=Path("models/laya-base"))
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
