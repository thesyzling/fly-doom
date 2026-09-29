"""Training-only, balanced memorization probe; never produces a gameplay model."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.learning_data import read_prepared
from flydoom.laya_teacher import load_base, encode_states, batch_logits, class_weights, metrics
from flydoom.laya_features import cache_features, cached_logits, cached_probabilities


def unique_balanced_indices(states, labels, per_class=4, seed=17):
    rng = np.random.default_rng(seed)
    signatures = [json.dumps(state, sort_keys=True) for state in states]
    used, chosen = set(), []
    for action in range(4):
        candidates = rng.permutation(np.flatnonzero(labels == action))
        group = []
        for index in candidates:
            if signatures[index] not in used:
                group.append(int(index))
                used.add(signatures[index])
            if len(group) == per_class:
                break
        if len(group) != per_class:
            raise ValueError("Not enough distinct training observations for the balanced probe")
        chosen.extend(group)
    return np.asarray(chosen)


def diagnose(dataset, base, output, *, steps=200, seed=17):
    if not 1 <= steps <= 1000:
        raise ValueError("Require 1..1000 diagnostic steps")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    _, splits = read_prepared(dataset)
    part = splits["train"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    buckets = defaultdict(Counter)
    for state, label in zip(part["states"], part["actions"]):
        buckets[json.dumps(state, sort_keys=True)][int(label)] += 1
    chosen = unique_balanced_indices(part["states"], part["actions"], seed=seed)
    labels = part["actions"][chosen]
    agent, provenance = load_base(base)
    for name, parameter in agent.model.named_parameters():
        parameter.requires_grad_(name.startswith(("head.", "scorer.", "type_emb.")))
    parameters = [p for p in agent.model.parameters() if p.requires_grad]
    initial_parameter = agent.model.scorer[-1].weight.detach().clone()
    items = encode_states(agent, [part["states"][i] for i in chosen])
    features = cache_features(agent, items)
    agent.model.eval()
    with torch.no_grad():
        live = torch.cat([batch_logits(agent, items[i:i + 2]).cpu() for i in range(0, len(items), 2)])
        cached = torch.cat([cached_logits(agent, features[i:i + 2]).cpu() for i in range(0, len(items), 2)])
    error = float((live - cached).abs().max())
    if not torch.allclose(live, cached, rtol=1e-4, atol=1e-4):
        raise ValueError(f"Cached head differs from upstream forward: {error}")
    report = {"schema": "laya_memorization_probe_v1", "status": "running", "diagnostic_only": True,
        "dataset_sha256": digest(Path(dataset) / "manifest.json", "sha256"), "base": provenance,
        "source_split": "train", "validation_or_test_used": False, "game_policy_saved": False,
        "selected_training_indices": chosen.tolist(), "class_counts": np.bincount(labels, minlength=4).tolist(),
        "conflicting_training_observations": sum(len(v) > 1 for v in buckets.values()),
        "deterministic_training_accuracy_upper_bound": sum(max(v.values()) for v in buckets.values()) / len(part["actions"]),
        "cached_logit_max_abs_error": error, "learning_rate": 1e-4, "dropout_enabled": False,
        "initial": metrics(cached.softmax(-1).numpy(), labels), "history": []}
    weights = class_weights(part["actions"])
    report["old_style_single_example_gradient_audit"] = []
    for action in range(4):
        agent.model.zero_grad(set_to_none=True)
        index = int(np.flatnonzero(labels == action)[0])
        loss = F.cross_entropy(cached_logits(agent, [features[index]]), torch.tensor([action], device=agent.device)) * float(weights[action])
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        report["old_style_single_example_gradient_audit"].append({"action": action,
            "class_weight": float(weights[action]), "weighted_grad_norm_before_clip": norm,
            "clip_multiplier": min(1.0, 1.0 / (norm + 1e-6))})
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    targets = torch.tensor(labels, device=agent.device)
    try:
        for step in range(steps):
            # One optimizer step contains all four classes before gradient clipping.
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for start in range(0, len(features), 2):
                loss = F.cross_entropy(cached_logits(agent, features[start:start + 2]), targets[start:start + 2], reduction="sum") / len(features)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite probe loss")
                loss.backward()
                loss_sum += float(loss.detach())
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
            optimizer.step()
            if step == 0 or (step + 1) % 10 == 0:
                measured = metrics(cached_probabilities(agent, features), labels)
                report["history"].append({"step": step + 1, "loss": loss_sum,
                                          "gradient_norm": norm, "memorization": measured})
                print(f"Balanced probe {step + 1}/{steps}: accuracy={measured['accuracy']:.3f}, loss={measured['nll']:.4f}", flush=True)
                write_json(output / "report.json", report)
                if measured["accuracy"] >= 0.99 and measured["nll"] < 0.1:
                    break
        report["final"] = metrics(cached_probabilities(agent, features), labels)
        report["memorization_passed"] = report["final"]["accuracy"] >= 0.95
        report["head_weights_changed"] = not torch.equal(initial_parameter, agent.model.scorer[-1].weight)
        report["encoder_has_gradients"] = any(p.grad is not None for p in agent.model.encoder.parameters())
        report["status"] = "completed"
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("models/laya-base"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    diagnose(**vars(args))


if __name__ == "__main__":
    main()
