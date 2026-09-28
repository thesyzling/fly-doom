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
        ids, markers = build_sequence(agent.tok, state, question, max_len=512, head_max_len=160)
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
    return {"samples": len(labels), "accuracy": float(np.mean(predicted == labels)),
            "balanced_accuracy": float(np.mean(recalls)),
            "nll": float(-np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-8)).mean()),
            "class_counts": np.bincount(labels, minlength=4).tolist()}


def teacher_gate(validation, baseline_accuracy, shuffled_accuracy):
    # This gate is a pilot screen, not a statistical significance claim.
    return (validation["samples"] >= 30 and min(validation["class_counts"][1:]) >= 5
            and validation["accuracy"] >= baseline_accuracy + 0.05
            and validation["balanced_accuracy"] >= 0.5
            and validation["accuracy"] >= shuffled_accuracy + 0.05)


def train(dataset, base, output, *, epochs=5, batch_size=2, seed=7, device="auto"):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    if not 1 <= epochs <= 100 or not 1 <= batch_size <= 16:
        raise ValueError("Require 1..100 epochs and 1..16 examples per batch")
    source, splits = read_prepared(dataset)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent, provenance = load_base(base, device)
    for name, parameter in agent.model.named_parameters():
        parameter.requires_grad_(name.startswith(("head.", "scorer.", "type_emb.")))
    optimizer = torch.optim.AdamW([p for p in agent.model.parameters() if p.requires_grad], lr=2e-5)
    items = {key: encode_states(agent, part["states"]) for key, part in splits.items() if key != "test"}
    train_part, validation = splits["train"], splits["validation"]
    majority = int(np.bincount(train_part["actions"], minlength=4).argmax())
    baseline = max(float(np.mean(validation["actions"] == majority)),
                   float(np.mean(validation["actions"] == validation["previous"])))
    initial = probabilities(agent, items["validation"], batch_size)
    report = {"schema": "laya_teacher_v1", "status": "running", "teacher_accepted": False,
              "training_method": "supervised cross-entropy; encoder frozen; no RLCD or game-reward update",
              "base": provenance, "dataset_sha256": digest(Path(dataset) / "manifest.json", "sha256"),
              "actions": ACTIONS, "question": QUESTION, "seed": seed,
              "initial_validation": metrics(initial, validation["actions"]),
              "validation_baseline_accuracy": baseline, "epochs": [], "temperature": 1.0,
              "probabilities_calibrated": False, "calibration_sha256": source["calibration_sha256"]}
    best = float("inf")
    try:
        for epoch in range(epochs):
            agent.model.train()
            agent.model.encoder.eval()
            losses = []
            order = rng.permutation(len(train_part["actions"]))
            for offset in range(0, len(order), batch_size):
                chosen = order[offset:offset + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = batch_logits(agent, [items["train"][i] for i in chosen])
                loss = F.cross_entropy(logits, torch.tensor(train_part["actions"][chosen], device=agent.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite teacher loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in agent.model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                if (offset // batch_size + 1) % 25 == 0:
                    print(f"Teacher epoch {epoch + 1}: {offset + len(chosen)}/{len(order)} rows", flush=True)
            val = metrics(probabilities(agent, items["validation"], batch_size), validation["actions"])
            report["epochs"].append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": val})
            print(f"Teacher epoch {epoch + 1}: validation accuracy={val['accuracy']:.3f}", flush=True)
            if val["nll"] < best:
                best = val["nll"]
                report["selected_epoch"] = epoch + 1
                save_file({name: tensor.detach().cpu().contiguous() for name, tensor in agent.model.state_dict().items()
                           if name.startswith(("head.", "scorer.", "type_emb."))}, str(output / "head.safetensors"))
            write_json(output / "report.json", report)
        restore_head(agent, output / "head.safetensors")
        val_probs = probabilities(agent, items["validation"], batch_size)
        val = metrics(val_probs, validation["actions"])
        # Keep previous-action hints fixed; shuffle only image content to expose shortcut policies.
        permutation = rng.permutation(len(validation["states"]))
        shuffled_states = [{**validation["states"][i], "rows": validation["states"][j]["rows"]}
                           for i, j in enumerate(permutation)]
        shuffled = metrics(probabilities(agent, encode_states(agent, shuffled_states), batch_size), validation["actions"])
        report["validation"] = val
        report["shuffled_image_validation"] = shuffled
        report["teacher_accepted"] = teacher_gate(val, baseline, shuffled["accuracy"])
        # Test is evaluated once after selection; its labels never select a checkpoint or a gate.
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
