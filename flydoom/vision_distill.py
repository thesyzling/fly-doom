"""Offline Vision distillation and teacher-free, paired native-game evaluation."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
import torch
from torch import nn
import torch.nn.functional as F

from flydoom import action_memory, student
from flydoom.calibration import output_features, write_json
from flydoom.data import digest
from flydoom.learning_data import ACTIONS, make_game
from flydoom.live_activity import ReadoutTelemetry, frame_png
from flydoom.live_brain import load_checkpoint
from flydoom.replay import finish_live_replay, safe_path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verified_parent(path):
    path = Path(path)
    report = read_json(path / "report.json")
    if (report.get("schema") != action_memory.SCHEMA or report.get("status") != "completed"
            or not report.get("student_trained")
            or digest(path / "student.safetensors", "sha256") != report["student_sha256"]
            or digest(student.__file__, "sha256") != report["implementation_sha256"]
            or digest(action_memory.__file__, "sha256") != report["memory_implementation_sha256"]
            or report["memory_feature_names"] != list(action_memory.NAMES)):
        raise ValueError("Require an intact completed action-memory parent")
    return report


def valid_distribution(value):
    p = np.asarray(value, dtype=np.float32)
    if p.shape != (4,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-5):
        raise ValueError("Invalid recorded probability distribution")
    return p


def signature(vector):
    return hashlib.sha256(np.asarray(vector, dtype="<f4").tobytes()).hexdigest()


def exclude_training_duplicates(train, validation):
    """Exclude either exact RGB or exact student-input repeats across splits."""
    images = set(train["rgb_hashes"])
    inputs = {signature(x) for x in train["features"]}
    return np.array([h not in images and signature(x) not in inputs
                     for h, x in zip(validation["rgb_hashes"], validation["features"])])


def prepare(plan_path, parent):
    plan = read_json(plan_path)
    if plan.get("status") != "completed":
        raise ValueError("Collection plan is not complete")
    sets = [set(plan[k + "_seeds"]) for k in ("train", "validation", "evaluation")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Training, validation and evaluation seeds overlap")
    parts, sources, teacher_identity = {}, [], None
    for split in ("train", "validation"):
        values = {k: [] for k in ("features", "targets", "rgb_hashes", "seeds")}
        seen = set()
        for directory in plan[split + "_runs"]:
            run = Path(directory)
            report = read_json(run / "report.json")
            if (report.get("schema") != "vision_fly_research_v1" or report.get("status") != "completed"
                    or report["checkpoint_sha256"] != parent["student_sha256"]
                    or report.get("disconnected", False)):
                raise ValueError("Recording does not belong to this intact parent and connected graph")
            identity = {k: report["vision"][k] for k in ("repo", "revision", "source_commit", "manifest_sha256")}
            if teacher_identity is not None and identity != teacher_identity:
                raise ValueError("Different Vision checkpoints in the dataset")
            teacher_identity = identity
            sources.append({"run": str(run), "split": split, "report_sha256": digest(run / "report.json", "sha256"),
                            "trace_sha256": digest(run / "decisions.jsonl", "sha256")})
            memories, numbers = {}, {}
            for line in (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                seed = row["seed"]
                if seed not in plan[split + "_seeds"]:
                    raise ValueError("Recorded seed violates the declared split")
                memory = memories.setdefault(seed, action_memory.ActionMemory())
                numbers[seed] = numbers.get(seed, 0) + 1
                if row["decision"] != numbers[seed]:
                    raise ValueError("Missing or reordered decision history")
                path = safe_path(run, row["observation"] + ".npz")
                if digest(path, "sha256") != row["observation_sha256"]:
                    raise ValueError("Observation checksum changed")
                with np.load(path, allow_pickle=False) as stored:
                    x = stored["features"].astype(np.float32)
                    target = valid_distribution(stored["teacher_probabilities"])
                    np.testing.assert_allclose(target, valid_distribution(row["vision"]["probabilities"]), atol=1e-7)
                    np.testing.assert_allclose(stored["student_probabilities"], valid_distribution(row["probabilities"]), atol=1e-7)
                if x.shape != (parent["input_size"],) or not np.isfinite(x).all():
                    raise ValueError("Invalid student feature vector")
                previous = memory.actions[-1] if memory.actions else 0
                np.testing.assert_array_equal(x[-4:], np.eye(4)[previous])
                np.testing.assert_array_equal(x[-21:-4], action_memory.encode(memory.observe()))
                mixed = row["alpha"] * target + (1 - row["alpha"]) * np.asarray(row["probabilities"])
                np.testing.assert_allclose(mixed, row["applied_probabilities"], atol=1e-6)
                if ACTIONS[int(mixed.argmax())] != row["action"]:
                    raise ValueError("Recorded action disagrees with fusion")
                memory.advance(ACTIONS.index(row["action"]))
                with Image.open(safe_path(run, row["observation"] + ".png")) as image:
                    rgb_hash = hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()
                if rgb_hash != row["vision"]["frame_sha256"]:
                    raise ValueError("RGB observation differs from the teacher input")
                key = (signature(x), rgb_hash)
                if key in seen:
                    continue
                seen.add(key)
                for name, value in (("features", x), ("targets", target), ("rgb_hashes", rgb_hash), ("seeds", seed)):
                    values[name].append(value)
        if not values["features"]:
            raise ValueError("Empty dataset split")
        parts[split] = {k: np.asarray(v) for k, v in values.items()}
    keep = exclude_training_duplicates(parts["train"], parts["validation"])
    removed = int((~keep).sum())
    if not keep.any():
        raise ValueError("Validation entirely repeats training observations; collect different episodes")
    parts["validation"] = {k: v[keep] for k, v in parts["validation"].items()}
    return plan, parts, sources, teacher_identity, removed


def measure(model, x, target):
    with torch.no_grad():
        logits = model(x)
        labels, predicted = target.argmax(-1), logits.argmax(-1)
        per_action = {ACTIONS[i]: float((predicted[labels == i] == i).float().mean())
                      for i in range(4) if (labels == i).any()}
        return {"kl": float(F.kl_div(logits.log_softmax(-1), target, reduction="batchmean")),
                "teacher_agreement": float((predicted == labels).float().mean()),
                "macro_agreement": float(np.mean(list(per_action.values()))), "per_action_agreement": per_action,
                "samples": len(x), "teacher_action_counts": torch.bincount(labels, minlength=4).tolist(),
                "student_action_counts": torch.bincount(predicted, minlength=4).tolist()}


def fit(model, parts, *, epochs, learning_rate, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = {s: torch.tensor(p["features"], dtype=torch.float32) for s, p in parts.items()}
    targets = {s: torch.tensor(p["targets"], dtype=torch.float32) for s, p in parts.items()}
    model.eval()
    initial = {s: measure(model, x[s], targets[s]) for s in parts}
    best, selected, history = initial["validation"]["kl"], 0, []
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        order = rng.permutation(len(x["train"]))
        for start in range(0, len(order), 32):
            indices = order[start:start + 32]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x["train"][indices])
            loss = F.kl_div(logits.log_softmax(-1), targets["train"][indices], reduction="batchmean")
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite distillation loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        validation = measure(model, x["validation"], targets["validation"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation})
        if validation["kl"] < best:
            best, selected = validation["kl"], epoch
            saved = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}: validation KL={validation['kl']:.4f}, teacher agreement={validation['teacher_agreement']:.3f}", flush=True)
    model.load_state_dict(saved)
    return {"initial": initial, "final": {s: measure(model, x[s], targets[s]) for s in parts},
            "selected_epoch": selected, "epochs": history}


def train(plan_path, output):
    plan = read_json(plan_path)
    parent = verified_parent(plan["parent"])
    plan, parts, sources, teacher, removed = prepare(plan_path, parent)
    if not 1 <= plan["epochs"] <= 500 or not 0 < plan["learning_rate"] <= .001:
        raise ValueError("Invalid training settings")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = student.SpikingReadout(parent["input_size"], parent["hidden"])
    original = load_file(str(Path(plan["parent"]) / "student.safetensors"))
    model.load_state_dict(original)
    keys = ("schema", "input_size", "hidden", "dataset_sha256", "calibration_sha256", "output_root_ids",
            "implementation_sha256", "memory_implementation_sha256", "memory_feature_names")
    report = {k: parent[k] for k in keys}
    report.update(status="running", student_trained=False, action_memory=True,
        parent_student_sha256=parent["student_sha256"], parent_report_sha256=digest(Path(plan["parent"]) / "report.json", "sha256"),
        training_method="KL distillation of pinned Laya Vision probabilities into the existing spiking readout",
        vision_distillation={"teacher": teacher, "plan_sha256": digest(plan_path, "sha256"), "sources": sources},
        teacher_accepted=None, experimental_teacher=True,
        teacher_scope="Visual teacher measured on basic scenario; earlier text teacher gate is unrelated",
        normalization="Frozen parent mean and scale", connectome_weights_trained=False, laya_needed_at_inference=False,
        game_skill_validated=False, test_evaluated=False, default_checkpoint_changed=False,
        selection_rule=plan["selection"], reserved_evaluation_seeds=plan["evaluation_seeds"],
        learning_rate=plan["learning_rate"], seed=plan["seed"], validation_duplicates_removed=removed,
        training_implementation_sha256=digest(__file__, "sha256"))
    write_json(output / "report.json", report)
    try:
        for split, part in parts.items():
            np.savez_compressed(output / f"{split}.npz", **part)
        report["dataset_files"] = {s: digest(output / f"{s}.npz", "sha256") for s in parts}
        report.update(fit(model, parts, epochs=plan["epochs"], learning_rate=plan["learning_rate"], seed=plan["seed"]))
        if not torch.equal(model.mean, original["mean"]) or not torch.equal(model.scale, original["scale"]):
            raise ValueError("Input normalization changed")
        report["parameter_changes"] = {k: {"changed": int(torch.count_nonzero(v - original[k])),
            "max_absolute_delta": float((v - original[k]).abs().max())} for k, v in model.state_dict().items()}
        save_file(model.state_dict(), str(output / "student.safetensors"))
        report.update(status="completed", student_trained=True, student_sha256=digest(output / "student.safetensors", "sha256"))
        if verified_parent(plan["parent"])["student_sha256"] != parent["student_sha256"]:
            raise ValueError("Parent changed during training")
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        write_json(output / "report.json", report)
    return report


def evaluate_policy(controller, model, checkpoint_report, seeds, output, max_decisions=75):
    """No Vision object exists here. Only the student controls the native engine."""
    import vizdoom as vzd
    if (not 1 <= len(seeds) <= 20 or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or not 1 <= max_decisions <= 75):
        raise ValueError("Require 1..20 distinct uint32 seeds and 1..75 decisions")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "observations").mkdir()
    report = {"schema": "vision_student_evaluation_v1", "status": "running", "episodes": [], "alpha": 0.,
              "checkpoint_sha256": checkpoint_report["student_sha256"], "laya_used_during_play": False,
              "training_during_run": False, "connectome_weights_trained": False, "opening_hashes": [],
              "planned_seeds": list(seeds), "max_decisions": max_decisions,
              "disconnected": getattr(controller, "control", "connected") == "disconnected",
              "source_sha256": digest(__file__, "sha256")}
    game, telemetry = None, ReadoutTelemetry(model)
    sequence = 0
    try:
        game, buttons = make_game(seeds[0], False)
        with (output / "decisions.jsonl").open("w", encoding="utf-8") as trace:
            for episode, seed in enumerate(seeds, 1):
                game.set_seed(seed)
                game.new_episode()
                controller.reset()
                memory, previous = action_memory.ActionMemory(), 0
                counts = dict.fromkeys(ACTIONS, 0)
                report["opening_hashes"].append(hashlib.sha256(game.get_state().screen_buffer.tobytes()).hexdigest())
                for number in range(1, max_decisions + 1):
                    frame = game.get_state().screen_buffer.copy()
                    decision = controller.decide(frame)
                    if decision["voltage_min_mv"] < -90:
                        raise ValueError("Fly voltage bound exceeded")
                    vector = np.concatenate((output_features(controller), np.eye(4, dtype=np.float32)[previous]))
                    vector = action_memory.augment(vector[None], [memory.observe()])[0]
                    with torch.inference_mode():
                        p = model(torch.tensor(vector)[None]).softmax(-1)[0].numpy()
                    action = int(p.argmax())
                    reward, kills = game.get_total_reward(), int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
                    frames, tics = [frame], 0
                    for _ in range(4):
                        if game.is_episode_finished():
                            break
                        game.make_action([int(b == ACTIONS[action]) for b in buttons], 1)
                        tics += 1
                        state = game.get_state()
                        if state is not None:
                            frames.append(state.screen_buffer.copy())
                    after_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
                    effect = {"reward_delta": game.get_total_reward() - reward, "kill_delta": after_kills - kills,
                              "kills": after_kills, "episode_finished": game.is_episode_finished(),
                              "game_tics": tics, "terminal_image_available": not game.is_episode_finished()}
                    sequence += 1
                    stem = f"observations/{sequence:05d}"
                    paths = []
                    for tic, image in enumerate(frames):
                        name = f"{stem}-tic{tic}.png"
                        (output / name).write_bytes(base64.b64decode(frame_png(image).split(",", 1)[1]))
                        paths.append(name)
                    np.savez_compressed(output / f"{stem}.npz", features=vector, student_probabilities=p,
                                        student_activity=telemetry.activity)
                    row = {"episode": episode, "decision": number, "seed": seed, "action": ACTIONS[action],
                           "student_action": ACTIONS[action], "probabilities": p.tolist(), "applied_probabilities": p.tolist(),
                           "alpha": 0., "agreement": None, "return": game.get_total_reward(), "brain_spikes": decision["spikes"],
                           "vision": {"probabilities": None}, "effect": effect, "playback_frames": paths,
                           "observation": stem, "observation_sha256": digest(output / f"{stem}.npz", "sha256")}
                    trace.write(json.dumps(row, allow_nan=False) + "\n")
                    trace.flush()
                    counts[ACTIONS[action]] += 1
                    memory.advance(action)
                    previous = action
                    if game.is_episode_finished():
                        break
                summary = {"seed": seed, "decisions": number, "return": game.get_total_reward(), "kills": after_kills,
                           "action_counts": counts, "end_reason": "game_finished" if game.is_episode_finished() else "decision_limit"}
                report["episodes"].append(summary)
                write_json(output / "report.json", report)
                print(f"{output.name}: {summary}", flush=True)
        report["status"] = "completed"
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        telemetry.close()
        if game:
            game.close()
        write_json(output / "report.json", report)
    finish_live_replay(output, report)
    return report


def compare(plan_path, candidate, output, calibration, data_dir):
    plan, output = read_json(plan_path), Path(output)
    candidate_report = verified_parent(candidate)
    if candidate_report["vision_distillation"]["plan_sha256"] != digest(plan_path, "sha256"):
        raise ValueError("Candidate belongs to a different split plan")
    output.mkdir(parents=True, exist_ok=False)
    _, controller, model, parent = load_checkpoint(plan["parent"], calibration, data_dir)
    if parent["student_sha256"] != candidate_report["parent_student_sha256"]:
        raise ValueError("Evaluation parent differs from the training parent")
    results = {}
    for name, checkpoint, report in (("parent", plan["parent"], parent), ("candidate", candidate, candidate_report)):
        model.load_state_dict(load_file(str(Path(checkpoint) / "student.safetensors")))
        model.eval()
        results[name] = evaluate_policy(controller, model, report, plan["evaluation_seeds"], output / ("run-" + name), plan["max_decisions"])
    with np.load(Path(candidate) / "train.npz", allow_pickle=False) as data:
        known = set(data["rgb_hashes"])
    scores = {k: {"kills": sum(e["kills"] for e in v["episodes"]),
                  "mean_return": float(np.mean([e["return"] for e in v["episodes"]])),
                  "decisions": sum(e["decisions"] for e in v["episodes"])} for k, v in results.items()}
    def rank(name):
        return scores[name]["kills"], scores[name]["mean_return"]
    comparison = {"schema": "vision_distillation_comparison_v1", "status": "completed", "scores": scores,
                  "winner": "candidate" if rank("candidate") > rank("parent") else "parent", "selection_rule": plan["promotion"],
                  "plan_sha256": digest(plan_path, "sha256"), "candidate_sha256": candidate_report["student_sha256"],
                  "parent_sha256": parent["student_sha256"], "laya_used_during_play": False,
                  "training_opening_overlaps": [s for s, h in zip(plan["evaluation_seeds"], results["parent"]["opening_hashes"]) if h in known],
                  "episodes": {k: v["episodes"] for k, v in results.items()}, "default_checkpoint_changed": False,
                  "note": "Small development pilot. New seeds do not guarantee unique scenes or establish a biological-topology advantage."}
    if results["parent"]["opening_hashes"] != results["candidate"]["opening_hashes"]:
        raise ValueError("Paired evaluation openings differ")
    write_json(output / "report.json", comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--plan", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    evaluation = sub.add_parser("compare")
    evaluation.add_argument("--plan", type=Path, required=True)
    evaluation.add_argument("--candidate", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--calibration", type=Path, default=Path("runs/calibration-training-v1/report.json"))
    evaluation.add_argument("--data-dir", type=Path, default=Path("data/processed/fafb783"))
    args = vars(parser.parse_args())
    command = args.pop("command")
    args["plan_path"] = args.pop("plan")
    result = train(**args) if command == "train" else compare(**args)
    print(json.dumps({k: result[k] for k in ("status", "selected_epoch", "final", "winner", "scores") if k in result}, indent=2))


if __name__ == "__main__":
    main()
