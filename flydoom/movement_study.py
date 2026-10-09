"""Bounded six-action pilot: reserve, collect Vision labels, train, then benchmark."""

import argparse
import base64
import hashlib
from pathlib import Path

import numpy as np
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.live_activity import frame_png
from flydoom.live_brain import load_checkpoint
from flydoom.movement_core import ACTIONS, MEMORY, SCHEMA, Memory, apply_action, expand, make_game, vector
from flydoom.movement_teacher import MovementTeacher
from flydoom.reserve_recovery import image_hashes
from flydoom.vision_distill import read_json, signature
from flydoom.vision_recovery import select_openings
from flydoom.vision_validation import fingerprint, paired_summary

PARENT = Path("runs/fly-student-vision-v1")
CALIBRATION = Path("runs/calibration-training-v1/report.json")
DATA = Path("data/processed/fafb783")


def metrics(model, part):
    with torch.inference_mode():
        logits = model(torch.tensor(part["features"], dtype=torch.float32))
        targets = torch.tensor(part["targets"], dtype=torch.float32)
        labels, predicted = targets.argmax(1), logits.argmax(1)
        return {"kl": float(F.kl_div(logits.log_softmax(1), targets, reduction="batchmean")),
                "teacher_agreement": float((labels == predicted).float().mean()),
                "teacher_actions": torch.bincount(labels, minlength=6).tolist(),
                "student_actions": torch.bincount(predicted, minlength=6).tolist(), "samples": len(labels)}


def reserve(output):
    excluded = set()
    for path in Path("runs").glob("*/opening-scan.json"):
        excluded.update(image_hashes(read_json(path)))
    for pattern in ("*/train.npz", "*/validation.npz"):
        for path in Path("runs").glob(pattern):
            with np.load(path, allow_pickle=False) as data:
                if "rgb_hashes" in data.files:
                    excluded.update(map(str, data["rgb_hashes"]))
    game, rows = make_game(33000000), []
    sizes = {"evaluation": 8, "train": 8, "validation": 4}
    try:
        for seed in np.random.default_rng(1910).choice(np.arange(33000000, 33100000), 1000, replace=False):
            game.set_seed(int(seed)); game.new_episode()
            rows.append({"seed": int(seed), "rgb_sha256": hashlib.sha256(game.get_state().screen_buffer.tobytes()).hexdigest()})
            selected = select_openings(rows, excluded, set(), sizes=sizes)
            if all(len(selected[s]) == n for s, n in sizes.items()):
                break
        else:
            raise ValueError("Not enough distinct pilot openings")
    finally:
        game.close()
    write_json(output / "opening-scan.json", {"rows": rows, "excluded_rgb_hashes": sorted(excluded)})
    import vizdoom
    package = Path(vizdoom.__file__).parent
    artifacts = [PARENT / "student.safetensors", PARENT / "report.json", CALIBRATION,
                 output / "opening-scan.json", package / "scenarios/basic.cfg", package / "scenarios/basic.wad",
                 package / "freedoom2.wad", *Path("flydoom").glob("*.py")]
    plan = {"schema": "six_action_pilot_v1", "actions": ACTIONS, "splits": selected,
            "collection_decisions": 24, "evaluation_decisions": 48, "epochs": 80, "learning_rate": .0001,
            "seed": 1909, "selection": "Lowest validation KL, including epoch zero. No gameplay selection.",
            "collection": "First six actions: forward, forward, left, backward, backward, right; then expanded parent argmax. Vision labels every pre-action image.",
            "scope": "Single-seed basic-scenario engineering pilot. Six-action teacher quality is unvalidated; no navigation mastery claim.",
            "artifact_sha256": fingerprint(artifacts)}
    write_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(digest(output / "plan.json", "sha256"), encoding="ascii")
    return plan


def collect(controller, model, teacher, openings, folder, limit):
    folder.mkdir(); parts = {k: [] for k in ("features", "targets", "rgb_hashes", "seeds", "applied_actions")}
    trace, game = [], make_game(openings[0]["seed"])
    try:
        for opening in openings:
            game.set_seed(opening["seed"]); game.new_episode(); controller.reset(); memory = Memory()
            assert hashlib.sha256(game.get_state().screen_buffer.tobytes()).hexdigest() == opening["rgb_sha256"]
            for number in range(limit):
                if game.is_episode_finished(): break
                frame = game.get_state().screen_buffer.copy()
                answer = teacher.predict(frame)
                decision = controller.decide(frame)
                if decision["voltage_min_mv"] < -90: raise ValueError("Engineering voltage bound exceeded")
                features = vector(controller, memory)
                with torch.inference_mode(): probs = model(torch.tensor(features)[None]).softmax(1)[0].numpy()
                action = (4, 4, 1, 5, 5, 2)[number] if number < 6 else int(probs.argmax())
                image = folder / f"{len(trace):05d}.png"
                image.write_bytes(base64.b64decode(frame_png(frame).split(",", 1)[1]))
                effect = apply_action(game, action)
                trace.append({"seed": opening["seed"], "decision": number + 1, "action": ACTIONS[action],
                              "teacher": answer, "image": image.name, "png_sha256": digest(image, "sha256"), "effect": effect})
                for key, value in (("features", features), ("targets", answer["probabilities"]),
                                   ("rgb_hashes", answer["frame_sha256"]), ("seeds", opening["seed"]), ("applied_actions", action)):
                    parts[key].append(value)
                memory.advance(action)
            print(f"Collected {folder.name}: seed {opening['seed']}, {len(memory.actions)} decisions", flush=True)
    finally:
        game.close(); write_json(folder / "trace.json", trace)
    return {key: np.asarray(values) for key, values in parts.items()}


def evaluate(controller, model, openings, limit):
    game, episodes = make_game(openings[0]["seed"]), []
    try:
        for opening in openings:
            game.set_seed(opening["seed"]); game.new_episode(); controller.reset(); memory = Memory(); counts = dict.fromkeys(ACTIONS, 0)
            assert hashlib.sha256(game.get_state().screen_buffer.tobytes()).hexdigest() == opening["rgb_sha256"]
            rows = []
            for _ in range(limit):
                if game.is_episode_finished(): break
                frame = game.get_state().screen_buffer.copy(); controller.decide(frame); features = vector(controller, memory)
                with torch.inference_mode(): probs = model(torch.tensor(features)[None]).softmax(1)[0].numpy()
                action = int(probs.argmax()); counts[ACTIONS[action]] += 1
                effect = apply_action(game, action)
                rows.append({"action": ACTIONS[action], "probabilities": probs.tolist(), "effect": effect,
                             "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
                memory.advance(action)
            episodes.append({"seed": opening["seed"], "kills": rows[-1]["effect"]["kills"],
                             "return": game.get_total_reward(), "decisions": len(rows), "action_counts": counts, "trace": rows})
            print(f"Evaluation: {opening['seed']} / kills {episodes[-1]['kills']} / {len(rows)} decisions", flush=True)
    finally:
        game.close()
    return episodes


def run(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    plan = reserve(output)
    report = {"schema": SCHEMA, "status": "collecting", "student_trained": False, "actions": ACTIONS,
              "memory_feature_names": MEMORY, "connectome_weights_trained": False,
              "plan_sha256": digest(output / "plan.json", "sha256"), "scope": plan["scope"]}
    write_json(output / "report.json", report)
    teacher = None
    try:
        _, controller, parent, identity = load_checkpoint(PARENT, CALIBRATION, DATA)
        model = expand(parent, len(identity["output_root_ids"]))
        baseline = {k: v.clone() for k, v in model.state_dict().items()}
        report.update(input_size=model.input.in_features, hidden=model.input.out_features,
                      output_root_ids=identity["output_root_ids"], calibration_sha256=identity["calibration_sha256"],
                      parent_sha256=identity["student_sha256"], core_sha256=digest(Path(__file__).with_name("movement_core.py"), "sha256"))
        teacher = MovementTeacher(output / "teacher.log"); report["teacher"] = teacher.metadata
        parts = {s: collect(controller, model, teacher, plan["splits"][s], output / s, plan["collection_decisions"])
                 for s in ("train", "validation")}
        teacher.close(); teacher = None
        train_images = set(parts["train"]["rgb_hashes"]); train_inputs = {signature(x) for x in parts["train"]["features"]}
        keep = [h not in train_images and signature(x) not in train_inputs for h, x in zip(parts["validation"]["rgb_hashes"], parts["validation"]["features"])]
        report["validation_duplicates_removed"] = int(len(keep) - sum(keep))
        parts["validation"] = {k: v[keep] for k, v in parts["validation"].items()}
        if not sum(keep): raise ValueError("No distinct validation examples")
        for split, part in parts.items(): np.savez_compressed(output / (split + ".npz"), **part)
        report["initial"] = {s: metrics(model, p) for s, p in parts.items()}
        best, selected, saved = report["initial"]["validation"]["kl"], 0, baseline
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan["learning_rate"])
        rng = np.random.default_rng(plan["seed"]); history = []
        x, y = torch.tensor(parts["train"]["features"], dtype=torch.float32), torch.tensor(parts["train"]["targets"], dtype=torch.float32)
        for epoch in range(1, plan["epochs"] + 1):
            model.train()
            for indices in np.array_split(rng.permutation(len(x)), max(1, (len(x) + 31) // 32)):
                optimizer.zero_grad(set_to_none=True)
                loss = F.kl_div(model(x[indices]).log_softmax(1), y[indices], reduction="batchmean")
                if not torch.isfinite(loss): raise ValueError("Non-finite loss")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            model.eval(); validation = metrics(model, parts["validation"]); history.append({"epoch": epoch, **validation})
            if validation["kl"] < best:
                best, selected, saved = validation["kl"], epoch, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(saved); model.eval()
        report.update(selected_epoch=selected, epochs=history, final={s: metrics(model, p) for s, p in parts.items()})
        save_file(model.state_dict(), str(output / "student.safetensors"))
        report.update(student_trained=True, student_sha256=digest(output / "student.safetensors", "sha256"), status="evaluating")
        write_json(output / "report.json", report)
        results = {}
        for name, weights in (("expanded_parent", baseline), ("trained", saved)):
            model.load_state_dict(weights)
            episodes = evaluate(controller, model, plan["splits"]["evaluation"], plan["evaluation_decisions"])
            results[name] = {"episodes": episodes, "hits": sum(e["kills"] > 0 for e in episodes),
                             "mean_return": float(np.mean([e["return"] for e in episodes]))}
        report["benchmark"] = results
        report["paired"] = paired_summary(results["expanded_parent"]["episodes"], results["trained"]["episodes"], [r["rgb_sha256"] for r in plan["splits"]["evaluation"]])
        collected = set(parts["train"]["rgb_hashes"]) | set(parts["validation"]["rgb_hashes"])
        report["test_opening_overlap"] = sum(r["rgb_sha256"] in collected for r in plan["splits"]["evaluation"])
        report["test_trajectory_rgb_matches"] = {name: sum(row["rgb_sha256"] in collected for e in result["episodes"] for row in e["trace"]) for name, result in results.items()}
        if any(digest(path, "sha256") != sha for path, sha in plan["artifact_sha256"].items()):
            raise ValueError("Pilot source or artifact changed during execution")
        report["status"] = "completed"
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        if teacher: teacher.close()
        write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); result = run(args.output)
    print({"status": result["status"], "selected_epoch": result["selected_epoch"], "benchmark": {k: {n: v[n] for n in ("hits", "mean_return")} for k, v in result["benchmark"].items()}})
