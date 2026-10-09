"""Bounded, auditable experience -> synaptic candidate -> benchmark -> promotion loop.

Run `python -m flydoom.learning_cycle --cycles 1`. Ctrl+C cancels the candidate;
an active checkpoint changes only after a successful paired promotion gate.
"""

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import secrets

import numpy as np
from PIL import Image
import torch

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.learning_curriculum import make_game, act, learning_target
from flydoom.learning_features import ContrastMotionController
from flydoom.movement_core import ACTIONS, Memory, vector
from flydoom.movement_teacher import MovementTeacher
from flydoom.reserve_recovery import image_hashes
from flydoom.synaptic import load_system, replay as full_replay, array_hash, CALIBRATION
from flydoom.synaptic_eligibility import restore, gradient, SCHEMA
from flydoom.vision_validation import collect_seeds

ROOT = Path("runs/learning")
INITIAL = Path("runs/synaptic-eligibility-v1")


def replay(controller, model, episodes):
    result = full_replay(controller, model, episodes)
    rows = [r for episode in episodes for r in episode]
    keep = np.array([r.get("score", True) for r in rows], bool)
    if not keep.any(): raise ValueError("No independent frames remain for evaluation")
    if not keep.all():
        p = np.asarray(result["probabilities"], np.float64)[keep]
        q = np.asarray([r["teacher"]["probabilities"] for r in rows], np.float64)[keep]
        result["kl"] = float(np.sum(q * (np.log(np.maximum(q, 1e-30)) - np.log(np.maximum(p, 1e-30)))) / len(p))
        result["teacher_agreement"] = float(np.mean(p.argmax(1) == q.argmax(1)))
    result.update(samples=int(keep.sum()), replayed_frames=len(rows), excluded_duplicate_frames=int((~keep).sum()))
    return result


def load_experience(folder, split):
    rows = json.loads((Path(folder) / split / "trace.json").read_text())
    episodes = []
    for seed in dict.fromkeys(r["seed"] for r in rows):
        episode = []
        for row in (r for r in rows if r["seed"] == seed):
            path = Path(folder) / split / row["image"]
            if digest(path, "sha256") != row["png_sha256"]: raise ValueError("Experience PNG changed")
            rgb = np.array(Image.open(path).convert("RGB"))
            if array_hash(rgb) != row["vision"]["frame_sha256"]: raise ValueError("Experience RGB changed")
            episode.append({**row, "rgb": rgb})
        if [r["decision"] for r in episode] != list(range(1, len(episode)+1)): raise ValueError("Experience must retain complete causal prefixes")
        episodes.append(episode)
    return episodes, {"frames": len(rows), "episodes": len(episodes),
                      "action_counts": {a: sum(r["action"] == a for r in rows) for a in ACTIONS},
                      "reward_targets": sum(r["reward_target_mix"] > 0 for r in rows), "reused_from": str(folder)}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def checkpoint_record(folder):
    folder = Path(folder).resolve()
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "completed" or digest(folder / "synapses.npz", "sha256") != report["synapses_sha256"]:
        raise ValueError("Incomplete or changed checkpoint")
    return {"path": str(folder), "report_sha256": digest(folder / "report.json", "sha256"),
            "synapses_sha256": report["synapses_sha256"], "encoder_mix": report.get("encoder_mix", 0.)}


def champion(root=ROOT):
    path = Path(root) / "registry.json"
    if not path.exists(): return checkpoint_record(INITIAL)
    record = json.loads(path.read_text())["champion"]
    if record != checkpoint_record(record["path"]): raise ValueError("Champion identity changed")
    return record


def promote(root, candidate, previous, gate):
    if not gate.get("accepted"): return False
    if champion(root) != previous: raise ValueError("Champion changed during training")
    registry = Path(root) / "registry.json"
    history = json.loads(registry.read_text()).get("history", []) if registry.exists() else []
    history.append({"previous": previous, "candidate": checkpoint_record(candidate), "gate": gate,
                    "utc": datetime.now(timezone.utc).isoformat()})
    atomic_json(registry, {"champion": checkpoint_record(candidate), "history": history})
    return True


def rollback(root=ROOT):
    root = Path(root)
    with CycleLock(root):
        path = root / "registry.json"
        data = json.loads(path.read_text())
        if not data.get("history"): raise ValueError("No previous promoted checkpoint")
        last = data["history"].pop()
        previous = checkpoint_record(last["previous"]["path"])
        if previous != last["previous"]: raise ValueError("Previous checkpoint changed")
        data["champion"] = previous
        data.setdefault("rollbacks", []).append(last)
        atomic_json(path, data)
        return data


class CycleLock:
    def __init__(self, root): self.path = Path(root) / "cycle.lock"
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="ascii")
        self.stream.write(str(os.getpid())); self.stream.flush()
        return self
    def __exit__(self, *args):
        self.stream.close(); self.path.unlink(missing_ok=True)


def paired_gate(before, after, initial_kl, final_kl):
    """A conservative engineering gate; finite paired results, no score regression."""
    keys = lambda rows: [(r["task"], r["seed"]) for r in rows]
    if not before or keys(before) != keys(after) or len(set(keys(before))) != len(before):
        raise ValueError("Benchmark pairs do not match")
    checks, deltas = {}, {}
    for task in sorted({r["task"] for r in before}):
        pairs = [(a, b) for a, b in zip(before, after) if a["task"] == task]
        deltas[task] = {metric: float(np.mean([b[metric] - a[metric] for a, b in pairs]))
                        for metric in ("return", "kills", "range_progress")}
        checks[task] = all(deltas[task][metric] >= -1e-7 for metric in ("return", "kills", "range_progress"))
    finite = all(np.isfinite(v) for d in deltas.values() for v in d.values()) and np.isfinite([initial_kl, final_kl]).all()
    improved = any(d["return"] > 0 or d["kills"] > 0 or d["range_progress"] > .01 for d in deltas.values())
    validation = final_kl <= initial_kl + 1e-6
    return {"accepted": bool(finite and all(checks.values()) and improved and validation),
            "paired_episodes": len(before), "task_deltas": deltas, "nonregression": checks,
            "validation_nonregression": bool(validation), "gameplay_improved": bool(improved),
            "rule": "Finite results; validation KL and every task's mean return, kills and range progress cannot regress; at least one gameplay metric must improve. Small-sample engineering gate, not statistical proof."}


def known_experience():
    seeds, hashes = set(), set()
    paths = set()
    for pattern in ("*/report.json", "*/plan.json", "*/opening-scan.json", "*/train/trace.json",
                    "*/validation/trace.json", "*/run-*/report.json", "learning/*/plan.json"):
        paths.update(Path("runs").glob(pattern))
    for path in sorted(paths):
        value = json.loads(path.read_text(encoding="utf-8"))
        seeds.update(collect_seeds(value)); hashes.update(image_hashes(value))
    return seeds, hashes


def reserve(counts, rng):
    seeds, hashes = known_experience()
    splits = {}
    # Test and promotion starts are fixed before any teacher labels or outcomes.
    game = make_game(1)
    try:
        for split in ("test", "gate", "validation", "train"):
            records = []
            for attempt in range(3000):
                seed = int(rng.integers(80_000_000, 2**31))
                if seed in seeds: continue
                task = "basic" if len(records) % 2 == 0 else "distance"
                game.set_seed(seed); game.new_episode()
                if task == "distance" and seed % 2 == 0:
                    for _ in range(10): act(game, 4, task)
                rgb_hash = array_hash(game.get_state().screen_buffer)
                if rgb_hash in hashes: continue
                hashes.add(rgb_hash); seeds.add(seed)
                records.append({"seed": seed, "task": task, "rgb_sha256": rgb_hash})
                if len(records) == counts[split]: break
            else: raise ValueError("Could not reserve distinct unobserved openings")
            splits[split] = records
    finally: game.close()
    return splits


def check_cancel(folder):
    if (Path(folder) / "cancel.request").exists(): raise InterruptedError("Cancellation requested")


def collect(controller, model, specs, folder, limit, rng, teacher, cancel):
    folder.mkdir()
    episodes, saved, counts, shaped = [], [], dict.fromkeys(ACTIONS, 0), 0
    for spec in specs:
        check_cancel(cancel)
        game = make_game(spec["seed"], spec["task"])
        episode, memory = [], Memory(); controller.reset()
        try:
            if array_hash(game.get_state().screen_buffer) != spec["rgb_sha256"]: raise ValueError("Opening changed")
            for step in range(limit):
                check_cancel(cancel)
                if game.is_episode_finished(): break
                rgb = game.get_state().screen_buffer.copy()
                label = teacher.predict(rgb)
                measured = controller.decide(rgb)
                if measured["voltage_min_mv"] < -90: raise ValueError("Collection voltage bound")
                with torch.inference_mode(): probabilities = model(torch.tensor(vector(controller, memory))[None]).softmax(1)[0].numpy()
                # Fixed exploration schedule guarantees that both translation
                # buttons are sampled without labeling every such action as good.
                action = [4, 5, 1, 2][step % 4] if step % 3 == 0 else int(probabilities.argmax())
                effect = act(game, action, spec["task"])
                target, alpha = learning_target(label["probabilities"], action, effect)
                shaped += int(alpha > 0); counts[ACTIONS[action]] += 1
                name = f"{spec['seed']}-{step+1:03d}.png"
                Image.fromarray(rgb).save(folder / name)
                row = {**spec, "decision": step+1, "action": ACTIONS[action], "image": name,
                       "png_sha256": digest(folder / name, "sha256"), "vision": label,
                       "teacher": {**label, "probabilities": target}, "reward_target_mix": alpha,
                       "effect": effect, "student_probabilities": probabilities.tolist()}
                saved.append(row); episode.append({**row, "rgb": rgb}); memory.advance(action)
            episodes.append(episode)
            atomic_json(folder / "trace.json", saved)
            print(f"Collected {folder.name}: {len(episodes)}/{len(specs)} episodes, {len(saved)} frames", flush=True)
        finally: game.close()
    return episodes, {"frames": len(saved), "episodes": len(episodes), "action_counts": counts, "reward_targets": shaped}


def benchmark(controller, model, specs, limit, cancel):
    rows = []
    for spec in specs:
        check_cancel(cancel)
        game = make_game(spec["seed"], spec["task"])
        controller.reset(); memory = Memory(); trace = []
        try:
            if array_hash(game.get_state().screen_buffer) != spec["rgb_sha256"]: raise ValueError("Benchmark opening changed")
            for step in range(limit):
                check_cancel(cancel)
                if game.is_episode_finished(): break
                rgb = game.get_state().screen_buffer.copy()
                d = controller.decide(rgb)
                if d["voltage_min_mv"] < -90: raise ValueError("Benchmark voltage bound")
                with torch.inference_mode(): p = model(torch.tensor(vector(controller, memory))[None]).softmax(1)[0].numpy()
                action = int(p.argmax()); effect = act(game, action, spec["task"]); memory.advance(action)
                trace.append({"action": ACTIONS[action], "probabilities": p.tolist(), "effect": effect,
                              "rgb_sha256": array_hash(rgb)})
            rows.append({**spec, "return": game.get_total_reward(), "kills": trace[-1]["effect"]["kills"],
                         "range_progress": sum(t["effect"]["range_progress"] for t in trace),
                         "decisions": len(trace), "terminal": game.is_episode_finished(), "trace": trace})
            print(f"Benchmark {len(rows)}/{len(specs)}: {spec['task']} return {rows[-1]['return']}", flush=True)
        finally: game.close()
    return rows


def run_cycle(root=ROOT, train_scenes=4, validation_scenes=2, gate_scenes=4, test_scenes=4,
              decisions=12, benchmark_decisions=32, proposals=2, encoder_mix=.2, experience=None):
    root = Path(root)
    if any(type(n) is not int or not 2 <= n <= 64 or n % 2 for n in (train_scenes, validation_scenes, gate_scenes, test_scenes)):
        raise ValueError("Split sizes must be even integers in [2,64]")
    if not 6 <= decisions <= 75 or not 6 <= benchmark_decisions <= 75 or not 1 <= proposals <= 16:
        raise ValueError("Invalid bounded cycle budget")
    torch.set_num_threads(2)
    with CycleLock(root):
        folder = root / datetime.now().strftime("cycle-%Y%m%d-%H%M%S-%f"); folder.mkdir()
        previous = champion(root)
        status = {"status": "running", "phase": "reserve", "output": str(folder), "champion": previous}
        def phase(name, **values):
            status.update(phase=name, **values); atomic_json(folder / "status.json", status)
            atomic_json(root / "latest.json", status); print(f"Learning phase: {name}", flush=True)
        try:
            phase("reserve")
            seed = secrets.randbits(32); rng = np.random.default_rng(seed)
            splits = reserve(dict(train=train_scenes, validation=validation_scenes, gate=gate_scenes, test=test_scenes), rng)
            if experience is not None:
                origin = json.loads((Path(experience) / "plan.json").read_text())
                if origin["champion"] != previous: raise ValueError("Recovery experience belongs to a different champion")
                for split in ("train", "validation"): splits[split] = origin["splits"][split]
            sources = [Path(__file__), Path("flydoom/learning_features.py"), Path("flydoom/learning_curriculum.py"),
                       Path("flydoom/simulation.py"), Path("flydoom/bridge.py"), Path("flydoom/synaptic.py"),
                       Path("flydoom/synaptic_eligibility.py"), Path("flydoom/movement_core.py"), Path("flydoom/movement_teacher.py")]
            import vizdoom
            assets = Path(vizdoom.__file__).parent
            sources += [assets / "scenarios/basic.cfg", assets / "scenarios/basic.wad", assets / "freedoom2.wad"]
            plan = {"schema": "automatic_synaptic_cycle_v1", "random_seed": seed, "splits": splits,
                    "decisions": decisions, "benchmark_decisions": benchmark_decisions, "proposals": proposals,
                    "encoder_candidate_mix": encoder_mix, "champion": previous,
                    "selection": "Training KL only; validation veto and independent paired promotion gate; test reported once after gate decision",
                    "reward": "Distance-band potential improvement, capped 0.35 target mix; engine geometry never enters the policy",
                    "optimizer": "Direct eligibility proposal plus antithetic full-graph recurrent rollout directions; hard spike/reset feedback included in finite differences, not exact BPTT",
                    "source_sha256": {str(p): digest(p, "sha256") for p in sources}}
            if experience is not None:
                plan["recovered_experience"] = {"path": str(experience), "sha256": {
                    str(Path(experience) / s / "trace.json"): digest(Path(experience) / s / "trace.json", "sha256") for s in ("train", "validation")}}
            atomic_json(folder / "plan.json", plan)
            ids, raw_controller, _, model, rows, identity = load_system()
            patch, parent_report = restore(raw_controller, ids, previous["path"], identity)
            controller = ContrastMotionController(raw_controller, previous["encoder_mix"])
            original_gains = patch.edge_gains.copy(); original_mix = controller.encoder_mix
            topology = (array_hash(patch.weights.indices), array_hash(patch.weights.indptr))
            mask = np.ones(patch.weights.nnz, bool); mask[patch.offsets] = False
            outside = array_hash(patch.weights.data[mask])
            phase("collect")
            if experience is None:
                teacher = MovementTeacher(folder / "teacher.log")
                try:
                    train_data, train_info = collect(controller, model, splits["train"], folder / "train", decisions, rng, teacher, folder)
                    val_data, val_info = collect(controller, model, splits["validation"], folder / "validation", decisions, rng, teacher, folder)
                    atomic_json(folder / "teacher.json", teacher.metadata)
                finally: teacher.close()
            else:
                train_data, train_info = load_experience(experience, "train")
                val_data, val_info = load_experience(experience, "validation")
            # Keep every frame in recurrent state evolution, but exclude exact
            # training duplicates from validation scoring. Never splice time.
            th = {r["teacher"]["frame_sha256"] for ep in train_data for r in ep}
            for ep in val_data:
                for row in ep: row["score"] = row["teacher"]["frame_sha256"] not in th
            vh = {r["teacher"]["frame_sha256"] for ep in val_data for r in ep}
            val_info["excluded_training_duplicate_frames"] = sum(not r["score"] for ep in val_data for r in ep)
            if sum(r["score"] for ep in val_data for r in ep) < 4: raise ValueError("Insufficient independent validation frames")
            phase("baseline", experience={"train": train_info, "validation": val_info})
            report = {"schema": SCHEMA, "status": "running", "student_sha256": identity["student_sha256"],
                      "calibration_sha256": digest(CALIBRATION, "sha256"), "base_weights_sha256": parent_report["base_weights_sha256"],
                      "group_labels": patch.labels, "group_edge_counts": np.bincount(patch.groups).tolist(),
                      "plastic_edges": len(patch.offsets), "total_edges": patch.weights.nnz, "readout_trained": False,
                      "teacher_trained": False, "biology_validated": False, "history": [],
                      "parameterization": "Individual bounded biological edge gains and a versioned image adapter",
                      "scope": "Bounded automatic learning pilot. A rejected candidate is retained for inspection, never deployed.",
                      "plan_sha256": digest(folder / "plan.json", "sha256")}
            report["initial"] = {"train": replay(controller, model, train_data), "validation": replay(controller, model, val_data)}
            best = report["initial"]["train"]["kl"]; best_gains = original_gains.copy(); best_mix = original_mix
            report["history"].append({"evaluation": 0, "kl": best, "gains": patch.gains.tolist(), "accepted": True, "method": "champion"})
            def evaluate(gains, mix, method):
                nonlocal best, best_gains, best_mix
                check_cancel(folder); patch.apply(np.clip(gains, .75, 1.25)); controller.encoder_mix = mix
                event = {"evaluation": len(report["history"]), "method": method, "gains": patch.gains.tolist(), "accepted": False}
                try:
                    metric = replay(controller, model, train_data); event["kl"] = metric["kl"]
                    if metric["kl"] < best:
                        best, best_gains, best_mix = metric["kl"], patch.edge_gains.copy(), mix; event["accepted"] = True
                except ValueError as error: event["rejected"] = str(error)
                report["history"].append(event); atomic_json(folder / "report.json", report)
                phase("optimize", evaluation=event)
                return event.get("kl", float("inf"))
            phase("encoder_ablation")
            evaluate(original_gains, encoder_mix, "contrast_motion_ablation")
            patch.apply(best_gains); controller.encoder_mix = best_mix
            phase("eligibility")
            derivative = gradient(controller, model, patch, train_data)
            nonzero = np.abs(derivative[derivative != 0])
            if len(nonzero):
                direction = -np.clip(derivative / max(float(np.quantile(nonzero, .95)), 1e-12), -1, 1)
                evaluate(best_gains + .0001 * direction, best_mix, "direct_eligibility")
            phase("recurrent_optimization")
            for k in range(proposals):
                center, mix = best_gains.copy(), best_mix
                # Random per-edge directions perturb the actual full recurrent
                # graph. Antithetic losses include indirect feedback and spikes.
                direction = rng.choice(np.array([-1., 1.], np.float32), len(center))
                radius = .0001 * (k + 1)
                plus = evaluate(center + radius * direction, mix, "recurrent_plus")
                minus = evaluate(center - radius * direction, mix, "recurrent_minus")
                report["history"][-1]["directional_derivative"] = (plus - minus) / (2 * radius) if np.isfinite([plus, minus]).all() else None
            patch.apply(best_gains); controller.encoder_mix = best_mix
            phase("validation")
            report["final"] = {"train": replay(controller, model, train_data), "validation": replay(controller, model, val_data)}
            report.update(encoder_mix=best_mix, gains=patch.gains.tolist(),
                          group_gain_ranges=[[float(best_gains[patch.groups == i].min()), float(best_gains[patch.groups == i].max())] for i in range(len(patch.labels))],
                          changed_edges=int(np.count_nonzero(patch.weights.data[patch.offsets] != patch.base)),
                          changed_from_champion=int(np.count_nonzero(best_gains != original_gains)),
                          trained_weights_sha256=array_hash(patch.weights.data))
            report["connectome_weights_trained"] = bool(report["changed_edges"])
            patch.save(folder / "synapses.npz", ids); report["synapses_sha256"] = digest(folder / "synapses.npz", "sha256")
            evaluations = {}
            for split in ("gate", "test"):
                phase(split)
                evaluations[split] = {}
                for name, gains, mix in (("champion", original_gains, original_mix), ("candidate", best_gains, best_mix)):
                    patch.apply(gains); controller.encoder_mix = mix
                    evaluations[split][name] = benchmark(controller, model, splits[split], benchmark_decisions, folder)
                    atomic_json(folder / "benchmarks.json", evaluations)
                if split == "gate":
                    gate = paired_gate(evaluations[split]["champion"], evaluations[split]["candidate"],
                                       report["initial"]["validation"]["kl"], report["final"]["validation"]["kl"])
                    atomic_json(folder / "gate.json", gate)
            # Test outcomes never alter the already recorded promotion decision.
            patch.apply(best_gains)
            if topology != (array_hash(patch.weights.indices), array_hash(patch.weights.indptr)) or outside != array_hash(patch.weights.data[mask]):
                raise ValueError("Graph topology or unselected weights changed")
            if not np.array_equal(np.sign(patch.base), np.sign(patch.weights.data[patch.offsets])): raise ValueError("Synaptic signs changed")
            if any(digest(p, "sha256") != h for p, h in plan["source_sha256"].items()): raise ValueError("Locked cycle source changed")
            benchmark_hashes = {t["rgb_sha256"] for pair in evaluations.values() for episodes in pair.values() for ep in episodes for t in ep["trace"]}
            overlap = len((th | vh) & benchmark_hashes)
            if overlap:
                gate["accepted"] = False; gate["trajectory_overlap_veto"] = overlap
            report.update(status="completed", gate=gate, experience=status["experience"],
                          topology_unchanged=True, signs_unchanged=True, outside_mask_unchanged=True,
                          development_gameplay={"base_graph": evaluations["gate"]["champion"], "trained_graph": evaluations["gate"]["candidate"]})
            atomic_json(folder / "report.json", report); atomic_json(folder / "gate.json", gate)
            promoted = promote(root, folder, previous, gate)
            phase("completed", status="completed", promoted=promoted, gate=gate, candidate=str(folder))
            del controller, raw_controller, patch, model, rows, train_data, val_data
            gc.collect()
            return status
        except BaseException as error:
            phase("cancelled" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
                  status="cancelled" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed", error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--train-scenes", type=int, default=4)
    parser.add_argument("--validation-scenes", type=int, default=2)
    parser.add_argument("--gate-scenes", type=int, default=4)
    parser.add_argument("--test-scenes", type=int, default=4)
    parser.add_argument("--decisions", type=int, default=12)
    parser.add_argument("--benchmark-decisions", type=int, default=32)
    parser.add_argument("--proposals", type=int, default=2)
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--experience", type=Path, help="Recover verified complete training/validation trajectories from an interrupted cycle")
    args = parser.parse_args()
    if args.rollback: print(json.dumps(rollback(args.root))); return
    if not 1 <= args.cycles <= 100: parser.error("cycles must be in [1,100]")
    for _ in range(args.cycles):
        run_cycle(args.root, args.train_scenes, args.validation_scenes, args.gate_scenes,
                  args.test_scenes, args.decisions, args.benchmark_decisions, args.proposals, experience=args.experience)


if __name__ == "__main__": main()
