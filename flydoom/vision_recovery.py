"""Collect diverse student trajectories and repeat offline Vision teaching."""

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.learning_data import make_game
from flydoom.live_brain import load_checkpoint
from flydoom.research import ResearchSession
from flydoom.vision import VisionClient
from flydoom.vision_distill import read_json, train, verified_parent
from flydoom.vision_validation import collect_seeds, fingerprint


SPLIT_SIZES = {"train": 12, "validation": 6, "evaluation": 20}
TRAINING_SEEDS = (101, 202, 303)


def select_openings(rows, excluded_images, excluded_seeds, sizes=SPLIT_SIZES):
    """Allocate unique opening images before seeing actions, labels or rewards."""
    seen_images, seen_seeds = set(excluded_images), set(excluded_seeds)
    selected = {split: [] for split in sizes}
    for row in rows:
        if row["rgb_sha256"] in seen_images or row["seed"] in seen_seeds:
            continue
        split = next((name for name, size in sizes.items() if len(selected[name]) < size), None)
        if split is None:
            break
        selected[split].append(row)
        seen_images.add(row["rgb_sha256"])
        seen_seeds.add(row["seed"])
    return selected


def plan_study(parent, output, calibration, data_dir, prior_evaluation):
    parent, output = Path(parent), Path(output)
    identity = verified_parent(parent)
    prior = read_json(prior_evaluation)
    if prior.get("status") != "completed":
        raise ValueError("Prior evaluation must be complete")
    excluded, artifacts = set(), [parent / "report.json", parent / "student.safetensors", Path(calibration), Path(prior_evaluation)]
    for split in ("train", "validation"):
        path = parent / (split + ".npz")
        if digest(path, "sha256") != identity["dataset_files"][split]:
            raise ValueError("Parent teaching data changed")
        artifacts.append(path)
        with np.load(path, allow_pickle=False) as data:
            excluded.update(str(h) for h in data["rgb_hashes"])
    for result in prior["results"].values():
        excluded.update(result["opening_hashes"])
    used = set()
    for pattern in ("report.json", "plan.json"):
        for path in Path("runs").rglob(pattern):
            if any("profile" in p or "tests-" in p for p in path.parts):
                continue
            used.update(collect_seeds(read_json(path)))
    output.mkdir(parents=True, exist_ok=False)
    game = None
    rows = []
    try:
        game, _ = make_game(11000000, False)
        rng = np.random.default_rng(902)
        selected = None
        for seed in rng.choice(np.arange(11000000, 11010000), 400, replace=False):
            seed = int(seed)
            if seed in used:
                continue
            game.set_seed(seed)
            game.new_episode()
            frame = game.get_state().screen_buffer
            rows.append({"seed": seed, "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
            selected = select_openings(rows, excluded, used)
            if all(len(selected[s]) == n for s, n in SPLIT_SIZES.items()):
                break
        else:
            raise ValueError("Not enough unique openings; do not silently weaken scene separation")
    finally:
        if game:
            game.close()
        write_json(output / "opening-scan.json", {"rows": rows, "excluded_rgb_hashes": sorted(excluded),
                   "scope": "Opening images only; no policy decisions, teacher labels or rewards used for selection"})
    import vizdoom
    package = Path(vizdoom.__file__).parent
    artifacts += list(Path(__file__).parent.glob("*.py"))
    artifacts += [output / "opening-scan.json", package / "scenarios/basic.cfg", package / "scenarios/basic.wad", package / "freedoom2.wad"]
    result = {"schema": "vision_recovery_plan_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "parent": str(parent.resolve()), "calibration": str(Path(calibration).resolve()),
              "data_dir": str(Path(data_dir).resolve()), "splits": selected,
              "training_seeds": list(TRAINING_SEEDS), "epochs": 100, "learning_rate": .0001,
              "collection_decisions": 24, "alpha": 0., "artifact_sha256": fingerprint(artifacts),
              "collection_rule": "Student controls all actions; pinned Vision labels the same causal observation. No optimizer during collection.",
              "selection_rule": "Lowest validation KL per repetition, including epoch zero; report all three repetitions; no automatic gameplay promotion",
              "evaluation_rule": "Twenty opening-disjoint seeds reserved; no gameplay evaluation or outcome-based selection in this teaching command",
              "scope": "Distinct opening pixels, not guaranteed distinct complete trajectories or independence from earlier parent training. Same warm-start weights; independent minibatch-order seeds."}
    write_json(output / "plan.json", result)
    (output / "plan.sha256").write_text(digest(output / "plan.json", "sha256") + "\n", encoding="ascii")
    return result


def verify_study(study):
    study = Path(study)
    if digest(study / "plan.json", "sha256") != (study / "plan.sha256").read_text().strip():
        raise ValueError("Recovery plan changed")
    plan = read_json(study / "plan.json")
    if (plan.get("schema") != "vision_recovery_plan_v1" or plan["training_seeds"] != list(TRAINING_SEEDS)
            or plan["alpha"] != 0 or plan["collection_decisions"] != 24):
        raise ValueError("Unsupported recovery protocol")
    rows = [r for split in plan["splits"].values() for r in split]
    if (any(len(plan["splits"][s]) != n for s, n in SPLIT_SIZES.items())
            or len({r["rgb_sha256"] for r in rows}) != len(rows)
            or len({r["seed"] for r in rows}) != len(rows)):
        raise ValueError("Recovery opening splits overlap")
    for path, checksum in plan["artifact_sha256"].items():
        if digest(path, "sha256") != checksum:
            raise ValueError(f"Recovery artifact changed: {path}")
    return plan


def verify_collected_openings(directory, expected):
    rows = [json.loads(line) for line in (Path(directory) / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    actual = {r["seed"]: r["vision"]["frame_sha256"] for r in rows if r["decision"] == 1}
    if actual != {r["seed"]: r["rgb_sha256"] for r in expected}:
        raise ValueError("Collected openings differ from the locked scan")


def execute(study, checkpoint_prefix):
    study = Path(study)
    plan = verify_study(study)
    if (study / "report.json").exists():
        raise FileExistsError("This study already ran; preserve its outcomes")
    outputs = [Path(str(checkpoint_prefix) + f"-seed-{seed}") for seed in TRAINING_SEEDS]
    if any(path.exists() for path in outputs):
        raise FileExistsError("A repetition checkpoint already exists")
    report = {"schema": "vision_recovery_study_v1", "status": "collecting", "collection": {}, "repetitions": [],
              "plan_sha256": digest(study / "plan.json", "sha256"), "gameplay_evaluated": False,
              "default_checkpoint_changed": False, "biological_weights_trained": False}
    write_json(study / "report.json", report)
    vision = None
    try:
        _, controller, model, parent = load_checkpoint(plan["parent"], plan["calibration"], plan["data_dir"])
        catalog = SimpleNamespace(controller=controller, model=model)
        vision = VisionClient(log=study / "vision-worker.log")
        for split in ("train", "validation"):
            seeds = [r["seed"] for r in plan["splits"][split]]
            session = ResearchSession(catalog, study / ("run-" + split), parent, vision,
                                      alpha=0., seeds=seeds, seed=seeds[0], max_decisions=24,
                                      feedback_split=split, autoplay=True)
            # Batch teaching needs no retained browser snapshots of the full graph.
            session.samples = deque(maxlen=2)
            session.run()
            if session.report["status"] != "completed" or session.report.get("replay_error"):
                raise RuntimeError(f"Collection failed: {session.report}")
            verify_collected_openings(session.output, plan["splits"][split])
            report["collection"][split] = str(session.output)
            write_json(study / "report.json", report)
        vision.close()
        vision = None
        report["status"] = "training"
        write_json(study / "report.json", report)
        for seed, output in zip(TRAINING_SEEDS, outputs):
            training_plan = {"schema": "vision_distillation_plan_v1", "status": "completed", "parent": plan["parent"],
                "epochs": plan["epochs"], "learning_rate": plan["learning_rate"], "seed": seed,
                "train_runs": [report["collection"]["train"]], "validation_runs": [report["collection"]["validation"]],
                **{split + "_seeds": [r["seed"] for r in rows] for split, rows in plan["splits"].items()},
                "max_decisions": 75, "selection": plan["selection_rule"], "promotion": "No automatic promotion",
                "recovery_plan_sha256": report["plan_sha256"]}
            path = study / f"training-plan-{seed}.json"
            write_json(path, training_plan)
            trained = train(path, output)
            report["repetitions"].append({"seed": seed, "checkpoint": str(output), "sha256": trained["student_sha256"],
                "selected_epoch": trained["selected_epoch"], "initial": trained["initial"], "final": trained["final"],
                "validation_duplicates_removed": trained["validation_duplicates_removed"]})
            write_json(study / "report.json", report)
        verify_study(study)
        report["status"] = "completed"
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        if vision:
            vision.close()
        write_json(study / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan")
    planning.add_argument("--parent", type=Path, default=Path("runs/fly-student-vision-v1"))
    planning.add_argument("--output", type=Path, required=True)
    planning.add_argument("--calibration", type=Path, default=Path("runs/calibration-training-v1/report.json"))
    planning.add_argument("--data-dir", type=Path, default=Path("data/processed/fafb783"))
    planning.add_argument("--prior-evaluation", type=Path, default=Path("runs/vision-validation-v1/report.json"))
    running = commands.add_parser("run")
    running.add_argument("--study", type=Path, required=True)
    running.add_argument("--checkpoint-prefix", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = plan_study(**args) if command == "plan" else execute(**args)
    print(json.dumps({k: result[k] for k in ("status", "splits", "repetitions") if k in result}, indent=2))


if __name__ == "__main__":
    main()
