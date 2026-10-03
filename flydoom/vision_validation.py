"""Lock and execute a larger paired evaluation without tuning or online learning."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.live_brain import load_checkpoint
from flydoom.replay import safe_path
from flydoom.vision_distill import evaluate_policy, read_json, signature, verified_parent


CONDITIONS = ("parent", "candidate", "candidate-disconnected")


def collect_seeds(value):
    """Find declared and executed seeds in existing local experiment metadata."""
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "seed" and type(item) is int:
                result.add(item)
            elif key.endswith("seeds") and isinstance(item, list):
                result.update(s for s in item if type(s) is int)
            result.update(collect_seeds(item))
    elif isinstance(value, list):
        for item in value:
            result.update(collect_seeds(item))
    return result


def fingerprint(paths):
    return {str(Path(p).resolve()): digest(p, "sha256") for p in paths}


def create_plan(candidate, parent, output, runs, calibration, data_dir):
    candidate = Path(candidate).resolve()
    report = verified_parent(candidate)
    parent = Path(parent).resolve()
    parent_report = verified_parent(parent)
    if report.get("parent_student_sha256") != parent_report["student_sha256"]:
        raise ValueError("Candidate was not distilled from the declared parent")
    if not report.get("vision_distillation"):
        raise ValueError("Require a Vision-distilled candidate")
    artifacts = []
    for split in ("train", "validation"):
        path = candidate / (split + ".npz")
        if digest(path, "sha256") != report["dataset_files"][split]:
            raise ValueError("Distillation dataset changed")
        artifacts.append(path)
    used, history = set(), []
    for pattern in ("report.json", "plan.json", "decisions.jsonl"):
        for path in sorted(Path(runs).rglob(pattern)):
            # Browser profiles and test fixtures are not experiment provenance.
            if any("profile" in p or "tests-" in p for p in path.parts):
                continue
            if path.suffix == ".jsonl":
                values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                values = read_json(path)
            found = collect_seeds(values)
            if found:
                used.update(found)
                history.append({"path": str(path), "sha256": digest(path, "sha256")})
    rng = np.random.default_rng(20261003)
    seeds = []
    while len(seeds) < 20:
        seed = int(rng.integers(900000, 9000000))
        if seed not in used and seed not in seeds:
            seeds.append(seed)
    import vizdoom
    package = Path(vizdoom.__file__).parent
    artifacts += [parent / "student.safetensors", parent / "report.json",
                  candidate / "student.safetensors", candidate / "report.json", Path(calibration)]
    artifacts += list(Path(__file__).parent.glob("*.py"))
    artifacts += [package / "scenarios/basic.cfg", package / "scenarios/basic.wad", package / "freedoom2.wad"]
    plan = {"schema": "vision_validation_plan_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "parent": str(parent.resolve()), "candidate": str(candidate),
            "calibration": str(Path(calibration).resolve()), "data_dir": str(Path(data_dir).resolve()),
            "evaluation_seeds": seeds, "max_decisions": 75, "conditions": list(CONDITIONS),
            "artifacts": fingerprint(artifacts), "prior_seed_count": len(used), "seed_audit_sources": history,
            "selection_rule": "No model selection or promotion; report all predeclared conditions and seeds",
            "primary": "Paired target-hit rate and mean return; all 20 seeds retained",
            "uncertainty": "Paired bootstrap over unique opening RGB groups, 10000 draws, RNG seed 401",
            "overlap_rule": "Report exact opening and trajectory RGB/input overlap with both distillation splits; never replace seeds after seeing outcomes",
            "scope": "One fixed trained checkpoint. Training-seed replication and retrained topology controls remain outstanding. Earlier parent-training scene independence is not established.",
            "versions": {"vizdoom": vizdoom.__version__, "numpy": np.__version__}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(digest(output / "plan.json", "sha256") + "\n", encoding="ascii")
    return plan


def verify_plan(path):
    path = Path(path)
    if digest(path, "sha256") != path.with_suffix(".sha256").read_text(encoding="ascii").strip():
        raise ValueError("Locked plan checksum changed")
    plan = read_json(path)
    seeds = plan["evaluation_seeds"]
    if (plan.get("schema") != "vision_validation_plan_v1" or plan["conditions"] != list(CONDITIONS)
            or len(seeds) != 20 or len(set(seeds)) != 20
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or plan["max_decisions"] != 75):
        raise ValueError("Invalid locked evaluation protocol")
    for file, checksum in plan["artifacts"].items():
        if digest(file, "sha256") != checksum:
            raise ValueError(f"Locked artifact changed: {file}")
    return plan


def paired_summary(reference, other, openings):
    """Bootstrap paired opening groups, retaining within-group episode counts."""
    if ([e["seed"] for e in reference] != [e["seed"] for e in other]
            or len(openings) != len(reference) or not reference):
        raise ValueError("Paired episode identities differ")
    delta = np.array([[int(b["kills"] > 0) - int(a["kills"] > 0), b["return"] - a["return"]]
                      for a, b in zip(reference, other)], dtype=float)
    groups = [np.flatnonzero(np.asarray(openings) == h) for h in dict.fromkeys(openings)]
    totals = np.array([delta[g].sum(axis=0) for g in groups])
    counts = np.array([len(g) for g in groups])
    draws = np.random.default_rng(401).integers(0, len(groups), size=(10000, len(groups)))
    values = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    bounds = np.quantile(values, [.025, .975], axis=0)
    return {"hit_rate_difference": float(delta[:, 0].mean()), "mean_return_difference": float(delta[:, 1].mean()),
            "hit_rate_interval95": bounds[:, 0].tolist(), "return_interval95": bounds[:, 1].tolist(),
            "unique_opening_groups": len(groups), "episodes": len(reference),
            "note": "Descriptive paired cluster bootstrap; identical openings are grouped. Different images do not establish independent scenes or unseen parent training."}


def overlap_audit(output, candidate, report):
    known = {}
    for split in ("train", "validation"):
        with np.load(Path(candidate) / (split + ".npz"), allow_pickle=False) as data:
            known[split] = (set(data["rgb_hashes"]), {signature(x) for x in data["features"]})
    result = {split: {"opening_seeds": [e["seed"] for e, h in zip(report["episodes"], report["opening_hashes"]) if h in images],
                      "rgb_decisions": 0, "input_decisions": 0, "trajectory_seeds": []}
              for split, (images, _) in known.items()}
    for line in (Path(output) / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        artifact = safe_path(output, row["observation"] + ".npz")
        if digest(artifact, "sha256") != row["observation_sha256"]:
            raise ValueError("Evaluation observation changed")
        with np.load(artifact, allow_pickle=False) as data:
            features = signature(data["features"])
        with Image.open(safe_path(output, row["playback_frames"][0])) as frame:
            rgb = hashlib.sha256(np.asarray(frame.convert("RGB")).tobytes()).hexdigest()
        for split, (images, inputs) in known.items():
            found_image, found_input = rgb in images, features in inputs
            result[split]["rgb_decisions"] += int(found_image)
            result[split]["input_decisions"] += int(found_input)
            if (found_image or found_input) and row["seed"] not in result[split]["trajectory_seeds"]:
                result[split]["trajectory_seeds"].append(row["seed"])
    return result


def run(plan_path, output):
    plan = verify_plan(plan_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "vision_validation_v1", "status": "running", "plan_sha256": digest(plan_path, "sha256"),
              "plan": plan, "results": {}, "laya_used_during_play": False, "training_during_run": False,
              "default_checkpoint_changed": False}
    write_json(output / "report.json", report)
    try:
        _, controller, model, parent = load_checkpoint(plan["parent"], plan["calibration"], plan["data_dir"])
        candidate = verified_parent(plan["candidate"])
        if candidate["parent_student_sha256"] != parent["student_sha256"]:
            raise ValueError("Evaluation checkpoint lineage differs")
        for name in CONDITIONS:
            checkpoint = plan["parent"] if name == "parent" else plan["candidate"]
            identity = parent if name == "parent" else candidate
            model.load_state_dict(load_file(str(Path(checkpoint) / "student.safetensors")))
            model.eval()
            controller.control = "disconnected" if name.endswith("disconnected") else "connected"
            folder = output / ("run-" + name)
            measured = evaluate_policy(controller, model, identity, plan["evaluation_seeds"], folder, plan["max_decisions"])
            if name != "parent" and measured["opening_hashes"] != report["results"]["parent"]["opening_hashes"]:
                raise ValueError("Native game openings differ between paired conditions")
            measured["overlap"] = overlap_audit(folder, plan["candidate"], measured)
            episodes = measured["episodes"]
            measured["scores"] = {"hits": sum(e["kills"] > 0 for e in episodes), "episodes": len(episodes),
                                  "mean_return": float(np.mean([e["return"] for e in episodes])),
                                  "total_decisions": sum(e["decisions"] for e in episodes)}
            report["results"][name] = measured
            write_json(output / "report.json", report)
        results = report["results"]
        openings = results["parent"]["opening_hashes"]
        report["paired"] = {"candidate_minus_parent": paired_summary(results["parent"]["episodes"], results["candidate"]["episodes"], openings),
                            "connected_minus_disconnected": paired_summary(results["candidate-disconnected"]["episodes"], results["candidate"]["episodes"], openings)}
        report["unique_openings"] = len(set(openings))
        report["opening_groups"] = [{"sha256": h, "seeds": [s for s, image in zip(plan["evaluation_seeds"], openings) if image == h]}
                                    for h in dict.fromkeys(openings)]
        verify_plan(plan_path)
        report["status"] = "completed"
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan")
    planning.add_argument("--candidate", type=Path, default=Path("runs/fly-student-vision-v1"))
    planning.add_argument("--parent", type=Path, default=Path("runs/fly-student-memory-v1"))
    planning.add_argument("--output", type=Path, required=True)
    planning.add_argument("--runs", type=Path, default=Path("runs"))
    planning.add_argument("--calibration", type=Path, default=Path("runs/calibration-training-v1/report.json"))
    planning.add_argument("--data-dir", type=Path, default=Path("data/processed/fafb783"))
    executing = commands.add_parser("run")
    executing.add_argument("--plan", dest="plan_path", type=Path, required=True)
    executing.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = create_plan(**args) if command == "plan" else run(**args)
    print(json.dumps({k: result[k] for k in ("status", "evaluation_seeds", "paired") if k in result}, indent=2))


if __name__ == "__main__":
    main()
