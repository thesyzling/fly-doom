"""Evaluate all recovery repetitions on their reserved native-game openings."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.live_brain import load_checkpoint
from flydoom.vision_distill import evaluate_policy, read_json, verified_parent
from flydoom.vision_recovery import TRAINING_SEEDS, verify_study
from flydoom.vision_validation import fingerprint, overlap_audit, paired_summary


CONDITIONS = ("parent", "seed-101", "seed-202", "seed-303")


def make_plan(study, output):
    study, output = Path(study), Path(output)
    teaching = verify_study(study)
    summary = read_json(study / "report.json")
    if summary.get("status") != "completed" or [r["seed"] for r in summary["repetitions"]] != list(TRAINING_SEEDS):
        raise ValueError("Require the completed three-repetition teaching study")
    conditions = [{"name": "parent", "checkpoint": teaching["parent"]}]
    artifacts = [study / "plan.json", study / "plan.sha256", study / "report.json", Path(teaching["calibration"])]
    for repetition in summary["repetitions"]:
        path = Path(repetition["checkpoint"])
        identity = verified_parent(path)
        training_plan = study / f"training-plan-{repetition['seed']}.json"
        if (identity["student_sha256"] != repetition["sha256"]
                or identity["vision_distillation"]["plan_sha256"] != digest(training_plan, "sha256")):
            raise ValueError("Recovery checkpoint or training plan differs")
        conditions.append({"name": f"seed-{repetition['seed']}", "checkpoint": str(path.resolve())})
        artifacts.append(training_plan)
    parent = verified_parent(teaching["parent"])
    reference_parts = None
    for condition in conditions:
        path = Path(condition["checkpoint"])
        identity = verified_parent(path)
        if condition["name"] != "parent" and identity["parent_student_sha256"] != parent["student_sha256"]:
            raise ValueError("Recovery checkpoint has a different parent")
        artifacts += [path / "report.json", path / "student.safetensors"]
        parts = {}
        for split in ("train", "validation"):
            file = path / (split + ".npz")
            if digest(file, "sha256") != identity["dataset_files"][split]:
                raise ValueError("Checkpoint teaching dataset changed")
            artifacts.append(file)
            if condition["name"] != "parent":
                with np.load(file, allow_pickle=False) as data:
                    parts[split] = {k: data[k] for k in data.files}
        if condition["name"] != "parent":
            if reference_parts is None:
                reference_parts = parts
            else:
                for split in parts:
                    for key in parts[split]:
                        np.testing.assert_array_equal(parts[split][key], reference_parts[split][key])
    artifacts += list(Path(__file__).parent.glob("*.py"))
    plan = {"schema": "vision_recovery_gameplan_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "study": str(study.resolve()), "conditions": conditions,
            "openings": teaching["splits"]["evaluation"], "max_decisions": 75,
            "calibration": teaching["calibration"], "data_dir": teaching["data_dir"],
            "artifact_sha256": fingerprint(artifacts),
            "primary": "Report target hits, mean return and total decisions for every condition on all 20 starts",
            "selection": "All three preselected checkpoints reported; no gameplay-based checkpoint selection or automatic promotion",
            "uncertainty": "Paired opening-group bootstrap per repetition, 10000 draws, RNG seed 401",
            "overlap": "Report full trajectory RGB/input matches against recovery and earlier Vision teaching; retain all planned episodes",
            "scope": "Three continuation seeds share a warm start and evaluation scenes. Twenty distinct opening images; broader scene independence is not established."}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(digest(output / "plan.json", "sha256") + "\n", encoding="ascii")
    return plan


def verify_plan(path):
    path = Path(path)
    if digest(path, "sha256") != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Recovery gameplay plan changed")
    plan = read_json(path)
    openings = plan["openings"]
    if (plan.get("schema") != "vision_recovery_gameplan_v1"
            or tuple(c["name"] for c in plan["conditions"]) != CONDITIONS
            or plan["max_decisions"] != 75 or len(openings) != 20
            or len({r["seed"] for r in openings}) != 20
            or len({r["rgb_sha256"] for r in openings}) != 20
            or any(type(r["seed"]) is not int or not 0 <= r["seed"] < 2**32 for r in openings)):
        raise ValueError("Invalid recovery gameplay protocol")
    for file, checksum in plan["artifact_sha256"].items():
        if digest(file, "sha256") != checksum:
            raise ValueError(f"Recovery gameplay artifact changed: {file}")
    return plan


def score(episodes):
    return {"hits": sum(e["kills"] > 0 for e in episodes), "episodes": len(episodes),
            "mean_return": float(np.mean([e["return"] for e in episodes])),
            "decisions": sum(e["decisions"] for e in episodes),
            "action_counts": {a: sum(e["action_counts"][a] for e in episodes)
                              for a in ("WAIT", "MOVE_LEFT", "MOVE_RIGHT", "ATTACK")}}


def run(plan_path, output):
    plan = verify_plan(plan_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "vision_recovery_gameplay_v1", "status": "running", "plan": plan,
              "plan_sha256": digest(plan_path, "sha256"), "results": {}, "paired": {},
              "training_during_run": False, "laya_used_during_play": False,
              "default_checkpoint_changed": False}
    write_json(output / "report.json", report)
    try:
        parent_path = plan["conditions"][0]["checkpoint"]
        _, controller, model, parent = load_checkpoint(parent_path, plan["calibration"], plan["data_dir"])
        expected = [r["rgb_sha256"] for r in plan["openings"]]
        seeds = [r["seed"] for r in plan["openings"]]
        recovery_data = plan["conditions"][1]["checkpoint"]
        for condition in plan["conditions"]:
            name, checkpoint = condition["name"], Path(condition["checkpoint"])
            identity = verified_parent(checkpoint)
            for key in ("input_size", "hidden", "output_root_ids", "calibration_sha256"):
                if identity[key] != parent[key]:
                    raise ValueError("Paired policy feature mapping differs")
            model.load_state_dict(load_file(str(checkpoint / "student.safetensors")))
            model.eval()
            folder = output / ("run-" + name)
            measured = evaluate_policy(controller, model, identity, seeds, folder, plan["max_decisions"])
            if measured["opening_hashes"] != expected or [e["seed"] for e in measured["episodes"]] != seeds:
                raise ValueError("Actual game openings differ from the reserved scan")
            measured["overlap_recovery"] = overlap_audit(folder, recovery_data, measured)
            measured["overlap_earlier_vision"] = overlap_audit(folder, parent_path, measured)
            measured["scores"] = score(measured["episodes"])
            report["results"][name] = measured
            if name != "parent":
                report["paired"][name] = paired_summary(report["results"]["parent"]["episodes"], measured["episodes"], expected)
            write_json(output / "report.json", report)
        candidates = [report["results"][name]["scores"] for name in CONDITIONS[1:]]
        report["repetition_summary"] = {"mean_hits": float(np.mean([r["hits"] for r in candidates])),
            "sample_sd_hits": float(np.std([r["hits"] for r in candidates], ddof=1)),
            "mean_return": float(np.mean([r["mean_return"] for r in candidates])),
            "sample_sd_mean_return": float(np.std([r["mean_return"] for r in candidates], ddof=1)),
            "note": "Variation across three continuation seeds on the same openings; these are not 60 independent test scenes."}
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
    planning.add_argument("--study", type=Path, default=Path("runs/vision-recovery-v1"))
    planning.add_argument("--output", type=Path, required=True)
    executing = commands.add_parser("run")
    executing.add_argument("--plan", dest="plan_path", type=Path, required=True)
    executing.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = make_plan(**args) if command == "plan" else run(**args)
    if command == "run":
        result = {"status": result["status"], "scores": {k: v["scores"] for k, v in result["results"].items()}}
    else:
        result = {"conditions": [r["name"] for r in result["conditions"]], "openings": len(result["openings"])}
    print(result)


if __name__ == "__main__":
    main()
