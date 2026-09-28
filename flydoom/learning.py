"""Prepare, record, train, and inspect the staged local Laya learning pipeline."""

import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from flydoom.calibration import write_json
from flydoom.learning_data import image_state, prepare, record


DEFAULT_CALIBRATION = Path("runs/calibration-training-v1/report.json")
DEFAULT_BASE = Path("models/laya-base")


def doctor(base, calibration, output):
    import torch
    import torch.nn.functional as F
    from flydoom.laya_teacher import batch_logits, encode_states, load_base
    from flydoom.student import SpikingReadout
    torch.set_num_threads(4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"versions": {name: version(name) for name in ("laya", "torch", "transformers", "vizdoom")},
              "cuda_available": torch.cuda.is_available(), "doom_policy_trained": False}
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
    agent, provenance = load_base(base)
    report["laya_revision"] = provenance["revision"]
    for name, parameter in agent.model.named_parameters():
        parameter.requires_grad_(name.startswith(("head.", "scorer.", "type_emb.")))
    states = [image_state(np.full((8, 8, 3), value, dtype=np.uint8), 0) for value in (0, 255)]
    items = encode_states(agent, states)
    before = agent.model.scorer[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in agent.model.parameters() if p.requires_grad], lr=2e-5)
    started = perf_counter()
    logits = batch_logits(agent, items)
    report["probabilities"] = logits.detach().softmax(-1).cpu().tolist()
    # Synthetic labels exercise backpropagation only. Nothing from this model is saved for play.
    loss = F.cross_entropy(logits, torch.tensor([0, 3], device=agent.device))
    loss.backward()
    optimizer.step()
    report["laya_head_update_verified"] = not torch.equal(before, agent.model.scorer[-1].weight)
    report["laya_smoke_seconds"] = perf_counter() - started
    report["laya_smoke_loss"] = float(loss.detach())
    report["synthetic_smoke_not_training_data"] = True
    if torch.cuda.is_available():
        report["peak_torch_gpu_mib"] = torch.cuda.max_memory_allocated() / 2**20
    model = SpikingReadout(12)
    x = torch.randn(8, 12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.input.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), torch.arange(8) % 4)
        loss.backward()
        optimizer.step()
    report["spiking_group_update_verified"] = not torch.equal(before, model.input.weight)
    calibration_report = json.loads(Path(calibration).read_text(encoding="utf-8"))
    report["engineering_calibration_passed"] = calibration_report["engineering_gate_passed"]
    report["setup_passed"] = all(report[key] for key in ("laya_head_update_verified", "spiking_group_update_verified", "engineering_calibration_passed"))
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def run_pipeline(output, *, episodes=10, base=DEFAULT_BASE, calibration=DEFAULT_CALIBRATION,
                 teacher_epochs=5, student_epochs=50):
    from flydoom import laya_teacher, student
    if not 5 <= episodes <= 100 or not 1 <= teacher_epochs <= 100 or not 1 <= student_epochs <= 1000:
        raise ValueError("Require 5..100 episodes, 1..100 teacher epochs and 1..1000 student epochs")
    # Check existing artifacts before asking the player to spend time recording.
    calibration_report = json.loads(Path(calibration).read_text(encoding="utf-8"))
    if not calibration_report.get("engineering_gate_passed"):
        raise ValueError("Run engineering calibration before recording for this pipeline")
    from flydoom.data import digest
    if calibration_report["simulation_sha256"] != digest(Path(__file__).with_name("simulation.py"), "sha256"):
        raise ValueError("Simulation changed; rerun calibration first")
    provenance = json.loads((Path(base) / "download.json").read_text(encoding="utf-8"))
    for name, expected in provenance["sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or digest(Path(base) / name, "sha256") != expected:
            raise ValueError("Laya model checksum mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "recording", "root": str(output)}
    write_json(output / "pipeline.json", report)
    recording = record(output / "demonstrations", episodes=episodes)
    if recording["status"] != "completed":
        report["status"] = "recording_interrupted"
        write_json(output / "pipeline.json", report)
        return
    report["status"] = "preparing_neural_features"
    write_json(output / "pipeline.json", report)
    prepare(output / "demonstrations", calibration, output / "prepared")
    report["status"] = "training_laya"
    write_json(output / "pipeline.json", report)
    teacher_report = laya_teacher.train(output / "prepared", base, output / "teacher", epochs=teacher_epochs)
    if not teacher_report["teacher_accepted"]:
        report["status"] = "teacher_gate_failed"
        write_json(output / "pipeline.json", report)
        print("Laya did not beat the baselines and image-shuffle control. No student was trained. Inspect teacher/report.json.", flush=True)
        return
    report["status"] = "training_student"
    write_json(output / "pipeline.json", report)
    student.train(output / "prepared", base, output / "teacher", output / "student", epochs=student_epochs)
    report["status"] = "ready_for_game_evaluation"
    report["play_command"] = f'.venv/Scripts/python.exe -m flydoom.learning play --checkpoint "{output / "student"}" --calibration "{calibration}"'
    write_json(output / "pipeline.json", report)
    print(report["play_command"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="Record your play, prepare features, train Laya, then distill only if the teacher passes")
    p.add_argument("--output", type=Path)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    p.add_argument("--teacher-epochs", type=int, default=5)
    p.add_argument("--student-epochs", type=int, default=50)
    p = sub.add_parser("record", help="Record real keyboard demonstrations; no model required")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max-decisions", type=int, default=75)
    p.add_argument("--random-smoke", action="store_true", help="Headless infrastructure test; explicitly ineligible for training")
    p = sub.add_parser("prepare", help="Replay saved frames through the calibrated frozen connectome")
    p.add_argument("--recording", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed/fafb783"))
    p.add_argument("--allow-smoke", action="store_true", help="Extract infrastructure-test features, still ineligible for training")
    p = sub.add_parser("teacher", help="Adapt the local Laya decision head using human demonstrations")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p = sub.add_parser("student", help="Train extra spiking cells from an accepted Laya teacher")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("play", help="Watch the trained extra spiking cells control Doom without Laya")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed/fafb783"))
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--max-decisions", type=int, default=24)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--disconnected", action="store_true")
    p = sub.add_parser("doctor", help="Verify local Laya inference and gradient updates using synthetic inputs only")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    p.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command in {"run", "play"} and args["output"] is None:
        args["output"] = Path("runs") / datetime.now().strftime(f"learning-{command}-%Y%m%d-%H%M%S-%f")
    try:
        if command == "run":
            run_pipeline(**args)
        elif command == "record":
            record(**args)
        elif command == "prepare":
            prepare(**args)
        elif command == "doctor":
            doctor(**args)
        elif command == "teacher":
            from flydoom.laya_teacher import train
            train(**args)
        elif command == "student":
            from flydoom.student import train
            train(**args)
        elif command == "play":
            from flydoom.student import play
            args["visible"] = not args.pop("headless")
            play(**args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("Stopped. Completed artifacts remain in the selected output directory.")


if __name__ == "__main__":
    main()
