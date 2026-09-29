"""Train an additional 64-cell spiking readout on a frozen fly connectome."""

import json
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from flydoom.calibration import load_calibrated, output_features, write_json
from flydoom.data import digest
from flydoom.learning_data import ACTIONS, make_game, read_prepared
from flydoom.laya_teacher import encode_states, load_base, metrics, probabilities, restore_head


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (voltage,) = ctx.saved_tensors
        return gradient / (1 + 5 * voltage.abs()).square()


class SpikingReadout(nn.Module):
    """Dimensionless LIF-like cells; eight internal steps per game decision.

    State resets for each decision. The frozen connectome itself retains state
    across decisions. These added cells are engineering units, not fly cells.
    """

    def __init__(self, input_size, hidden=64):
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_size))
        self.register_buffer("scale", torch.ones(input_size))
        self.input = nn.Linear(input_size, hidden)
        self.recurrent = nn.Linear(hidden, hidden, bias=False)
        self.readout = nn.Linear(hidden, 4)
        nn.init.normal_(self.recurrent.weight, std=0.01)
        nn.init.constant_(self.input.bias, 0.2)

    def forward(self, x):
        drive = self.input((x - self.mean) / self.scale)
        voltage = torch.zeros_like(drive)
        spikes = torch.zeros_like(drive)
        total = torch.zeros_like(drive)
        for _ in range(8):
            voltage = 0.8 * voltage + drive + 0.1 * torch.tanh(self.recurrent(spikes))
            spikes = SurrogateSpike.apply(voltage - 1.0)
            voltage = voltage * (1 - spikes.detach())
            total = total + spikes
        return self.readout(total / 8)


def train(dataset, base, teacher, output, *, epochs=50, seed=7, hidden=64,
          experimental_teacher=False, evaluate_test=True):
    if not 1 <= epochs <= 1000 or not 4 <= hidden <= 512:
        raise ValueError("Require 1..1000 epochs and 4..512 additional cells")
    source, parts = read_prepared(dataset)
    teacher = Path(teacher)
    teacher_report = json.loads((teacher / "report.json").read_text(encoding="utf-8"))
    if teacher_report.get("status") != "completed":
        raise ValueError("Teacher checkpoint is not complete")
    if not teacher_report.get("teacher_accepted") and not experimental_teacher:
        raise ValueError("Laya did not pass the teacher gate. Collect better demonstrations before distillation.")
    if Path(output).exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    dataset_hash = digest(Path(dataset) / "manifest.json", "sha256")
    if teacher_report["dataset_sha256"] != dataset_hash or digest(teacher / "head.safetensors", "sha256") != teacher_report["head_sha256"]:
        raise ValueError("Teacher/data provenance mismatch")
    agent, provenance = load_base(base)
    if provenance != teacher_report["base"]:
        raise ValueError("Teacher belongs to a different Laya base")
    restore_head(agent, teacher / "head.safetensors")
    targets = probabilities(agent, encode_states(agent, parts["train"]["states"]))
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpikingReadout(parts["train"]["features"].shape[1], hidden)
    with torch.no_grad():
        model.mean.copy_(torch.tensor(parts["train"]["features"].mean(axis=0)))
        model.scale.copy_(torch.tensor(np.maximum(parts["train"]["features"].std(axis=0), 1e-3)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = {key: torch.tensor(part["features"]) for key, part in parts.items()}
    y = torch.tensor(parts["train"]["actions"])
    targets = torch.tensor(targets)
    with torch.no_grad():
        initial = metrics(model(x["validation"]).softmax(-1).numpy(), parts["validation"]["actions"])
    report = {"schema": "spiking_student_v1", "status": "running", "student_trained": False,
              "connectome_weights_trained": False, "laya_needed_at_inference": False,
              "experimental_teacher": experimental_teacher,
              "teacher_accepted": bool(teacher_report.get("teacher_accepted")),
              "teacher_rejection_reasons": teacher_report.get("rejection_reasons", []),
              "test_evaluated": evaluate_test,
              "training_method": "Laya probability distillation plus 0.25 weighted human cross-entropy",
              "input_size": x["train"].shape[1], "hidden": hidden, "seed": seed,
              "dataset_sha256": dataset_hash, "teacher_report_sha256": digest(teacher / "report.json", "sha256"),
              "calibration_sha256": source["calibration_sha256"], "output_root_ids": source["output_root_ids"],
              "initial_validation": initial, "epochs": [],
              "implementation_sha256": digest(Path(__file__), "sha256")}
    best = float("inf")
    try:
        for epoch in range(epochs):
            model.train()
            losses = []
            order = rng.permutation(len(y))
            for offset in range(0, len(order), 32):
                indices = order[offset:offset + 32]
                optimizer.zero_grad(set_to_none=True)
                logits = model(x["train"][indices])
                loss = F.kl_div(F.log_softmax(logits, -1), targets[indices], reduction="batchmean") + 0.25 * F.cross_entropy(logits, y[indices])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite student loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                val = metrics(model(x["validation"]).softmax(-1).numpy(), parts["validation"]["actions"])
            report["epochs"].append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": val})
            if val["nll"] < best:
                best = val["nll"]
                report["selected_epoch"] = epoch + 1
                save_file(model.state_dict(), str(output / "student.safetensors"))
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"Student epoch {epoch + 1}: validation accuracy={val['accuracy']:.3f}", flush=True)
            write_json(output / "report.json", report)
        model.load_state_dict(load_file(str(output / "student.safetensors")))
        with torch.no_grad():
            report["validation"] = metrics(model(x["validation"]).softmax(-1).numpy(), parts["validation"]["actions"])
            for split in ("validation", "test") if evaluate_test else ("validation",):
                report[split] = metrics(model(x[split]).softmax(-1).numpy(), parts[split]["actions"])
                silenced = x[split].clone()
                silenced[:, :-4] = 0
                report[f"{split}_zero_neural_features"] = metrics(model(silenced).softmax(-1).numpy(), parts[split]["actions"])
        report["student_sha256"] = digest(output / "student.safetensors", "sha256")
        report["student_trained"] = True
        report["status"] = "completed"
        report["game_skill_validated"] = False
    except BaseException:
        report["status"] = "interrupted_or_failed"
        raise
    finally:
        write_json(output / "report.json", report)
    return report


def play(checkpoint, calibration, output, *, data_dir="data/processed/fafb783", episodes=1,
         seed=20000, max_decisions=24, visible=True, disconnected=False):
    import vizdoom as vzd
    if not 1 <= episodes <= 20 or not 1 <= max_decisions <= 75 or not 0 <= seed <= 2**32 - episodes:
        raise ValueError("Invalid game limits or seed")
    checkpoint, output = Path(checkpoint), Path(output)
    report = json.loads((checkpoint / "report.json").read_text(encoding="utf-8"))
    if report.get("schema") != "spiking_student_v1" or report.get("status") != "completed" or not report["student_trained"]:
        raise ValueError("Student checkpoint is not complete")
    if (digest(checkpoint / "student.safetensors", "sha256") != report["student_sha256"]
            or digest(calibration, "sha256") != report["calibration_sha256"]
            or digest(Path(__file__), "sha256") != report["implementation_sha256"]):
        raise ValueError("Student artifact, calibration, or implementation changed")
    ids, controller, _ = load_calibrated(data_dir, calibration)
    actual_ids = [str(ids[i]) for i in np.sort(np.concatenate(controller.mapping.output_groups))]
    if actual_ids != report["output_root_ids"]:
        raise ValueError("Student neuron mapping mismatch")
    if disconnected:
        controller.control = "disconnected"
    torch.set_num_threads(4)
    model = SpikingReadout(report["input_size"], report["hidden"])
    model.load_state_dict(load_file(str(checkpoint / "student.safetensors")))
    model.eval()
    output.mkdir(parents=True, exist_ok=False)
    game = None
    result = {"status": "running", "student_trained": True, "connectome_weights_trained": False,
              "laya_used_during_play": False, "disconnected": disconnected,
              "experimental_teacher": report.get("experimental_teacher", False),
              "teacher_accepted": report.get("teacher_accepted"),
              "checkpoint_sha256": report["student_sha256"], "episodes": []}
    try:
        game, buttons = make_game(seed, visible)
        with (output / "decisions.jsonl").open("w", encoding="utf-8") as trace:
            for episode in range(episodes):
                game.set_seed(seed + episode)
                game.new_episode()
                controller.reset()
                previous = 0
                summary = {"seed": seed + episode, "decisions": 0, "return": 0.0,
                           "action_counts": dict.fromkeys(ACTIONS, 0), "end_reason": "interrupted"}
                result["episodes"].append(summary)
                while not game.is_episode_finished() and summary["decisions"] < max_decisions:
                    decision = controller.decide(game.get_state().screen_buffer)
                    if decision["voltage_min_mv"] < -90:
                        raise ValueError("Live game failed the engineering voltage gate")
                    vector = np.concatenate((output_features(controller), np.eye(4, dtype=np.float32)[previous]))
                    with torch.no_grad():
                        probs = model(torch.tensor(vector).unsqueeze(0)).softmax(-1)[0].numpy()
                    action = int(probs.argmax())
                    for _ in range(4):
                        if game.is_episode_finished():
                            break
                        started = perf_counter()
                        game.make_action([int(b == ACTIONS[action]) for b in buttons], 1)
                        if visible:
                            sleep(max(0, 1 / 35 - (perf_counter() - started)))
                    summary["decisions"] += 1
                    summary["action_counts"][ACTIONS[action]] += 1
                    summary["return"] = game.get_total_reward()
                    previous = action
                    trace.write(json.dumps({"episode": episode + 1, "decision": summary["decisions"],
                        "action": ACTIONS[action], "probabilities": probs.tolist(),
                        "brain_spikes": decision["spikes"], "descending_spikes": decision["descending_spikes"],
                        "neural_feature_norm": float(np.linalg.norm(vector[:-4])),
                        "brain_compute_seconds": decision["compute_seconds"],
                        "voltage_min_mv": decision["voltage_min_mv"], "return": summary["return"]}) + "\n")
                    trace.flush()
                    print(f"{summary['decisions']:03d}: {ACTIONS[action]} | return={summary['return']:.0f}"
                          f" | brain spikes={decision['spikes']} | brain compute={decision['compute_seconds']:.2f}s", flush=True)
                summary["end_reason"] = "game_finished" if game.is_episode_finished() else "decision_limit"
                # Evaluation only: kill count is never part of the policy input.
                summary["kills"] = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
            result["status"] = "completed"
    except KeyboardInterrupt:
        result["status"] = "interrupted"
    except Exception:
        result["status"] = "error"
        raise
    finally:
        if game:
            game.close()
        write_json(output / "report.json", result)
    return result
