"""Build a local interactive review of recorded student decisions and teacher advice."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch

from flydoom import action_memory, student
from flydoom.calibration import write_json
from flydoom.correction_data import read_collection
from flydoom.correction_training import read_labels
from flydoom.data import digest
from flydoom.learning_data import ACTIONS
from flydoom.live_activity import frame_png


def summarize(actions, advice):
    longest = streak = 0
    start = best_start = None
    for i, action in enumerate(actions):
        if action == 0:
            if not streak:
                start = i + 1
            streak += 1
            if streak > longest:
                longest, best_start = streak, start
        else:
            streak = 0
    teacher = np.asarray(advice).argmax(1)
    return {"decisions": len(actions), "student_action_counts": np.bincount(actions, minlength=4).tolist(),
            "teacher_action_counts": np.bincount(teacher, minlength=4).tolist(),
            "teacher_disagreements": int(np.count_nonzero(np.asarray(actions) != teacher)),
            "teacher_active_when_student_waited": int(np.count_nonzero((np.asarray(actions) == 0) & (teacher != 0))),
            "longest_wait": longest, "longest_wait_start": best_start}


def build(collection, labels, checkpoint, output):
    collection, labels, checkpoint, output = map(Path, (collection, labels, checkpoint, output))
    if output.exists():
        raise FileExistsError(output)
    recording, parts = read_collection(collection)
    parent = json.loads((checkpoint / "report.json").read_text(encoding="utf-8"))
    if (parent.get("schema") != "spiking_student_v1" or parent.get("status") != "completed"
            or parent["student_sha256"] != recording["parent_student_sha256"]
            or digest(checkpoint / "student.safetensors", "sha256") != parent["student_sha256"]
            or digest(checkpoint / "report.json", "sha256") != recording["parent_report_sha256"]
            or digest(student.__file__, "sha256") != parent["implementation_sha256"]):
        raise ValueError("Recorded policy checkpoint changed")
    label_report = json.loads((labels / "manifest.json").read_text(encoding="utf-8"))
    sizes = {s: len(p["actions"]) for s, p in parts.items()}
    sizes["human_train"] = label_report["files"]["human_train"]["samples"]
    _, targets = read_labels(labels, collection, parent, sizes)
    torch.set_num_threads(4)
    model = student.SpikingReadout(parent["input_size"], parent["hidden"])
    model.load_state_dict(load_file(str(checkpoint / "student.safetensors")))
    model.eval()
    activity = []
    handle = model.readout.register_forward_pre_hook(lambda module, inputs: activity.append(inputs[0][0].detach().numpy().tolist()))
    offsets = {"train": 0, "validation": 0}
    episodes, summaries = [], []
    try:
        for episode in recording["episodes"]:
            split, n = episode["split"], episode["decisions"]
            begin = offsets[split]
            advice = targets[split][begin:begin + n]
            offsets[split] += n
            states = json.loads((collection / episode["states_file"]).read_text(encoding="utf-8"))
            with np.load(collection / episode["file"], allow_pickle=False) as arrays:
                actions = arrays["actions"]
                frames = []
                for i in range(n):
                    with torch.no_grad():
                        replay = model(torch.tensor(arrays["features"][i:i + 1])).softmax(-1)[0].numpy()
                    if not np.allclose(replay, arrays["student_probabilities"][i], atol=1e-6):
                        raise ValueError("Policy replay differs from recorded probabilities")
                    frames.append({"image": frame_png(arrays["frames"][i]), "decision": i + 1,
                        "action": int(actions[i]), "student": replay.tolist(), "teacher": advice[i].tolist(),
                        "activity": activity.pop(), "blue_region": states[i]["blue_region"],
                        "recent_actions": states[i]["recent_actions"], "timing": states[i]["timing"],
                        "neural_feature_norm": float(np.linalg.norm(arrays["features"][i, :-4])),
                        "return_after_action": float(arrays["returns"][i])})
                summary = {"seed": episode["seed"], "kills": episode["kills"], "return": episode["return"],
                           **summarize(actions, advice)}
                summaries.append(summary)
                episodes.append({"summary": summary, "frames": frames})
    finally:
        handle.remove()
    report = {"status": "completed", "checkpoint_sha256": parent["student_sha256"],
              "collection_manifest_sha256": digest(collection / "manifest.json", "sha256"),
              "labels_manifest_sha256": digest(labels / "manifest.json", "sha256"),
              "teacher_accepted": label_report["teacher_accepted"], "episodes": summaries,
              "training_during_review": False,
              "note": "Recorded actions and aligned pre-action images; teacher suggestions are not verified correct actions. Added-cell activity is verified checkpoint replay."}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    template = (Path(__file__).parent / "web/recovery.html").read_text(encoding="utf-8")
    encoded = json.dumps({"actions": ACTIONS, "episodes": episodes}).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__REVIEW_DATA__", encoded), encoding="utf-8")
    print(f"Review: {(output / 'index.html').resolve()}", flush=True)
    print(json.dumps(summaries, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=Path("runs/recovery-diagnosis-observations-v1"))
    parser.add_argument("--labels", type=Path, default=Path("runs/recovery-diagnosis-labels-v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/fly-student-corrected-v2"))
    parser.add_argument("--output", type=Path, required=True)
    build(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
