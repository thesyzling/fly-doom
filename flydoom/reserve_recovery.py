"""Reserve fresh recovery splits before collecting labels or tuning another model."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from flydoom.calibration import write_json
from flydoom.data import digest
from flydoom.learning_data import make_game
from flydoom.vision_recovery import select_openings
from flydoom.vision_validation import collect_seeds, fingerprint


def image_hashes(value, key=""):
    """Collect explicit RGB identities, never confuse PNG or model hashes with RGB."""
    found = set()
    if isinstance(value, dict):
        for name, child in value.items():
            found.update(image_hashes(child, name))
    elif isinstance(value, list):
        for child in value:
            found.update(image_hashes(child, key))
    elif isinstance(value, str) and key in {"rgb_sha256", "frame_sha256", "opening_hashes", "rgb_hashes", "excluded_rgb_hashes"}:
        if len(value) == 64:
            found.add(value)
    return found


def reserve(output, runs=Path("runs")):
    output, runs = Path(output), Path(runs)
    if output.exists():
        raise FileExistsError("Preserve existing reservations; use a fresh output directory")
    seeds, images, sources = set(), set(), []
    patterns = ("*/report.json", "*/plan.json", "*/opening-scan.json", "*/run-*/report.json",
                "*/run-*/decisions.jsonl", "replays/*/report.json", "replays/*/decisions.jsonl")
    paths = sorted({p for pattern in patterns for p in runs.glob(pattern)
                    if not any("tests-" in part or "profile" in part for part in p.parts)})
    for path in paths:
        if path.suffix == ".jsonl":
            values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            values = [json.loads(path.read_text(encoding="utf-8"))]
        for value in values:
            seeds.update(collect_seeds(value))
            images.update(image_hashes(value))
        sources.append(path)
    for pattern in ("*/train.npz", "*/validation.npz"):
        for path in sorted(runs.glob(pattern)):
            with np.load(path, allow_pickle=False) as dataset:
                if "rgb_hashes" in dataset.files:
                    images.update(str(h) for h in dataset["rgb_hashes"])
                    sources.append(path)
    # Select the untouched evaluation reservation first, without policy outcomes.
    sizes = {"evaluation": 20, "train": 24, "validation": 8}
    rows, game = [], None
    try:
        game, _ = make_game(22000000, False)
        for seed in np.random.default_rng(1909).choice(np.arange(22000000, 22100000), 1200, replace=False):
            seed = int(seed)
            if seed in seeds:
                continue
            game.set_seed(seed)
            game.new_episode()
            frame = game.get_state().screen_buffer
            rows.append({"seed": seed, "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
            selected = select_openings(rows, images, seeds, sizes=sizes)
            if all(len(selected[name]) == count for name, count in sizes.items()):
                break
        else:
            raise ValueError("Insufficient distinct openings; do not weaken split separation")
    finally:
        if game:
            game.close()
    import vizdoom
    package = Path(vizdoom.__file__).parent
    assets = [package / "scenarios/basic.cfg", package / "scenarios/basic.wad", package / "freedoom2.wad"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "opening-scan.json", {"rows": rows, "excluded_rgb_hashes": sorted(images),
               "excluded_seeds": sorted(seeds), "source_sha256": fingerprint(sources)})
    plan = {"schema": "recovery_reservation_v2", "status": "reserved_not_collected",
            "created_utc": datetime.now(timezone.utc).isoformat(), "splits": selected,
            "artifact_sha256": fingerprint(assets + [Path(__file__), output / "opening-scan.json"]),
            "collection": "Future collection: rotate all three continuation policies equally across training starts; retain causal features and pinned Vision targets. No winner chosen using the prior benchmark.",
            "training": "Not started. Define and lock rehearsal, validation selection and training budget before collection/training.",
            "evaluation": "No actions, labels or rewards inspected on these 20 reserved starts. Keep unavailable to tuning and live demonstrations.",
            "scope": "Exact opening RGB separation from discovered retained teaching data and recorded RGB/opening hashes. Does not audit every historical PNG, older unrecorded history or semantic scene similarity. Audit complete trajectories before interpreting later results."}
    write_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(digest(output / "plan.json", "sha256") + "\n", encoding="ascii")
    return {"status": plan["status"], "scanned": len(rows), "known_images": len(images),
            "splits": {name: len(values) for name, values in selected.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reserve(args.output), indent=2))


if __name__ == "__main__":
    main()
