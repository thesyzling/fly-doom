"""Held-out morphology checks and provisional R1-6 -> L1 -> Mi1 columns.

Assignments are anatomical hypotheses, not segmentation-verified identities or
measured receptive fields. All competing L1 cells participate in dominance.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from flydoom.anatomy import skeleton_arrays
from flydoom.data import digest
from flydoom.eye_mapping import (ASSET_CACHE, DATA, OUTPUT, read_sources,
                                rank_candidates, transformed_samples, write_json)

RESULT = Path("runs/retinal-mapping-v1")


def heldout_points(xyz, count=32):
    """Exclude initial sample coordinates, then sample the remaining unique nodes."""
    xyz = np.asarray(xyz, dtype=float)
    original = xyz[np.linspace(0, len(xyz) - 1, 32).astype(int)]
    used = set(map(tuple, original))
    candidates = np.array([p for p in np.unique(xyz, axis=0) if tuple(p) not in used])
    if len(candidates) < count:
        raise ValueError("Insufficient independent skeleton coordinates")
    return candidates[np.linspace(0, len(candidates) - 1, count).astype(int)]


def load_counts():
    manifest = json.loads((DATA / "manifest.json").read_text())
    annotation = json.loads((DATA / "annotations_manifest.json").read_text())
    expected = {**manifest["output_sha256"], "neuron_annotations.tsv": annotation["aligned_sha256"]}
    for name, checksum in expected.items():
        if digest(DATA / name, "sha256") != checksum:
            raise ValueError(f"Prepared data mismatch: {name}")
    ids = np.load(DATA / "root_ids.npy", allow_pickle=False)
    with (DATA / "neuron_annotations.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [str(i) for i in ids] != [r["root_id"] for r in rows]:
        raise ValueError("Annotation order mismatch")
    counts = sparse.load_npz(DATA / "synapse_counts.npz")
    if counts.shape != (len(ids), len(ids)):
        raise ValueError("Graph shape mismatch")
    return ids, rows, counts, expected


def dominant(values, minimum=5, fraction=.8):
    values = np.asarray(values).ravel()
    if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Expected nonnegative finite contact counts")
    winner = int(values.argmax())
    total = float(values.sum())
    share = float(values[winner] / total) if total else 0.0
    return winner, int(values[winner]), share, bool(values[winner] >= minimum and share >= fraction)


def infer_paths(ids, rows, counts, supported):
    lookup = {str(n): i for i, n in enumerate(ids)}
    l1 = np.array([i for i, r in enumerate(rows) if r["cell_type"] == "L1"], dtype=int)
    receptors = np.array([i for i, r in enumerate(rows) if r["cell_type"] == "R1-6"], dtype=int)
    columns = []
    for row in supported:
        mi = lookup[row["candidate_root_id"]]
        j, contacts, fraction, good = dominant(counts[mi, l1].toarray())
        idx = int(l1[j])
        columns.append({**row, "mi1_root_id": str(ids[mi]), "l1_root_id": str(ids[idx]),
                        "l1_to_mi1_contacts": contacts, "l1_input_fraction": fraction,
                        "l1_supported": good and rows[idx]["side"] == row["side"]})
    occurrences = Counter(c["l1_root_id"] for c in columns if c["l1_supported"])
    for c in columns:
        c["l1_supported"] &= occurrences[c["l1_root_id"]] == 1
    by_l1 = {c["l1_root_id"]: c for c in columns if c["l1_supported"]}
    # Include every L1 target, including unmatched columns and the other eye.
    incoming = counts[l1][:, receptors].tocsc()
    assignments = []
    for j, root_idx in enumerate(receptors):
        column = incoming[:, j].toarray().ravel()
        k, contacts, fraction, good = dominant(column)
        target = by_l1.get(str(ids[l1[k]]))
        if not good or target is None or rows[root_idx]["side"] != target["side"]:
            continue
        assignments.append({"root_id": str(ids[root_idx]), "cell_type": "R1-6", "side": target["side"],
                            "column_index": target["column_index"], "optical_direction": target["optical_direction"],
                            "l1_root_id": target["l1_root_id"], "mi1_root_id": target["mi1_root_id"],
                            "receptor_to_l1_contacts": contacts, "l1_output_fraction": fraction,
                            "status": "provisional_connectivity_assignment"})
    return columns, assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Fetch only the new public coordinate transform")
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prior_report = json.loads((OUTPUT / "report.json").read_text())
    for name, checksum in prior_report["output_sha256"].items():
        if digest(OUTPUT / name, "sha256") != checksum:
            raise ValueError(f"Previous mapping changed: {name}")
    prior = json.loads((OUTPUT / "columns.json").read_text())
    _, cells = read_sources()
    samples = np.stack([heldout_points(cells["Mi1"][r["catmaid_skid"]]["d"][["X", "Y", "Z"]].to_numpy()) for r in prior])
    write_json(args.output / "plan.json", {"schema": "retinal_mapping_plan_v1", "sample_count": 32,
               "selection": "Unique coordinates excluding all initial sample coordinates; deterministic ordered spacing",
               "morphology_thresholds": prior_report["thresholds"], "path_minimum_contacts": 5,
               "path_minimum_dominance": .8, "same_candidate_required": True,
               "new_candidates_promoted": False, "prior_report_sha256": digest(OUTPUT / "report.json", "sha256")})
    print("Checking 778 columns with disjoint morphology coordinates", flush=True)
    transformed = transformed_samples(samples, args.output, args.fetch)
    manifest = json.loads((OUTPUT / "skeleton_manifest.json").read_text())
    skeletons = []
    for root in sorted(manifest):
        path = ASSET_CACHE / "skeletons" / root
        if digest(path, "sha256") != manifest[root]["sha256"]:
            raise ValueError("Skeleton checksum mismatch")
        skeletons.append((root, skeleton_arrays(path.read_bytes())[0]))
    checks = rank_candidates(transformed, skeletons)
    accepted = []
    for original, checked in zip(prior, checks):
        checked.update(column_index=original["column_index"], prior_supported=original["geometry_supported"],
                       same_candidate=checked["candidate_root_id"] == original["candidate_root_id"])
        checked["heldout_supported"] = original["geometry_supported"] and checked["geometry_supported"] and checked["same_candidate"]
        if checked["heldout_supported"]:
            accepted.append(original)
    print(f"Repeated support for {len(accepted)} / {prior_report['geometry_supported']} prior candidates", flush=True)
    del skeletons, cells
    ids, rows, counts, provenance = load_counts()
    columns, assignments = infer_paths(ids, rows, counts, accepted)
    write_json(args.output / "heldout.json", checks)
    write_json(args.output / "columns.json", columns)
    write_json(args.output / "receptors.json", assignments)
    report = {"schema": "retinal_mapping_v1", "prior_supported": prior_report["geometry_supported"],
              "heldout_supported": len(accepted), "supported_l1_columns": sum(c["l1_supported"] for c in columns),
              "mapped_receptors": len(assignments), "receptor_columns": len({a["column_index"] for a in assignments}),
              "segmentation_verified": 0, "trained": False, "live_policy_changed": False,
              "provenance": provenance, "prior_report_sha256": digest(OUTPUT / "report.json", "sha256"),
              "source_sha256": {n: digest(Path(__file__).parent / n, "sha256") for n in ["retinal_mapping.py", "eye_mapping.py"]},
              "output_sha256": {n: digest(args.output / n, "sha256") for n in
                                ["plan.json", "transform.json", "heldout.json", "columns.json", "receptors.json"]},
              "limitations": ["Disjoint points from the same skeletons, not independent animals or segmentation verification",
                              "Receptive fields inferred from R1-6 to L1 to Mi1 dominance; no direct optical calibration",
                              "Only R1-6 channels and supported columns; R7/R8 remain unmapped",
                              "Missing and ambiguous columns stay excluded; no symmetry filling"],
              "gates": {"provisional_retinal_probe_ready": len(assignments) > 0,
                        "retinal_identity_verified": False, "physiology_validated": False}}
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
