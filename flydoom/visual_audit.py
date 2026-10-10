"""Independently check saved visual-probe counts, identities and controls."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from flydoom.data import digest
from flydoom.eye_mapping import DATA, OUTPUT, write_json


def audit(output, mapping):
    report = json.loads((output / "report.json").read_text())
    m = json.loads((mapping / "report.json").read_text())
    if digest(DATA / "neuron_annotations.tsv", "sha256") != m["annotation_sha256"]:
        raise ValueError("Annotation identity changed")
    if digest(Path(__file__).with_name("eye_mapping.py"), "sha256") != m["source_sha256"]:
        raise ValueError("Mapping implementation changed")
    if report["mapping_report_sha256"] != digest(mapping / "report.json", "sha256"):
        raise ValueError("Wrong mapping report")
    for folder, hashes in [(output, report["output_sha256"]), (mapping, m["output_sha256"])]:
        for name, expected in hashes.items():
            if digest(folder / name, "sha256") != expected:
                raise ValueError(f"Artifact changed: {name}")
    for name, expected in report["source_sha256"].items():
        if digest(Path(__file__).parent / name, "sha256") != expected:
            raise ValueError(f"Source changed: {name}")
    with (DATA / "neuron_annotations.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    ids = np.array([int(r["root_id"]) for r in rows], dtype=np.uint64)
    columns = json.loads((mapping / "columns.json").read_text())
    supported = [r for r in columns if r["geometry_supported"]]
    expected_input = [r["candidate_root_id"] for r in supported]
    if len(expected_input) != len(set(expected_input)):
        raise ValueError("Duplicate supported input")
    control_counts = {}
    for condition in report["conditions"]:
        name = condition["condition"]
        with np.load(output / (name + ".npz"), allow_pickle=False) as stored:
            if not np.array_equal(stored["root_ids"], ids):
                raise ValueError("Root order mismatch")
            counts = stored["spike_counts"]
            if counts.shape != ids.shape or counts.dtype.kind not in 'iu' or np.any(counts < 0):
                raise ValueError("Invalid recorded counts")
            if name != "photoreceptor_flash" and list(map(str, stored["input_root_ids"])) != expected_input:
                raise ValueError("Spatial input identity mismatch")
            for population, expected in condition["spikes_by_population"].items():
                if population == "descending":
                    selected = [i for i, r in enumerate(rows) if r["super_class"] == "descending"]
                else:
                    kind, side = population.rsplit('_', 1)
                    selected = [i for i, r in enumerate(rows) if r["cell_type"] == kind and r["side"] == side]
                if int(counts[selected].sum()) != expected:
                    raise ValueError("Population count mismatch")
            if name in {"no_input", "Mi1_disconnected"}:
                direct = np.isin(ids, stored["input_root_ids"])
                control_counts[name] = int(counts.sum() if name == "no_input" else counts[~direct].sum())
    if set(control_counts) != {"no_input", "Mi1_disconnected"} or any(control_counts.values()):
        raise ValueError("No-input or disconnected control failed")
    result = {"schema": "visual_audit_v1", "passed": True, "conditions": len(report["conditions"]),
              "supported_columns": len(supported), "control_counts": control_counts,
              "physiology_validated": False, "note": "Artifact consistency and transmission controls only"}
    write_json(output / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/visual-integration-v1"))
    parser.add_argument("--mapping", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.output, args.mapping), indent=2))
