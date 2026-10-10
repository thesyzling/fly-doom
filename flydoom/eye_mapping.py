"""Audit author eye columns and propose morphology-supported Mi1 correspondences.

This is not a photoreceptor mapping. Geometric candidates remain distinct from
segmentation-verified identity and are never installed into the live policy.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import urllib.request
import warnings

import numpy as np
from scipy.spatial import cKDTree

from flydoom.anatomy import SKELETONS, skeleton_arrays
from flydoom.data import digest
from flydoom.eye_sources import ROOT, REVISION

DATA = Path("data/processed/fafb783")
OUTPUT = Path("runs/eye-mapping-v2")
ASSET_CACHE = Path("runs/eye-mapping-v1")
TRANSFORM = "https://services.itanna.io/app/transform-service/transform/dataset/flywire_v1_inverse/s/4/values_array"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_sources():
    import rdata
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest["revision"] != REVISION:
        raise ValueError("Unexpected eye-source revision")
    for item in manifest["files"]:
        if digest(ROOT / item["path"], "sha256") != item["sha256"]:
            raise ValueError("Eye-source checksum mismatch")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Missing constructor")
        eye = rdata.read_rda(ROOT / "data/eyemap.RData")
        cells = rdata.read_rda(ROOT / "data/neu_Mi1.RData")
    return eye, cells


def source_columns(eye, cells):
    mapping = np.asarray(eye["eyemap"])
    if mapping.shape[1] != 2 or not np.equal(mapping, np.floor(mapping)).all():
        raise ValueError("Eye columns require integer one-based indices")
    index = mapping[:, 0].astype(int) - 1
    if np.any(index < 0) or np.any(index >= len(cells["Mi1_neu_ind"])):
        raise ValueError("Eye index outside the one-based Mi1 array")
    cell_index = np.asarray(cells["Mi1_neu_ind"], dtype=int)[index] - 1
    if np.any(cell_index < 0) or np.any(cell_index >= len(cells["anno_Mi1"])):
        raise ValueError("Mi1 index outside the one-based annotation array")
    skids = cells["anno_Mi1"].iloc[cell_index]["skid"].to_numpy(dtype=np.int64)
    directions = np.asarray(eye["ucl_rot_sm"], dtype=float)
    medulla = np.asarray(eye["med_xyz"], dtype=float)
    expected = np.asarray(cells["Mi1_M10_xyz"])[index]
    if not np.allclose(medulla, expected, atol=1e-6):
        raise ValueError("Column and Mi1 index chains disagree")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-6):
        raise ValueError("Optical directions are not unit vectors")
    if len(set(skids)) != len(skids):
        raise ValueError("Repeated source skeleton in eye map")
    samples = []
    for skid in skids:
        xyz = cells["Mi1"][str(skid)]["d"][["X", "Y", "Z"]].to_numpy(dtype=float)
        # Deterministic coverage over the stored skeleton order; no random fit.
        samples.append(xyz[np.linspace(0, len(xyz) - 1, 32).astype(int)])
    return mapping, skids, directions, medulla, np.stack(samples)


def transformed_samples(samples, output, online):
    path = output / "transform.json"
    source_sha = hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest()
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["source_sha256"] != source_sha or saved["url"] != TRANSFORM:
            raise ValueError("Stale coordinate transform cache")
        result = np.asarray(saved["flywire_nm"], dtype=float)
    elif online:
        points = samples.reshape(-1, 3) / [4, 4, 40]
        # The service consumes base-resolution voxels, independent of mip.
        payload = {axis: points[:, i].tolist() for i, axis in enumerate("xyz")}
        request = urllib.request.Request(TRANSFORM, json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=55) as response:
            raw = json.load(response)
        result = np.column_stack([raw[a] for a in "xyz"]) * [4, 4, 40]
        write_json(path, {"url": TRANSFORM, "source_sha256": source_sha,
                         "input_space": "FAFB14 nanometers", "output_space": "FAFB14.1 nanometers",
                         "flywire_nm": result.tolist(), "response": raw})
    else:
        raise ValueError("Run with --fetch once to obtain coordinate transforms")
    if result.shape != (samples.shape[0] * samples.shape[1], 3) or not np.isfinite(result).all():
        raise ValueError("Invalid coordinate transform response")
    return result.reshape(samples.shape)


def fetch_skeletons(roots, output, online):
    folder = ASSET_CACHE / "skeletons"
    folder.mkdir(exist_ok=True)
    manifest_path = output / "skeleton_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    def one(root):
        path = folder / root
        url = SKELETONS + "/" + root
        if path.exists():
            body = path.read_bytes()
            if root in manifest and hashlib.sha256(body).hexdigest() != manifest[root]["sha256"]:
                raise ValueError("Skeleton cache changed")
        elif online:
            with urllib.request.urlopen(url, timeout=25) as response:
                body = response.read(4_000_001)
            if len(body) > 4_000_000:
                raise ValueError("Skeleton exceeds size bound")
            skeleton_arrays(body)
            path.write_bytes(body)
        else:
            raise ValueError(f"Missing skeleton {root}; use --fetch")
        xyz, _ = skeleton_arrays(body)
        return root, xyz, {"url": url, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, (root, xyz, record) in enumerate(pool.map(one, roots)):
            manifest[root] = record
            results.append((root, xyz))
            if (i + 1) % 100 == 0:
                write_json(manifest_path, manifest)
                print(f"Verified {i + 1}/{len(roots)} Mi1 skeletons", flush=True)
    write_json(manifest_path, manifest)
    return results


def rank_candidates(samples, skeletons):
    xyz = np.concatenate([p for _, p in skeletons])
    labels = np.concatenate([np.full(len(p), i) for i, (_, p) in enumerate(skeletons)])
    tree = cKDTree(xyz)
    distances, nearest = tree.query(samples.reshape(-1, 3), k=1)
    labels = labels[nearest].reshape(samples.shape[:2])
    distances = distances.reshape(samples.shape[:2])
    result = []
    for votes, dist in zip(labels, distances):
        counts = Counter(votes.tolist()).most_common(2)
        winner, count = counts[0]
        runner = counts[1][1] if len(counts) > 1 else 0
        winner_dist = dist[votes == winner]
        median = float(np.median(winner_dist))
        p90 = float(np.percentile(winner_dist, 90))
        fraction = count / len(votes)
        margin = (count - runner) / len(votes)
        result.append({"candidate_root_id": skeletons[winner][0], "vote_fraction": fraction,
                       "vote_margin": margin, "median_distance_nm": median, "p90_distance_nm": p90,
                       "geometry_supported": bool(fraction >= .8 and margin >= .6 and p90 <= 1000),
                       "segmentation_verified": False})
    # Reject duplicate assignments instead of forcing a one-to-one permutation.
    duplicates = Counter(r["candidate_root_id"] for r in result)
    for row in result:
        row["unique_candidate"] = duplicates[row["candidate_root_id"]] == 1
        row["geometry_supported"] &= row["unique_candidate"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    eye, cells = read_sources()
    mapping, skids, directions, medulla, samples = source_columns(eye, cells)
    with (DATA / "neuron_annotations.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    roots = sorted(r["root_id"] for r in rows if r["cell_type"] == "Mi1")
    sides = {r["root_id"]: r["side"] for r in rows}
    print(f"Auditing {len(skids)} source columns against {len(roots)} bilateral Mi1 cells", flush=True)
    transformed = transformed_samples(samples, ASSET_CACHE, args.fetch)
    (args.output / "transform.json").write_bytes((ASSET_CACHE / "transform.json").read_bytes())
    skeletons = fetch_skeletons(roots, args.output, args.fetch)
    candidates = rank_candidates(transformed, skeletons)
    for i, row in enumerate(candidates):
        row.update(column_index=i, medulla_index=int(mapping[i, 0]), lens_index=int(mapping[i, 1]),
                   catmaid_skid=str(skids[i]), optical_direction=directions[i].tolist(),
                   medulla_fafb14_nm=medulla[i].tolist(), side=sides[row["candidate_root_id"]], cell_type="Mi1")
    write_json(args.output / "columns.json", candidates)
    report = {"schema": "eye_mapping_v1", "source_revision": REVISION, "columns": len(candidates),
              "candidate_cells": len(roots), "geometry_supported": sum(r["geometry_supported"] for r in candidates),
              "segmentation_verified": 0, "live_policy_changed": False,
              "source_manifest_sha256": digest(ROOT / "manifest.json", "sha256"),
              "annotation_sha256": digest(DATA / "neuron_annotations.tsv", "sha256"),
              "output_sha256": {n: digest(args.output / n, "sha256") for n in
                                ("columns.json", "transform.json", "skeleton_manifest.json")},
              "source_sha256": digest(Path(__file__), "sha256"),
              "supported_sides": dict(Counter(r["side"] for r in candidates if r["geometry_supported"])),
              "method": "32 transformed CATMAID skeleton samples; nearest-vertex votes over all bilateral Mi1 skeletons",
              "thresholds": {"vote_fraction": .8, "vote_margin": .6, "p90_distance_nm": 1000, "unique": True},
              "limitations": ["Mi1 correspondence is not a photoreceptor-to-column map",
                              "Morphological support is not segmentation-verified identity",
                              "Nearest vertices depend on skeleton sampling density",
                              "Author optical-axis naming and FlyWire anatomical laterality must be distinguished",
                              "Camera axes and FOV are not calibrated"],
              "gates": {"source_index_chain": True, "unit_optical_vectors": True,
                        "retinal_input_ready": False, "physiology_validated": False}}
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
