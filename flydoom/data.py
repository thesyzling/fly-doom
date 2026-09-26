"""Download and audit the publication-time FAFB v783 connectivity archive."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np
import pyarrow.feather as feather
from scipy import sparse

SOURCE = "https://zenodo.org/records/10676866"
FILES = {
    "proofread_root_ids_783.npy": "e0e6c19732fd8c7a4e39a2d170105421",
    "proofread_connections_783.feather": "f48f972d262323a102aed49af1396b8a",
}
ANNOTATION_FILE = "neuron_annotations_v2.1.0.tsv"
ANNOTATION_COMMIT = "ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0"
ANNOTATION_URL = (
    f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{ANNOTATION_COMMIT}"
    "/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
# Observed SHA-256 of the file at the pinned author-repository commit.
ANNOTATION_SHA256 = "30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6"


def digest(path, algorithm):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def verify(path):
    if digest(path, "md5") != FILES[path.name]:
        raise ValueError(f"Archive checksum mismatch: {path}")


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        target = directory / filename
        if target.exists():
            verify(target)
            continue
        partial = target.with_suffix(target.suffix + ".partial")
        print(f"Downloading {filename}", flush=True)
        request = urllib.request.Request(
            f"{SOURCE}/files/{filename}?download=1",
            headers={"User-Agent": "fly-doom-research/0.1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            with partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        if digest(partial, "md5") != FILES[filename]:
            raise ValueError(f"Download checksum mismatch: {filename}")
        partial.replace(target)


def build_graph(root_ids, pre, post, counts):
    """Return sorted IDs and CSR A[post, pre], retaining isolated neurons.

    Region-specific rows are summed into neuron pairs. No edge threshold,
    sign assignment, normalization or self-loop removal is applied.
    """
    arrays = [np.asarray(x) for x in (root_ids, pre, post, counts)]
    if any(x.ndim != 1 or x.dtype.kind not in "iu" for x in arrays):
        raise ValueError("IDs and counts must be one-dimensional integer arrays")
    root_ids, pre, post, counts = arrays
    if len(pre) != len(post) or len(pre) != len(counts):
        raise ValueError("Edge column lengths differ")
    if any(np.any(x < 0) for x in (root_ids, pre, post)):
        raise ValueError("Neuron IDs must be nonnegative")
    # Mixing signed/unsigned int64 in searchsorted can promote to float64,
    # which cannot represent these ~7e17 IDs exactly. Normalize losslessly.
    root_ids, pre, post = (x.astype(np.uint64, copy=False) for x in (root_ids, pre, post))
    ids = np.sort(root_ids)
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("Neuron IDs must be nonempty and unique")
    if np.any(counts <= 0):
        raise ValueError("Synapse counts must be positive")
    indices = []
    for endpoints in (post, pre):
        idx = np.searchsorted(ids, endpoints)
        if np.any(idx >= len(ids)) or not np.array_equal(ids[idx], endpoints):
            raise ValueError("Connection endpoint absent from proofread neuron IDs")
        indices.append(idx)
    matrix = sparse.coo_matrix(
        (counts.astype(np.int64), tuple(indices)), shape=(len(ids), len(ids))
    ).tocsr()
    return ids, matrix


def prepare(directory, output):
    for filename in FILES:
        verify(directory / filename)
    roots = np.load(directory / "proofread_root_ids_783.npy", allow_pickle=False)
    columns = ["pre_pt_root_id", "post_pt_root_id", "syn_count"]
    table = feather.read_table(directory / "proofread_connections_783.feather", columns=columns)
    ids, matrix = build_graph(roots, *(table[c].to_numpy() for c in columns))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "root_ids.npy", ids, allow_pickle=False)
    sparse.save_npz(output / "synapse_counts.npz", matrix)
    report = {
        "dataset": "FlyWire FAFB v783",
        "source": SOURCE,
        "neurons": len(ids),
        "input_rows": table.num_rows,
        "directed_neuron_pairs": matrix.nnz,
        "synaptic_contacts": int(matrix.sum()),
        "matrix_orientation": "row=postsynaptic, column=presynaptic",
        "minimum_synapses": 1,
        "self_connections": "retained",
        "region_rows": "summed per directed pair",
        "weights": "unsigned integer counts, not physiological strengths",
        "source_sha256": {name: digest(directory / name, "sha256") for name in FILES},
        "output_sha256": {name: digest(output / name, "sha256") for name in ("root_ids.npy", "synapse_counts.npz")},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def align_annotations(root_ids, rows):
    """Match by exact integer identity, never by spreadsheet row position."""
    mapping = {}
    for row in rows:
        root = int(row["root_id"])
        if root in mapping:
            raise ValueError(f"Duplicate annotation ID: {root}")
        mapping[root] = row
    roots = [int(root) for root in root_ids]
    missing = set(roots) - mapping.keys()
    if missing:
        raise ValueError(f"Missing annotations for {len(missing)} graph neurons")
    return [mapping[root] for root in roots], len(mapping.keys() - set(roots))


def annotate(directory, output):
    graph_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if digest(output / "root_ids.npy", "sha256") != graph_manifest["output_sha256"]["root_ids.npy"]:
        raise ValueError("Prepared neuron ID checksum mismatch")
    path = directory / ANNOTATION_FILE
    if not path.exists():
        partial = path.with_suffix(".tsv.partial")
        with urllib.request.urlopen(ANNOTATION_URL, timeout=60) as response:
            with partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        if digest(partial, "sha256") != ANNOTATION_SHA256:
            raise ValueError("Annotation download checksum mismatch")
        partial.replace(path)
    if digest(path, "sha256") != ANNOTATION_SHA256:
        raise ValueError("Annotation checksum mismatch")
    roots = np.load(output / "root_ids.npy", allow_pickle=False)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames
        rows, extra = align_annotations(roots, reader)
    aligned = output / "neuron_annotations.tsv"
    with aligned.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "source_url": ANNOTATION_URL, "source_commit": ANNOTATION_COMMIT,
        "source_sha256": ANNOTATION_SHA256,
        "graph_root_ids_sha256": digest(output / "root_ids.npy", "sha256"),
        "aligned_sha256": digest(aligned, "sha256"),
        "matched_neurons": len(rows), "extra_source_neurons": extra,
        "row_order": "same as root_ids.npy and graph matrix",
        "counts": {field: dict(Counter(row[field] or "(missing)" for row in rows))
                   for field in ("flow", "super_class", "cell_class", "top_nt", "side")},
        "missing_cell_type": sum(not row["cell_type"] for row in rows),
        "note": "top_nt is a prediction; no physiological signs or input/output mappings assigned",
    }
    (output / "annotations_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "prepare", "annotate"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/fafb783"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/fafb783"))
    args = parser.parse_args()
    if args.command == "download":
        download(args.data_dir)
    elif args.command == "prepare":
        prepare(args.data_dir, args.output)
    else:
        annotate(args.data_dir, args.output)


if __name__ == "__main__":
    main()
