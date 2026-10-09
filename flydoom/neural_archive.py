"""Atomic, checksum-verified per-decision neural recordings with bounded memory."""

import json
import os
from pathlib import Path

import numpy as np

from flydoom.data import digest

ARRAYS = ("features", "activity", "drive", "voltage", "counts", "current", "refractory")
MAX_BYTES = 512 * 1024**2


class NeuralArchive:
    def __init__(self, folder, identity=None):
        self.folder = Path(folder)
        self.cache = None
        if identity is not None:
            self.folder.mkdir(exist_ok=False)
            self.manifest = {"schema": "neural_archive_v1", "identity": identity, "records": {}}
            self.flush()
        else:
            self.manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
            if self.manifest.get("schema") != "neural_archive_v1": raise ValueError("Unknown neural archive")

    def flush(self):
        temporary = self.folder / "manifest.tmp"
        temporary.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        os.replace(temporary, self.folder / "manifest.json")

    def save(self, sample):
        sequence = sample["sequence"]
        if type(sequence) is not int or sequence < 1 or str(sequence) in self.manifest["records"]:
            raise ValueError("Invalid or duplicate archive sequence")
        if sum(r["bytes"] for r in self.manifest["records"].values()) >= MAX_BYTES:
            raise ValueError("Neural recording reached its 512 MiB disk budget")
        metadata = {k: v for k, v in sample.items() if k not in ARRAYS or v is None}
        arrays = {k: sample[k] for k in ARRAYS if sample.get(k) is not None}
        path = self.folder / f"{sequence:04d}.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, metadata=np.array(json.dumps(metadata)), **arrays)
        if sum(r["bytes"] for r in self.manifest["records"].values()) + temporary.stat().st_size > MAX_BYTES:
            temporary.unlink(); raise ValueError("Neural recording exceeds the 512 MiB disk budget")
        os.replace(temporary, path)
        self.manifest["records"][str(sequence)] = {"file": path.name, "bytes": path.stat().st_size,
                                                   "sha256": digest(path, "sha256"), "decision": sample["decision"]}
        self.flush()

    def load(self, sequence):
        if type(sequence) is not int or str(sequence) not in self.manifest["records"]:
            raise ValueError("Decision is not present in the neural archive")
        if self.cache is not None and self.cache["sequence"] == sequence: return self.cache
        record = self.manifest["records"][str(sequence)]
        if record["file"] != f"{sequence:04d}.npz": raise ValueError("Invalid archive path")
        path = self.folder / record["file"]
        if digest(path, "sha256") != record["sha256"]: raise ValueError("Neural recording checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            sample = json.loads(str(data["metadata"]))
            sample.update({k: data[k].copy() for k in ARRAYS if k in data})
        if sample["sequence"] != sequence: raise ValueError("Neural sequence identity mismatch")
        self.cache = sample
        return sample

    def sequences(self): return sorted(map(int, self.manifest["records"]))
