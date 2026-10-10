"""Fetch a small, pinned author dataset for the visual integration audit."""

import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "99d2a43123db636cedb55af9ff31a59657e7d17e"
ROOT = Path("data/raw/eyemap_T4") / REVISION
FILES = {
    "data/eyemap.RData": "f3ee4a5aeb5be159de7071952dc52eb20d8b42d6",
    "data/neu_Mi1.RData": "ee941396f53e813b0e12d5a1fcc64abdfec321db",
    "data/neu_PR.RData": "bdc0d3d3d7df9593058619bd3fac6423c961418b",
    "data/neu_R7.RData": "8f2627d291b8ae894daecb19ecb891da2ba8d76e",
    "data/med_ixy.RData": "66d5b9f1e9d06d65a77bfa04d305396e3ff21a9f",
    "data/H2_tuning_2023.rda": "b0219862b75a8253ced57c5811e6035ff88f72aa",
    "LICENSE": "f288702d2fa16d3cdf0035b15a9fcbc552cd88e7",
}


def fetch():
    records = []
    for name, blob in FILES.items():
        path = ROOT / name
        url = f"https://raw.githubusercontent.com/reiserlab/eyemap_T4/{REVISION}/{name}"
        if path.exists():
            body = path.read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=45) as response:
                body = response.read(25_000_001)
        if len(body) > 25_000_000:
            raise ValueError("Source exceeds the bounded download size")
        actual = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
        if actual != blob:
            raise ValueError(f"Pinned Git blob mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(body)
        records.append({"path": name, "url": url, "bytes": len(body),
                        "git_blob": blob, "sha256": hashlib.sha256(body).hexdigest()})
        print(f"Verified {name}: {len(body):,} bytes", flush=True)
    manifest = {"schema": "eye_sources_v1", "revision": REVISION,
                "repository": "https://github.com/reiserlab/eyemap_T4",
                "paper": "https://doi.org/10.1038/s41586-025-09276-5",
                "files": records}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    fetch()
