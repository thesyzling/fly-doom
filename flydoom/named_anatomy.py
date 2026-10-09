"""Named FlyWire neuropil surfaces from a pinned, checksum-verified fafbseg release."""

import base64
import hashlib
from pathlib import Path
import zipfile

import numpy as np

from flydoom.anatomy import download, normalize

COMMIT = "d0da95123ee606e204ae2c702e7bc78538646fbd"
SOURCE = f"https://raw.githubusercontent.com/navis-org/fafbseg-py/{COMMIT}/fafbseg/data/JFRC2NP.surf.fw.zip"
SHA256 = "75637dc4aee119211ab20788da0cb6c5a0495cc4bdaf9ae09cd17a7a44cf34b1"
PATH = Path("data/anatomy/flywire/named-surfaces.zip")


def read_ply(body):
    header, raw = body.split(b"end_header\n", 1)
    lines = header.decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in lines or "property list int int vertex_indices" not in lines:
        raise ValueError("Unsupported named surface encoding")
    vertices = int(next(l.split()[-1] for l in lines if l.startswith("element vertex ")))
    faces = int(next(l.split()[-1] for l in lines if l.startswith("element face ")))
    if vertices < 1 or faces < 1 or len(raw) != vertices * 12 + faces * 16: raise ValueError("Invalid named surface size")
    points = np.frombuffer(raw, "<f4", vertices * 3).reshape(-1, 3)
    triangles = np.frombuffer(raw, "<i4", faces * 4, vertices * 12).reshape(-1, 4)
    if not np.all(triangles[:, 0] == 3) or triangles[:, 1:].min() < 0 or triangles[:, 1:].max() >= vertices or not np.isfinite(points).all():
        raise ValueError("Invalid named surface geometry")
    return points, triangles[:, 1:]


def surfaces(map_data):
    if not PATH.exists():
        body = download(SOURCE, limit=2_000_000)
        if hashlib.sha256(body).hexdigest() != SHA256: raise ValueError("Named anatomy download changed")
        PATH.parent.mkdir(parents=True, exist_ok=True); PATH.write_bytes(body)
    if hashlib.sha256(PATH.read_bytes()).hexdigest() != SHA256: raise ValueError("Named anatomy checksum mismatch")
    chunks, parts = [], []
    with zipfile.ZipFile(PATH) as archive:
        for name in sorted(n for n in archive.namelist() if "/" not in n and n.endswith(".ply")):
            vertices, triangles = read_ply(archive.read(name))
            chunks.append(normalize(vertices, map_data)[triangles].reshape(-1, 3))
            parts.append({"id": str(len(parts)), "name": name[:-4], "count": len(triangles) * 3})
    return {"positions_f32": base64.b64encode(np.concatenate(chunks).tobytes()).decode(), "parts": parts,
            "source": SOURCE, "sha256": SHA256,
            "note": "78 named neuropil surfaces from fafbseg, in FlyWire space. Names come from the mesh filenames; old numeric mesh IDs are not reused."}
