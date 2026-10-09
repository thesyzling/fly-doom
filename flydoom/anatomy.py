"""Real FlyWire-space neuropil meshes and bounded, exact-ID v783 skeleton loading."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path("data/anatomy/flywire")
BUCKET = "https://storage.googleapis.com/flywire_neuropil_meshes"
PREFIX = "neuropils/neuropil_mesh_v141.surf_v2/mesh/"
SKELETONS = "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783"


def download(url, limit=8_000_000):
    with urllib.request.urlopen(url, timeout=20) as response:
        body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError("Anatomical asset exceeds its download bound")
    return body


def mesh_arrays(body):
    count = struct.unpack_from("<I", body)[0]
    offset = 4 + count * 12
    if not count or offset >= len(body) or (len(body) - offset) % 12:
        raise ValueError("Invalid precomputed mesh")
    vertices = np.frombuffer(body, "<f4", count * 3, 4).reshape(-1, 3)
    faces = np.frombuffer(body, "<u4", offset=offset).reshape(-1, 3)
    if not np.isfinite(vertices).all() or faces.max() >= count:
        raise ValueError("Invalid mesh coordinates or indices")
    return vertices, faces


def skeleton_arrays(body):
    vertices, edges = struct.unpack_from("<II", body)
    end = 8 + vertices * 12 + edges * 8
    if not vertices or end > len(body):
        raise ValueError("Invalid precomputed skeleton")
    positions = np.frombuffer(body, "<f4", vertices * 3, 8).reshape(-1, 3)
    links = np.frombuffer(body, "<u4", edges * 2, 8 + vertices * 12).reshape(-1, 2)
    if not np.isfinite(positions).all() or (len(links) and links.max() >= vertices):
        raise ValueError("Invalid skeleton coordinates or indices")
    return positions, links


def fetch_surfaces():
    listing = download(BUCKET + "?prefix=" + PREFIX + "&max-keys=1000")
    xml = ET.fromstring(listing)
    ns = {"g": "http://doc.s3.amazonaws.com/2006-03-01"}
    if xml.findtext("g:IsTruncated", namespaces=ns) != "false":
        raise ValueError("Incomplete mesh listing")
    keys = [node.findtext("g:Key", namespaces=ns) for node in xml.findall("g:Contents", ns)]
    keys = sorted(k for k in keys if k.rsplit("/", 1)[-1].count(":") == 2)
    ROOT.mkdir(parents=True, exist_ok=True)
    def fetch(key):
        url = BUCKET + "/" + key
        body = download(url)
        vertices, faces = mesh_arrays(body)
        name = key.rsplit("/", 1)[-1].replace(":", "-") + ".mesh"
        (ROOT / name).write_bytes(body)
        return {"file": name, "region_id": key.rsplit("/", 1)[-1].split(":")[0], "url": url,
                "sha256": hashlib.sha256(body).hexdigest(), "vertices": len(vertices), "triangles": len(faces)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(fetch, keys))
    manifest = {"schema": "flywire_anatomy_v1", "coordinate_space": "FlyWire FAFB14.1", "units": "nanometers",
                "source": BUCKET + "/" + PREFIX, "listing_sha256": hashlib.sha256(listing).hexdigest(),
                "note": "Neuropil region surfaces, not neuron membranes or inferred activity. Region IDs retain upstream numbering.",
                "parts": parts}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"regions": len(parts), "triangles": sum(p["triangles"] for p in parts)}


def normalize(vertices, map_data):
    return ((vertices / 1000 - np.asarray(map_data["center_um"])) * (1.7 / map_data["extent_um"])).astype("<f4")


def surfaces(map_data):
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    chunks, parts = [], []
    for part in manifest["parts"]:
        body = (ROOT / part["file"]).read_bytes()
        if hashlib.sha256(body).hexdigest() != part["sha256"]:
            raise ValueError("Anatomical mesh checksum changed")
        vertices, faces = mesh_arrays(body)
        chunks.append(normalize(vertices, map_data)[faces].reshape(-1, 3))
        parts.append({"id": part["region_id"], "count": len(faces) * 3})
    vertices = np.concatenate(chunks)
    return {"positions_f32": base64.b64encode(vertices.tobytes()).decode(), "parts": parts,
            "note": manifest["note"], "source": manifest["source"]}


def skeleton(root_id, map_data):
    if root_id not in map_data["ids"] or not root_id.isdigit():
        raise ValueError("Skeleton ID is not a neuron in the loaded v783 graph")
    folder = ROOT / "skeletons"
    folder.mkdir(exist_ok=True)
    path = folder / (root_id + ".bin")
    url = SKELETONS + "/" + root_id
    if path.exists():
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != path.with_suffix(".sha256").read_text().strip():
            raise ValueError("Cached skeleton checksum changed")
    else:
        body = download(url)
        skeleton_arrays(body)
        path.write_bytes(body)
        path.with_suffix(".sha256").write_text(hashlib.sha256(body).hexdigest(), encoding="ascii")
        # Raw downloads are a bounded cache; the model and recordings are separate.
        for old in sorted(folder.glob("*.bin"), key=lambda p: p.stat().st_mtime, reverse=True)[16:]:
            old.unlink()
            old.with_suffix(".sha256").unlink(missing_ok=True)
    positions, edges = skeleton_arrays(body)
    lines = normalize(positions, map_data)[edges].reshape(-1, 3)
    return {"id": root_id, "positions_f32": base64.b64encode(lines.tobytes()).decode(),
            "nodes": len(positions), "segments": len(edges), "source": url,
            "sha256": hashlib.sha256(body).hexdigest(), "note": "Actual v783 neuron skeleton; a geometric tree, not synapse locations."}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(fetch_surfaces())
