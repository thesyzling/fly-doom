"""Read-only anatomical anchors and bounded learned connections for the live map."""
import base64
import numpy as np


def packed(array, dtype):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode("ascii")


def build_map(catalog):
    """Preserve exact IDs; engineered cells deliberately have no anatomical position."""
    positions = np.array([[float(row[f"pos_{axis}"]) for axis in "xyz"] for row in catalog.rows])
    positions *= [0.004, 0.004, 0.040]
    if not np.isfinite(positions).all():
        raise ValueError("Map requires finite neuron anchor coordinates")
    extent = float(np.ptp(positions, axis=0).max())
    if extent <= 0:
        raise ValueError("Map coordinates have no spatial extent")
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    normalized = ((positions - center) * (1.7 / extent)).astype(np.float32)
    roles = np.zeros(len(positions), dtype=np.uint8)
    roles[list(catalog.input_lookup)] = 1
    roles[catalog.outputs] = 2
    edges = []
    def edge(source, target, weight, channel):
        if weight:
            edges.append({"source": source, "target": target, "weight": float(weight), "channel": channel})
    for i, weights in enumerate(catalog.input_weights):
        for j in catalog.strongest(weights[:2 * len(catalog.outputs)], 2):
            source, channel = catalog.feature(int(j))
            edge(source, f"student:{i}", weights[j], channel)
    for i, name in enumerate(catalog.memory_names):
        weights = catalog.input_weights[:, 2 * len(catalog.outputs) + i]
        j = int(np.argmax(np.abs(weights)))
        edge("memory:" + name, f"student:{j}", weights[j], "causal action memory")
    for i, name in enumerate(("WAIT", "MOVE_LEFT", "MOVE_RIGHT", "ATTACK")):
        for j in catalog.strongest(catalog.action_weights[i], 4):
            edge(f"student:{j}", "action:" + name, catalog.action_weights[i, j], "action readout")
    return {"ids": [str(root) for root in catalog.ids], "positions_f32": packed(normalized, "<f4"),
            "roles_u8": packed(roles, "u1"), "edges": edges,
            "center_um": center.tolist(), "extent_um": extent,
            "coordinate_note": "Real neuron anchor positions; not cell morphology. Source axes retained. Engineered cells use a schematic layout.",
            "edge_note": "Two strongest neural inputs per added cell, strongest destination per memory input, four strongest inputs per action. Select a node for local directed connections."}


def activity_snapshot(sample):
    counts = np.asarray(sample["counts"])
    if np.any(counts < 0) or np.any(counts > 65535):
        raise ValueError("Spike counts exceed map encoding bounds")
    return {"sequence": sample["sequence"], "counts_u16": packed(counts, "<u2")}
