"""Read-only neural telemetry and exact directed connection inspection."""

import base64
import csv
from pathlib import Path
import struct
import zlib

import numpy as np

from flydoom.learning_data import ACTIONS
from flydoom.action_memory import NAMES as MEMORY_NAMES


def frame_png(frame):
    """Encode an RGB observation with the standard library; no image dependency."""
    frame = np.asarray(frame, dtype=np.uint8)
    height, width, channels = frame.shape
    if channels != 3:
        raise ValueError("Expected an RGB observation")

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    scanlines = np.column_stack((np.zeros(height, dtype=np.uint8), frame.reshape(height, -1)))
    body = b"\x89PNG\r\n\x1a\n"
    body += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    body += chunk(b"IDAT", zlib.compress(scanlines.tobytes(), 3)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(body).decode("ascii")


class ReadoutTelemetry:
    """Capture the actual forward pass without changing it or replaying dynamics."""

    def __init__(self, model):
        self.drive = np.zeros(model.input.out_features, dtype=np.float32)
        self.activity = self.drive.copy()
        self.handles = [model.input.register_forward_hook(self._drive),
                        model.readout.register_forward_hook(self._activity)]

    def _drive(self, module, inputs, output):
        self.drive = output[0].detach().cpu().numpy().copy()

    def _activity(self, module, inputs, output):
        self.activity = inputs[0][0].detach().cpu().numpy().copy()

    def close(self):
        for handle in self.handles:
            handle.remove()


class ConnectionCatalog:
    """Fly CSR rows are targets; CSC columns expose the same outgoing edges."""

    def __init__(self, ids, controller, model, rows):
        self.ids, self.controller, self.model, self.rows = ids, controller, model, rows
        if len(rows) != len(ids) or any(str(i) != r["root_id"] for i, r in zip(ids, rows)):
            raise ValueError("Neuron annotation order mismatch")
        self.lookup = {str(root): i for i, root in enumerate(ids)}
        self.incoming = controller.network.weights
        self.outgoing = self.incoming.tocsc()
        self.outputs = np.sort(np.concatenate(controller.mapping.output_groups))
        self.output_lookup = {int(i): offset for offset, i in enumerate(self.outputs)}
        self.input_lookup = dict(zip(map(int, controller.mapping.input_indices),
                                     map(int, controller.mapping.feature_indices)))
        self.hidden = model.input.out_features
        self.input_weights = model.input.weight.detach().numpy().copy()
        extra = self.input_weights.shape[1] - 2 * len(self.outputs) - 4
        if extra not in (0, len(MEMORY_NAMES)):
            raise ValueError("Unsupported observer input mapping")
        self.memory_names = MEMORY_NAMES if extra else ()
        self.recurrent = model.recurrent.weight.detach().numpy().copy()
        self.action_weights = model.readout.weight.detach().numpy().copy()
        self.action_bias = model.readout.bias.detach().numpy().copy()

    @classmethod
    def from_directory(cls, ids, controller, model, data_dir):
        with (Path(data_dir) / "neuron_annotations.tsv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        return cls(ids, controller, model, rows)

    def label(self, key):
        if key in self.lookup:
            i = self.lookup[key]
            row = self.rows[i]
            return {"id": key, "label": row.get("cell_type") or row.get("super_class") or "Fly neuron",
                    "kind": "input" if i in self.input_lookup else "descending" if i in self.output_lookup else "fly"}
        if key.startswith("student:"):
            i = int(key.split(":")[1])
            if 0 <= i < self.hidden:
                return {"id": key, "label": f"Added cell {i + 1:02d}", "kind": "student"}
        if key.startswith("action:") and key.split(":")[1] in ACTIONS:
            return {"id": key, "label": key.split(":")[1], "kind": "action"}
        if key.startswith("previous:") and key.split(":")[1] in ACTIONS:
            return {"id": key, "label": "Previous " + key.split(":")[1], "kind": "previous"}
        if key.startswith("memory:") and key.split(":")[1] in self.memory_names:
            return {"id": key, "label": key.split(":")[1].replace("_", " "), "kind": "memory"}
        raise KeyError("Unknown neuron or action")

    @staticmethod
    def strongest(values, limit=12):
        return np.argsort(-np.abs(values), kind="stable")[:limit]

    def feature(self, index):
        count = len(self.outputs)
        if index < 2 * count:
            return str(self.ids[self.outputs[index % count]]), "voltage" if index < count else "spike rate"
        offset = index - 2 * count
        if offset < len(self.memory_names):
            return "memory:" + self.memory_names[offset], "causal action memory"
        return "previous:" + ACTIONS[offset - len(self.memory_names)], "previous action indicator"

    def inspect(self, key, sample):
        result = self.label(key)
        edges = []

        def edge(source, target, weight, channel, unit):
            edges.append({"source": self.label(source), "target": self.label(target),
                          "weight": float(weight), "channel": channel, "unit": unit})

        if key in self.lookup:
            i = self.lookup[key]
            row = self.rows[i]
            result.update({"super_class": row.get("super_class"), "cell_type": row.get("cell_type"),
                           "transmitter": row.get("known_nt") or row.get("top_nt") or "Unspecified",
                           "voltage_mv": float(sample["voltage"][i]), "spikes": int(sample["counts"][i]),
                           "rest_mv": self.controller.network.params.rest_mv,
                           "threshold_mv": self.controller.network.params.threshold_mv,
                           "input_pixel_bin": self.input_lookup.get(i)})
            for matrix, direction in ((self.incoming, "incoming"), (self.outgoing, "outgoing")):
                start, end = matrix.indptr[i:i + 2]
                values, neighbors = matrix.data[start:end], matrix.indices[start:end]
                result[direction + "_count"] = len(values)
                for j in self.strongest(values):
                    neighbor = str(self.ids[neighbors[j]])
                    source, target = (neighbor, key) if direction == "incoming" else (key, neighbor)
                    edge(source, target, values[j], "fly synaptic weight", "mV equivalent")
            if i in self.output_lookup:
                offset = self.output_lookup[i]
                for channel, feature in (("normalized voltage", offset), ("normalized spike rate", offset + len(self.outputs))):
                    weights = self.input_weights[:, feature]
                    for j in self.strongest(weights, 4):
                        edge(key, f"student:{j}", weights[j], channel, "learned weight")
            result["explanation"] = "Fly voltages and spikes come from the current 50 ms simulation window. Connection signs are model assumptions; this is not a measured living brain."
        elif result["kind"] == "student":
            i = int(key.split(":")[1])
            result.update({"spikes": int(round(float(sample["activity"][i]) * 8)),
                           "internal_steps": 8, "drive": float(sample["drive"][i])})
            for j in self.strongest(self.input_weights[i]):
                source, channel = self.feature(int(j))
                edge(source, key, self.input_weights[i, j], "normalized " + channel, "learned weight")
            for j in self.strongest(self.recurrent[i], 4):
                edge(f"student:{j}", key, self.recurrent[i, j], "recurrent: 0.1 * tanh(weighted spikes)", "learned weight")
            for j, name in enumerate(ACTIONS):
                edge(key, "action:" + name, self.action_weights[j, i], "mean spike activity to action logit", "learned weight")
            result["explanation"] = "An engineered cell, not a fly neuron. Counts cover eight internal steps, not milliseconds. Inputs are normalized with training statistics; state resets each decision."
        elif result["kind"] == "action":
            i = ACTIONS.index(key.split(":")[1])
            contributions = self.action_weights[i] * sample["activity"]
            result.update({"probability": float(sample["probabilities"][i]),
                           "bias": float(self.action_bias[i]), "logit": float(contributions.sum() + self.action_bias[i])})
            for j in self.strongest(contributions):
                edge(f"student:{j}", key, self.action_weights[i, j], "mean spike activity to action logit", "learned weight")
                edges[-1]["current_logit_contribution"] = float(contributions[j])
            result["explanation"] = "Probability comes from all cells plus the action bias. Listed contributions are arithmetic terms, not proof that a cell caused the behavior. Probabilities are not calibrated confidence."
        elif result["kind"] == "memory":
            i = self.memory_names.index(key.split(":")[1])
            result["value"] = sample["memory"][i]
            feature = 2 * len(self.outputs) + i
            for j in self.strongest(self.input_weights[:, feature]):
                edge(key, f"student:{j}", self.input_weights[j, feature], "causal action memory", "learned weight")
            result["explanation"] = "An engineered record of past applied buttons or elapsed decisions, not a biological neuron. It resets every episode and contains no reward or future action."
        else:
            i = ACTIONS.index(key.split(":")[1])
            result["value"] = int(sample["previous"] == i)
            feature = self.input_weights.shape[1] - 4 + i
            for j in self.strongest(self.input_weights[:, feature]):
                edge(key, f"student:{j}", self.input_weights[j, feature], "normalized previous action indicator", "learned weight")
            result["explanation"] = "An engineering memory input containing the previous applied action. It is not a biological neuron."
        result["edges"] = edges
        result["sequence"] = sample["sequence"]
        result["connection_note"] = "Only a bounded selection of strongest weights is drawn. Arrows point from source to target; layout is schematic, not anatomical."
        return result
