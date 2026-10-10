"""Standalone, offline inspection of measured mapping and visual diagnostics."""

import json
from pathlib import Path


def render(output, mapping):
    output, mapping = Path(output), Path(mapping)
    payload = {"mapping": json.loads((mapping / "report.json").read_text()),
               "columns": json.loads((mapping / "columns.json").read_text()),
               "probe": json.loads((output / "report.json").read_text())}
    template = Path(__file__).with_name("visual_report.html").read_text(encoding="utf-8")
    text = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__REPORT_DATA__", text), encoding="utf-8")
