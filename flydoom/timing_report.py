"""Offline measured-versus-simulated temporal physiology report."""

import argparse
import json
from pathlib import Path

import numpy as np


def render(output):
    output = Path(output)
    report = json.loads((output / 'report.json').read_text())
    with np.load(output / 'traces.npz') as data:
        traces = {k:data[k].tolist() for k in data.files}
    body = json.dumps({'report': report, 'traces': traces}).replace('<', '\\u003c')
    template = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
    (output / 'index.html').write_text(template.replace('__PAYLOAD__', body), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    render(parser.parse_args().output)
