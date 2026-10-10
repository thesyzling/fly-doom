"""Render an offline, inspectable report for the qualitative fitting pilot."""

import json
from pathlib import Path

import numpy as np


def render(output):
    output = Path(output)
    report = json.loads((output / 'report.json').read_text())
    parameters = json.loads((output / 'parameters.json').read_text())
    with np.load(output / 'traces.npz') as saved:
        data = {key: saved[key].tolist() for key in saved.files if key != 'root_ids'}
    payload = json.dumps({'report': report, 'parameters': parameters, 'traces': data}).replace('<', '\\u003c')
    page = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
    (output / 'index.html').write_text(page.replace('__PAYLOAD__', payload), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    render(parser.parse_args().output)
