"""Pinned measured L1/L2 impulse responses and their author export protocol.

Public author data is retained locally with attribution, without relicensing or
executing upstream analysis code. The repository has no explicit data license.
"""

import argparse
import hashlib
from pathlib import Path
import urllib.request

import numpy as np
from scipy.io import loadmat

from flydoom.eye_mapping import write_json

REVISION = '7fa5829e37d566e02beaaa87efd6a0f1de4e48c0'
ROOT = Path('data/raw/pang-timing') / REVISION
REPOSITORY = 'https://github.com/ClandininLab/L1L2-recurrent-feedback'
FILES = {
    'computational-model/L1_2responses_to_flashes.ipynb': '2bc8d15e349d6c764cf17fee28bda55c6477defe',
    'computational-model/data/L1_highLum.mat': '7c392525939485066bd2488004c495656cfd90a3',
    'computational-model/data/L1_lowLum.mat': 'e4b75cf1cee86c7c46ddaca3f2a6d427605b28e5',
    'computational-model/data/L2_highLum.mat': '745fc159b808f14b7876cdfcdfad12a96ea7473a',
    'computational-model/data/L2_lowLum.mat': '1d6cc9f26c015a85c5dfbbce687d4449a79b65fc',
    'imaging-analysis/HHY_stimulusSpecificAnalysisScripts/shortFlashProcessed_saveMean.m': '1faaf2b042ae4165128f76561d03ea0fbf116ef1',
    'imaging-analysis/HHY_stimulusSpecificAnalysisScripts/selectTimeSeries_L2Project_FFFoG.m': '19c63a9da516be4c704abf930a9262ffb75f3733',
    'imaging-analysis/compute_meanResp_err_moveAvg_FFFoG.m': '58fb0dddc6f44d48ccbcefa9a580fbaf1f795f61',
}


def verify(body, expected):
    if hashlib.sha1(f'blob {len(body)}\0'.encode() + body).hexdigest() != expected:
        raise ValueError('Pinned timing-source Git blob mismatch')


def fetch():
    records = []
    for name, blob in FILES.items():
        path = ROOT / name
        url = f'https://raw.githubusercontent.com/ClandininLab/L1L2-recurrent-feedback/{REVISION}/{name}'
        if path.exists():
            body = path.read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=40) as response:
                body = response.read(1_000_001)
        if len(body) > 1_000_000:
            raise ValueError('Timing source exceeds declared download bound')
        verify(body, blob)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(body)
        records.append({'path': name, 'url': url, 'git_blob': blob, 'bytes': len(body),
                        'sha256': hashlib.sha256(body).hexdigest()})
    write_json(ROOT / 'timing_manifest.json', {
        'schema': 'measured_timing_sources_v1', 'repository': REPOSITORY, 'revision': REVISION,
        'authors': 'Pang, Chen, Xie, Druckmann, Clandinin and Yang',
        'publication': 'A recurrent neural circuit in Drosophila temporally sharpens visual inputs',
        'publication_url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11769683/',
        'methods_preprint': 'https://doi.org/10.1101/2024.04.19.590352',
        'license': 'No explicit repository license found; retained author files for local research',
        'files': records, 'dryad_required': False})
    print(f'Verified {len(records)} timing source files; {sum(r["bytes"] for r in records):,} bytes')


def decode_measurements(path):
    """Return dark/light baseline-centered negative dF/F on the source time axis."""
    data = loadmat(path, simplify_cells=True)
    time_ms = np.asarray(data['t'], dtype=np.float64) * 1000
    raw = np.asarray(data['meanResp'], dtype=np.float64)
    if raw.shape != (2, 63) or time_ms.shape != (63,) or not np.isfinite(raw).all():
        raise ValueError('Expected two finite 63-sample population-mean traces')
    if not np.allclose(time_ms, np.arange(63) * 1000 / 120, atol=1e-8):
        raise ValueError('Unexpected timing axis; do not silently resample')
    # Export rows are dark then light. ASAP2f fluorescence falls on depolarization.
    voltage_sign = -(raw - raw[:, :1])
    return time_ms, voltage_sign


def load_split(split):
    if split not in ('train', 'test'):
        raise ValueError('Unknown timing split')
    suffix = 'highLum' if split == 'train' else 'lowLum'
    responses = []
    for cell in ('L1', 'L2'):
        name = f'computational-model/data/{cell}_{suffix}.mat'
        verify((ROOT / name).read_bytes(), FILES[name])
        time_ms, values = decode_measurements(ROOT / name)
        responses.append(values)
    return time_ms, np.stack(responses)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    fetch()
