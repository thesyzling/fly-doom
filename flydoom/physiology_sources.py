"""Fetch the versioned, CC0 Gou et al. visual physiology archive."""

import hashlib
import argparse
import ast
from pathlib import Path
import urllib.request
import zipfile

from flydoom.eye_mapping import write_json

ROOT = Path('data/raw/dryad-sparsity-v401319')
URL = 'https://datadryad.org/downloads/file_stream/4404307'
SIZE = 180331689
SHA256 = '83a2bc0c5e1ce64787a30183f486d219c377014014b5bac5448f6b17502bc35f'
REFERENCE_ROOT = Path('data/raw/flyvis-physiology-92b3845')
REFERENCE_REVISION = '92b3845cc426dd309a1a0e1b3890156c42e14021'
REFERENCE_BLOB = '999012cea4bcf67a997d3ecd72bb5f11c7bb0b30'


def reference_targets():
    """Read a pinned qualitative literature table without executing author code."""
    path = REFERENCE_ROOT / 'groundtruth_utils.py'
    body = path.read_bytes()
    blob = hashlib.sha1(f'blob {len(body)}\0'.encode() + body).hexdigest()
    if blob != REFERENCE_BLOB:
        raise ValueError('Reference Git blob mismatch')
    for node in ast.parse(body).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'polarity' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing qualitative polarity table')


def fetch_reference():
    REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
    url = f'https://raw.githubusercontent.com/TuragaLab/flyvis/{REFERENCE_REVISION}/flyvis/utils/groundtruth_utils.py'
    path = REFERENCE_ROOT / 'groundtruth_utils.py'
    if not path.exists():
        with urllib.request.urlopen(url, timeout=45) as response:
            body = response.read(100_001)
        if len(body) > 100_000 or hashlib.sha1(f'blob {len(body)}\0'.encode() + body).hexdigest() != REFERENCE_BLOB:
            raise ValueError('Reference download integrity check failed')
        path.write_bytes(body)
    reference_targets()
    license_path = REFERENCE_ROOT / 'license'
    if not license_path.exists():
        with urllib.request.urlopen(f'https://raw.githubusercontent.com/TuragaLab/flyvis/{REFERENCE_REVISION}/license', timeout=45) as response:
            license_path.write_bytes(response.read(100_000))
    write_json(REFERENCE_ROOT / 'manifest.json', {
        'schema': 'qualitative_physiology_reference_v1', 'revision': REFERENCE_REVISION,
        'url': url, 'git_blob': REFERENCE_BLOB,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'license_sha256': hashlib.sha256(license_path.read_bytes()).hexdigest(),
        'paper': 'https://doi.org/10.1038/s41586-024-07939-3',
        'evidence': 'Author literature synthesis of qualitative ON/OFF polarity; not raw recordings',
        'numeric_tau_or_amplitude_targets': False})
    print('Verified qualitative reference (no measured response traces)', flush=True)


def fetch():
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'sparsity_Dryad_upload.zip'
    if not archive.exists():
        partial = ROOT / 'download.partial'
        with urllib.request.urlopen(URL, timeout=45) as response, partial.open('wb') as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > SIZE:
                    raise ValueError('Archive exceeds declared size')
                output.write(chunk)
        if total != SIZE or hashlib.sha256(partial.read_bytes()).hexdigest() != SHA256:
            raise ValueError('Dryad archive integrity check failed')
        partial.replace(archive)
    if archive.stat().st_size != SIZE or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Cached archive integrity check failed')
    files = []
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            name = entry.filename
            if name.startswith('__MACOSX/') or entry.is_dir():
                continue
            # Keep author analysis and flash data; do not expand unrelated recordings.
            if not ('/scripts/' in name or '/utilities/' in name or '/fig4_otherNeurons_flash/' in name):
                continue
            target = (ROOT / name).resolve()
            if not target.is_relative_to(ROOT.resolve()) or entry.file_size > 30_000_000:
                raise ValueError('Unsafe or oversized archive member')
            body = source.read(entry)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(body)
            if target.read_bytes() != body:
                raise ValueError('Cached extraction differs from archive')
            files.append({'path': name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()})
    write_json(ROOT / 'manifest.json', {
        'schema': 'physiology_source_v1', 'doi': '10.5061/dryad.t1g1jwtbs',
        'paper': 'https://doi.org/10.1016/j.cub.2024.10.053',
        'license': 'CC0-1.0', 'file_version': 401319, 'metadata_version': 402168,
        'url': URL, 'archive_bytes': SIZE, 'archive_sha256': SHA256, 'files': files})
    print(f'Verified archive and {len(files)} extracted files', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', action='store_true', help='Also attempt the measured Dryad archive download')
    args = parser.parse_args()
    fetch_reference()
    if args.recordings:
        fetch()
