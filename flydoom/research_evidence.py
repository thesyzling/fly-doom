"""Verified, compact research summaries for the live desk."""
from functools import lru_cache
import json
from pathlib import Path

from flydoom.data import digest


@lru_cache(maxsize=8)
def _read_report(path, modified_ns):
    path = Path(path)
    report = json.loads(path.read_text(encoding='utf-8'))
    for name, sha in report['output_sha256'].items():
        artifact = (path.parent/name).resolve()
        if not artifact.is_relative_to(path.parent.resolve()) or digest(artifact, 'sha256') != sha:
            raise ValueError('Research evidence identity changed: '+name)
    if digest(Path(report['checkpoint'])/'report.json', 'sha256') != report['checkpoint_sha256']:
        raise ValueError('Research evidence checkpoint changed')
    if report.get('schema') == 'independent_calcium_audit_v1':
        if digest(Path(__file__).with_name('independent_physiology.py'), 'sha256') != report['source_sha256']:
            raise ValueError('Independent physiology implementation changed')
    elif report.get('schema') == 'frozen_structure_transfer_v1':
        plan = json.loads((path.parent/'plan.json').read_text(encoding='utf-8'))
        for name, sha in plan['sources'].items():
            if digest(Path(__file__).with_name(name), 'sha256') != sha:
                raise ValueError('Structure audit implementation changed')
    return report


def extension_evidence():
    result = {}
    for key, folder, fields in [
        ('structure', 'runs/structural-transfer-v1', ('checkpoint','topology','transfer','rewiring','scope')),
        ('physiology', 'runs/independent-physiology-v1', ('checkpoint','paper','metrics','blocking_gaps','scope','biological_validation_passed'))]:
        path = Path(folder)/'report.json'
        if path.exists():
            report = _read_report(str(path), path.stat().st_mtime_ns)
            result[key] = {k: report[k] for k in fields}
    return result
