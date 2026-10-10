"""Replay the C2/C3 candidate and compare feedback at a matched time step."""

import argparse
import json

import numpy as np

from flydoom.data import digest
from flydoom.eye_mapping import write_json
from flydoom.retinal_feedback import OUTPUT, PARENT, prepare, build, simulate, load_checkpoint
from flydoom.timing_audit import audit as audit_parent
from flydoom.timing_sources import load_split
from flydoom.timing_training import fit_scales, metrics


def audit(output=OUTPUT):
    ids, model, report = load_checkpoint(output)
    audit_parent(PARENT)
    assert digest(PARENT/'report.json','sha256') == report['parent_report_sha256']
    system = prepare()
    np.testing.assert_array_equal(ids, system['ids'])
    pars = report['selected']['parameters']
    rebuilt = build(system, pars)
    np.testing.assert_array_equal(rebuilt.weights.indptr, model.weights.indptr)
    np.testing.assert_array_equal(rebuilt.weights.indices, model.weights.indices)
    np.testing.assert_allclose(rebuilt.weights.data, model.weights.data, atol=0, rtol=0)
    np.testing.assert_array_equal(rebuilt.tau, model.tau)
    time, train = load_split('train')
    _, comparison = load_split('test')
    scales = np.sqrt(np.mean(train**2, axis=(1,2)))
    gains, prediction = fit_scales(simulate(system, pars, time), train)
    with np.load(output/'traces.npz') as saved:
        np.testing.assert_allclose(saved['fitted'], prediction, atol=1e-10, rtol=1e-7)
        np.testing.assert_array_equal(saved['train'], train)
        np.testing.assert_array_equal(saved['comparison'], comparison)
    _, baseline = fit_scales(simulate(system, [*pars[:2], 0, 0], time), train)
    result = {'passed': True, 'report_sha256': digest(output/'report.json','sha256'),
              'matched_dt_no_feedback_train': metrics(baseline, train, time, scales),
              'matched_dt_no_feedback_seen_comparison': metrics(baseline, comparison, time, scales),
              'candidate_seen_comparison': metrics(prediction, comparison, time, scales),
              'time_constants_near_lower_bound': bool(any(x < 2.01 for x in pars[:2])),
              'interpretation': 'Replay and graph identity verified. Bound-hitting taus are not identified biological constants. Late phase remains unresolved.'}
    write_json(output/'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(audit(), indent=2))
