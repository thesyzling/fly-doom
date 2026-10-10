"""Verify the saved retinal actor, causal graph dependency and real reward replay."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from flydoom.data import digest
from flydoom.eye_mapping import write_json
from flydoom.retinal_policy import load_policy, RetinalActor, TYPES, ACTIONS
from flydoom.retinal_train import rollout, metrics, TEACHER


def audit(folder):
    folder=Path(folder);encoder,actor,report=load_policy(folder)
    plan=json.loads((folder/'plan.json').read_text());evaluation=json.loads((folder/'evaluation.json').read_text())
    teacher=json.loads((folder/'distillation.json').read_text())
    np.testing.assert_array_equal(np.unique(encoder.types[encoder.outputs[actor.indices]]),np.array(sorted(TYPES)))
    parts={}
    for split in ('train','validation'):
        with np.load(folder/(split+'.npz')) as data:parts[split]={k:data[k].copy() for k in data.files}
    assert not set(parts['train']['hashes'])&set(parts['validation']['hashes'])
    assert not set(parts['train']['seeds'])&set(parts['validation']['seeds'])
    assert len({r['opening_sha256'] for rows in plan['splits'].values() for r in rows})==24
    distilled=RetinalActor.load(folder/'distilled.npz')
    for split,part in parts.items():
        actual=metrics(distilled,part['raw'],part['targets'])
        assert abs(actual['kl']-teacher['final'][split]['kl'])<1e-10
    raw_row=json.loads((TEACHER/'train/trace.json').read_text())[0]
    frame=np.asarray(Image.open(TEACHER/'train'/raw_row['image']).convert('RGB'))
    encoder.reset();raw,_=encoder.encode(frame)
    np.testing.assert_allclose(raw,parts['train']['raw'][0],atol=1e-8,rtol=1e-5)
    encoder.reset();cut,_=encoder.encode(frame,connected=False)
    assert not cut.any(), 'Disconnected T4/T5 cells must remain silent'
    assert np.linalg.norm(actor.decide(raw)[1]-actor.decide(cut)[1])>1e-6
    # Replay real engine reward and stochastic decoder updates from the initial
    # distilled checkpoint. There is no teacher or scripted action in this loop.
    training=json.loads((folder/'reward-training.json').read_text())
    replay=rollout(encoder,distilled,plan['splits']['reward_train'][0],learn=True,random_seed=90210)
    assert replay['return']==training[0]['return'] and replay['decisions']==training[0]['decisions']
    for actual,saved in zip(replay['trace'],training[0]['trace']):
        assert actual['action']==saved['action'] and actual['reward']==saved['reward']
        np.testing.assert_allclose(actual['probabilities'],saved['probabilities'],atol=1e-8,rtol=1e-7)
        assert abs(actual['update']['weight_delta_l2']-saved['update']['weight_delta_l2'])<1e-10
    for mode in ('test','disconnected'):
        for episode in evaluation[mode]:
            for row in episode['trace']:
                assert row['action']==ACTIONS[int(np.argmax(row['probabilities']))]
                assert row['update'] is None
    result={'passed':True,'report_sha256':digest(folder/'report.json','sha256'),
            'exact_root_ids':True,'teacher_split_disjoint':True,'raw_neural_features_replayed':True,
            'disconnected_output_zero':True,'connected_vs_disconnected_probabilities_differ':True,
            'real_reward_episode_replayed':replay['decisions'],'frozen_test_argmax_verified':True,
            'limitations':'Engineering closed-loop audit, not a physiological or gameplay superiority claim'}
    write_json(folder/'audit.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',type=Path,default=Path('runs/retinal-control-v1'))
    args=parser.parse_args();print(json.dumps(audit(args.checkpoint),indent=2))
