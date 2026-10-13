import json

import numpy as np
import pytest

from flydoom.adaptive_policy import AdaptiveActor
from flydoom.reward_student_v2 import RewardStudent, distribution
from flydoom.self_learning import assess, publish
from flydoom.stable_bridge import StableBridge
from flydoom.stable_vision import evoked_release, upgrade
from test_bridge_training import circuit


def stable_circuit():
    e = upgrade(circuit())
    e.reset = lambda: e.model.reset()
    return e


def test_corrected_partition_tracks_native_tiny_signals_with_non_neutral_gains():
    e = stable_circuit(); b = StableBridge(e, [11]); gains = b.neutral()
    gains[0] = 1.04; gains[1] = .96; gains[2] = 1.02
    b.apply(gains)
    drive = np.empty((4, e.steps, 2, len(b.nodes)), np.float32)
    reference = []
    for i in range(4):
        for tick in range(e.steps):
            release = evoked_release(e.model.baseline, e.model.delta)
            drive[i,tick,0] = b.positive@release[b.visual]
            drive[i,tick,1] = b.negative@release[b.visual]
            e.model.step(np.array([1e-8*(i+1), 0, 0], np.float32))
        reference.append(e.model.delta[e.outputs].copy())
    replay = b.replay(drive, [True, False, False, False], gains)
    np.testing.assert_allclose(replay, reference, rtol=3e-5, atol=1e-15)
    assert np.any(abs(replay) > 0)
    with pytest.raises(ValueError, match='reset'):
        b.replay(drive, [False]*4, gains)
    with pytest.raises(ValueError, match='corrected'):
        StableBridge(circuit(), [11])


def test_corrected_reward_updates_existing_edges_and_readout_but_not_timing():
    e = stable_circuit(); b = StableBridge(e, [11])
    actor = AdaptiveActor([0], [33], [0], [.1]); actor.weights[1,0] = 1
    drive = np.zeros((4,4,2,2), np.float32); drive[:,:,0,:] = np.arange(1,5)[:,None,None]*.03
    resets = np.array([True,False,False,False]); raw = b.replay(drive,resets,b.neutral())
    p = distribution(actor,raw)[1]
    learner = RewardStudent(b,actor,drive,resets,p,seed=17,relay_directions=4)
    episode = dict(drive=drive,resets=resets,raw=raw,actions=np.ones(4,int),probabilities=p,rewards=np.array([0.,0.,0.,100.]),kills=1)
    before = e.model.weights.copy(); tau=e.model.tau.copy()
    result = learner.update([episode])
    assert result['accepted'] and result['relay_change_l2'] > 0 and result['decoder_change_l2'] > 0
    np.testing.assert_array_equal(e.model.weights.indices,before.indices)
    np.testing.assert_array_equal(np.sign(e.model.weights.data),np.sign(before.data))
    np.testing.assert_array_equal(e.model.weights[0].toarray(),before[0].toarray())
    np.testing.assert_array_equal(e.model.tau,tau)
    with pytest.raises(ValueError,match='Stale'):
        learner.update([episode])


def games(count, hits):
    return [dict(seed=i,kills=int(i in hits),**{'return':100 if i in hits else -100}) for i in range(count)]


def test_training_progress_does_not_qualify_a_candidate_without_game_improvement():
    gate = dict(candidate=games(8,{0,1,2,3}),incumbent=games(8,{0,1,2,3}),decoder_only=games(8,{0,1,2}),disconnected=games(8,{0,1}))
    final = dict(candidate=games(32,set(range(22))),incumbent=games(32,set(range(22))),decoder_only=games(32,set(range(20))),disconnected=games(32,set(range(16))))
    assert not assess(gate,final)['passed']
    final['candidate']=games(32,set(range(24)))
    assert assess(gate,final)['passed']
    final['disconnected']=games(32,set(range(23)))
    assert not assess(gate,final)['passed']


def test_rejection_continues_research_cursor_without_replacing_preferred_model(tmp_path):
    parent=tmp_path/'parent'; first=tmp_path/'first'; second=tmp_path/'second'
    for p in (parent,first,second):
        p.mkdir(); (p/'report.json').write_text(json.dumps({'identity':p.name}))
    assert publish(tmp_path,first,parent,{'passed':False}) == parent
    assert json.loads((tmp_path/'cursor.json').read_text())['path'] == str(first.resolve())
    assert json.loads((tmp_path/'selected.json').read_text())['path'] == str(parent.resolve())
    assert publish(tmp_path,second,parent,{'passed':True}) == second
    assert json.loads((tmp_path/'selected.json').read_text())['path'] == str(second.resolve())
    assert len(json.loads((tmp_path/'history.json').read_text())) == 2
