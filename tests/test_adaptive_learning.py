import copy
from threading import Condition
from types import SimpleNamespace

import numpy as np
import pytest

from flydoom.adaptive_policy import AdaptiveActor, ACTIONS
from flydoom.retinal_policy import RetinalActor
from flydoom.reward_plasticity import RewardPlasticity
from test_bridge_training import circuit


def test_eight_actions_preserve_legacy_and_turn_roundtrip(tmp_path):
    old=RetinalActor([0],[11],[0],[1]);old.weights[3,-1]=2
    actor=AdaptiveActor.from_legacy(old)
    assert actor.decide([0])[0]==old.decide([0])[0]==3
    actor.weights[7,-1]=5
    selected,p,f=actor.decide([0]);assert selected==7 and len(p)==len(ACTIONS)==8
    before=actor.weights.copy();actor.feedback(f,selected,p,100)
    assert actor.weights[7,-1]>before[7,-1]
    actor.save(tmp_path/'actor.npz');restored=AdaptiveActor.load(tmp_path/'actor.npz')
    np.testing.assert_array_equal(actor.weights,restored.weights)
    assert restored.contributions(f,7)['cells']


def test_paired_reward_has_no_update_until_matched_minus_and_preserves_anatomy():
    e=circuit();weights=e.model.weights.copy();tau=e.model.tau.copy();learner=RewardPlasticity(e,[11])
    assert not learner.active[3].any()
    learner.begin(12,'basic')
    with pytest.raises(ValueError):learner.begin(12,'basic')
    assert not learner.observe(100,12,'basic')['updated']
    with pytest.raises(ValueError):learner.observe(0,12,'basic')
    with pytest.raises(ValueError):learner.begin(13,'basic')
    learner.begin(12,'basic');event=learner.observe(0,12,'basic')
    assert event['updated'] and learner.updates==1
    learner.bridge.apply(learner.center)
    np.testing.assert_array_equal(e.model.weights.indices,weights.indices)
    np.testing.assert_array_equal(np.sign(e.model.weights.data),np.sign(weights.data))
    np.testing.assert_array_equal(e.model.weights[0].toarray(),weights[0].toarray())
    np.testing.assert_array_equal(e.model.tau,tau)
    assert np.all(learner.center[~learner.active]==1)


def test_resume_pending_pair_and_rng_exactly():
    a=RewardPlasticity(circuit(),[11],seed=2);a.begin(4,'basic');a.observe(100,4,'basic')
    b=RewardPlasticity(circuit(),[11],seed=99);b.restore(copy.deepcopy(a.snapshot()))
    assert a.begin(4,'basic')==b.begin(4,'basic')
    assert a.observe(20,4,'basic')==b.observe(20,4,'basic')
    assert a.begin(5,'basic')==b.begin(5,'basic')
    a.cancel();assert a.direction is None and not a.trial_open


def test_equal_returns_do_not_learn_and_invalid_state_rejected():
    a=RewardPlasticity(circuit(),[11]);a.begin(1,'basic');a.observe(3,1,'basic')
    a.begin(1,'basic');assert not a.observe(3,1,'basic')['updated']
    np.testing.assert_array_equal(a.center,np.ones_like(a.center))
    s=a.snapshot();s['center'][3][0]=1.01
    with pytest.raises(ValueError):a.restore(s)


def live_stub(tmp_path):
    from flydoom.adaptive_live import AdaptiveSession
    s=AdaptiveSession.__new__(AdaptiveSession);s.condition=Condition()
    s.relay_learner=RewardPlasticity(circuit(),[11]);s.relay_trial=s.relay_learner.begin(4,'basic')
    s.seed=4;s.scenario='basic';s.folder=tmp_path;s.remaining=2;s.auto_learning=True
    s.save_learner=lambda:None;s.log_event=lambda *a,**k:None
    s.paused=False;s.pending=0;s.phase='ready';s.restarting=False;s.teacher_busy=False;s.busy=False
    return s


def test_live_records_preserve_trial_graph_until_the_next_episode(tmp_path):
    s=live_stub(tmp_path);weights=s.relay_learner.bridge.encoder.model.weights.data.copy()
    record={'termination':{'reason':'target_eliminated'},'return':100}
    s.complete_record(record)
    assert record['relay_feedback']['status']=='awaiting_matched_minus'
    assert s.remaining==1 and s.auto_learning
    s.relay_trial=s.relay_learner.begin(4,'basic')
    weights=s.relay_learner.bridge.encoder.model.weights.data.copy()
    record={'termination':{'reason':'decision_limit'},'return':0};s.complete_record(record)
    assert record['relay_feedback']['updated'] and not s.auto_learning and s.remaining==0
    # Updating the learned center cannot rewrite the graph behind recorded decisions.
    np.testing.assert_array_equal(s.relay_learner.bridge.encoder.model.weights.data,weights)


def test_end_discards_incomplete_pair_without_a_false_reward_update(tmp_path):
    s=live_stub(tmp_path);s.command('stop')
    s.complete_record({'termination':{'reason':'time_limit'},'return':100})
    assert not s.auto_learning and s.relay_learner.updates==0 and s.relay_trial is None
    assert not s.relay_learner.trial_open


def test_pause_keeps_trial_but_stops_automatic_advance(tmp_path):
    s=live_stub(tmp_path);s.command('pause')
    assert not s.auto_learning and s.relay_learner.trial_open and s.paused


def test_corridor_decision_limit_is_not_the_basic_limit():
    from flydoom.live_diagnostics import episode_end
    game=SimpleNamespace(get_episode_time=lambda:400,get_episode_timeout=lambda:2100,is_player_dead=lambda:False)
    effect={'episode_finished':False,'kill_delta':0}
    assert episode_end(game,effect,75,decision_limit=150) is None
    assert episode_end(game,effect,150,decision_limit=150)['decision_limit']==150


def test_invalid_new_episode_does_not_discard_pending_learning(tmp_path,monkeypatch):
    from flydoom.integrated_desk import IntegratedSession
    s=live_stub(tmp_path);s.paused=True
    def reject(*a,**kw):raise ValueError('Reserved seed')
    monkeypatch.setattr(IntegratedSession,'new_run',reject)
    with pytest.raises(ValueError,match='Reserved'):s.new_run(12)
    assert s.auto_learning and s.relay_learner.trial_open and s.relay_trial is not None
