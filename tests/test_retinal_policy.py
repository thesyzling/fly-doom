import numpy as np
import pytest

from flydoom.retinal_policy import RetinalActor


def actor():
    return RetinalActor([0,2],np.array([720575940600010668,720575940600010669],np.uint64),[0,0],[1,1])


def test_policy_inputs_only_selected_neural_states_and_exact_root_ids(tmp_path):
    policy=actor();policy.weights[3,:2]=[2,-1]
    a=policy.decide(np.array([1.,10000.,-.5]))
    b=policy.decide(np.array([1.,-10000.,-.5]))
    np.testing.assert_array_equal(a[1],b[1])
    assert a[0]==3
    policy.save(tmp_path/'policy.npz');loaded=RetinalActor.load(tmp_path/'policy.npz')
    assert loaded.root_ids.dtype==np.uint64
    np.testing.assert_array_equal(loaded.root_ids,policy.root_ids)
    np.testing.assert_array_equal(loaded.decide(np.array([1.,0.,-.5]))[1],a[1])


def test_reward_returns_to_policy_with_correct_sign_and_bounded_change():
    for reward in (100.,-100.):
        policy=actor();a,p,x=policy.decide(np.array([.7,0,.1]),np.random.default_rng(1))
        before=policy.weights.copy();update=policy.feedback(x,a,p,reward)
        assert 0<np.linalg.norm(policy.weights-before)<=.002+1e-12
        assert (update['action_probability_after']-p[a])*reward>0
        with pytest.raises(ValueError,match='Stale'):
            policy.feedback(x,a,p,reward)


def test_eligibility_transfers_delayed_reward_and_resets_between_episodes():
    policy=actor();rng=np.random.default_rng(1)
    a,p,x=policy.decide(np.array([1,0,0]),rng);policy.feedback(x,a,p,0)
    assert not policy.weights.any() and policy.eligibility.any()
    first=policy.eligibility.copy()
    a,p,x=policy.decide(np.array([0,0,1]),rng);policy.feedback(x,a,p,100)
    expected=.97*.9*first+np.outer(np.eye(6)[a]-p,x)
    np.testing.assert_allclose(policy.eligibility,expected)
    learned=policy.weights.copy();policy.reset()
    assert not policy.eligibility.any() and policy.updates==0
    np.testing.assert_array_equal(policy.weights,learned)


def test_signed_contributions_reconstruct_selected_action_logit():
    policy=actor();policy.weights=np.arange(18).reshape(6,3)/10
    a,p,x=policy.decide(np.array([.2,0,-.5]))
    terms=policy.contributions(x,a,limit=1)
    assert terms['logit']==pytest.approx(sum(c['contribution'] for c in terms['cells'])+terms['other_terms']+terms['bias'])
    assert terms['all_neural_terms']+terms['bias']==pytest.approx(terms['logit'])


def test_live_learning_toggle_clears_credit_but_keeps_update_count():
    from threading import Condition
    from flydoom.retinal_play import RetinalSession
    session=RetinalSession.__new__(RetinalSession)
    session.condition=Condition();session.paused=True;session.busy=False;session.actor=actor()
    session.actor.eligibility.fill(1);session.actor.updates=7
    session.set_learning(False)
    assert session.learning is False and session.actor.updates==7 and not session.actor.eligibility.any()
    session.busy=True
    with pytest.raises(ValueError):session.set_learning(True)
