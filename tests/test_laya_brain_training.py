import numpy as np
import pytest

from flydoom.adaptive_policy import ACTIONS,AdaptiveActor
from flydoom.bridge_training import Bridge
from flydoom.laya_brain_fit import active_groups,fit,joint_fit,targets
from flydoom.laya_brain_teacher import map_probabilities,teacher_actions
from flydoom.laya_brain_training import teacher_label
from flydoom.research_cycle import frame_hash
from test_bridge_training import circuit


def test_teacher_mapping_retains_action_identity_and_no_invented_options():
    names=teacher_actions('basic');assert names==('MOVE_LEFT','MOVE_RIGHT','ATTACK')
    mapped=map_probabilities([.2,.3,.5],names)
    np.testing.assert_array_equal(mapped,[0,.2,.3,.5,0,0,0,0])
    for p,n in [([.3,.3],names),([.2,.3,.6],names),([.5,.5],['ATTACK','ATTACK']),([1],['TELEPORT'])]:
        with pytest.raises(ValueError):map_probabilities(p,n)


def test_soft_targets_require_real_distributions_not_hard_rule_labels():
    with pytest.raises(KeyError):targets([{'action':3}])
    with pytest.raises(ValueError):targets([{'teacher':{'probabilities':[0]*8}}])
    y=map_probabilities([.2,.3,.5],teacher_actions('basic'))
    np.testing.assert_array_equal(targets([{'teacher':{'probabilities':y}}])[0],y)


def test_teacher_rejects_different_frame_or_action_mapping():
    frame=np.zeros((240,320,3),np.uint8)
    class Teacher:
        def predict(self,f):return self.answer
    teacher=Teacher();teacher.answer={'frame_sha256':frame_hash(frame),'actions':ACTIONS,
                                     'probabilities':[0,.2,.3,.5,0,0,0,0]}
    assert teacher_label(teacher,frame) is teacher.answer
    teacher.answer={**teacher.answer,'frame_sha256':'wrong'}
    with pytest.raises(ValueError,match='mismatch'):teacher_label(teacher,frame)


def test_training_starts_from_parent_and_can_keep_epoch_zero():
    a=AdaptiveActor([0],[33],[.2],[.5]);a.weights[3,-1]=2
    raw=np.array([[.1],[.2],[.3]]);y=np.tile([0,0,0,1,0,0,0,0],(3,1))
    frozen,report=fit(raw,y,a,(raw,y),epochs=0)
    np.testing.assert_array_equal(a.weights,frozen.weights)
    np.testing.assert_array_equal(a.mean,frozen.mean)
    assert report['selected_epoch']==0
    learned,report=fit(raw,y,a,(raw,y),epochs=30)
    assert report['final']['validation']['kl']<report['initial']['validation']['kl']
    assert a.weights[3,-1]==2  # Fitting cannot mutate the verified parent.


def test_joint_laya_objective_preserves_visual_rows_and_response_times():
    e=circuit();b=Bridge(e,[11]);a=AdaptiveActor([0],[33],[0],[.1]);a.weights[3,-1]=1
    before=e.model.weights.copy();tau=e.model.tau.copy();active=active_groups(b)
    assert not active[3].any()
    drive=np.zeros((3,4,2,2),np.float32);drive[:,:,0,:]=.1
    rows=[{'teacher':{'probabilities':[0,.1,.2,.7,0,0,0,0]}} for _ in range(3)]
    gains,report=joint_fit(b,drive,[True,False,False],rows,a,pairs=1)
    assert report['final_training_objective']<=report['initial_training_objective']
    np.testing.assert_array_equal(gains[~active],1)
    b.apply(gains)
    np.testing.assert_array_equal(e.model.weights[0].toarray(),before[0].toarray())
    np.testing.assert_array_equal(e.model.tau,tau)
    np.testing.assert_array_equal(e.model.weights.indices,before.indices)
