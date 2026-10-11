from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from flydoom.bridge_training import Bridge, ordered
from flydoom.graded_vision import GradedVision
from flydoom.retinal_policy import WINDOW_MS


def circuit():
    weights=csr_matrix(np.array([[0,0,0],[.2,.1,-.1],[-.1,.3,0]],np.float32))
    model=GradedVision(weights,np.array([10,20,20],np.float32),dt_ms=WINDOW_MS/4)
    encoder=SimpleNamespace(ids=np.array([11,22,33],np.uint64),types=np.array(['visual','relay','descending']),
                            inputs=np.array([0]),outputs=np.array([2]),steps=4,model=model,reset=model.reset)
    return encoder


def test_joint_parameters_preserve_visual_rows_anatomy_signs_and_bounds():
    e=circuit();b=Bridge(e,[11]);before=e.model.weights.copy();tau=e.model.tau.copy()
    p=b.neutral();p[0]=1.2;p[1]=.8;p[2]=1.1;p[3]=1.25;b.apply(p)
    np.testing.assert_array_equal(e.model.weights[0].toarray(),before[0].toarray())
    np.testing.assert_array_equal(e.model.weights.indices,before.indices)
    np.testing.assert_array_equal(np.sign(e.model.weights.data),np.sign(before.data))
    assert e.model.tau[0]==tau[0] and np.all(e.model.tau[1:]==tau[1:]*1.25)
    assert np.asarray(abs(e.model.weights).sum(1)).max()<.99
    with pytest.raises(ValueError,match='multipliers'):b.apply(p*2)
    b.apply(b.neutral());np.testing.assert_array_equal(e.model.weights.data,before.data)


def test_cached_drive_matches_full_dynamics_with_learned_gains_and_timing():
    e=circuit();b=Bridge(e,[11]);p=b.neutral();p[0]=1.2;p[1]=.8;p[2]=1.1;p[3]=1.25;b.apply(p)
    drive=np.empty((3,4,2,2),np.float32);reference=[]
    for frame in range(3):
        for tick in range(4):
            release=np.maximum(e.model.baseline+e.model.delta,0)-np.maximum(e.model.baseline,0)
            drive[frame,tick,0]=b.positive@release[b.visual]
            drive[frame,tick,1]=b.negative@release[b.visual]
            e.model.step(np.array([.1*(frame+1),0,0],np.float32))
        reference.append(e.model.delta[e.outputs].copy())
    replay=b.replay(drive,[True,False,False],p)
    np.testing.assert_allclose(replay,reference,rtol=2e-6,atol=1e-8)
    assert not b.replay(np.zeros_like(drive),[True,False,False],p).any()


def test_visual_feedback_blocks_caching_and_missing_frames_block_training():
    e=circuit();w=e.model.weights.toarray();w[0,2]=.1;e.model.weights=csr_matrix(w)
    with pytest.raises(ValueError,match='feedback'):Bridge(e,[11])
    with pytest.raises(ValueError,match='Missing causal'):ordered([{'seed':1,'step':0},{'seed':1,'step':2}])
    assert [r['step'] for r in ordered([{'seed':1,'step':1},{'seed':1,'step':0}])]==[0,1]
