import json
from pathlib import Path
from threading import Condition
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from flydoom.data import digest
from flydoom.graded_vision import GradedVision
from flydoom.retinal_policy import RetinalEncoder, RetinalActor, WINDOW_MS
from flydoom.sensorimotor import SensorimotorEncoder, save_candidate, load_candidate, SCHEMA
from flydoom.research_cycle import paired_gate, decoder_fit
from flydoom import research_registry as registry


def encoder():
    parent=RetinalEncoder.__new__(RetinalEncoder)
    parent.ids=np.arange(720575940600010660,720575940600010665,dtype=np.uint64)
    parent.types=np.array(['R1-6','L1','L2','C2','C3'])
    parent.inputs=np.array([0]);parent.outputs=np.array([3,4]);parent.steps=138
    parent.uv=np.array([[.5,.5]]);parent.visible=np.array([True]);parent.directions=np.array([[0,0,1.]])
    parent.camera={};parent.gains=np.ones(8)
    weights=csr_matrix(([-.2,-.3,.4,-.1],([1,2,3,4],[0,0,1,2])),shape=(5,5),dtype=np.float32)
    parent.model=GradedVision(weights,np.full(5,20.,np.float32),dt_ms=1000/120/10)
    parent.model.decay=np.exp(-(WINDOW_MS/parent.steps)/parent.model.tau);parent.reset()
    return parent


def test_no_adaptation_reproduces_parent_and_disconnection_removes_outputs():
    parent=encoder();candidate=SensorimotorEncoder.wrap(parent);frame=np.full((240,320,3),200,np.uint8)
    expected,_=parent.encode(frame);actual,_=candidate.encode(frame)
    np.testing.assert_array_equal(expected,actual)
    candidate.reset();actual,_=candidate.encode(frame,connected=False)
    assert not actual.any()


def test_positive_plasticity_preserves_support_signs_and_enforces_bounds():
    model=SensorimotorEncoder.wrap(encoder());old=model.model.weights.copy()
    model.set_plastic(np.full(len(model.gains),.9))
    np.testing.assert_array_equal(old.indices,model.model.weights.indices)
    np.testing.assert_array_equal(np.sign(old.data),np.sign(model.model.weights.data))
    np.testing.assert_allclose(model.model.weights.data,.9*old.data)
    with pytest.raises(ValueError):model.set_plastic(np.full(len(model.gains),1.2))
    with pytest.raises(ValueError):model.set_plastic(np.full(len(model.gains),np.nan))


def test_timing_changes_state_and_reset_clears_adaptation():
    model=SensorimotorEncoder.wrap(encoder());frame=np.full((240,320,3),200,np.uint8)
    baseline,_=model.encode(frame);model.reset();model.set_timing({'L1':5},.9,40,1.2)
    changed,_=model.encode(frame)
    assert model.adaptation.any() and not np.allclose(baseline,changed)
    model.reset();assert not model.adaptation.any() and not model.model.delta.any()
    assert model.model.tau[1]==5


def test_checkpoint_roundtrip_exact_states_and_rejects_changed_artifacts(tmp_path):
    model=SensorimotorEncoder.wrap(encoder());model.set_timing({'L1':8},.7,30,1.3)
    actor=RetinalActor([0,1],model.ids[model.outputs],[0,0],[1,1]);save_candidate(tmp_path,model,actor)
    names=['weights.npz','circuit.npz','circuit.json','actor.npz']
    (tmp_path/'report.json').write_text(json.dumps({'schema':SCHEMA,'status':'completed','source_sha256':{},
                                                'output_sha256':{n:digest(tmp_path/n,'sha256') for n in names}}))
    loaded,other,_=load_candidate(tmp_path);frame=np.full((240,320,3),50,np.uint8)
    np.testing.assert_array_equal(loaded.encode(frame)[0],model.encode(frame)[0])
    np.testing.assert_array_equal(loaded.ids,model.ids)
    (tmp_path/'actor.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='artifact changed'):load_candidate(tmp_path)


def rows(value=1):return [{'task':task,'seed':i,'return':value,'kills':1} for i,task in enumerate(('basic','deadly_corridor'))]


def test_gate_requires_improvement_without_cross_map_regression():
    assert not paired_gate(rows(),rows())['accepted']
    assert paired_gate(rows(),rows(2))['accepted']
    after=rows(2);after[0]['return']=0
    assert not paired_gate(rows(),after)['accepted']
    after=rows();after[0]['return']=float('nan')
    with pytest.raises(ValueError):paired_gate(rows(),after)
    with pytest.raises(ValueError):paired_gate(rows(),rows()+rows())


def test_registry_preserves_champion_on_rejection_and_rolls_back_verified_promotion(tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.mkdir();(parent/'report.json').write_text('{}')
    monkeypatch.setattr(registry,'INITIAL',parent);monkeypatch.setattr(registry,'verify',lambda item:Path(item['path']))
    candidate=tmp_path/'candidate';candidate.mkdir();report=candidate/'report.json'
    report.write_text(json.dumps({'promotion':{'accepted':False}}))
    state=registry.register(candidate,parent,tmp_path/'registry')
    assert state['champion']==registry.record(parent) and state['latest']==registry.record(candidate)
    report.write_text(json.dumps({'promotion':{'accepted':True}}))
    state=registry.register(candidate,parent,tmp_path/'registry');assert state['champion']==registry.record(candidate)
    state=registry.rollback(tmp_path/'registry');assert state['champion']==registry.record(parent)


def test_cycle_lock_is_exclusive_and_retains_other_files(tmp_path):
    (tmp_path/'cache.bin').write_bytes(b'retain')
    with registry.CycleLock(tmp_path):
        with pytest.raises(FileExistsError):
            with registry.CycleLock(tmp_path):pass
    assert (tmp_path/'cache.bin').read_bytes()==b'retain' and not (tmp_path/'cycle.lock').exists()


def test_continued_decoder_keeps_learned_normalization_and_initial_weights():
    model=SensorimotorEncoder.wrap(encoder());model.output_kind='descending'
    parent=RetinalActor([0,1],model.ids[model.outputs],[.2,.3],[2,3]);parent.weights[3,-1]=1
    raw=np.array([[.1,.2],[.2,.3],[.3,.5]])
    target=np.array([parent.decide(r)[1] for r in raw]);parts={s:{'raw':raw,'targets':target} for s in ('train','validation')}
    actor,report=decoder_fit(model,parts,parent)
    assert report['initial']['validation']['kl']==pytest.approx(0,abs=1e-12)
    np.testing.assert_array_equal(actor.mean,parent.mean);np.testing.assert_array_equal(actor.scale,parent.scale)


def test_resume_rejects_another_circuit_before_loading_actor(tmp_path):
    from flydoom.retinal_play import RetinalSession
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir();(checkpoint/'report.json').write_text('{}')
    old=tmp_path/'live';old.mkdir();(old/'identity.json').write_text(json.dumps({'checkpoint_sha256':'different'}))
    session=RetinalSession.__new__(RetinalSession);session.checkpoint=checkpoint
    with pytest.raises(ValueError,match='another circuit'):session.resume_live(old)
