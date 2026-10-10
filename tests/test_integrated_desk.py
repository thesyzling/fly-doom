import base64
from hashlib import sha256
import json
from pathlib import Path
from threading import Condition, Lock
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from flydoom.data import digest
from flydoom.integrated_desk import DeskServer, IntegratedSession, handler
from flydoom.retinal_play import frame_png
from flydoom.retinal_policy import RetinalActor


@pytest.fixture
def desk(tmp_path):
    session=IntegratedSession.__new__(IntegratedSession)
    session.condition=Condition();session.asset_lock=Lock()
    session.folder=tmp_path;session.sequence=1;session.seed=74000
    session.phase='ready';session.paused=True;session.busy=False;session.teacher_busy=False
    session.learning=True;session.error=None;session.record_cache=None
    ids=np.arange(720575940600010660,720575940600010664,dtype=np.uint64)
    session.full_ids=ids;session.lookup={str(root):i for i,root in enumerate(ids)}
    session.visual_lookup=session.lookup.copy();session.visual_full_indices=np.arange(4)
    session.decoder_lookup={str(ids[3]):0}
    weights=csr_matrix(([.2,-.3,.4],([1,2,3],[0,1,2])),shape=(4,4))
    session.encoder=SimpleNamespace(ids=ids,inputs=np.array([0]),model=SimpleNamespace(
        weights=weights,baseline=np.ones(4),tau=np.array([5.,6.,7.,8.])))
    session.contacts=weights.copy();session.contacts.data=np.array([2,3,4])
    session.outgoing=weights.T.tocsr()
    session.rows=[{'cell_type':label,'pos_x':str(i),'pos_y':'0','pos_z':'0'} for i,label in enumerate(['R1-6','L1','Mi1','T4a'])]
    session.receptors=[{'root_id':str(ids[0]),'l1_root_id':str(ids[1]),'mi1_root_id':str(ids[2])}]
    session.actor=RetinalActor([3],ids[3:],[0],[1])
    session.actor.save(tmp_path/'actor-start.npz')
    frame=np.full((240,320,3),127,np.uint8)
    before=np.arange(12,dtype=float).reshape(6,2)/10
    np.savez_compressed(tmp_path/'0001.npz',root_ids=ids,delta=[.1,-.2,.3,-.4],frame=frame,
                        weights_before=before,weights_after=before+.01,features=[.5,1])
    session.latest={'sequence':1,'frame':frame_png(frame),'probabilities':[1/6]*6,
                    'recording_sha256':digest(tmp_path/'0001.npz','sha256')}
    (tmp_path/'0000.json').write_text(json.dumps({'sequence':0,'frame':frame_png(frame),'probabilities':[1/6]*6}))
    return session


def test_replay_maps_signed_states_and_exact_ids(desk):
    values=np.frombuffer(base64.b64decode(desk.activity(1)['delta_f32']),'<f4')
    np.testing.assert_allclose(values,[.1,-.2,.3,-.4])
    mapping=desk.build_map()
    assert mapping['ids']==list(desk.lookup)
    assert mapping['visual_indices']==[0,1,2,3]
    assert list(base64.b64decode(mapping['roles_u8']))==[1,3,3,2]


def test_neuron_inspection_uses_recorded_weights_and_endpoint_release(desk):
    desk.actor.weights.fill(100)
    detail=desk.inspect(str(desk.full_ids[3]),1)
    np.testing.assert_allclose(detail['decoder']['contributions'],np.arange(6)/10)
    np.testing.assert_allclose(detail['decoder']['weight_changes'],.01)
    assert detail['edges'][0]['contribution']==pytest.approx(.4*.3)
    assert detail['edges'][0]['contacts']==4
    assert detail['tau_ms']==8
    _,initial=desk.record(0)
    assert not initial['weights_before'].any()


def test_replay_rejects_stale_run_and_tampered_record(desk):
    with pytest.raises(ValueError,match='no longer active'):desk.record(1,'old-run')
    with pytest.raises(ValueError,match='Unknown recorded'):desk.record(-1)
    with (desk.folder/'0001.npz').open('ab') as output:output.write(b'tampered')
    with pytest.raises(ValueError,match='checksum'):desk.record(1)


def test_retinal_path_uses_existing_directed_edges(desk):
    path,edges=desk.retinal_path(0)
    assert path==list(desk.lookup)
    assert [e['contacts'] for e in edges]==[2,3,4]
    np.testing.assert_allclose([e['weight'] for e in edges],[.2,-.3,.4])
    desk.outgoing=csr_matrix((4,4))
    path,_=desk.retinal_path(0)
    assert path==list(desk.lookup)[:3]


def test_laya_comparison_is_same_frame_without_action_or_weight_mutation(desk,monkeypatch):
    instances=[]
    class Teacher:
        def __init__(self,log):self.metadata={'revision':'test'};self.closed=False;instances.append(self)
        def predict(self,frame):
            assert desk.teacher_busy
            with pytest.raises(ValueError,match='comparison'):desk.command('step')
            return {'probabilities':[1/6]*6,'frame_sha256':sha256(frame.tobytes()).hexdigest()}
        def close(self):self.closed=True
    monkeypatch.setattr('flydoom.movement_teacher.MovementTeacher',Teacher)
    before=desk.actor.weights.copy()
    result=desk.compare_teacher(1,str(desk.folder))
    assert not result['policy_overridden'] and result['sequence']==1
    np.testing.assert_array_equal(before,desk.actor.weights)
    assert desk.sequence==1 and not desk.teacher_busy and instances[0].closed
    assert desk.compare_teacher(1,str(desk.folder))==result and len(instances)==1
    with pytest.raises(ValueError,match='no longer active'):desk.compare_teacher(1,'old-run')


def test_laya_failure_releases_busy_flag(desk,monkeypatch):
    def fail(log):raise RuntimeError('Unavailable teacher')
    monkeypatch.setattr('flydoom.movement_teacher.MovementTeacher',fail)
    with pytest.raises(RuntimeError,match='Unavailable'):desk.compare_teacher(1)
    assert not desk.teacher_busy
    desk.busy=True
    with pytest.raises(ValueError,match='Pause'):desk.compare_teacher(1)


def test_skeleton_loading_preserves_existing_caches(desk,tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache=Path('data/anatomy/flywire/skeletons');cache.mkdir(parents=True)
    root=str(desk.full_ids[0]);body=b'cached-morphology'
    (cache/(root+'.bin')).write_bytes(body)
    (cache/(root+'.sha256')).write_text(sha256(body).hexdigest())
    for i in range(20):(cache/(str(i)+'.bin')).write_bytes(b'retain')
    before={p.name:p.read_bytes() for p in cache.iterdir()}
    monkeypatch.setattr('flydoom.anatomy.skeleton_arrays',lambda body:(np.array([[0.,0,0],[1000.,0,0]]),np.array([[0,1]])))
    def forbid_download(*args):pytest.fail('Cached morphology must not be downloaded again')
    monkeypatch.setattr('flydoom.anatomy.download',forbid_download)
    desk.map_data={'center_um':[0,0,0],'extent_um':1}
    result=desk.skeleton(root)
    assert result['segments']==1
    assert before=={p.name:p.read_bytes() for p in cache.iterdir()}


def test_server_cannot_silently_share_a_listening_port():
    first=DeskServer(('127.0.0.1',0),handler(None))
    try:
        with pytest.raises(OSError):DeskServer(('127.0.0.1',first.server_port),handler(None))
    finally:first.server_close()


def test_mi1_morphology_reuses_eye_mapping_cache_with_manifest(desk,tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache=Path('runs/eye-mapping-v1/skeletons');cache.mkdir(parents=True)
    root=str(desk.full_ids[2]);body=b'existing-eye-morphology'
    (cache/root).write_bytes(body)
    manifest=cache.parent/'skeleton_manifest.json'
    manifest.write_text(json.dumps({root:{'sha256':sha256(body).hexdigest()}}))
    monkeypatch.setattr('flydoom.anatomy.skeleton_arrays',lambda body:(np.array([[0.,0,0],[1000.,0,0]]),np.array([[0,1]])))
    monkeypatch.setattr('flydoom.anatomy.download',lambda *args:pytest.fail('Do not redownload retained eye morphology'))
    desk.map_data={'center_um':[0,0,0],'extent_um':1}
    assert desk.skeleton(root)['segments']==1
    assert (cache/root).read_bytes()==body
    (cache/root).write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):desk.skeleton(root)
