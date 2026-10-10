import hashlib
from http.server import ThreadingHTTPServer
import json
from threading import Thread
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
import pytest

from flydoom.live_recording import write_arrays, observed_action, responsive_clock
from flydoom.movement_core import apply_action
from flydoom.retinal_play import handler


def test_buffered_archive_is_lossless_and_checksum_matches_disk(tmp_path):
    arrays={'root_ids':np.array([720575940600010668],dtype=np.uint64),
            'state':np.array([-.2,0,.1],np.float32),'weights':np.arange(18).reshape(6,3)/10}
    path=tmp_path/'record.npz';checksum=write_arrays(path,**arrays)
    assert checksum==hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as saved:
        for name,value in arrays.items():
            np.testing.assert_array_equal(saved[name],value)
            assert saved[name].dtype==value.dtype


class Game:
    def __init__(self,end=100):self.tic=0;self.end=end;self.buttons=[]
    def get_total_reward(self):return -self.tic
    def get_game_variable(self,variable):return 0
    def is_episode_finished(self):return self.tic>=self.end
    def make_action(self,buttons,tics):self.tic+=tics;self.buttons.append(buttons);return -tics
    def get_state(self):return None if self.is_episode_finished() else SimpleNamespace(screen_buffer=np.full((240,320,3),self.tic,np.uint8))


def test_intermediate_frames_do_not_change_action_or_reward():
    for end in (2,100):
        reference=Game(end);observed=Game(end)
        expected=apply_action(reference,4);actual,frames=observed_action(observed,4)
        assert actual==expected and reference.buttons==observed.buttons
        assert len(frames)==min(end-1,4)
        assert [int(frame[0,0,0]) for frame in frames]==list(range(1,len(frames)+1))
        frames[0][:]=99
        if len(frames)>1:assert frames[1][0,0,0]==2


def test_unchanged_status_omits_frames_but_run_change_forces_full_snapshot():
    snapshot={'sequence':2,'run':'one','phase':'ready','paused':False,'busy':True,
              'learning':True,'error':None,'latest_sequence':2,'seed':1,'frame':'image','action_frames':['a','b'],
              'termination':{'reason':'time_limit'},'diagnostics_log':'session.log'}
    session=SimpleNamespace(state=lambda sequence=None:snapshot)
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(session))
    worker=Thread(target=server.serve_forever,daemon=True);worker.start()
    try:
        def get(query):
            with urlopen(f'http://127.0.0.1:{server.server_port}/api/state'+query) as response:return json.load(response)
        status=get('?since=2&run=one')
        assert status['unchanged'] and status['busy'] and 'frame' not in status and 'action_frames' not in status
        assert status['termination']==snapshot['termination'] and status['diagnostics_log']=='session.log'
        assert get('?since=2&run=older')['frame']=='image'
        assert get('?sequence=2&since=2&run=one')['frame']=='image'
    finally:server.shutdown();server.server_close();worker.join()


def test_timer_request_is_balanced_even_on_engine_error():
    calls=[]
    timer=SimpleNamespace(timeBeginPeriod=lambda n:calls.append(('begin',n)) or 0,
                          timeEndPeriod=lambda n:calls.append(('end',n)))
    with pytest.raises(RuntimeError):
        with responsive_clock(timer):raise RuntimeError('Engine failed')
    assert calls==[('begin',1),('end',1)]
    calls.clear();timer.timeBeginPeriod=lambda n:1
    with responsive_clock(timer):pass
    assert calls==[]
