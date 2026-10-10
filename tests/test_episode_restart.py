import json
from threading import Condition, Thread
from types import MethodType

import numpy as np
import pytest

from flydoom.retinal_play import RetinalSession
from flydoom.retinal_policy import RetinalActor


def session(tmp_path):
    s=RetinalSession.__new__(RetinalSession)
    s.condition=Condition();s.closed=False;s.busy=False;s.paused=False;s.restarting=False
    s.teacher_busy=False;s.pending=2;s.seed=100;s.phase='ready';s.learning=False;s.checkpoint=tmp_path
    (tmp_path/'plan.json').write_text(json.dumps({'splits':{'test':[{'seed':101}],'validation':[{'seed':102}],'train':[{'seed':103}]}}))
    s.actor=RetinalActor([0],np.array([123],np.uint64),[0],[1]);s.actor.weights.fill(.5)
    s.actor.eligibility.fill(1);s.actor.updates=4;s.resets=[]
    def new_run(self,seed):
        assert not self.busy and self.paused
        self.actor.reset();self.seed=seed;self.phase='ready';self.resets.append(seed)
        return {'run':f'run-{seed}'}
    s.new_run=MethodType(new_run,s)
    return s


@pytest.mark.parametrize('phase',['ready','completed','stopped','error'])
def test_restart_resets_episode_credit_but_retains_weights_and_learning_mode(tmp_path,phase):
    s=session(tmp_path);s.phase=phase;weights=s.actor.weights.copy()
    assert s.restart_episode()=={'run':'run-103','seed':103}
    assert s.phase=='ready' and not s.paused and not s.restarting and not s.pending
    assert not s.learning and not s.actor.eligibility.any() and s.actor.updates==0
    np.testing.assert_array_equal(s.actor.weights,weights)
    assert s.restart_episode()['seed']==104


def test_restart_waits_for_current_record_and_rejects_concurrent_commands(tmp_path):
    s=session(tmp_path);s.busy=True;result=[]
    thread=Thread(target=lambda:result.append(s.restart_episode()));thread.start()
    with s.condition:
        assert s.condition.wait_for(lambda:s.restarting,timeout=2)
        assert s.paused and not s.pending and not s.resets
        with pytest.raises(ValueError,match='progress'):s.restart_episode()
        with pytest.raises(ValueError,match='progress'):s.command('run')
        s.actor.weights+=.1  # Final feedback belongs to the old completed record.
        s.busy=False;s.condition.notify_all()
    thread.join(timeout=2)
    assert not thread.is_alive() and result[0]['seed']==103
    np.testing.assert_allclose(s.actor.weights,.6)


def test_restart_refuses_teacher_work_and_closed_session(tmp_path):
    s=session(tmp_path);s.teacher_busy=True
    with pytest.raises(ValueError,match='Laya'):s.restart_episode()
    assert not s.restarting and not s.resets
    s.teacher_busy=False;s.closed=True
    with pytest.raises(ValueError,match='closed'):s.restart_episode()
