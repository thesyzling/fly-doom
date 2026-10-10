import json
from threading import Condition, Thread
from types import SimpleNamespace

import pytest

from flydoom.live_diagnostics import LiveDiagnostics, episode_end
from flydoom.retinal_play import RetinalSession


@pytest.mark.parametrize('finished,kills,dead,tick,sequence,reason', [
    (False, 0, False, 40, 10, None),
    (True, 1, False, 210, 50, 'target_eliminated'),
    (True, 0, False, 300, 75, 'time_limit'),
    (True, 0, True, 42, 11, 'player_dead'),
    (True, 0, False, 42, 11, 'engine_finished'),
    (False, 0, False, 299, 75, 'decision_limit'),
])
def test_completion_has_an_explicit_reason(finished,kills,dead,tick,sequence,reason):
    game=SimpleNamespace(get_episode_time=lambda:tick,get_episode_timeout=lambda:300,
                         is_player_dead=lambda:dead)
    result=episode_end(game,{'episode_finished':finished,'kill_delta':kills},sequence)
    assert (result['reason'] if result else None)==reason


def test_worker_error_is_saved_with_traceback_and_releases_restart_waiters(tmp_path):
    session=RetinalSession.__new__(RetinalSession)
    session.condition=Condition();session.closed=False;session.paused=False;session.pending=0
    session.phase='ready';session.busy=False;session.sequence=7;session.folder=tmp_path
    session.diagnostics=LiveDiagnostics(tmp_path)
    def fail():raise RuntimeError('engine transport failed')
    session.step=fail
    worker=Thread(target=session.loop,daemon=True);worker.start()
    try:
        with session.condition:
            assert session.condition.wait_for(lambda:session.phase=='error',timeout=3)
            assert session.paused and not session.busy and not session.pending
        records=[json.loads(line) for line in session.diagnostics.path.read_text().splitlines()]
        assert records[-1]['event']=='decision_failed' and records[-1]['sequence']==8
        assert 'RuntimeError: engine transport failed' in records[-1]['traceback']
        assert 'in fail' in records[-1]['traceback']
    finally:
        with session.condition:session.closed=True;session.condition.notify_all()
        worker.join(timeout=3);session.diagnostics.close()
    assert not worker.is_alive()


def test_lifecycle_log_is_durable_and_does_not_overwrite(tmp_path):
    log=LiveDiagnostics(tmp_path);log.event('episode_started',seed=1);log.close()
    log=LiveDiagnostics(tmp_path);log.event('episode_completed',kills=1);log.close()
    rows=[json.loads(line) for line in (tmp_path/'session.log').read_text().splitlines()]
    assert [row['event'] for row in rows]==['episode_started','episode_completed']


def test_terminal_engine_clock_reset_does_not_erase_elapsed_tics():
    game=SimpleNamespace(get_episode_time=lambda:0,get_episode_timeout=lambda:300,
                         is_player_dead=lambda:False)
    effect={'episode_finished':True,'kill_delta':1,'game_tics':2}
    result=episode_end(game,effect,50,action_start_tick=210)
    assert result['game_tick']==212 and result['reason']=='target_eliminated'
