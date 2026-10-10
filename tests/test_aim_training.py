from types import SimpleNamespace
import numpy as np
import pytest

from flydoom import aim_training as aim
from flydoom.retinal_policy import RetinalActor


def scene(x,width=30):
    return SimpleNamespace(labels=[SimpleNamespace(object_name='Cacodemon',x=x,y=100,width=width,height=30)],
                           screen_buffer=np.zeros((240,320,3),np.uint8))


@pytest.mark.parametrize('x,expected',[(50,1),(230,2),(145,3)])
def test_training_annotation_aligns_before_firing(x,expected):
    assert aim.target(scene(x))['action']==expected


def test_no_target_produces_no_fabricated_training_label():
    state=scene(145);state.labels[0].object_name='DoomPlayer'
    assert aim.target(state) is None


def test_evaluation_never_overrides_neural_action_with_target_metadata(monkeypatch):
    actor=RetinalActor([0],np.array([123],np.uint64),[0],[1]);actor.weights[3,-1]=10
    encoder=SimpleNamespace(reset=lambda:None,encode=lambda frame:(np.array([.1]),{}))
    actions=[]
    for x in (50,230):
        state=scene(x);game=SimpleNamespace(get_state=lambda:state,is_episode_finished=lambda:False,
                                          get_total_reward=lambda:0,close=lambda:None)
        monkeypatch.setattr(aim,'make_game',lambda seed:game)
        def action(game,chosen):
            actions.append(chosen);return {'reward_delta':0,'kill_delta':0}
        monkeypatch.setattr(aim,'apply_action',action)
        row,examples=aim.episode(encoder,actor,{'seed':1,'task':'basic','opening_sha256':aim.frame_hash(state.screen_buffer)},limit=1)
        assert not examples
    assert actions==[3,3]
