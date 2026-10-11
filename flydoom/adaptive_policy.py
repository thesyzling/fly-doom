"""Versioned eight-action deployment; legacy six-action checkpoints stay intact."""
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse
import vizdoom as vzd

from flydoom.data import digest
from flydoom.graded_vision import GradedVision
from flydoom.retinal_policy import RetinalActor, WINDOW_MS
from flydoom.sensorimotor import SensorimotorEncoder, plastic_groups, load_any as load_legacy, save_candidate

ACTIONS=('WAIT','MOVE_LEFT','MOVE_RIGHT','ATTACK','MOVE_FORWARD','MOVE_BACKWARD','TURN_LEFT','TURN_RIGHT')
TASKS=('basic','predict_position','deadly_corridor')
SCHEMA='adaptive_sensorimotor_v1'


class AdaptiveActor(RetinalActor):
    actions=ACTIONS

    def __init__(self,indices,root_ids,mean,scale,weights=None):
        super().__init__(indices,root_ids,mean,scale)
        self.weights=np.zeros((len(ACTIONS),len(self.indices)+1)) if weights is None else np.asarray(weights,float).copy()
        if self.weights.shape!=(len(ACTIONS),len(self.indices)+1) or not np.isfinite(self.weights).all():raise ValueError('Invalid eight-action decoder')
        self.reset()

    @classmethod
    def from_legacy(cls,actor):
        weights=np.zeros((8,len(actor.indices)+1));weights[:6]=actor.weights;weights[6:,-1]=-20
        return cls(actor.indices,actor.root_ids,actor.mean,actor.scale,weights)

    def decide(self,raw,rng=None):
        from flydoom.retinal_policy import softmax
        features=self.features(raw);p=softmax(self.weights@features)
        action=int(np.argmax(p)) if rng is None else int(rng.choice(len(ACTIONS),p=p))
        return action,p,features

    def feedback(self,features,action,p,reward,*,learning_rate=.002):
        from flydoom.retinal_policy import softmax
        if not 0<=action<len(ACTIONS) or not np.isfinite(reward) or not 0<learning_rate<=.01:raise ValueError('Invalid feedback')
        if not np.allclose(softmax(self.weights@features),p,atol=1e-8):raise ValueError('Stale policy feedback')
        self.eligibility=.97*.9*self.eligibility+np.outer(np.eye(len(ACTIONS))[action]-p,features)
        gradient=np.clip(reward/100,-1,1)*self.eligibility;gradient/=max(1,float(np.linalg.norm(gradient)))
        change=learning_rate*gradient;self.weights+=change;self.updates+=1
        return {'reward':float(reward),'weight_delta_l2':float(np.linalg.norm(change)),'updates':self.updates,
                'action_probability_before':float(p[action]),'action_probability_after':float(softmax(self.weights@features)[action])}

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:return cls(d['indices'],d['root_ids'],d['mean'],d['scale'],d['weights'])


def load_any(folder):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text())
    if report.get('schema')!=SCHEMA:return load_legacy(folder)
    if report.get('status')!='completed':raise ValueError('Require a completed adaptive checkpoint')
    for name,sha in report['output_sha256'].items():
        if digest(folder/name,'sha256')!=sha:raise ValueError('Adaptive artifact changed: '+name)
    for name,sha in report['source_sha256'].items():
        if digest(Path(__file__).parent/name,'sha256')!=sha:raise ValueError('Adaptive implementation changed: '+name)
    meta=json.loads((folder/'circuit.json').read_text());e=SensorimotorEncoder.__new__(SensorimotorEncoder)
    with np.load(folder/'circuit.npz',allow_pickle=False) as d:
        for key in ('ids','types','inputs','outputs','uv','visible','directions'):setattr(e,key,d[key].copy())
        e.model=GradedVision(sparse.load_npz(folder/'weights.npz'),d['tau'],dt_ms=1000/120/10)
        e.model.baseline=d['baseline'].copy();e.steps=int(d['steps'])
        for key in ('adaptation_gain','adaptation_tau_ms','dark_gain'):setattr(e,key,float(d[key]))
    e.model.decay=np.exp(-(WINDOW_MS/e.steps)/e.model.tau)
    e.camera=meta['camera'];e.output_kind=meta['output_kind'];e.bridge=meta['bridge']
    e.base_weights=e.model.weights.data.copy();e.base_tau=e.model.tau.copy()
    e.group_names,e.edge_group=plastic_groups(e.model.weights,e.types,128);e.gains=np.ones(len(e.group_names));e.reset()
    actor=AdaptiveActor.load(folder/'actor.npz')
    np.testing.assert_array_equal(e.ids[e.outputs[actor.indices]],actor.root_ids)
    return e,actor,report


def make_game(seed,task='basic',actions=ACTIONS,*,labels=False):
    if task not in TASKS:raise ValueError('Unsupported adaptive scenario')
    package=Path(os.path.relpath(Path(vzd.__file__).parent));game=vzd.DoomGame()
    try:
        game.load_config(str(package/'scenarios'/f'{task}.cfg'))
        game.set_vizdoom_path(str(package/('vizdoom.exe' if os.name=='nt' else 'vizdoom')))
        game.set_doom_game_path(str(package/'freedoom2.wad'))
        game.set_available_buttons([getattr(vzd.Button,n) for n in actions[1:]])
        game.set_available_game_variables([vzd.GameVariable.KILLCOUNT,vzd.GameVariable.POSITION_X,vzd.GameVariable.POSITION_Y])
        game.set_screen_format(vzd.ScreenFormat.RGB24);game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.set_mode(vzd.Mode.PLAYER);game.set_window_visible(False);game.set_render_hud(False)
        game.set_labels_buffer_enabled(labels);game.set_sound_enabled(False);game.set_seed(seed);game.init();game.new_episode()
        return game
    except BaseException:game.close();raise


def action(game,index,actions=ACTIONS,observe=False):
    if type(index)!=int or not 0<=index<len(actions):raise ValueError('Invalid action')
    from flydoom.live_recording import responsive_clock
    pos=lambda:[float(game.get_game_variable(v)) for v in (vzd.GameVariable.POSITION_X,vzd.GameVariable.POSITION_Y)]
    before=game.get_total_reward();start=pos();kills=int(game.get_game_variable(vzd.GameVariable.KILLCOUNT));frames=[];ticks=0
    with responsive_clock():
        for _ in range(4):
            if game.is_episode_finished():break
            game.make_action([int(i==index) for i in range(1,len(actions))],1);ticks+=1
            if observe:
                state=game.get_state()
                if state is not None:frames.append(state.screen_buffer.copy())
    final=int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
    return {'reward_delta':float(game.get_total_reward()-before),'kill_delta':final-kills,'kills':final,
            'game_tics':ticks,'episode_finished':game.is_episode_finished(),'position_before':start,'position_after':pos(),
            'note':'Position is diagnostic only, never a deployed policy input.'},frames


def fit(raw,targets,validation,roots,epochs=240):
    actor=AdaptiveActor(np.arange(len(roots)),roots,raw.mean(0),np.maximum(raw.std(0),1e-9))
    x=np.array([actor.features(v) for v in raw]);vx=np.array([actor.features(v) for v in validation[0]])
    y=np.eye(8)[targets]*.99+.01/8;vy=np.eye(8)[validation[1]]*.99+.01/8
    counts=np.bincount(targets,minlength=8);weight=1/np.sqrt(np.maximum(counts[targets],1));weight/=weight.mean()
    best=float('inf');saved=actor.weights.copy();history=[];selected=0;m=saved.copy();v=saved.copy()
    def stats(features,truth):
        z=features@actor.weights.T;z-=z.max(1,keepdims=True);logp=z-np.log(np.exp(z).sum(1,keepdims=True))
        return {'kl':float(np.mean(np.sum(truth*(np.log(truth)-logp),axis=1))),
                'agreement':float(np.mean(np.argmax(z,axis=1)==np.argmax(truth,axis=1))),'samples':len(truth)}
    initial={'train':stats(x,y),'validation':stats(vx,vy)}
    for epoch in range(1,epochs+1):
        z=x@actor.weights.T;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
        grad=((p-y)*weight[:,None]).T@x/len(x)+.0005*actor.weights
        m=.9*m+.1*grad;v=.999*v+.001*grad*grad
        actor.weights-=.015*m/(1-.9**epoch)/(np.sqrt(v/(1-.999**epoch))+1e-8)
        if epoch%10==0:
            score=stats(vx,vy)['kl'];history.append({'epoch':epoch,'validation_kl':score})
            if score<best:best=score;saved=actor.weights.copy();selected=epoch
    actor.weights=saved
    return actor,{'initial':initial,'final':{'train':stats(x,y),'validation':stats(vx,vy)},'history':history,
                  'selected_epoch':selected,'target_source':'Training-only annotated aiming/navigation curriculum; eight learned actions'}
