"""Paired episode reward learning of existing relay gains; no biological claim."""
import numpy as np
from flydoom.bridge_training import Bridge


class RewardPlasticity:
    def __init__(self,encoder,visual_ids,seed=0):
        self.bridge=Bridge(encoder,visual_ids);self.center=self.bridge.neutral();self.rng=np.random.default_rng(seed)
        channels=np.where(self.bridge.edge_internal,2,np.where(self.bridge.edge_positive,0,1))
        groups=self.bridge.groups[self.bridge.edge_rows]
        self.active=np.zeros_like(self.center,dtype=bool);self.active[channels,groups]=True
        self.direction=None;self.plus=None;self.pair_seed=None;self.task=None;self.updates=0;self.history=[];self.trial_open=False

    def begin(self,seed,task):
        if self.trial_open:raise ValueError('Finish or cancel the current reward trial first')
        if self.direction is None:
            self.direction=self.rng.choice([-1.,1.],size=self.center.shape)*self.active
            self.pair_seed=int(seed);self.task=task
        if self.pair_seed!=seed or self.task!=task:raise ValueError('Paired reward episodes require identical seed and scenario')
        sign=1 if self.plus is None else -1
        trial=np.clip(self.center+sign*.02*self.direction,.9,1.1)
        self.bridge.apply(trial)
        self.trial_open=True
        return {'side':'plus' if sign==1 else 'minus','pair':self.updates+1,'seed':seed,'task':task,'parameters':trial.tolist()}

    def observe(self,reward,seed,task):
        if not self.trial_open or not np.isfinite(reward) or self.direction is None or seed!=self.pair_seed or task!=self.task:raise ValueError('Invalid paired reward')
        self.trial_open=False
        if self.plus is None:self.plus=float(reward);return {'status':'awaiting_matched_minus','updated':False}
        delta=float(self.plus-reward);before=self.center.copy()
        self.center=np.clip(self.center+.005*np.clip(delta/100,-1,1)*self.direction,.9,1.1)
        self.center[~self.active]=1
        self.updates+=1
        event={'pair':self.updates,'seed':seed,'task':task,'plus_return':self.plus,'minus_return':float(reward),
               'return_difference':delta,'gain_delta_l2':float(np.linalg.norm(self.center-before)),
               'updated':bool(np.any(self.center!=before)),'status':'pair_completed'}
        self.history.append(event);self.direction=None;self.plus=None;self.pair_seed=None;self.task=None
        return event

    def snapshot(self):
        return {'center':self.center.tolist(),'direction':None if self.direction is None else self.direction.tolist(),
                'plus':self.plus,'pair_seed':self.pair_seed,'task':self.task,'updates':self.updates,
                'history':self.history,'rng':self.rng.bit_generator.state,'trial_open':self.trial_open}

    def cancel(self):
        """Discard an incomplete pair without interpreting an interruption as reward."""
        self.direction=None;self.plus=None;self.pair_seed=None;self.task=None;self.trial_open=False

    def restore(self,state):
        center=np.asarray(state['center'],float)
        if center.shape!=self.center.shape or not np.isfinite(center).all() or np.any(center<.9) or np.any(center>1.1) or np.any(center[~self.active]!=1):raise ValueError('Invalid saved relay gains')
        direction=None if state['direction'] is None else np.asarray(state['direction'],float)
        if direction is not None and (direction.shape!=center.shape or not np.isfinite(direction).all() or np.any(abs(direction[self.active])!=1) or np.any(direction[~self.active]!=0)):raise ValueError('Invalid saved perturbation')
        if state['plus'] is not None and (direction is None or not np.isfinite(state['plus'])):raise ValueError('Invalid pending return')
        if direction is None and (state['pair_seed'] is not None or state['task'] is not None or state['trial_open']):raise ValueError('Invalid trial state')
        if direction is not None and (type(state['pair_seed'])!=int or not 0<=state['pair_seed']<2**32 or not isinstance(state['task'],str)):raise ValueError('Invalid pair identity')
        self.center=center;self.direction=direction;self.plus=state['plus'];self.pair_seed=state['pair_seed'];self.task=state['task']
        self.updates=int(state['updates']);self.history=state['history'];self.rng.bit_generator.state=state['rng'];self.trial_open=state['trial_open']
