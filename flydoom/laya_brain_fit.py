"""Joint teacher-supervised relay/readout fitting with a retained decoder start."""
import numpy as np

from flydoom.adaptive_policy import AdaptiveActor
from flydoom.retinal_train import metrics


def targets(rows):
    y=np.array([r['teacher']['probabilities'] for r in rows],float)
    if y.shape!=(len(rows),8) or not len(rows) or not np.isfinite(y).all() or np.any(y<0) or not np.allclose(y.sum(1),1):
        raise ValueError('Require actual eight-action mapped teacher distributions')
    return y


def fit(raw,y,parent,validation=None,epochs=240):
    """Keep existing neuron identities, normalization and weights at epoch zero."""
    actor=AdaptiveActor(parent.indices,parent.root_ids,parent.mean,parent.scale,parent.weights)
    x=np.array([actor.features(r) for r in raw]);initial_weights=actor.weights.copy()
    if y.shape!=(len(x),8) or not np.isfinite(y).all() or np.any(y<0) or not np.allclose(y.sum(1),1):raise ValueError('Invalid soft teaching targets')
    def scores():
        result={'train':metrics(actor,raw,y)}
        if validation is not None:result['validation']=metrics(actor,*validation)
        return result
    initial=scores();key='validation' if validation is not None else 'train'
    best=initial[key]['kl'];saved=actor.weights.copy();selected=0;history=[];m=np.zeros_like(saved);v=m.copy()
    for epoch in range(1,epochs+1):
        z=x@actor.weights.T;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
        # A small anchor limits forgetting; epoch zero is always eligible.
        gradient=(p-y).T@x/len(x)+1e-5*(actor.weights-initial_weights)
        m=.9*m+.1*gradient;v=.999*v+.001*gradient*gradient
        actor.weights-=.015*m/(1-.9**epoch)/(np.sqrt(v/(1-.999**epoch))+1e-8)
        if epoch%10==0:
            score=scores()[key]['kl'];history.append({'epoch':epoch,'validation_kl':score})
            if score<best:best=score;saved=actor.weights.copy();selected=epoch
    actor.weights=saved
    return actor,{'initial':initial,'final':scores(),'selected_epoch':selected,'history':history,
                  'target_source':'Frozen Laya Vision image probabilities; no engine annotations or scripted action targets',
                  'initialization':'Verified parent decoder, neuron identities and normalization retained',
                  'selection':key+' teacher KL including epoch zero'}


def active_groups(bridge):
    active=np.zeros_like(bridge.neutral(),dtype=bool)
    channel=np.where(bridge.edge_internal,2,np.where(bridge.edge_positive,0,1))
    active[channel,bridge.groups[bridge.edge_rows]]=True
    return active


def joint_fit(bridge,drive,resets,rows,parent,pairs=8):
    """Profile the decoder under each relay proposal using TRAIN targets only."""
    # Preserve complete causal episodes in the bounded search prefix.
    ends=np.flatnonzero(np.r_[resets[1:],True])+1
    count=int(ends[np.searchsorted(ends,min(256,len(rows)))])
    y=targets(rows[:count]);active=active_groups(bridge);rng=np.random.default_rng(121026)
    best=bridge.neutral();history=[]
    def objective(gains):
        raw=bridge.replay(drive[:count],resets[:count],gains)
        _,report=fit(raw,y,parent,epochs=160)
        return report['final']['train']['kl']+.002*float(np.mean((gains[active]-1)**2))
    loss=initial=objective(best)
    for pair in range(pairs):
        direction=rng.choice([-1.,1.],size=best.shape)*active;radius=.04 if pair<4 else .02
        proposals=[np.clip(best+sign*radius*direction,.9,1.1) for sign in (-1,1)]
        values=[objective(p) for p in proposals];index=int(np.argmin(values));accepted=values[index]<loss
        if accepted:best=proposals[index];loss=values[index]
        history.append({'pair':pair+1,'minus_training_objective':values[0],'plus_training_objective':values[1],
                        'accepted':bool(accepted),'selected_training_objective':loss})
        print(f'Laya joint fit {pair+1}/{pairs}: training objective {loss:.6f}; accepted={accepted}',flush=True)
    return best,{'initial_training_objective':initial,'final_training_objective':loss,
                 'training_prefix_frames':count,'active_groups':int(active.sum()),'history':history,
                 'supervision':'Laya soft probabilities; both relay proposals and their fitted readouts are scored together',
                 'timing_trained':False,'validation_used_for_gain_search':False}
