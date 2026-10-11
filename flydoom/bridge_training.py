"""Bounded joint learning of anatomical relay transmission, timing and readout.

Only rows outside the inherited visual circuit can change. Cached visual drive
therefore remains causal and exact up to floating-point summation order. Engine
annotations are training supervision only; deployment still uses neural states.
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from flydoom.aim_training import episode, evaluate, reserve, fit
from flydoom.data import digest
from flydoom.research_cycle import summarize
from flydoom.research_registry import ROOT, CycleLock, atomic_json, champion, register
from flydoom.retinal_policy import RetinalActor, WINDOW_MS
from flydoom.sensorimotor import load_any, save_candidate
from flydoom.sensorimotor_timing import flash
from flydoom.timing_sources import load_split
from flydoom.visual_observer import sample_frame

PARENT=ROOT/'aim-20261010-191727-751332'


class Bridge:
    """Partition the graph only after proving absence of relay-to-visual feedback."""
    def __init__(self, encoder, visual_ids):
        self.encoder=encoder
        self.visual=np.flatnonzero(np.isin(encoder.ids,visual_ids))
        self.nodes=np.flatnonzero(~np.isin(encoder.ids,visual_ids))
        if not len(self.nodes):raise ValueError('No relay cells to train')
        w=encoder.model.weights
        if w[self.visual][:,self.nodes].nnz:raise ValueError('Cached visual drive requires no bridge feedback into visual rows')
        if np.intersect1d(encoder.inputs,self.nodes).size:raise ValueError('Relay cells cannot receive a pixel shortcut')
        if not np.isin(encoder.outputs,self.nodes).all():raise ValueError('Require exclusively descending outputs')
        self.outputs=np.searchsorted(self.nodes,encoder.outputs)
        self.incoming=w[self.nodes][:,self.visual].tocsr()
        self.positive=self.incoming.copy();self.positive.data=np.maximum(self.positive.data,0);self.positive.eliminate_zeros()
        self.negative=self.incoming.copy();self.negative.data=np.minimum(self.negative.data,0);self.negative.eliminate_zeros()
        self.internal=w[self.nodes][:,self.nodes].tocsr()
        self.types,self.groups=np.unique(encoder.types[self.nodes],return_inverse=True)
        self.baseline=encoder.model.baseline[self.nodes].copy()
        self.tau=encoder.model.tau[self.nodes].copy()
        self.base_weights=w.data.copy();self.base_tau=encoder.model.tau.copy()
        self.targets=np.repeat(np.arange(len(encoder.ids)),np.diff(w.indptr))
        self.edges=np.flatnonzero(np.isin(self.targets,self.nodes))
        self.edge_rows=np.searchsorted(self.nodes,self.targets[self.edges])
        self.edge_internal=np.isin(w.indices[self.edges],self.nodes)
        self.edge_positive=self.base_weights[self.edges]>0

    def neutral(self):return np.ones((4,len(self.types)),dtype=np.float64)

    def validate(self, parameters):
        p=np.asarray(parameters,float)
        if p.shape!=self.neutral().shape or not np.isfinite(p).all() or np.any(p<.7) or np.any(p>1.3):
            raise ValueError('Relay transmission and tau multipliers require [0.7, 1.3]')
        return p

    def apply(self,parameters):
        p=self.validate(parameters);w=self.encoder.model.weights
        g=self.groups[self.edge_rows]
        channels=np.where(self.edge_internal,2,np.where(self.edge_positive,0,1))
        values=self.base_weights.copy();values[self.edges]*=p[channels,g]
        candidate=w.copy();candidate.data=values
        bound=float(np.asarray(abs(candidate).sum(axis=1)).max())
        if bound>=.99:raise ValueError('Relay update violates contraction bound')
        w.data[:]=values;self.encoder.model.row_bound=bound
        self.encoder.model.tau=self.base_tau.copy()
        self.encoder.model.tau[self.nodes]*=p[3,self.groups]
        self.encoder.model.decay=np.exp(-(WINDOW_MS/self.encoder.steps)/self.encoder.model.tau)
        self.encoder.reset()

    def replay(self,drive,resets,parameters):
        p=self.validate(parameters);g=p[:,self.groups].astype(np.float32)
        decay=np.exp(-(WINDOW_MS/self.encoder.steps)/(self.tau*g[3])).astype(np.float32)
        state=np.zeros(len(self.nodes),np.float32);positive=np.maximum(self.baseline,0)
        output=np.empty((len(drive),len(self.outputs)),np.float32)
        for i in range(len(drive)):
            if resets[i]:state.fill(0)
            for incoming in drive[i]:
                release=np.maximum(self.baseline+state,0)-positive
                coupling=g[0]*incoming[0]+g[1]*incoming[1]+g[2]*(self.internal@release)
                state=decay*state+(1-decay)*coupling
            output[i]=state[self.outputs]
        if not np.isfinite(output).all():raise FloatingPointError('Non-finite relay states')
        return output


def ordered(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['seed']].append(row)
    result=[]
    for seed,values in groups.items():
        values=sorted(values,key=lambda r:r['step'])
        if [r['step'] for r in values]!=list(range(len(values))):
            raise ValueError(f'Missing causal frames for episode {seed}')
        result.extend(values)
    return result


def cache_drive(bridge,rows,path):
    e=bridge.encoder;bridge.apply(bridge.neutral())
    drive=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,
                                   shape=(len(rows),e.steps,2,len(bridge.nodes)))
    reference=[];resets=np.array([r['step']==0 for r in rows]);started=perf_counter()
    for i,row in enumerate(rows):
        if resets[i]:e.reset()
        image=Path(row['path'])
        if digest(image,'sha256')!=row['png_sha256']:raise ValueError('Training image checksum changed')
        with Image.open(image) as im:frame=np.asarray(im.convert('RGB'))
        contrast=sample_frame(frame,e.uv,e.visible)
        for tick in range(e.steps):
            release=np.maximum(e.model.baseline+e.model.delta,0)-np.maximum(e.model.baseline,0)
            drive[i,tick,0]=bridge.positive@release[bridge.visual]
            drive[i,tick,1]=bridge.negative@release[bridge.visual]
            e.advance(contrast,WINDOW_MS/e.steps)
        reference.append(e.model.delta[e.outputs].copy())
        if (i+1)%100==0:print(f'Visual cache {path.stem}: {i+1}/{len(rows)} frames, {perf_counter()-started:.1f}s',flush=True)
    drive.flush();reference=np.array(reference)
    replay=bridge.replay(drive,resets,bridge.neutral())
    error=float(np.max(abs(replay-reference)))
    if not np.allclose(replay,reference,rtol=2e-3,atol=2e-7):raise ValueError(f'Partition replay mismatch: {error}')
    return drive,resets,reference,{'frames':len(rows),'max_absolute_error':error,'cache_sha256':digest(path,'sha256')}


def train_decoder(raw,actions,epochs=160):
    """Fit using training loss only; no validation or test selects a perturbation."""
    mean=raw.mean(0);scale=np.maximum(raw.std(0),1e-9)
    x=np.c_[np.tanh((raw-mean)/scale),np.ones(len(raw))].astype(float)
    y=np.eye(6)[actions]*.99+.01/6;w=np.zeros((6,x.shape[1]));m=w.copy();v=w.copy()
    counts=np.bincount(actions,minlength=6);sw=1/np.sqrt(np.maximum(counts[actions],1));sw/=sw.mean()
    for epoch in range(1,epochs+1):
        z=x@w.T;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
        grad=((p-y)*sw[:,None]).T@x/len(x)+.0005*w
        m=.9*m+.1*grad;v=.999*v+.001*grad*grad
        w-=.02*m/(1-.9**epoch)/(np.sqrt(v/(1-.999**epoch))+1e-8)
    z=x@w.T;z-=z.max(1,keepdims=True);logp=z-np.log(np.exp(z).sum(1,keepdims=True))
    loss=float(-np.mean(sw*np.sum(y*logp,axis=1))+.00025*np.sum(w*w))
    return loss


def train_bridge(bridge,drive,resets,rows,pairs=8):
    # Use complete episodes within a bounded training prefix, never isolated states.
    ends=np.flatnonzero(np.r_[resets[1:],True])+1
    count=int(ends[np.searchsorted(ends,min(256,len(rows)))])
    actions=np.array([r['action'] for r in rows[:count]])
    rng=np.random.default_rng(110026);best=bridge.neutral();history=[]
    def score(p):
        raw=bridge.replay(drive[:count],resets[:count],p)
        return train_decoder(raw,actions)+.002*float(np.mean((p-1)**2))
    initial=loss=score(best)
    for pair in range(pairs):
        direction=rng.choice([-1.,1.],size=best.shape);radius=.08 if pair<4 else .04
        trials=[np.clip(best+sign*radius*direction,.7,1.3) for sign in (-1,1)]
        values=[score(p) for p in trials];j=int(np.argmin(values));accepted=values[j]<loss
        if accepted:best=trials[j];loss=values[j]
        history.append({'pair':pair+1,'minus_loss':values[0],'plus_loss':values[1],
                        'accepted':bool(accepted),'selected_training_loss':loss})
        print(f'Relay fit {pair+1}/{pairs}: objective {loss:.6f}, accepted={accepted}',flush=True)
    return best,{'initial_training_loss':initial,'final_training_loss':loss,'training_prefix_frames':count,'history':history}


def examples(raw,rows):return [{'raw':v,'action':r['action']} for v,r in zip(raw,rows)]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=PARENT)
    args=parser.parse_args(argv)
    with CycleLock():
        folder=ROOT/datetime.now().strftime('bridge-%Y%m%d-%H%M%S-%f');folder.mkdir()
        def progress(stage,message):
            atomic_json(ROOT/'progress.json',{'status':'running','stage':stage,'folder':str(folder),'message':message})
            print(message,flush=True)
        try:
            incumbent=champion();e,parent_actor,parent_report=load_any(args.parent)
            baseline_encoder,baseline_actor,_=load_any(incumbent)
            visual_ids=load_any('runs/retinal-control-v1')[0].ids
            bridge=Bridge(e,visual_ids);plan=reserve()
            plan.pop('rounds',None);plan['fresh_training_episodes_used']=24
            plan.update(schema='relay_joint_plan_v1',parent=str(args.parent),champion=str(incumbent),
                        training='Inherited train frames plus 24 fresh supervised episodes; 8 paired relay perturbations on a complete-episode train prefix; final decoder uses full train',
                        policy_inputs='Descending states only; no live engine annotations',
                        relay_bounds=[.7,1.3],trainable='Positive/negative visual transmission, recurrent transmission, and tau multipliers by relay target type',
                        timing='Visual timing fixed; relay time constants optimized for task loss, not measured physiology',
                        selection='Biology uses training objective only; decoder epoch uses fresh validation KL. Gate >=80% kills with no regression vs champion and frozen-bridge decoder ablation',
                        physiology='Visual flash trajectories must remain numerically unchanged; existing failed waveform gate remains failed')
            atomic_json(folder/'plan.json',plan)
            progress(1,'Collect fresh supervision and retain causal training-only trajectories')
            source=json.loads((args.parent/'experience.json').read_text())
            rows=[{**r,'path':str((args.parent/'frames'/r['image']).resolve()),'origin':'inherited_train'} for r in source if r['split']=='train']
            frames=folder/'frames';frames.mkdir()
            for split,openings in [('train',plan['splits']['train'][:24]),('validation',plan['splits']['validation'])]:
                for opening in openings:
                    _,values=episode(e,parent_actor,opening,collect=True,expert_mix=1.,limit=24)
                    for row in values:
                        name=f'{split}-{row["seed"]}-{row["step"]:03d}.png';path=frames/name
                        Image.fromarray(row.pop('frame')).save(path)
                        row.pop('raw');rows.append({**row,'split':split,'path':str(path.resolve()),'png_sha256':digest(path,'sha256'),'origin':'fresh_engine_supervision'})
                print(f'Collected {split}: {len(openings)} episodes',flush=True)
            train=ordered([r for r in rows if r['split']=='train']);validation=ordered([r for r in rows if r['split']=='validation'])
            train_hashes={r['png_sha256'] for r in train}
            validation_mask=np.array([r['png_sha256'] not in train_hashes for r in validation])
            if not validation_mask.any():raise ValueError('No distinct validation images')
            atomic_json(folder/'experience.json',rows)
            progress(2,'Cache frozen visual drive and verify split-circuit replay')
            train_drive,train_reset,train_raw,train_audit=cache_drive(bridge,train,folder/'train-drive.npy')
            val_drive,val_reset,val_raw,val_audit=cache_drive(bridge,validation,folder/'validation-drive.npy')
            atomic_json(folder/'cache-audit.json',{'train':train_audit,'validation':val_audit})
            val_rows=[r for r,keep in zip(validation,validation_mask) if keep]
            fixed_actor,fixed_fit=fit(e,examples(train_raw,train),examples(val_raw[validation_mask],val_rows))
            fixed_actor.save(folder/'fixed-bridge-actor.npz')
            atomic_json(folder/'fixed-bridge-fit.json',fixed_fit)
            progress(3,'Joint task fitting of relay transmission, response time and decoder')
            parameters,optimization=train_bridge(bridge,train_drive,train_reset,train)
            raw=bridge.replay(train_drive,train_reset,parameters);val=bridge.replay(val_drive,val_reset,parameters)
            actor,distillation=fit(e,examples(raw,train),examples(val[validation_mask],val_rows))
            distillation['target_source']='Training-only engine aiming annotations; joint relay and decoder fit'
            distillation['inherited_teacher']='Inherited Laya-trained visual circuit; no Laya model fine-tuning'
            # This guard measures inherited visual dynamics before and after intervention.
            times,_=load_split('train');bridge.apply(bridge.neutral());before_flash=flash(e,times)
            bridge.apply(parameters);after_flash=flash(e,times)
            if not np.array_equal(before_flash,after_flash):raise ValueError('Visual physiology changed')
            # Verify the optimized partition against the unmodified full deployment integrator.
            reference=[]
            for row in train[:24]:
                if row['step']==0:e.reset()
                with Image.open(row['path']) as im:frame=np.asarray(im.convert('RGB'))
                value,_=e.encode(frame);reference.append(value)
            reference=np.array(reference)
            if not np.allclose(reference,raw[:24],rtol=2e-3,atol=2e-7):raise ValueError('Optimized deployment replay mismatch')
            e.reset();disconnected,_=e.encode(np.full((240,320,3),255,np.uint8),connected=False)
            if np.any(disconnected):raise ValueError('Descending output bypasses anatomical transmission')
            groups=[f'{channel} -> {kind}' for channel in ('visual positive','visual negative','relay recurrent','tau multiplier') for kind in bridge.types]
            changed=bridge.base_weights!=e.model.weights.data
            plastic={**optimization,'groups':groups,'gains':parameters.ravel().tolist(),
                     'changed_edges':int(changed.sum()),'changed_groups':int(np.count_nonzero(parameters[:3]!=1)),
                     'changed_tau_cells':int(np.count_nonzero(e.model.tau!=bridge.base_tau)),
                     'visual_flash_bitwise_unchanged':True,'deployment_max_absolute_error':float(np.max(abs(reference-raw[:24]))),
                     'scope':'Task-supervised positive magnitude gains on existing relay-target edges; fixed anatomy and visual rows. Relay timing is an engineering fit, not measured physiology.'}
            atomic_json(folder/'plasticity.json',plastic);atomic_json(folder/'distillation.json',distillation)
            # Inherited measured timing remains explicit; it was not refitted in this stage.
            for name in ('timing.json','bridge.json'):(folder/name).write_bytes((args.parent/name).read_bytes())
            progress(4,'Frozen paired gate and untouched final benchmark')
            gate={}
            gate['parent']=evaluate(baseline_encoder,baseline_actor,plan['splits']['gate'],'Champion gate')
            bridge.apply(bridge.neutral());gate['fixed_bridge']=evaluate(e,fixed_actor,plan['splits']['gate'],'Fixed bridge gate')
            bridge.apply(parameters);gate['candidate']=evaluate(e,actor,plan['splits']['gate'],'Joint bridge gate')
            scores={k:summarize(v) for k,v in gate.items()};a=scores['candidate'];b=scores['parent'];f=scores['fixed_bridge']
            accepted=a['kill_episode_rate']>=.8 and all(a['kills']>=s['kills'] and a['mean_return']>=s['mean_return'] for s in (b,f))
            promotion={'accepted':bool(accepted),'rule':plan['selection'],'scope':'Basic scenario only; no biological validation',
                       'by_task':{'basic':{'parent':b,'candidate':a,'fixed_bridge':f,'return_delta':a['mean_return']-b['mean_return'],'kill_delta':a['kills']-b['kills']}}}
            atomic_json(folder/'gate.json',{**gate,'decision':promotion})
            tests={}
            for name,encoder,decoder in [('candidate',e,actor),('fixed_bridge',e,fixed_actor),('parent',baseline_encoder,baseline_actor)]:
                if name!='parent':bridge.apply(parameters if name=='candidate' else bridge.neutral())
                tests[name]=evaluate(encoder,decoder,plan['splits']['test'],name+' test')
                atomic_json(folder/'benchmark.json',tests)
            progress(5,'Save verified relay candidate and retain the champion unless the gate passed')
            bridge.apply(parameters)
            e.group_names=groups[:3*len(bridge.types)];e.gains=parameters[:3].ravel().copy()
            save_candidate(folder,e,actor)
            report={**parent_report,'status':'completed','parent':str(args.parent),'parent_report_sha256':digest(args.parent/'report.json','sha256'),
                    'plasticity':plastic,'distillation':distillation,'promotion':promotion,
                    'bridge_training':{'training_frames':len(train),'validation_frames':int(validation_mask.sum()),
                                       'inherited_frames':sum(r['origin']=='inherited_train' for r in train),
                                       'source':'Training-only engine annotations','frozen_bridge_ablation':True,'visual_physiology_unchanged':True},
                    'test':{k:summarize(v) for k,v in tests.items()},'test_by_task':{k:{'basic':summarize(v)} for k,v in tests.items()},
                    'reward':{'updates':0,'scope':'Supervised joint relay training; no reward optimizer in this stage'},
                    'scope':'Jointly trained anatomical relay weights, relay time constants and engineered descending decoder. Visual graph and measured flash responses unchanged. Inherited physiological validation still fails. Engine annotations train the policy but are absent at deployment. No whole-brain or biological learning claim.'}
            report.pop('aim_training',None)
            files=['weights.npz','circuit.npz','circuit.json','actor.npz','plan.json','distillation.json','plasticity.json','timing.json','bridge.json','experience.json','cache-audit.json','fixed-bridge-actor.npz','fixed-bridge-fit.json','gate.json','benchmark.json']
            report['output_sha256']={n:digest(folder/n,'sha256') for n in files}
            report['source_sha256']={**parent_report['source_sha256'],'bridge_training.py':digest(Path(__file__),'sha256')}
            atomic_json(folder/'report.json',report);load_any(folder);register(folder,incumbent)
            atomic_json(ROOT/'progress.json',{'status':'completed','stage':5,'folder':str(folder),'promoted':bool(accepted)})
            print(json.dumps({'folder':str(folder),'changed_edges':plastic['changed_edges'],'changed_tau_cells':plastic['changed_tau_cells'],'promotion':promotion,'test':report['test']},indent=2),flush=True)
        except BaseException as error:
            atomic_json(ROOT/'progress.json',{'status':'failed','folder':str(folder),'error':str(error)})
            raise


if __name__=='__main__':main()
