"""Versioned graded sensorimotor candidates on existing anatomical edges.

Descending outputs are anatomical candidates, not identified Doom motor units.
The ventral nerve cord and muscles are absent; action decoding is engineered.
"""

from collections import deque
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse

from flydoom.data import digest
from flydoom.eye_mapping import write_json
from flydoom.graded_vision import GradedVision
from flydoom.retinal_policy import RetinalEncoder, RetinalActor, WINDOW_MS, load_policy
from flydoom.retinal_mapping import load_counts
from flydoom.simulation import transmitter_signs
from flydoom.visual_observer import sample_frame

SCHEMA='sensorimotor_v2'


def plastic_groups(weights, types, limit=96):
    """Share a positive magnitude gain within each existing type-pair group."""
    targets=np.repeat(np.arange(weights.shape[0]),np.diff(weights.indptr))
    names=np.char.add(np.char.add(types[weights.indices].astype(str),' -> '),types[targets].astype(str))
    labels,inverse,counts=np.unique(names,return_inverse=True,return_counts=True)
    order=np.argsort(-counts,kind='stable')[:limit]
    selected=np.full(len(labels),-1,dtype=np.int32);selected[order]=np.arange(len(order))
    return labels[order].tolist(),selected[inverse]


class SensorimotorEncoder(RetinalEncoder):
    @classmethod
    def wrap(cls,parent):
        self=cls.__new__(cls);self.__dict__.update(parent.__dict__)
        self.model=GradedVision(parent.model.weights.copy(),parent.model.tau.copy(),dt_ms=parent.model.dt_ms)
        self.model.baseline=parent.model.baseline.copy()
        self.model.decay=parent.model.decay.copy()
        self.base_weights=self.model.weights.data.copy()
        self.group_names,self.edge_group=plastic_groups(self.model.weights,self.types)
        self.gains=np.ones(len(self.group_names));self.base_tau=self.model.tau.copy()
        self.adaptation_gain=float(getattr(parent,'adaptation_gain',0.))
        self.adaptation_tau_ms=float(getattr(parent,'adaptation_tau_ms',60.))
        self.dark_gain=float(getattr(parent,'dark_gain',1.))
        self.output_kind=getattr(parent,'output_kind','T4/T5')
        self.bridge=getattr(parent,'bridge',None)
        self.reset();return self

    def reset(self):
        super().reset();self.adaptation=np.zeros(len(self.inputs),np.float32)

    def set_plastic(self,gains):
        gains=np.asarray(gains,float)
        if gains.shape!=(len(self.group_names),) or not np.isfinite(gains).all() or np.any(gains<.8) or np.any(gains>1.08):
            raise ValueError('Existing-edge gains must remain in [0.8, 1.08]')
        active=self.edge_group>=0
        proposed=self.base_weights.copy();proposed[active]*=gains[self.edge_group[active]]
        candidate=self.model.weights.copy();candidate.data=proposed
        bound=float(np.asarray(abs(candidate).sum(axis=1)).max())
        if bound>=.99:raise ValueError('Candidate loses the contraction safety margin')
        self.model.weights.data[:]=proposed;self.gains=gains.copy();self.model.row_bound=bound

    def set_timing(self,taus,adaptation_gain,adaptation_tau_ms,dark_gain=1.):
        tau=self.base_tau.copy()
        for name,value in taus.items():
            if not np.isfinite(value) or not 2<=value<=150:raise ValueError('Invalid conditional time constant')
            tau[self.types==name]=value
        if not 0<=adaptation_gain<=1.5 or not 5<=adaptation_tau_ms<=250 or not .25<=dark_gain<=4:
            raise ValueError('Invalid input adaptation candidate')
        self.model.tau=tau;self.model.decay=np.exp(-(WINDOW_MS/self.steps)/tau)
        self.adaptation_gain=float(adaptation_gain);self.adaptation_tau_ms=float(adaptation_tau_ms);self.dark_gain=float(dark_gain)

    def advance(self,contrast,dt_ms,*,connected=True):
        # Contrast adaptation is an engineering input model with fitted parameters.
        signal=np.asarray(contrast,np.float32).copy();signal[signal<0]*=self.dark_gain
        decay=np.float32(np.exp(-dt_ms/self.adaptation_tau_ms))
        self.adaptation[:]=decay*self.adaptation+(1-decay)*signal
        drive=np.zeros(len(self.ids),np.float32)
        drive[self.inputs]=signal-self.adaptation_gain*self.adaptation
        self.model.step(drive,connected=connected)

    def encode(self,frame,*,connected=True,telemetry=False):
        start=perf_counter()
        if np.asarray(frame).shape!=(240,320,3):raise ValueError('Require native 320 x 240 RGB')
        contrast=sample_frame(frame,self.uv,self.visible);trace=[]
        groups={kind:np.flatnonzero(self.types==kind) for kind in ('R1-6','L1','L2','Mi1','C2','C3','T4a','T5a')} if telemetry else {}
        if telemetry and self.output_kind=='descending':groups['Descending']=self.outputs
        dt=WINDOW_MS/self.steps
        for tick in range(self.steps):
            self.advance(contrast,dt,connected=connected)
            if telemetry and ((tick+1)%10==0 or tick==self.steps-1):
                trace.append({'ms':(tick+1)*dt,'values':{k:float(self.model.delta[g].mean()) for k,g in groups.items() if len(g)}})
        self.decisions+=1
        return self.model.delta[self.outputs].copy(),{'encoding_ms':(perf_counter()-start)*1000,'trace':trace,
            'neural_clock_ms':self.decisions*WINDOW_MS,'neural_processing_window_ms':WINDOW_MS,
            'visible_receptors':int(self.visible.sum()),'connected':connected}


def extend_to_descending(encoder,relay_budget=512,hops=3,outputs=96):
    if encoder.output_kind=='descending':return encoder
    full_ids,rows,counts,_=load_counts();signs,_=transmitter_signs(rows)
    old=np.searchsorted(full_ids,encoder.ids);oldset=set(old);front=old[encoder.outputs]
    eligible=np.array([i for i,r in enumerate(rows) if i not in oldset and signs[i]!=0 and r['super_class'] in ('central','visual_projection','descending')])
    selected=set(old);layers=[]
    for _ in range(hops):
        mass=np.asarray(counts[:,front].sum(axis=1)).ravel()
        rank=eligible[np.argsort(-mass[eligible],kind='stable')[:relay_budget]];rank=rank[mass[rank]>0]
        selected.update(map(int,rank));front=np.unique(np.r_[front,rank]);layers.append(len(rank))
    dn=np.array([i for i,r in enumerate(rows) if r['super_class']=='descending' and signs[i]!=0])
    mass=np.asarray(counts[dn][:,front].sum(axis=1)).ravel()
    dn=dn[np.argsort(-mass,kind='stable')[:outputs]];dn=dn[np.asarray(counts[dn][:,front].sum(axis=1)).ravel()>0]
    if len(dn)<8:raise ValueError('Insufficient anatomically connected descending outputs')
    selected.update(map(int,dn));chosen=np.array(sorted(selected));ids=full_ids[chosen]
    old_local=np.searchsorted(ids,encoder.ids);old_mask=np.zeros(len(ids),bool);old_mask[old_local]=True
    sub=counts[chosen][:,chosen].astype(np.float32).tocsr();target=np.repeat(np.arange(len(ids)),np.diff(sub.indptr))
    # Add existing inputs to bridge cells, preserving every visual parent row.
    sub.data*=signs[chosen][sub.indices];sub.data[old_mask[target]]=0;sub.eliminate_zeros()
    mass=np.asarray(abs(sub).sum(axis=1)).ravel();sub.data*=np.repeat(.6/np.maximum(mass,1),np.diff(sub.indptr))
    parent=encoder.model.weights.tocoo()
    weights=sub+sparse.csr_matrix((parent.data,(old_local[parent.row],old_local[parent.col])),shape=sub.shape)
    tau=np.full(len(ids),20.,np.float32);tau[old_local]=encoder.model.tau
    result=SensorimotorEncoder.wrap(encoder)
    result.ids=ids;result.types=np.array([rows[i]['cell_type'] or rows[i]['super_class'] for i in chosen])
    result.inputs=old_local[encoder.inputs];result.outputs=np.searchsorted(ids,full_ids[dn]);result.model=GradedVision(weights,tau,dt_ms=encoder.model.dt_ms)
    # The parent operating point remains unchanged in the visual rows.
    result.model.baseline[old_local]=encoder.model.baseline
    result.model.decay=np.exp(-(WINDOW_MS/result.steps)/tau)
    result.base_weights=weights.data.copy();result.base_tau=tau.copy()
    result.group_names,result.edge_group=plastic_groups(weights,result.types,128);result.gains=np.ones(len(result.group_names))
    result.output_kind='descending'
    result.bridge={'neurons_added':len(ids)-len(encoder.ids),'descending_outputs':len(dn),'layers':layers,
        'root_ids':[str(r) for r in full_ids[dn]],'source':'Existing FAFB v783 directed contacts',
        'scope':'Brain descending output candidates; no VNC, muscles or validated action identity',
        'parent_rows_preserved':True}
    difference=result.model.weights[old_local][:,old_local]-encoder.model.weights
    if difference.nnz and np.max(abs(difference.data))>1e-7:raise ValueError('Visual parent changed during extension')
    result.reset();return result


def save_candidate(folder,encoder,actor):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    sparse.save_npz(folder/'weights.npz',encoder.model.weights)
    np.savez_compressed(folder/'circuit.npz',ids=encoder.ids,types=encoder.types,inputs=encoder.inputs,outputs=encoder.outputs,
        tau=encoder.model.tau,baseline=encoder.model.baseline,uv=encoder.uv,visible=encoder.visible,
        directions=encoder.directions,steps=encoder.steps,adaptation_gain=encoder.adaptation_gain,
        adaptation_tau_ms=encoder.adaptation_tau_ms,dark_gain=encoder.dark_gain)
    actor.save(folder/'actor.npz')
    write_json(folder/'circuit.json',{'camera':encoder.camera,'output_kind':encoder.output_kind,'bridge':encoder.bridge,
        'group_names':encoder.group_names,'gains':encoder.gains.tolist()})


def load_candidate(folder):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text())
    if report.get('schema')!=SCHEMA or report.get('status')!='completed':raise ValueError('Require a completed sensorimotor candidate')
    for name,sha in report['output_sha256'].items():
        if digest(folder/name,'sha256')!=sha:raise ValueError('Sensorimotor artifact changed: '+name)
    for name,sha in report['source_sha256'].items():
        if digest(Path(__file__).parent/name,'sha256')!=sha:raise ValueError('Sensorimotor implementation changed: '+name)
    metadata=json.loads((folder/'circuit.json').read_text());self=SensorimotorEncoder.__new__(SensorimotorEncoder)
    with np.load(folder/'circuit.npz',allow_pickle=False) as data:
        for name in ('ids','types','inputs','outputs','uv','visible','directions'):setattr(self,name,data[name].copy())
        if self.ids.dtype!=np.uint64:raise ValueError('Exact cell IDs required')
        self.model=GradedVision(sparse.load_npz(folder/'weights.npz'),data['tau'],dt_ms=1000/120/10)
        self.model.baseline=data['baseline'].copy();self.steps=int(data['steps'])
        for name in ('adaptation_gain','adaptation_tau_ms','dark_gain'):setattr(self,name,float(data[name]))
    self.model.decay=np.exp(-(WINDOW_MS/self.steps)/self.model.tau)
    self.camera=metadata['camera'];self.output_kind=metadata['output_kind'];self.bridge=metadata['bridge']
    self.base_tau=self.model.tau.copy();self.base_weights=self.model.weights.data.copy()
    self.group_names,self.edge_group=plastic_groups(self.model.weights,self.types,128)
    self.gains=np.ones(len(self.group_names));self.reset()
    actor=RetinalActor.load(folder/'actor.npz')
    np.testing.assert_array_equal(self.ids[self.outputs[actor.indices]],actor.root_ids)
    return self,actor,report


def load_any(folder):
    schema=json.loads((Path(folder)/'report.json').read_text()).get('schema')
    return load_candidate(folder) if schema==SCHEMA else load_policy(folder)
