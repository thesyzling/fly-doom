"""Train and test a retinal closed loop using cached Laya targets and Doom reward."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from flydoom.data import digest
from flydoom.eye_mapping import write_json
from flydoom.movement_core import apply_action, make_game
from flydoom.retinal_policy import ACTIONS, TYPES, RetinalEncoder, RetinalActor, VISUAL_CHECKPOINT, softmax

OUTPUT = Path('runs/retinal-control-v1')
TEACHER = Path('runs/movement-pilot-v1')


def metrics(actor, raw, targets):
    x = np.array([actor.features(r) for r in raw])
    p = np.array([softmax(actor.weights @ row) for row in x])
    return {'kl':float(np.mean(np.sum(targets*np.log(np.maximum(targets,1e-12)/np.maximum(p,1e-12)),axis=1))),
            'agreement':float(np.mean(p.argmax(1)==targets.argmax(1))), 'samples':len(raw)}


def reserve(output):
    records = {s:json.loads((TEACHER/s/'trace.json').read_text()) for s in ('train','validation')}
    excluded = {r['teacher']['frame_sha256'] for values in records.values() for r in values}
    used_seeds = set()
    for path in Path('runs').rglob('plan.json'):
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        def visit(obj):
            if isinstance(obj,dict):
                for key,val in obj.items():
                    if key=='seed' and type(val)==int:used_seeds.add(val)
                    else:visit(val)
            elif isinstance(obj,list):
                for item in obj:visit(item)
        visit(value)
    game = make_game(42000000)
    selected = {'test':[], 'validation':[], 'reward_train':[]}
    try:
        for seed in np.random.default_rng(10102026).permutation(np.arange(42000000,42001000)):
            seed = int(seed)
            if seed in used_seeds:continue
            game.set_seed(seed);game.new_episode()
            sha = hashlib.sha256(game.get_state().screen_buffer.tobytes()).hexdigest()
            if sha in excluded:continue
            excluded.add(sha)
            split = next((s for s,n in [('test',4),('validation',4),('reward_train',16)] if len(selected[s])<n),None)
            if split is None:break
            selected[split].append({'seed':seed,'opening_sha256':sha})
        if [len(selected[s]) for s in ('test','validation','reward_train')]!=[4,4,16]:
            raise ValueError('Could not reserve distinct openings')
    finally:game.close()
    plan = {'schema':'retinal_closed_loop_plan_v1','splits':selected,'actions':ACTIONS,
            'teacher':'Existing pinned six-action Laya Vision image labels; no online teacher override',
            'feature_selection':'Top 32 training-variance cells per T4/T5 subtype; no pixel/memory shortcut',
            'decoder_fit':{'epochs':300,'lr':.02,'l2':.0001,'selection':'Lowest validation teacher KL, including epoch zero'},
            'reward_fit':{'episodes':16,'decision_limit':60,'lr':.002,'eligibility_decay':.97*.9,
                          'reward':'Engine reward / 100 clipped to [-1,1]; no privileged state features'},
            'synaptic_fit':{'pairs':4,'perturbation':.02,'bounds':[.95,1.05],
                            'groups':'Existing incoming edges to eight T4/T5 subtypes',
                            'method':'Paired stochastic perturbation, same training seed and random stream'},
            'selection':'Accept reward decoder / biological gains only if mean return on validation does not regress',
            'test':'Frozen selected policy, disconnected graph and always-ATTACK baseline; no selection on test',
            'scope':'Small basic-scenario engineering experiment, not biological validation or general Doom mastery',
            'teacher_sources':{s:digest(TEACHER/s/'trace.json','sha256') for s in records}}
    write_json(output/'plan.json',plan)
    return plan,records


def encode_records(encoder, records, split):
    features, targets, hashes, seeds = [], [], [], []
    previous_seed = None
    for r in records:
        if r['seed']!=previous_seed:
            encoder.reset();previous_seed=r['seed']
        path=TEACHER/split/r['image']
        if digest(path,'sha256')!=r['png_sha256']:raise ValueError('Teacher image checksum changed')
        frame=np.asarray(Image.open(path).convert('RGB'))
        sha=hashlib.sha256(frame.tobytes()).hexdigest()
        if sha!=r['teacher']['frame_sha256']:raise ValueError('Teacher frame mismatch')
        raw,_=encoder.encode(frame)
        features.append(raw);targets.append(r['teacher']['probabilities']);hashes.append(sha);seeds.append(r['seed'])
    return {'raw':np.array(features),'targets':np.array(targets),'hashes':np.array(hashes),'seeds':np.array(seeds)}


def fit_decoder(encoder, parts):
    train,validation=parts['train'],parts['validation']
    variance=train['raw'].var(axis=0)
    types=encoder.types[encoder.outputs]
    indices=[]
    for kind in TYPES:
        group=np.flatnonzero(types==kind)
        indices.extend(group[np.argsort(-variance[group],kind='stable')[:32]])
    indices=np.array(indices)
    actor=RetinalActor(indices,encoder.ids[encoder.outputs[indices]],train['raw'][:,indices].mean(axis=0),
                       np.maximum(train['raw'][:,indices].std(axis=0),1e-6))
    x=np.array([actor.features(r) for r in train['raw']]);y=train['targets']
    before={s:metrics(actor,p['raw'],p['targets']) for s,p in parts.items()}
    best=before['validation']['kl'];saved=actor.weights.copy();selected=0;history=[]
    m=np.zeros_like(actor.weights);v=m.copy()
    for epoch in range(1,301):
        p=np.array([softmax(z) for z in x@actor.weights.T])
        gradient=(p-y).T@x/len(x)+.0001*actor.weights
        m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
        actor.weights-=.02*(m/(1-.9**epoch))/(np.sqrt(v/(1-.999**epoch))+1e-8)
        score=metrics(actor,validation['raw'],validation['targets'])['kl']
        if score<best:best=score;saved=actor.weights.copy();selected=epoch
        if epoch%20==0:history.append({'epoch':epoch,'validation_kl':score})
    actor.weights=saved
    return actor,{'initial':before,'final':{s:metrics(actor,p['raw'],p['targets']) for s,p in parts.items()},
                  'selected_epoch':selected,'history':history}


def rollout(encoder, actor, opening, *, learn=False, connected=True, fixed_action=None, random_seed=0, limit=60):
    game=make_game(opening['seed']);rng=np.random.default_rng(random_seed)
    encoder.reset();actor.reset();rows=[];start=perf_counter()
    try:
        game.set_seed(opening['seed'])
        game.new_episode()
        sha=hashlib.sha256(game.get_state().screen_buffer.tobytes()).hexdigest()
        if sha!=opening['opening_sha256']:raise ValueError('Reserved opening changed')
        for step in range(limit):
            if game.is_episode_finished():break
            frame=game.get_state().screen_buffer.copy()
            raw,telemetry=encoder.encode(frame,connected=connected)
            action,p,x=actor.decide(raw,rng if learn else None)
            if fixed_action is not None:action=fixed_action
            effect=apply_action(game,action)
            update=actor.feedback(x,action,p,effect['reward_delta']) if learn else None
            rows.append({'step':step+1,'action':ACTIONS[action],'probabilities':p.tolist(),
                         'reward':effect['reward_delta'],'kill_delta':effect['kill_delta'],
                         'game_tics':effect['game_tics'],'update':update,'encoding_ms':telemetry['encoding_ms'],
                         'frame_sha256':hashlib.sha256(frame.tobytes()).hexdigest()})
        return {'seed':opening['seed'],'return':float(game.get_total_reward()),
                'kills':sum(r['kill_delta'] for r in rows),'decisions':len(rows),'seconds':perf_counter()-start,
                'trace':rows,'learned':learn,'connected':connected}
    finally:game.close()


def evaluate(encoder,actor,openings,label,**kwargs):
    result=[]
    for opening in openings:
        value=rollout(encoder,actor,opening,**kwargs);result.append(value)
        print(f"{label}: seed {value['seed']} return {value['return']} kills {value['kills']}",flush=True)
    return result


def average(rows):return float(np.mean([r['return'] for r in rows]))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--resume',action='store_true',help='Resume a pre-reward interrupted run using its reserved splits')
    args=parser.parse_args();output=args.output
    if args.resume:
        if (output/'report.json').exists() or (output/'reward-training.json').exists():
            raise ValueError('Resume is only supported before reward training starts')
        plan=json.loads((output/'plan.json').read_text())
        records={s:json.loads((TEACHER/s/'trace.json').read_text()) for s in ('train','validation')}
        for s,sha in plan['teacher_sources'].items():
            if digest(TEACHER/s/'trace.json','sha256')!=sha:raise ValueError('Teacher source changed')
    else:
        output.mkdir(parents=True,exist_ok=False)
        plan,records=reserve(output)
    encoder=RetinalEncoder();parts={}
    for split,rows in records.items():
        print(f'Encoding {len(rows)} Laya frames / {split}',flush=True)
        parts[split]=encode_records(encoder,rows,split)
    # Remove exact image duplicates across splits after preserving full causal trajectories.
    keep=~np.isin(parts['validation']['hashes'],parts['train']['hashes'])
    parts['validation']={k:v[keep] for k,v in parts['validation'].items()}
    if not keep.any():raise ValueError('No distinct teacher validation frames')
    for split,part in parts.items():np.savez_compressed(output/(split+'.npz'),**part)
    actor,distillation=fit_decoder(encoder,parts);actor.save(output/'distilled.npz')
    write_json(output/'distillation.json',distillation)
    print('Laya distillation: '+json.dumps(distillation['final']),flush=True)
    validation_before=evaluate(encoder,actor,plan['splits']['validation'],'Distilled validation')
    initial=actor.weights.copy();episodes=[]
    for j,opening in enumerate(plan['splits']['reward_train']):
        episode=rollout(encoder,actor,opening,learn=True,random_seed=90210+j);episodes.append(episode)
        write_json(output/'reward-training.json',episodes)
        print(f"Reward episode {j+1}: return {episode['return']}, updates {len(episode['trace'])}",flush=True)
    actor.save(output/'reward-candidate.npz')
    validation_reward=evaluate(encoder,actor,plan['splits']['validation'],'Reward validation')
    reward_accepted=average(validation_reward)>=average(validation_before)
    if not reward_accepted:actor.weights=initial
    actor.save(output/'decoder-selected.npz')
    current=average(validation_reward if reward_accepted else validation_before)
    gains=np.ones(8);perturbations=[];rng=np.random.default_rng(7102026)
    for pair,opening in enumerate(plan['splits']['reward_train'][:4]):
        direction=rng.choice([-1.,1.],8);plus=np.clip(gains+.02*direction,.95,1.05);minus=np.clip(gains-.02*direction,.95,1.05)
        encoder.set_gains(plus);a=rollout(encoder,actor,opening)
        encoder.set_gains(minus);b=rollout(encoder,actor,opening)
        # Bounded reward-driven update, with a fixed decoder and paired seed.
        signal=np.clip((a['return']-b['return'])/100,-1,1)
        gains=np.clip(gains+.01*signal*direction,.95,1.05)
        perturbations.append({'pair':pair,'plus':a,'minus':b,'updated_gains':gains.tolist()})
        write_json(output/'synaptic-reward.json',perturbations)
        print(f"Synaptic reward pair {pair+1}: {a['return']} / {b['return']}",flush=True)
    encoder.set_gains(gains)
    validation_graph=evaluate(encoder,actor,plan['splits']['validation'],'Synaptic validation')
    graph_accepted=average(validation_graph)>=current
    if not graph_accepted:gains=np.ones(8);encoder.set_gains(gains)
    actor.save(output/'actor.npz')
    # Freeze all parameters before the test split is touched.
    write_json(output/'selection.json',{'reward_decoder_accepted':reward_accepted,'graph_gains_accepted':graph_accepted,
                                      'gains':gains.tolist(),'decoder_sha256':digest(output/'actor.npz','sha256')})
    test=evaluate(encoder,actor,plan['splits']['test'],'Frozen test')
    disconnected=evaluate(encoder,actor,plan['splits']['test'],'Disconnected test',connected=False)
    attack=evaluate(encoder,actor,plan['splits']['test'],'Always ATTACK test',fixed_action=3)
    write_json(output/'evaluation.json',{'validation_initial':validation_before,'validation_reward':validation_reward,
                                        'validation_graph':validation_graph,'test':test,'disconnected':disconnected,'always_attack':attack})
    files=['plan.json','train.npz','validation.npz','distillation.json','distilled.npz','reward-candidate.npz',
           'decoder-selected.npz','actor.npz','selection.json','reward-training.json','synaptic-reward.json','evaluation.json']
    report={'schema':'retinal_closed_loop_v1','status':'completed','actions':ACTIONS,
            'visual_neurons':len(encoder.ids),'decoder_cells':len(actor.indices),
            'visual_parent_sha256':digest(VISUAL_CHECKPOINT/'report.json','sha256'),
            'teacher_report_sha256':digest(TEACHER/'report.json','sha256'),
            'reward_updates':sum(len(e['trace']) for e in episodes),
            'candidate_decoder_delta_l2':float(np.linalg.norm(RetinalActor.load(output/'reward-candidate.npz').weights-initial)),
            'reward_decoder_accepted':reward_accepted,'graph_gains_accepted':graph_accepted,'selected_gains':gains.tolist(),
            'selected_changed_biological_edges':int(sum(np.count_nonzero(mask & (encoder.base_weights!=0)) for mask,g in zip(encoder.gain_masks,gains) if g!=1)),
            'test':{name:{'kills':sum(r['kills'] for r in rows),'episodes':len(rows),'mean_return':average(rows)}
                    for name,rows in [('connected',test),('disconnected',disconnected),('always_attack',attack)]},
            'closed_loop_verified':True,'online_laya':False,'whole_brain_motor_path':False,
            'limitations':['Engineered decoder reads T4/T5 cells directly; no descending-neuron motor pathway',
                           'Retinal mapping and physiology remain provisional; earlier waveform gate still fails',
                           'Fixed neural response window per observed image; not continuous optics or calibrated reaction time',
                           'Small basic-scenario test; no general navigation or biological learning claim'],
            'source_sha256':{n:digest(Path(__file__).parent/n,'sha256') for n in
                             ['retinal_policy.py','retinal_train.py','visual_observer.py','movement_core.py']},
            'output_sha256':{n:digest(output/n,'sha256') for n in files}}
    write_json(output/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
