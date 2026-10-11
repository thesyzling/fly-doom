"""Four-stage research: causal ablation, physiology, curriculum, reward plasticity."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import secrets

import numpy as np
from PIL import Image

from flydoom.adaptive_policy import ACTIONS,TASKS,SCHEMA,AdaptiveActor,make_game,action,fit,load_any
from flydoom.aim_training import target
from flydoom.bridge_training import Bridge,ordered,examples
from flydoom.data import digest
from flydoom.research_cycle import summarize,frame_hash
from flydoom.research_registry import atomic_json,CycleLock,champion
from flydoom.retinal_train import metrics
from flydoom.reward_plasticity import RewardPlasticity
from flydoom.sensorimotor import save_candidate
from flydoom.sensorimotor_timing import fit as fit_timing
from flydoom.vision_validation import collect_seeds

ROOT=Path('runs/adaptive-research')
BRIDGE_RUN=Path('runs/sensorimotor-learning/bridge-20261011-023530-625108')
BIO_PARENT=Path('runs/sensorimotor-learning/aim-20261010-191727-751332')


def start_game(opening,labels=False):
    g=make_game(opening['seed'],opening['task'],labels=labels)
    rng=np.random.default_rng(opening['seed'])
    try:
        for _ in range(6):
            if g.is_episode_finished():raise ValueError('Episode ended during warmup')
            options=[1,2,4,5] if opening['task']=='basic' else [1,2,4,5,6,7]
            action(g,int(rng.choice(options)))
        return g
    except BaseException:g.close();raise


def reserve():
    used=set();hashes=set()
    def walk(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k=='opening_sha256':hashes.add(v)
                else:walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
    for path in Path('runs').rglob('plan.json'):
        try:d=json.loads(path.read_text(encoding='utf-8'));used.update(collect_seeds(d));walk(d)
        except (ValueError,OSError):continue
    splits={k:[] for k in ('train','validation','gate','test')}
    for task in TASKS:
        for split,count in [('test',6),('gate',4),('validation',3),('train',8)]:
            for _ in range(count):
                for attempt in range(10000):
                    seed=secrets.randbelow(2**31-1)
                    if seed in used:continue
                    row={'seed':seed,'task':task};g=start_game(row)
                    try:sha=frame_hash(g.get_state().screen_buffer)
                    finally:g.close()
                    if sha not in hashes:break
                else:raise ValueError('No distinct opening available')
                used.add(seed);hashes.add(sha);splits[split].append({**row,'opening_sha256':sha})
    return {'schema':'adaptive_research_plan_v1','splits':splits,'actions':ACTIONS,
            'limits':{'basic':75,'predict_position':75,'deadly_corridor':150},
            'training_steps':40,'warmup':'Six seeded actions; warmup reward excluded',
            'selection':'Decoder epoch uses fresh validation KL. Reward plasticity uses train episodes only. No gate/test optimization.',
            'promotion':'Every scenario must have no return or kill regression vs fixed biology and approved controller; at least one improvement; basic gate success >=80%. Independent physiology must also pass for biological validation.',
            'physiology':'Previously used high/low luminance means are development data; external cohorts require verified access and protocol matching.'}


def teaching_action(state,task,step,stuck):
    annotation=target(state)
    if annotation:
        choice=annotation['action']
        if task!='basic' and choice in (1,2):choice=6 if choice==1 else 7
        return choice,annotation
    # This curriculum rule is training supervision, never a deployed override.
    return (7 if stuck>=3 or task=='predict_position' else 4),None


def rollout(encoder,actor,opening,limit,*,collect=None):
    g=start_game(opening,labels=collect is not None);encoder.reset();actor.reset()
    rows=[];examples=[];stuck=0;initial=g.get_total_reward()
    try:
        if frame_hash(g.get_state().screen_buffer)!=opening['opening_sha256']:raise ValueError('Reserved opening changed')
        for step in range(limit):
            if g.is_episode_finished():break
            state=g.get_state();frame=state.screen_buffer.copy();raw,_=encoder.encode(frame)
            chosen,_,_=actor.decide(raw)
            if collect is not None:
                taught,annotation=teaching_action(state,opening['task'],step,stuck)
                file=collect/f'{opening["task"]}-{opening["seed"]}-{step:03d}.png';Image.fromarray(frame).save(file)
                examples.append({'seed':opening['seed'],'task':opening['task'],'step':step,'action':taught,'raw':raw,
                                 'path':str(file.resolve()),'png_sha256':digest(file,'sha256'),'annotation':annotation})
                chosen=taught
            effect,_=action(g,chosen)
            distance=float(np.linalg.norm(np.array(effect['position_after'])-effect['position_before']))
            stuck=stuck+1 if chosen in (1,2,4,5) and distance<.25 else 0
            rows.append({'action':ACTIONS[chosen],'reward':effect['reward_delta'],'kill_delta':effect['kill_delta'],
                         'travel':distance,'stuck':stuck>=3})
        return {**opening,'return':float(g.get_total_reward()-initial),'kills':sum(r['kill_delta'] for r in rows),
                'decisions':len(rows),'travel':sum(r['travel'] for r in rows),'stuck_decisions':sum(r['stuck'] for r in rows),
                'episode_finished':g.is_episode_finished(),'player_dead':g.is_player_dead(),'trace':rows},examples
    finally:g.close()


def evaluate(e,a,openings,limits,label):
    result=[]
    for i,o in enumerate(openings):
        r,_=rollout(e,a,o,limits[o['task']]);result.append(r)
        print(f'{label} {i+1}/{len(openings)} {o["task"]}: kills={r["kills"]}, return={r["return"]:.2f}',flush=True)
    return result


def ablation(folder):
    e,_,_=load_any(BIO_PARENT);v=load_any('runs/retinal-control-v1')[0]
    b=Bridge(e,v.ids);p=json.loads((BRIDGE_RUN/'plasticity.json').read_text())
    parameters=np.array(p['gains']).reshape(4,-1)
    rows=ordered([r for r in json.loads((BRIDGE_RUN/'experience.json').read_text()) if r['split']=='validation'])
    cache=BRIDGE_RUN/'validation-drive.npy';audit=json.loads((BRIDGE_RUN/'cache-audit.json').read_text())
    if digest(cache,'sha256')!=audit['validation']['cache_sha256']:raise ValueError('Ablation cache changed')
    drive=np.load(cache,mmap_mode='r');resets=np.array([r['step']==0 for r in rows]);targets=np.eye(6)[[r['action'] for r in rows]]*.99+.01/6
    from flydoom.retinal_policy import RetinalActor
    actors={'fixed_decoder':RetinalActor.load(BRIDGE_RUN/'fixed-bridge-actor.npz'),'joint_decoder':RetinalActor.load(BRIDGE_RUN/'actor.npz')}
    results={}
    for name,parts in [('neutral',()),('weights_only',(0,1,2)),('timing_only',(3,)),('joint',(0,1,2,3))]:
        x=b.neutral()
        for i in parts:x[i]=parameters[i]
        raw=b.replay(drive,resets,x);results[name]={key:metrics(a,raw,targets) for key,a in actors.items()}
    channels=np.where(b.edge_internal,2,np.where(b.edge_positive,0,1));nonempty=len(set(zip(channels,b.groups[b.edge_rows])))
    value={'development_only':True,'frames':len(rows),'matrix':results,'active_transmission_groups':nonempty,
           'prior_empty_transmission_parameters':int(3*len(b.types)-nonempty),
           'finding':'Separate weight/timing/decoder effects on fixed development observations; association is not proof of game benefit.',
           'new_method':'Only nonempty transmission groups receive paired episode-reward updates; response times stay frozen online.'}
    atomic_json(folder/'ablation.json',value);return value


def physiology(folder):
    e,_,parent=load_any(BIO_PARENT);before=parent['timing'];proposal=fit_timing(e,max_nfev=28)
    failures=[]
    for path in Path('data/raw/pang-external').glob('*.error.json'):failures.append({'file':str(path),'details':json.loads(path.read_text())})
    value={'initial':before,'proposal':proposal,'source':'https://datadryad.org/dataset/doi:10.5061/dryad.ngf1vhj4c',
           'external_download_failures':failures,'independent_validation_passed':False,
           'status':'external_data_unavailable','reason':'Dryad download returned 401/403. Existing population means have already informed development; no independent validation claim.',
           'deployed':False,'scope':'Training-only refit and development-condition re-evaluation. No fitted physiology is promoted without independent validation.'}
    atomic_json(folder/'physiology.json',value);return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv);ROOT.mkdir(exist_ok=True,parents=True)
    with CycleLock(ROOT):
        folder=args.output or ROOT/datetime.now().strftime('cycle-%Y%m%d-%H%M%S-%f');folder.mkdir(parents=True,exist_ok=False)
        def progress(stage,message):
            atomic_json(ROOT/'progress.json',{'status':'running','stage':stage,'folder':str(folder),'message':message});print(message,flush=True)
        try:
            plan=reserve();atomic_json(folder/'plan.json',plan)
            progress(1,'Factorial weight/timing/decoder diagnosis on retained development data');diagnosis=ablation(folder)
            progress(2,'Measured timing refit and independent-data availability audit');bio=physiology(folder)
            progress(3,'Three-scenario eight-action curriculum with fresh held-out openings')
            e,old,parent_report=load_any(BIO_PARENT);actor=AdaptiveActor.from_legacy(old)
            parts={};manifest=[];images=folder/'frames';images.mkdir()
            for split in ('train','validation'):
                raw=[];targets=[]
                for i,opening in enumerate(plan['splits'][split]):
                    _,values=rollout(e,actor,opening,plan['training_steps'],collect=images)
                    for row in values:raw.append(row.pop('raw'));targets.append(row['action']);manifest.append({**row,'split':split})
                    print(f'Curriculum {split} {i+1}/{len(plan["splits"][split])}: {len(raw)} frames',flush=True)
                parts[split]=(np.array(raw),np.array(targets))
            hashes={r['png_sha256'] for r in manifest if r['split']=='train'}
            mask=np.array([r['png_sha256'] not in hashes for r in manifest if r['split']=='validation'])
            if not mask.any():raise ValueError('Validation images overlap training')
            parts['validation']=(parts['validation'][0][mask],parts['validation'][1][mask])
            atomic_json(folder/'experience.json',manifest)
            actor,distillation=fit(*parts['train'],parts['validation'],e.ids[e.outputs]);actor.save(folder/'fixed-biology-actor.npz')
            atomic_json(folder/'distillation.json',distillation)
            progress(4,'Paired real-reward biological learning, independent gate and final benchmark')
            visual=load_any('runs/retinal-control-v1')[0].ids;learner=RewardPlasticity(e,visual,seed=111026)
            training=[]
            for opening in plan['splits']['train'][:3]+plan['splits']['train'][8:11]+plan['splits']['train'][16:19]:
                for _ in range(2):
                    trial=learner.begin(opening['seed'],opening['task'])
                    result,_=rollout(e,actor,opening,plan['limits'][opening['task']]);feedback=learner.observe(result['return'],opening['seed'],opening['task'])
                    training.append({'trial':trial,'result':result,'feedback':feedback})
                    atomic_json(folder/'reward-training.json',training);atomic_json(folder/'learner.json',learner.snapshot())
                    print(f'Reward pair {learner.updates}: {trial["side"]}, return={result["return"]:.2f}',flush=True)
            center=learner.center.copy();base=learner.bridge.base_weights.copy()
            old_e,old_a,_=load_any(champion());old_a=AdaptiveActor.from_legacy(old_a)
            gates={};tests={}
            for name,encoder,decoder in [('candidate',e,actor),('fixed_biology',e,actor),('approved',old_e,old_a)]:
                if name!='approved':learner.bridge.apply(center if name=='candidate' else learner.bridge.neutral())
                gates[name]=evaluate(encoder,decoder,plan['splits']['gate'],plan['limits'],name+' gate')
            by_task={};safe=True;improved=False
            for task in TASKS:
                scores={name:summarize([r for r in rows if r['task']==task]) for name,rows in gates.items()};by_task[task]=scores
                c=scores['candidate']
                for name in ('fixed_biology','approved'):
                    safe &= c['kills']>=scores[name]['kills'] and c['mean_return']>=scores[name]['mean_return']
                    improved |= c['kills']>scores[name]['kills'] or c['mean_return']>scores[name]['mean_return']
            game_gate=bool(safe and improved and by_task['basic']['candidate']['kill_episode_rate']>=.8)
            atomic_json(folder/'gate.json',{'episodes':gates,'by_task':by_task,'game_gate_passed':game_gate})
            for name,encoder,decoder in [('candidate',e,actor),('fixed_biology',e,actor),('approved',old_e,old_a)]:
                if name!='approved':learner.bridge.apply(center if name=='candidate' else learner.bridge.neutral())
                tests[name]=evaluate(encoder,decoder,plan['splits']['test'],plan['limits'],name+' test');atomic_json(folder/'benchmark.json',tests)
            learner.bridge.apply(center);save_candidate(folder,e,actor)
            changes=int(np.count_nonzero(e.model.weights.data!=base))
            report={'schema':SCHEMA,'status':'completed','parent':str(BIO_PARENT),'parent_report_sha256':digest(BIO_PARENT/'report.json','sha256'),
                    'neurons':len(e.ids),'readout_cells':len(actor.root_ids),'actions':ACTIONS,'tasks':TASKS,
                    'four_steps':{'causal_diagnosis':'completed','independent_physiology':'blocked_external_data',
                                  'multiscenario_training_and_test':'completed','persistent_reward_plasticity':'completed'},
                    'training_frames':len(parts['train'][0]),'validation_frames':len(parts['validation'][0]),
                    'plasticity':{'active_groups':int(learner.active.sum()),'pairs':learner.updates,'changed_edges':changes,
                                  'tau_changed':False,'rule':'Bounded antithetic episode-reward learning on nonempty relay transmission groups'},
                    'game_gate_passed':game_gate,'biological_validation_passed':False,'promoted':False,
                    'test':{k:summarize(v) for k,v in tests.items()},
                    'test_by_task':{k:{t:summarize([r for r in v if r['task']==t]) for t in TASKS} for k,v in tests.items()},
                    'scope':'Experimental eight-action, three-scenario neural policy. Anatomical relay gains learn from actual paired episode returns. Teacher metadata is training-only. Physiology refit is retained separately, not deployed. No mastery or biological validation claim.'}
            files=['weights.npz','circuit.npz','circuit.json','actor.npz','plan.json','distillation.json','ablation.json','physiology.json','experience.json','learner.json','reward-training.json','gate.json','benchmark.json','fixed-biology-actor.npz']
            report['output_sha256']={name:digest(folder/name,'sha256') for name in files}
            report['source_sha256']={**parent_report['source_sha256'],**{name:digest(Path(__file__).with_name(name),'sha256') for name in ('adaptive_policy.py','reward_plasticity.py','adaptive_experiment.py','bridge_training.py')}}
            atomic_json(folder/'report.json',report);load_any(folder)
            atomic_json(ROOT/'latest.json',{'path':str(folder.resolve()),'sha256':digest(folder/'report.json','sha256')})
            atomic_json(ROOT/'progress.json',{'status':'completed_with_open_scientific_requirements','stage':4,'folder':str(folder)})
            print(json.dumps(report['test_by_task'],indent=2),flush=True)
        except BaseException as error:
            atomic_json(ROOT/'progress.json',{'status':'failed','folder':str(folder),'error':str(error)});raise


if __name__=='__main__':main()
