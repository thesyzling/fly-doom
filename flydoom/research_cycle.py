"""Five-stage bounded sensorimotor learning, independent evaluation and persistence.

Run python -m flydoom.research_cycle --cycles 1. Each cycle reserves fresh seeds
before training. Interrupted evidence is retained; no incomplete model is promoted.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
from time import perf_counter

import numpy as np
from PIL import Image
import vizdoom as vzd

from flydoom.data import digest
from flydoom.movement_core import ACTIONS, apply_action
from flydoom.movement_teacher import MovementTeacher
from flydoom.retinal_policy import RetinalActor, softmax
from flydoom.retinal_train import metrics, TEACHER, encode_records
from flydoom.sensorimotor import SCHEMA, SensorimotorEncoder, load_any, extend_to_descending, save_candidate
from flydoom.sensorimotor_timing import fit as fit_timing
from flydoom.research_registry import ROOT, CycleLock, atomic_json, champion, register, rollback
from flydoom.live_recording import responsive_clock
from flydoom.vision_validation import collect_seeds

TASKS=('basic','deadly_corridor')


def reset_game(game,seed):
    game.set_seed(seed);game.new_episode()
    # Seeded neutral repositioning gives genuinely different images even on
    # maps with only a small number of distinct native spawn configurations.
    rng=np.random.default_rng(seed)
    with responsive_clock():
        for _ in range(6):
            if game.is_episode_finished():raise ValueError('Episode ended during reserved warmup')
            apply_action(game,int(rng.choice([1,2,4,5])))


def make_game(seed,task):
    if task not in TASKS:raise ValueError('Unknown research scenario')
    package=Path(os.path.relpath(Path(vzd.__file__).parent));game=vzd.DoomGame()
    try:
        game.load_config(str(package/'scenarios'/f'{task}.cfg'))
        game.set_vizdoom_path(str(package/('vizdoom.exe' if os.name=='nt' else 'vizdoom')))
        game.set_doom_game_path(str(package/'freedoom2.wad'))
        game.set_available_buttons([getattr(vzd.Button,n) for n in ACTIONS[1:]])
        game.set_available_game_variables([vzd.GameVariable.KILLCOUNT,vzd.GameVariable.POSITION_X,vzd.GameVariable.POSITION_Y])
        game.set_mode(vzd.Mode.PLAYER);game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240);game.set_render_hud(False)
        game.set_sound_enabled(False);game.set_window_visible(False);game.set_seed(seed);game.init()
        reset_game(game,seed);return game
    except BaseException:game.close();raise


def frame_hash(frame):return hashlib.sha256(frame.tobytes()).hexdigest()


def reserve():
    used=set();hashes=set()
    for path in Path('runs').rglob('plan.json'):
        try:
            value=json.loads(path.read_text(encoding='utf-8'));used.update(collect_seeds(value))
            def visit(v):
                if isinstance(v,dict):
                    for k,x in v.items():
                        if k=='opening_sha256':hashes.add(x)
                        else:visit(x)
                elif isinstance(v,list):
                    for x in v:visit(x)
            visit(value)
        except (ValueError,OSError):continue
    splits={s:[] for s in ('test','gate','validation','train')}
    # Counts are per map; freeze all openings before requesting teacher labels.
    for task in TASKS:
        game=make_game(1,task)
        try:
            for split,count in [('test',8),('gate',6),('validation',3),('train',6)]:
                for _ in range(count):
                    for attempt in range(10000):
                        seed=secrets.randbelow(2**31-1)
                        if seed in used:continue
                        reset_game(game,seed);sha=frame_hash(game.get_state().screen_buffer)
                        if sha not in hashes:break
                    else:raise ValueError('Cannot reserve distinct openings')
                    used.add(seed);hashes.add(sha);splits[split].append({'seed':seed,'task':task,'opening_sha256':sha})
        finally:game.close()
    return {'schema':'sensorimotor_plan_v2','splits':splits,'actions':ACTIONS,
            'opening_protocol':'Six seeded repositioning actions; warmup return excluded from reported policy return',
            'biological_fit':{'groups':96,'pairs':8,'training_prefix_frames':24,'gain_bounds':[.8,1.08]},
            'timing':'High-luminance measured L1/L2 training; previously seen low luminance is development only',
            'teacher_steps':{'train':16,'validation':12},'reward_training_episodes':24,'decision_limit':75,
            'selection':'Teacher validation selects decoder; paired gate selects promotion; final test never selects',
            'scope':'Two small scenarios, six mutually exclusive buttons; no turn controls, VNC or muscles'}


def plastic_fit(encoder,actor):
    records=json.loads((TEACHER/'train/trace.json').read_text())[:24]
    initial=encoder.gains.copy();best=initial.copy();rng=np.random.default_rng(513)
    def loss(gains):
        encoder.set_plastic(gains);part=encode_records(encoder,records,'train')
        return metrics(actor,part['raw'],part['targets'])['kl']
    first=score=loss(best);history=[]
    for pair in range(8):
        direction=rng.choice([-1,1],len(best));trials=[]
        for sign in (-1,1):
            gains=np.clip(best+sign*.012*direction,.8,1.08);trials.append((loss(gains),gains))
        candidate,gains=min(trials,key=lambda x:x[0])
        if candidate<score:score=candidate;best=gains.copy()
        history.append({'pair':pair+1,'training_kl':score})
        print(f'Plastic pair {pair+1}/8: KL {score:.6f}',flush=True)
    encoder.set_plastic(best)
    return {'groups':encoder.group_names,'gains':best.tolist(),'initial_training_kl':first,'final_training_kl':score,
            'changed_edges':int(np.count_nonzero(encoder.model.weights.data!=encoder.base_weights)),
            'changed_groups':int(np.count_nonzero(best!=initial)),'history':history,
            'scope':'Positive type-pair magnitude gains on existing edges; fixed parent tonic operating point; training-prefix optimization, not biological efficacy evidence'}


def collect(folder,plan):
    teacher=MovementTeacher(folder/'teacher.log');parts={}
    try:
        atomic_json(folder/'teacher.json',teacher.metadata)
        for split in ('train','validation'):
            dest=folder/split;dest.mkdir();records=[]
            for opening in plan['splits'][split]:
                game=make_game(opening['seed'],opening['task']);rng=np.random.default_rng(opening['seed'])
                try:
                    if frame_hash(game.get_state().screen_buffer)!=opening['opening_sha256']:raise ValueError('Opening changed')
                    for step in range(plan['teacher_steps'][split]):
                        if game.is_episode_finished():break
                        frame=game.get_state().screen_buffer.copy();answer=teacher.predict(frame)
                        if answer['frame_sha256']!=frame_hash(frame):raise ValueError('Teacher frame mismatch')
                        name=f'{opening["task"]}-{opening["seed"]}-{step:03d}.png';Image.fromarray(frame).save(dest/name)
                        # Exploration is confined to training, with the exact teacher target retained.
                        action=int(rng.choice(6,p=answer['probabilities'])) if split=='train' else int(np.argmax(answer['probabilities']))
                        with responsive_clock():effect=apply_action(game,action)
                        records.append({**opening,'step':step,'image':name,'png_sha256':digest(dest/name,'sha256'),
                                        'teacher':answer,'action':ACTIONS[action],'reward':effect['reward_delta']})
                finally:game.close()
                atomic_json(dest/'trace.json',records)
                print(f'Teacher {split}: {len(records)} native frames',flush=True)
            parts[split]=records
    finally:teacher.close()
    return parts


def encode_new(encoder,folder,records):
    raw=[];targets=[];previous=None
    for row in records:
        key=(row['task'],row['seed'])
        if key!=previous:encoder.reset();previous=key
        path=folder/row['image']
        if digest(path,'sha256')!=row['png_sha256']:raise ValueError('Experience image changed')
        frame=np.asarray(Image.open(path).convert('RGB'))
        if frame_hash(frame)!=row['teacher']['frame_sha256']:raise ValueError('Experience RGB changed')
        value,_=encoder.encode(frame);raw.append(value);targets.append(row['teacher']['probabilities'])
    return {'raw':np.array(raw),'targets':np.array(targets)}


def decoder_fit(encoder,parts,parent_actor=None):
    train=parts['train'];indices=np.arange(len(encoder.outputs))
    actor=RetinalActor(indices,encoder.ids[encoder.outputs],train['raw'].mean(0),np.maximum(train['raw'].std(0),1e-9))
    if parent_actor is not None and np.array_equal(actor.root_ids,parent_actor.root_ids):
        actor=RetinalActor(parent_actor.indices,parent_actor.root_ids,parent_actor.mean,parent_actor.scale,parent_actor.weights)
    before={s:metrics(actor,p['raw'],p['targets']) for s,p in parts.items()}
    x=np.array([actor.features(r) for r in train['raw']]);y=train['targets']
    best=before['validation']['kl'];saved=actor.weights.copy();selected=0;history=[]
    m=np.zeros_like(actor.weights);v=m.copy()
    for epoch in range(1,301):
        p=np.array([softmax(z) for z in x@actor.weights.T]);g=(p-y).T@x/len(x)+.0001*actor.weights
        m=.9*m+.1*g;v=.999*v+.001*g*g
        actor.weights-=.02*(m/(1-.9**epoch))/(np.sqrt(v/(1-.999**epoch))+1e-8)
        score=metrics(actor,parts['validation']['raw'],parts['validation']['targets'])['kl']
        if score<best:best=score;saved=actor.weights.copy();selected=epoch
        if epoch%20==0:history.append({'epoch':epoch,'validation_kl':score})
    actor.weights=saved
    return actor,{'initial':before,'final':{s:metrics(actor,p['raw'],p['targets']) for s,p in parts.items()},
                  'selected_epoch':selected,'history':history}


def rollout(encoder,actor,opening,*,learn=False,baseline=None,limit=75):
    game=make_game(opening['seed'],opening['task']);rng=np.random.default_rng(opening['seed'])
    encoder.reset();actor.reset();trace=[];start=perf_counter();initial_reward=game.get_total_reward()
    try:
        if frame_hash(game.get_state().screen_buffer)!=opening['opening_sha256']:raise ValueError('Reserved opening changed')
        for step in range(limit):
            if game.is_episode_finished():break
            if baseline:action=3 if baseline=='attack' else int(rng.integers(6))
            else:
                raw,_=encoder.encode(game.get_state().screen_buffer)
                action,p,x=actor.decide(raw,rng if learn else None)
            with responsive_clock():effect=apply_action(game,action)
            update=actor.feedback(x,action,p,effect['reward_delta']) if learn else None
            trace.append({'step':step+1,'action':ACTIONS[action],'reward':effect['reward_delta'],
                          'kill_delta':effect['kill_delta'],'update':update})
        return {**opening,'return':float(game.get_total_reward()-initial_reward),'kills':sum(r['kill_delta'] for r in trace),
                'decisions':len(trace),'trace':trace,'seconds':perf_counter()-start,'learn':learn}
    finally:game.close()


def evaluate(encoder,actor,openings,label,**kwargs):
    result=[]
    for i,opening in enumerate(openings):
        row=rollout(encoder,actor,opening,**kwargs);result.append(row)
        print(f'{label} {i+1}/{len(openings)} {opening["task"]}: return={row["return"]:.1f}, kills={row["kills"]}',flush=True)
    return result


def summarize(rows):
    return {'episodes':len(rows),'kills':sum(r['kills'] for r in rows),'mean_return':float(np.mean([r['return'] for r in rows])),
            'episodes_with_kill':sum(r['kills']>0 for r in rows),'kill_episode_rate':float(np.mean([r['kills']>0 for r in rows]))}


def paired_gate(before,after):
    key=lambda r:(r['task'],r['seed'])
    if len({key(r) for r in before})!=len(before) or {key(r) for r in before}!={key(r) for r in after}:raise ValueError('Unpaired gate')
    if len(after)!=len(before) or not all(np.isfinite(r['return']) and np.isfinite(r['kills']) for r in before+after):raise ValueError('Invalid gate measurements')
    comparisons={}
    for task in TASKS:
        b=summarize([r for r in before if r['task']==task]);a=summarize([r for r in after if r['task']==task])
        comparisons[task]={'parent':b,'candidate':a,'return_delta':a['mean_return']-b['mean_return'],'kill_delta':a['kills']-b['kills']}
    safe=all(v['return_delta']>=0 and v['kill_delta']>=0 for v in comparisons.values())
    improved=any(v['return_delta']>0 or v['kill_delta']>0 for v in comparisons.values())
    return {'accepted':safe and improved,'by_task':comparisons,'rule':'No mean-return or total-kill regression on either map, with at least one strict improvement. Small engineering gate, not statistical proof.'}


def cycle(root=ROOT):
    parent=champion(root);folder=Path(root)/datetime.now().strftime('cycle-%Y%m%d-%H%M%S-%f');folder.mkdir(parents=True)
    def progress(stage,message):
        print(f'Stage {stage}/5: {message}',flush=True)
        atomic_json(Path(root)/'progress.json',{'status':'running','stage':stage,'message':message,'folder':str(folder)})
    try:
        plan=reserve();plan['parent']=str(parent);plan['parent_report_sha256']=digest(parent/'report.json','sha256')
        atomic_json(folder/'plan.json',plan)
        p_encoder,p_actor,p_report=load_any(parent);encoder=SensorimotorEncoder.wrap(p_encoder)
        progress(1,'Optimize existing biological connection magnitudes')
        plastic=plastic_fit(encoder,p_actor);atomic_json(folder/'plasticity.json',plastic)
        progress(2,'Fit measured response timing and input adaptation')
        timing=fit_timing(encoder);atomic_json(folder/'timing.json',timing)
        progress(3,'Extend existing visual anatomy to descending output candidates')
        encoder=extend_to_descending(encoder);atomic_json(folder/'bridge.json',encoder.bridge)
        progress(4,'Collect new Laya labels, train, then run frozen independent benchmarks')
        records=collect(folder,plan);parts={s:encode_new(encoder,folder/s,r) for s,r in records.items()}
        # Retain earlier training only; the fresh validation set selects this decoder.
        old=json.loads((TEACHER/'train/trace.json').read_text());old_part=encode_records(encoder,old,'train')
        for key in ('raw','targets'):parts['train'][key]=np.concatenate([old_part[key],parts['train'][key]])
        # Replay every chronological frame, but never score a duplicate training image as validation.
        train_hashes={r['teacher']['frame_sha256'] for r in old+records['train']}
        keep=np.array([r['teacher']['frame_sha256'] not in train_hashes for r in records['validation']])
        if not keep.any():raise ValueError('No distinct teacher validation observations')
        for key in ('raw','targets'):parts['validation'][key]=parts['validation'][key][keep]
        actor,distillation=decoder_fit(encoder,parts,p_actor)
        distillation['excluded_validation_duplicates']=int((~keep).sum())
        atomic_json(folder/'distillation.json',distillation)
        saved=actor.weights.copy();reward_rows=[]
        for epoch in range(2):
            reward_rows.extend(evaluate(encoder,actor,plan['splits']['train'],f'Reward epoch {epoch+1}',learn=True))
        reward_kl=metrics(actor,parts['validation']['raw'],parts['validation']['targets'])['kl']
        accepted=reward_kl<=distillation['final']['validation']['kl']
        reward={'episodes':len(reward_rows),'updates':sum(r['decisions'] for r in reward_rows),
                'validation_kl_after':reward_kl,'accepted':accepted,'selection':'Nonregression in fresh teacher validation KL'}
        if not accepted:actor.weights=saved
        atomic_json(folder/'reward.json',{'summary':reward,'episodes':reward_rows})
        before=evaluate(p_encoder,p_actor,plan['splits']['gate'],'Parent gate')
        after=evaluate(encoder,actor,plan['splits']['gate'],'Candidate gate');promotion=paired_gate(before,after)
        atomic_json(folder/'gate.json',{'parent':before,'candidate':after,'decision':promotion})
        tests={}
        for label,e,a,baseline in [('candidate',encoder,actor,None),('parent',p_encoder,p_actor,None),
                                   ('always_attack',encoder,actor,'attack'),('random',encoder,actor,'random')]:
            tests[label]=evaluate(e,a,plan['splits']['test'],label,baseline=baseline)
            atomic_json(folder/'benchmark.json',tests)
        encoder.reset();raw,_=encoder.encode(np.full((240,320,3),255,np.uint8),connected=False)
        disconnected_zero=bool(np.all(raw==0));encoder.reset()
        if not disconnected_zero:raise ValueError('Descending output bypasses graph coupling')
        progress(5,'Verify checkpoint, publish gated registry and retain reproducible evidence')
        save_candidate(folder,encoder,actor)
        files=['weights.npz','circuit.npz','circuit.json','actor.npz','plan.json','distillation.json','plasticity.json',
               'timing.json','bridge.json','reward.json','gate.json','benchmark.json','teacher.json','train/trace.json','validation/trace.json']
        report={'schema':SCHEMA,'status':'completed','parent':str(parent),'parent_report_sha256':plan['parent_report_sha256'],
                'neurons':len(encoder.ids),'edges':encoder.model.weights.nnz,'readout_cells':len(actor.root_ids),
                'plasticity':plastic,'timing':{k:v for k,v in timing.items() if k not in ('time_ms','measured','predicted')},
                'bridge':encoder.bridge,'distillation':distillation,'reward':reward,'promotion':promotion,
                'test':{k:summarize(v) for k,v in tests.items()},
                'test_by_task':{k:{t:summarize([r for r in v if r['task']==t]) for t in TASKS} for k,v in tests.items()},
                'disconnected_outputs_zero':disconnected_zero,'phases_completed':[1,2,3,4,5],
                'scope':'Trainable visual/descending anatomical subgraph with engineered action decoder. Laya provides labels, not live commands. No VNC, muscles, whole-brain dynamics or independent physiological validation.',
                'output_sha256':{name:digest(folder/name,'sha256') for name in files},
                'source_sha256':{name:digest(Path(__file__).with_name(name),'sha256') for name in
                                 ('sensorimotor.py','sensorimotor_timing.py','research_cycle.py','retinal_policy.py','graded_vision.py','movement_core.py')}}
        atomic_json(folder/'report.json',report);register(folder,parent,root)
        atomic_json(Path(root)/'progress.json',{'status':'completed','stage':5,'folder':str(folder),'promoted':promotion['accepted']})
        print(json.dumps({'folder':str(folder),'promoted':promotion['accepted'],'test':report['test']},indent=2),flush=True)
        return folder
    except BaseException as error:
        atomic_json(Path(root)/'progress.json',{'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                                              'folder':str(folder),'error':str(error)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cycles',type=int,default=1)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--rollback',action='store_true')
    args=parser.parse_args()
    if not 1<=args.cycles<=20:parser.error('Use a bounded budget of 1 through 20 cycles')
    with CycleLock(args.root):
        if args.rollback:print(json.dumps(rollback(args.root),indent=2));return
        for _ in range(args.cycles):cycle(args.root)


if __name__=='__main__':main()
