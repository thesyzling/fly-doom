"""Learn basic-scenario aiming from training-only engine annotations.

The live policy receives only frozen descending-neuron states. Bounding boxes
supervise training; no object metadata, pixel shortcut or scripted aiming is
available to the deployed decoder. Acceptance uses real frozen game outcomes.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import secrets

import numpy as np
from PIL import Image
import vizdoom as vzd

from flydoom.data import digest
from flydoom.movement_core import ACTIONS, apply_action
from flydoom.live_recording import responsive_clock
from flydoom.research_cycle import reset_game, frame_hash, summarize
from flydoom.research_registry import ROOT, CycleLock, atomic_json, champion, register
from flydoom.retinal_policy import RetinalActor, softmax
from flydoom.retinal_train import metrics
from flydoom.sensorimotor import SCHEMA, load_any, save_candidate
from flydoom.vision_validation import collect_seeds

PARENT=ROOT/'cycle-20261010-184257-980939'


def make_game(seed):
    package=Path(os.path.relpath(Path(vzd.__file__).parent));game=vzd.DoomGame()
    try:
        game.load_config(str(package/'scenarios/basic.cfg'))
        game.set_vizdoom_path(str(package/('vizdoom.exe' if os.name=='nt' else 'vizdoom')))
        game.set_doom_game_path(str(package/'freedoom2.wad'))
        game.set_available_buttons([getattr(vzd.Button,k) for k in ACTIONS[1:]])
        game.set_available_game_variables([vzd.GameVariable.KILLCOUNT,vzd.GameVariable.POSITION_X,vzd.GameVariable.POSITION_Y])
        game.set_mode(vzd.Mode.PLAYER);game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240);game.set_render_hud(False)
        game.set_labels_buffer_enabled(True);game.set_window_visible(False);game.set_sound_enabled(False)
        game.set_seed(seed);game.init();reset_game(game,seed);return game
    except BaseException:game.close();raise


def target(state):
    labels=[o for o in state.labels if any(n in o.object_name.lower() for n in ('cacodemon','imp','zombie','shotgun','demon','baron','hellknight'))]
    if not labels:return None
    label=max(labels,key=lambda o:o.width*o.height);offset=label.x+label.width/2-160
    tolerance=max(4.,label.width*.28)
    action=3 if abs(offset)<=tolerance else (2 if offset>0 else 1)
    return {'action':action,'offset_pixels':float(offset),'tolerance_pixels':float(tolerance),
            'bbox':[int(label.x),int(label.y),int(label.width),int(label.height)],'object':label.object_name}


def reserve():
    used=set();hashes=set()
    for path in Path('runs').rglob('plan.json'):
        try:value=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError):continue
        used.update(collect_seeds(value))
        def walk(v):
            if isinstance(v,dict):
                for k,x in v.items():
                    if k=='opening_sha256':hashes.add(x)
                    else:walk(x)
            elif isinstance(v,list):
                for x in v:walk(x)
        walk(value)
    splits={};game=make_game(123)
    try:
        for split,count in [('test',24),('gate',16),('validation',8),('train',80)]:
            splits[split]=[]
            for _ in range(count):
                for attempt in range(10000):
                    seed=secrets.randbelow(2**31-1)
                    if seed in used:continue
                    reset_game(game,seed);sha=frame_hash(game.get_state().screen_buffer)
                    if sha not in hashes:break
                else:raise ValueError('Not enough new openings')
                used.add(seed);hashes.add(sha);splits[split].append({'seed':seed,'task':'basic','opening_sha256':sha})
    finally:game.close()
    return {'schema':'basic_aim_plan_v1','splits':splits,'rounds':[32,24,24],'training_steps':24,
            'labels':'Engine target bounding boxes available during training only; left/right/attack target, 6-way learned policy',
            'policy_inputs':'Only 96 descending neuron states; frozen inherited biological weights and timing',
            'selection':'Select training round by frozen validation kills then return; independent gate requires >=80% kill episodes and no regression against champion',
            'test':'24 fresh matched basic openings per policy, never used for selection',
            'scope':'Basic target-shooting specialization; no corridor or general Doom claim',
            'opening_protocol':'Six seeded repositioning actions, excluded from policy return'}


def episode(encoder,actor,opening,*,collect=False,expert_mix=0.,limit=75,oracle=False):
    game=make_game(opening['seed']);encoder.reset();actor.reset();rng=np.random.default_rng(opening['seed'])
    rows=[];examples=[];start_return=game.get_total_reward()
    try:
        if frame_hash(game.get_state().screen_buffer)!=opening['opening_sha256']:raise ValueError('Opening changed')
        for tick in range(limit):
            if game.is_episode_finished():break
            state=game.get_state();annotation=target(state)
            if oracle:action=annotation['action'] if annotation else 0
            else:
                raw,_=encoder.encode(state.screen_buffer);action,p,x=actor.decide(raw)
                if collect and annotation:
                    examples.append({'raw':raw,'action':annotation['action'],'frame':state.screen_buffer.copy(),
                                     'annotation':annotation,'seed':opening['seed'],'step':tick})
                    if rng.random()<expert_mix:action=annotation['action']
                    # Perturbed training trajectories expose recovery states.
                    if rng.random()<.12:action=int(rng.choice([1,2,3]))
            with responsive_clock():effect=apply_action(game,action)
            rows.append({'step':tick+1,'action':ACTIONS[action],'reward':effect['reward_delta'],'kill_delta':effect['kill_delta']})
        return {**opening,'return':float(game.get_total_reward()-start_return),'kills':sum(r['kill_delta'] for r in rows),
                'decisions':len(rows),'attack_decisions':sum(r['action']=='ATTACK' for r in rows),'trace':rows},examples
    finally:game.close()


def evaluate(encoder,actor,openings,label):
    rows=[]
    for i,opening in enumerate(openings):
        row,_=episode(encoder,actor,opening);rows.append(row)
        print(f'{label} {i+1}/{len(openings)}: kills={row["kills"]}, return={row["return"]}',flush=True)
    return rows


def fit(encoder,examples,validation):
    raw=np.array([r['raw'] for r in examples]);actions=np.array([r['action'] for r in examples])
    indices=np.arange(len(encoder.outputs));actor=RetinalActor(indices,encoder.ids[encoder.outputs],raw.mean(0),np.maximum(raw.std(0),1e-9))
    x=np.array([actor.features(r) for r in raw]);y=np.eye(6)[actions]*.99+.01/6
    val_raw=np.array([r['raw'] for r in validation]);val_y=np.eye(6)[[r['action'] for r in validation]]*.99+.01/6
    parts={'train':{'raw':raw,'targets':y},'validation':{'raw':val_raw,'targets':val_y}}
    before={s:metrics(actor,v['raw'],v['targets']) for s,v in parts.items()}
    best=before['validation']['kl'];saved=actor.weights.copy();selected=0;history=[];m=np.zeros_like(actor.weights);v=m.copy()
    counts=np.bincount(actions,minlength=6);sample_weights=1/np.sqrt(np.maximum(counts[actions],1));sample_weights/=sample_weights.mean()
    for epoch in range(1,601):
        logits=x@actor.weights.T;logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
        g=((p-y)*sample_weights[:,None]).T@x/len(x)+.00005*actor.weights
        m=.9*m+.1*g;v=.999*v+.001*g*g
        actor.weights-=.02*(m/(1-.9**epoch))/(np.sqrt(v/(1-.999**epoch))+1e-8)
        if epoch%10:continue
        score=metrics(actor,val_raw,val_y)['kl']
        if score<best:best=score;saved=actor.weights.copy();selected=epoch
        if epoch%20==0:history.append({'epoch':epoch,'validation_kl':score})
    actor.weights=saved
    return actor,{'initial':before,'final':{s:metrics(actor,v['raw'],v['targets']) for s,v in parts.items()},
                  'selected_epoch':selected,'history':history,'target_source':'Training-only engine aiming annotations, not Laya probabilities',
                  'inherited_teacher':'Laya-trained biological/descending parent retained; this stage corrects the engineered action decoder'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',type=Path,default=PARENT);args=parser.parse_args()
    with CycleLock():
        folder=ROOT/datetime.now().strftime('aim-%Y%m%d-%H%M%S-%f');folder.mkdir(parents=True)
        incumbent=champion();encoder,actor,parent_report=load_any(args.parent);old_encoder,old_actor,_=load_any(incumbent)
        plan=reserve();plan.update(parent=str(args.parent),champion=str(incumbent));atomic_json(folder/'plan.json',plan)
        atomic_json(ROOT/'progress.json',{'status':'running','stage':4,'folder':str(folder),'message':'Basic aiming correction; training-only target annotations'})
        examples=[];validation=[];saved_rounds=[];offset=0;frame_dir=folder/'frames';frame_dir.mkdir();manifest=[]
        def retain(values,split):
            for row in values:
                name=f'{split}-{row["seed"]}-{row["step"]:03d}.png';Image.fromarray(row.pop('frame')).save(frame_dir/name)
                manifest.append({k:v for k,v in row.items() if k!='raw'}|{'image':name,'split':split,'png_sha256':digest(frame_dir/name,'sha256')})
            atomic_json(folder/'experience.json',manifest)
        for opening in plan['splits']['validation']:
            _,values=episode(encoder,actor,opening,collect=True,expert_mix=1,limit=24);retain(values,'validation');validation.extend(values)
        for round_index,count in enumerate(plan['rounds']):
            for opening in plan['splits']['train'][offset:offset+count]:
                _,values=episode(encoder,actor,opening,collect=True,expert_mix=(1.,.5,.2)[round_index],limit=24)
                retain(values,'train');examples.extend(values)
            offset+=count
            # Preserve causal replay; remove any image duplicates from validation scoring.
            train_hashes={r['png_sha256'] for r in manifest if r['split']=='train'}
            excluded={(r['seed'],r['step']) for r in manifest if r['split']=='validation' and r['png_sha256'] in train_hashes}
            val=[r for r in validation if (r['seed'],r['step']) not in excluded]
            if not val:raise ValueError('No distinct validation observations')
            actor,distillation=fit(encoder,examples,val)
            scores=evaluate(encoder,actor,plan['splits']['validation'],f'Round {round_index+1} validation')
            actor.save(folder/f'round-{round_index+1}.npz')
            saved_rounds.append({'round':round_index+1,'metrics':summarize(scores),'episodes':scores,'distillation':distillation})
            atomic_json(folder/'rounds.json',saved_rounds)
            print(f'Round {round_index+1}: train={len(examples)}, validation={summarize(scores)}',flush=True)
        selected=max(saved_rounds,key=lambda r:(r['metrics']['kills'],r['metrics']['mean_return']))
        actor=RetinalActor.load(folder/f'round-{selected["round"]}.npz');distillation=selected['distillation']
        before=evaluate(old_encoder,old_actor,plan['splits']['gate'],'Champion gate');after=evaluate(encoder,actor,plan['splits']['gate'],'Aim gate')
        b,a=summarize(before),summarize(after)
        accepted=a['kill_episode_rate']>=.8 and a['kills']>=b['kills'] and a['mean_return']>=b['mean_return']
        promotion={'accepted':accepted,'scope':'Basic scenario only','rule':'At least 80% gate episodes with kill; no kill or return regression vs champion',
                   'by_task':{'basic':{'parent':b,'candidate':a,'return_delta':a['mean_return']-b['mean_return'],'kill_delta':a['kills']-b['kills']}}}
        atomic_json(folder/'gate.json',{'parent':before,'candidate':after,'decision':promotion})
        tests={'candidate':evaluate(encoder,actor,plan['splits']['test'],'Aim test'),
               'parent':evaluate(old_encoder,old_actor,plan['splits']['test'],'Champion test')}
        atomic_json(folder/'benchmark.json',tests);save_candidate(folder,encoder,actor)
        atomic_json(folder/'distillation.json',distillation)
        # Copy evidence as explicitly inherited provenance, not new measurements.
        for name in ('timing.json','plasticity.json','bridge.json'):
            (folder/name).write_bytes((args.parent/name).read_bytes())
        report={**parent_report,'status':'completed','parent':str(args.parent),'parent_report_sha256':digest(args.parent/'report.json','sha256'),
                'promotion':promotion,'distillation':distillation,'test':{k:summarize(v) for k,v in tests.items()},
                'test_by_task':{k:{'basic':summarize(v)} for k,v in tests.items()},
                'aim_training':{'training_frames':len(examples),'selected_round':selected['round'],'source':'Training-only engine bounding boxes',
                                'live_inputs':'Descending neuron states only','inherited_biology':str(args.parent),'trained_biological_edges_this_stage':0},
                'reward':{'updates':0,'scope':'Outcome-selected supervised aiming correction; no live reward update in this stage'},
                'scope':'Basic-scenario aiming specialization. Biological weights and timing inherited unchanged. Engine labels supervise training only; live decisions use descending cells exclusively. Laya parent retained; no new Laya fine-tuning or physiological validation.'}
        files=['weights.npz','circuit.npz','circuit.json','actor.npz','plan.json','distillation.json','timing.json','plasticity.json','bridge.json','experience.json','rounds.json','gate.json','benchmark.json']
        report['output_sha256']={n:digest(folder/n,'sha256') for n in files}
        report['source_sha256']={**parent_report['source_sha256'],'aim_training.py':digest(Path(__file__),'sha256')}
        atomic_json(folder/'report.json',report);register(folder,incumbent)
        atomic_json(ROOT/'progress.json',{'status':'completed','stage':5,'folder':str(folder),'promoted':accepted})
        print(json.dumps({'folder':str(folder),'promotion':promotion,'test':report['test']},indent=2),flush=True)


if __name__=='__main__':main()
