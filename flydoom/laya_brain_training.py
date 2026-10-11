"""Laya teaches anatomical relay/readout parameters; the eye drives deployment."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import secrets

import numpy as np
from PIL import Image

from flydoom.adaptive_policy import ACTIONS,SCHEMA,AdaptiveActor,make_game,action,load_any
from flydoom.bridge_training import Bridge,cache_drive,ordered
from flydoom.data import digest
from flydoom.laya_brain_fit import fit,joint_fit,targets,active_groups
from flydoom.laya_brain_teacher import LayaBrainTeacher
from flydoom.research_cycle import frame_hash,summarize
from flydoom.research_registry import atomic_json,CycleLock,champion
from flydoom.retinal_train import metrics
from flydoom.sensorimotor import save_candidate
from flydoom.vision_validation import collect_seeds

ROOT=Path('runs/laya-brain-learning')
PARENT=Path('runs/sensorimotor-learning/aim-20261010-191727-751332')
HISTORY=Path('runs/adaptive-research/cycle-20261011-030128-397107')
RETAINED=Path('runs/movement-pilot-v1/train')


def open_game(opening):
    game=make_game(opening['seed'],'basic',labels=False)
    rng=np.random.default_rng(opening['seed'])
    try:
        for _ in range(6):action(game,int(rng.choice([1,2])))
        if game.is_episode_finished():raise ValueError('Warmup ended episode')
        return game
    except BaseException:game.close();raise


def reserve():
    used={85012};hashes=set()
    def walk(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k=='opening_sha256':hashes.add(v)
                else:walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
    for file in Path('runs').rglob('plan.json'):
        try:d=json.loads(file.read_text(encoding='utf-8'));used.update(collect_seeds(d));walk(d)
        except (OSError,ValueError):continue
    old=json.loads((RETAINED/'trace.json').read_text());used.update(r['seed'] for r in old)
    hashes.update(r['teacher']['frame_sha256'] for r in old)
    splits={}
    for split,count in [('test',8),('gate',8),('validation',4),('train',16)]:
        splits[split]=[]
        for _ in range(count):
            for attempt in range(10000):
                seed=secrets.randbelow(2**31-1)
                if seed in used:continue
                row={'task':'basic','seed':seed};game=open_game(row)
                try:sha=frame_hash(game.get_state().screen_buffer)
                finally:game.close()
                if sha not in hashes:break
            else:raise ValueError('No unused opening')
            hashes.add(sha);used.add(seed);splits[split].append({**row,'opening_sha256':sha})
    return {'schema':'laya_brain_teaching_v1','splits':splits,'actions':ACTIONS,'teacher_actions':ACTIONS[1:4],
            'scope':'Basic target-shooting curriculum, using the author training question. Eight student outputs retained; other scenarios are not taught here.',
            'warmup':'Six seeded strafe actions; warmup rewards excluded','training_steps':24,'evaluation_steps':75,
            'labels':'Actual frozen Laya probabilities only; no engine annotations or rule-based action targets',
            'selection':'Relay search uses train only; decoder epoch uses fresh validation teacher KL including epoch zero. Gate >=80% kill episodes and no kills/return regression vs approved and fixed-relay policies. Final test never selects parameters.',
            'deployment':'Eye model -> anatomical visual/relay/descending states -> action readout. Teacher process is closed during student evaluation.'}


def teacher_label(teacher,frame):
    answer=teacher.predict(frame)
    if answer['frame_sha256']!=frame_hash(frame):raise ValueError('Teacher/input mismatch')
    if tuple(answer['actions'])!=ACTIONS:raise ValueError('Teacher action mapping changed')
    targets([{'teacher':answer}]);return answer


def collect(e,actor,teacher,opening,folder,split,mix,limit):
    game=open_game(opening);e.reset();actor.reset();rng=np.random.default_rng(opening['seed']);rows=[]
    try:
        if frame_hash(game.get_state().screen_buffer)!=opening['opening_sha256']:raise ValueError('Opening changed')
        for step in range(limit):
            if game.is_episode_finished():break
            frame=game.get_state().screen_buffer.copy();raw,_=e.encode(frame);student,_,_=actor.decide(raw)
            answer=teacher_label(teacher,frame);guided=bool(rng.random()<mix)
            choice=int(np.argmax(answer['probabilities'])) if guided else student
            image=folder/f'{split}-{opening["seed"]}-{step:03d}.png';Image.fromarray(frame).save(image)
            effect,_=action(game,choice)
            rows.append({'seed':opening['seed'],'task':'basic','step':step,'split':split,'path':str(image.resolve()),
                         'png_sha256':digest(image,'sha256'),'teacher':answer,'executed_action':ACTIONS[choice],
                         'behavior':'laya' if guided else 'parent_brain','effect':effect,'origin':'fresh_student_teacher_rollout'})
    finally:game.close()
    return rows


def retained(teacher):
    rows=[];last=None;step=0
    for item in json.loads((RETAINED/'trace.json').read_text()):
        if item['seed']!=last:step=0;last=item['seed']
        image=RETAINED/item['image']
        if digest(image,'sha256')!=item['png_sha256']:raise ValueError('Retained teacher image changed')
        with Image.open(image) as im:frame=np.array(im.convert('RGB'))
        if frame_hash(frame)!=item['teacher']['frame_sha256']:raise ValueError('Retained observation identity changed')
        # Relabel old causal training observations with this exact teacher question.
        rows.append({'seed':item['seed'],'task':'basic','step':step,'split':'train','path':str(image.resolve()),
                     'png_sha256':item['png_sha256'],'teacher':teacher_label(teacher,frame),
                     'executed_action':item['action'],'behavior':'retained_six_action_rollout','origin':'retained_train_relabeled'})
        step+=1
        if len(rows)%25==0:print(f'Retained observations retaught by Laya: {len(rows)}',flush=True)
    return rows


def evaluate(e,actor,openings,label,teacher=None):
    results=[]
    for opening in openings:
        game=open_game(opening);e.reset();actor.reset();initial=game.get_total_reward();trace=[]
        try:
            if frame_hash(game.get_state().screen_buffer)!=opening['opening_sha256']:raise ValueError('Evaluation opening changed')
            for step in range(75):
                if game.is_episode_finished():break
                frame=game.get_state().screen_buffer
                if teacher is None:raw,_=e.encode(frame);choice,_,_=actor.decide(raw)
                else:choice=int(np.argmax(teacher_label(teacher,frame)['probabilities']))
                effect,_=action(game,choice);trace.append({'action':ACTIONS[choice],'reward':effect['reward_delta'],'kills':effect['kill_delta']})
            result={**opening,'return':float(game.get_total_reward()-initial),'kills':sum(r['kills'] for r in trace),
                    'decisions':len(trace),'trace':trace,'teacher_controls_action':teacher is not None,
                    'labels_buffer_enabled':game.is_labels_buffer_enabled()}
            results.append(result);print(f'{label} {len(results)}/{len(openings)}: kills={result["kills"]}, return={result["return"]:.2f}',flush=True)
        finally:game.close()
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pairs',type=int,default=8)
    args=parser.parse_args(argv)
    if not 1<=args.pairs<=16:parser.error('--pairs must be 1 through 16')
    ROOT.mkdir(parents=True,exist_ok=True)
    with CycleLock(ROOT):
        folder=ROOT/datetime.now().strftime('cycle-%Y%m%d-%H%M%S-%f');folder.mkdir()
        def progress(stage,message):
            atomic_json(ROOT/'progress.json',{'status':'running','stage':stage,'folder':str(folder),'message':message});print(message,flush=True)
        try:
            progress(1,'Reserve fresh splits and collect actual Laya teaching targets')
            plan=reserve();plan['relay_search_pairs']=args.pairs;atomic_json(folder/'plan.json',plan)
            e,parent,parent_report=load_any(PARENT);parent=AdaptiveActor.from_legacy(parent)
            frames=folder/'frames';frames.mkdir();rows=[];teacher=LayaBrainTeacher(folder/'teacher-collection.log')
            try:
                atomic_json(folder/'teacher.json',teacher.metadata)
                rows=retained(teacher)
                for split in ('train','validation'):
                    for i,opening in enumerate(plan['splits'][split]):
                        # Alternating guided and student-only trajectories expose both correct behavior and the student's errors.
                        mix=(.5 if i%2 else 0.) if split=='train' else 0.
                        rows.extend(collect(e,parent,teacher,opening,frames,split,mix,plan['training_steps']))
                        atomic_json(folder/'experience.json',rows)
                        print(f'Laya {split} {i+1}/{len(plan["splits"][split])}: total {len(rows)} frames',flush=True)
            finally:teacher.close()
            if teacher.process.poll() is None:raise RuntimeError('Teacher must stop before student-only work')
            progress(2,'Verify eye-driven causal replay; retain the parent readout as the training start')
            visual=load_any('runs/retinal-control-v1')[0].ids;bridge=Bridge(e,visual)
            train=ordered([r for r in rows if r['split']=='train']);validation=ordered([r for r in rows if r['split']=='validation'])
            hashes={r['teacher']['frame_sha256'] for r in train};mask=np.array([r['teacher']['frame_sha256'] not in hashes for r in validation])
            if not mask.any():raise ValueError('No distinct teacher validation frames')
            drive,resets,raw,ta=cache_drive(bridge,train,folder/'train-drive.npy')
            vd,vr,vraw,va=cache_drive(bridge,validation,folder/'validation-drive.npy')
            atomic_json(folder/'cache-audit.json',{'train':ta,'validation':va})
            y=targets(train);vy=targets(validation)[mask]
            fixed,fixed_fit=fit(raw,y,parent,(vraw[mask],vy));fixed.save(folder/'fixed-biology-actor.npz')
            atomic_json(folder/'fixed-biology-fit.json',fixed_fit)
            progress(3,'Train relay gains and readout against the same Laya probabilities')
            gains,optimization=joint_fit(bridge,drive,resets,train,parent,pairs=args.pairs)
            learned=bridge.replay(drive,resets,gains);vlearned=bridge.replay(vd,vr,gains)
            actor,distillation=fit(learned,y,parent,(vlearned[mask],vy))
            bridge.apply(gains);reference=[]
            for row in train[:12]:
                if row['step']==0:e.reset()
                with Image.open(row['path']) as im:value,_=e.encode(np.asarray(im.convert('RGB')))
                reference.append(value)
            if not np.allclose(reference,learned[:12],rtol=2e-3,atol=2e-7):raise ValueError('Full eye-to-brain deployment disagrees with cached training')
            e.reset();disconnected,_=e.encode(np.full((240,320,3),255,np.uint8),connected=False)
            if np.any(disconnected):raise ValueError('Descending output bypassed anatomical transmission')
            if not np.array_equal(e.model.tau,bridge.base_tau):raise ValueError('Laya training must not alter physiology timing')
            changed=int(np.count_nonzero(e.model.weights.data!=bridge.base_weights))
            plasticity={**optimization,'gains':gains.tolist(),'changed_edges':changed,
                        'changed_groups':int(np.count_nonzero(gains!=1)),'visual_rows_unchanged':True,
                        'disconnected_output_zero':True,'deployment_max_absolute_error':float(np.max(abs(np.array(reference)-learned[:12])))}
            atomic_json(folder/'plasticity.json',plasticity);atomic_json(folder/'distillation.json',distillation)
            ablation={'frames':int(mask.sum()),'development_only':True,'active_transmission_groups':int(active_groups(bridge).sum()),
                      'prior_empty_transmission_parameters':int(3*len(bridge.types)-active_groups(bridge).sum()),
                      'finding':'Current Laya-labelled validation frames; separate fixed/trained relay and readout combinations.',
                      'matrix':{condition:{name:metrics(a,r[mask],vy) for name,a in [('fixed_decoder',fixed),('joint_decoder',actor)]}
                                for condition,r in [('neutral',vraw),('learned_connections',vlearned)]}}
            atomic_json(folder/'ablation.json',ablation)
            # Retain the earlier physiology audit as inherited evidence, without claiming a new fit.
            inherited=json.loads((HISTORY/'physiology.json').read_text());inherited['inherited_from']=str(HISTORY)
            atomic_json(folder/'physiology.json',inherited)
            progress(4,'Teacher process closed: frozen student game gates and untouched final tests')
            approved_e,approved_a,_=load_any(champion());approved_a=AdaptiveActor.from_legacy(approved_a)
            gate={};tests={}
            for name,encoder,decoder in [('candidate',e,actor),('fixed_biology',e,fixed),('approved',approved_e,approved_a)]:
                if name!='approved':bridge.apply(gains if name=='candidate' else bridge.neutral())
                gate[name]=evaluate(encoder,decoder,plan['splits']['gate'],name+' gate')
            scores={k:summarize(v) for k,v in gate.items()};c=scores['candidate']
            accepted=c['kill_episode_rate']>=.8 and all(c['kills']>=r['kills'] and c['mean_return']>=r['mean_return'] for r in (scores['fixed_biology'],scores['approved']))
            atomic_json(folder/'gate.json',{'episodes':gate,'scores':scores,'game_gate_passed':bool(accepted)})
            for name,encoder,decoder in [('candidate',e,actor),('fixed_biology',e,fixed),('approved',approved_e,approved_a)]:
                if name!='approved':bridge.apply(gains if name=='candidate' else bridge.neutral())
                tests[name]=evaluate(encoder,decoder,plan['splits']['test'],name+' test');atomic_json(folder/'benchmark.json',tests)
            # The separate teacher baseline never supplies student evaluation actions.
            teacher=LayaBrainTeacher(folder/'teacher-baseline.log')
            try:tests['laya_teacher']=evaluate(e,actor,plan['splits']['test'],'Laya alone test',teacher)
            finally:teacher.close()
            atomic_json(folder/'benchmark.json',tests);bridge.apply(gains);save_candidate(folder,e,actor)
            report={'schema':SCHEMA,'status':'completed','parent':str(PARENT),'parent_report_sha256':digest(PARENT/'report.json','sha256'),
                    'neurons':len(e.ids),'readout_cells':len(actor.root_ids),'actions':ACTIONS,'tasks':['basic'],
                    'four_steps':{'laya_supervision':'completed','joint_connections_and_decoder':'completed',
                                  'teacher_free_game_test':'completed','independent_physiology':'not_validated'},
                    'training_frames':len(train),'validation_frames':int(mask.sum()),'plasticity':plasticity,
                    'laya_training':{'teacher_revision':json.loads((folder/'teacher.json').read_text())['revision'],
                                     'teacher_actions':list(ACTIONS[1:4]),'target_source':'Laya image probabilities only',
                                     'engine_annotation_targets':False,'teacher_weights_trained':False,
                                     'student_live_teacher_required':False,'initial_decoder_retained':True,
                                     'teacher_process_closed_for_student_tests':True,'trained_scenarios':['basic']},
                    'game_gate_passed':bool(accepted),'biological_validation_passed':False,'promoted':False,
                    'test':{k:summarize(v) for k,v in tests.items()},
                    'test_by_task':{k:{'basic':summarize(v)} for k,v in tests.items()},
                    'scope':'Laya-supervised existing relay magnitudes and engineered readout; native images enter only through the retained eye model. Teacher absent from deployed policy. Basic target-shooting curriculum only. Physiological validity and general Doom mastery are not established.'}
            names=['weights.npz','circuit.npz','circuit.json','actor.npz','plan.json','teacher.json','experience.json','cache-audit.json',
                   'distillation.json','fixed-biology-actor.npz','fixed-biology-fit.json','plasticity.json','ablation.json','physiology.json','gate.json','benchmark.json']
            report['output_sha256']={name:digest(folder/name,'sha256') for name in names}
            report['source_sha256']={**parent_report['source_sha256'],**{name:digest(Path(__file__).with_name(name),'sha256') for name in
                                    ('adaptive_policy.py','bridge_training.py','laya_brain_training.py','laya_brain_fit.py','laya_brain_teacher.py')}}
            atomic_json(folder/'report.json',report);load_any(folder)
            atomic_json(ROOT/'latest.json',{'path':str(folder.resolve()),'sha256':digest(folder/'report.json','sha256')})
            atomic_json(ROOT/'progress.json',{'status':'completed','stage':4,'folder':str(folder)})
            print(json.dumps({'training':report['laya_training'],'changed_edges':changed,'test':report['test'],'game_gate_passed':accepted},indent=2),flush=True)
        except BaseException as error:
            atomic_json(ROOT/'progress.json',{'status':'failed','folder':str(folder),'error':str(error)});raise


if __name__=='__main__':main()
