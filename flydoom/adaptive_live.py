"""Observable, bounded paired-reward learning between complete live episodes."""
from pathlib import Path
from threading import Thread
import json

import numpy as np
from scipy import sparse

from flydoom.adaptive_policy import ACTIONS, TASKS, SCHEMA, make_game, action, load_any
from flydoom.data import digest
from flydoom.integrated_desk import IntegratedSession
from flydoom.research_registry import atomic_json
from flydoom.reward_plasticity import RewardPlasticity


class AdaptiveSession(IntegratedSession):
    def __init__(self,checkpoint,output,scenario='basic'):
        if scenario not in TASKS:raise ValueError('Unknown scenario')
        self.scenario=scenario;self.decision_limit=150 if scenario=='deadly_corridor' else 75
        self.relay_learner=None;self.relay_trial=None;self.auto_learning=False;self.remaining=0
        super().__init__(checkpoint,output)
        if self.report.get('schema') not in (SCHEMA, 'stable_sensorimotor_v1'):
            self.close();raise ValueError('Adaptive play requires an eight-action research checkpoint')
        visual=load_any('runs/retinal-control-v1')[0].ids
        self.relay_learner=RewardPlasticity(self.encoder,visual,seed=111026)
        timing=json.loads((self.checkpoint/'physiology.json').read_text())
        self.adaptive_evidence={'ablation':json.loads((self.checkpoint/'ablation.json').read_text()),
                               'timing':{k:timing['proposal'][k] for k in ('tau_ms','final','development_low_luminance','waveform_threshold_passed','scope')},
                               'physiology_status':timing['status']}
        if self.report.get('recovery_training'):
            from flydoom.recovery_evidence import teacher_coverage
            self.adaptive_evidence['teaching_coverage']=teacher_coverage(json.loads((self.checkpoint/'experience.json').read_text(encoding='utf-8')))
        self.save_learner()
        self.coordinator=Thread(target=self.advance_loop,daemon=True);self.coordinator.start()

    def create_game(self,seed):
        return make_game(seed,self.scenario,labels=False)

    def play_action(self,index):
        return action(self.game,index,observe=True)

    def save_learner(self):
        state={'checkpoint_sha256':digest(self.checkpoint/'report.json','sha256'),
               'decoder_sha256':digest(self.checkpoint/'actor.npz','sha256'),
               'rule_sha256':digest(Path(__file__).with_name('reward_plasticity.py'),'sha256'),
               'learner':self.relay_learner.snapshot(),'scenario':self.scenario,
               'scope':'Relative to the selected checkpoint. Training rewards only; no automatic promotion.'}
        path=self.output/'relay-learner.json';atomic_json(path,state)
        atomic_json(self.output/'relay-learner-identity.json',{'sha256':digest(path,'sha256')})

    def new_run(self,seed,*,paired=False):
        with self.condition:
            if self.busy or not self.paused:raise ValueError('Pause before starting another episode')
            learner=self.relay_learner
            result=super().new_run(seed)
            if learner and not paired:
                self.auto_learning=False;self.remaining=0;learner.cancel()
            self.relay_trial=None
            if learner:
                if self.auto_learning:self.relay_trial=learner.begin(seed,self.scenario)
                else:learner.bridge.apply(learner.center)
                self.outgoing=self.encoder.model.weights.T.tocsr()
                self.save_learner()
            # The graph is fixed throughout this episode, including historical inspection.
            sparse.save_npz(self.folder/'graph-weights.npz',self.encoder.model.weights)
            np.save(self.folder/'graph-tau.npy',self.encoder.model.tau,allow_pickle=False)
            atomic_json(self.folder/'relay-trial.json',{'trial':self.relay_trial,'scenario':self.scenario,
                        'graph_sha256':digest(self.folder/'graph-weights.npz','sha256'),
                        'tau_sha256':digest(self.folder/'graph-tau.npy','sha256')})
            return result

    def complete_record(self,record):
        with self.condition:
            record['relay_trial']=self.relay_trial
            record['scenario']=self.scenario
            if record['termination'] and self.relay_trial and self.relay_learner.trial_open:
                record['relay_feedback']=self.relay_learner.observe(record['return'],self.seed,self.scenario)
                self.remaining=max(0,self.remaining-1)
                if not self.remaining:self.auto_learning=False
                self.save_learner()
                atomic_json(self.folder/'relay-result.json',{'feedback':record['relay_feedback'],
                            'learner':self.relay_learner.snapshot()})

    def command(self,command):
        with self.condition:
            if command in ('pause','stop'):self.auto_learning=False
            if command=='stop' and self.relay_learner:
                self.relay_learner.cancel();self.relay_trial=None;self.save_learner()
            return super().command(command)

    def restart_episode(self):
        with self.condition:self.auto_learning=False
        return super().restart_episode()

    def set_learning(self,enabled):
        if enabled:raise ValueError('Use paired relay learning here; its decoder must remain frozen')
        return super().set_learning(enabled)

    def start_learning(self,pairs,scenario):
        with self.condition:
            if self.report.get('reward_training',{}).get('version',0)>=4:
                module='expanded_learning' if self.report['reward_training']['version']>=5 else 'self_learning'
                raise ValueError(f'Run python -m flydoom.{module} for bounded corrected reward learning; this desk stays frozen')
            if self.report.get('precision_training'):
                raise ValueError('Corrected arithmetic requires a compatible offline trainer; this session is frozen')
            if self.report.get('reward_training') or self.report.get('recovery_training') or self.report.get('sampling_evaluation'):
                raise ValueError('This desk tests a frozen reward-trained checkpoint. Run python -m flydoom.reward_training_v3 to continue learned weights in a bounded training experiment.')
            if type(pairs)!=int or not 1<=pairs<=10 or scenario not in TASKS:raise ValueError('Choose 1 to 10 pairs and a supported scenario')
            if self.busy or not self.paused or self.teacher_busy:raise ValueError('Pause and wait before starting paired learning')
            self.relay_learner.cancel();self.learning=False;self.scenario=scenario
            self.decision_limit=150 if scenario=='deadly_corridor' else 75
            self.remaining=2*pairs;self.auto_learning=True
            try:self.new_run(self.next_seed(),paired=True)
            except Exception:self.auto_learning=False;raise
            super().command('run')

    def next_seed(self):
        # Preserve all reserved evaluation seeds, including those in other experiments.
        reserved=set()
        for path in Path('runs').rglob('plan.json'):
            try:plan=json.loads(path.read_text(encoding='utf-8'))
            except (OSError,ValueError):continue
            for name,rows in plan.get('splits',{}).items():
                if name not in ('train','reward_train') and isinstance(rows,list):
                    reserved.update(r['seed'] for r in rows if isinstance(r,dict) and 'seed' in r)
        seed=(self.seed+1)%2**32
        while seed in reserved:seed=(seed+1)%2**32
        return seed

    def advance_loop(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda:self.closed or self.auto_learning and self.phase=='completed',timeout=.5)
                if self.closed:return
                if not self.auto_learning or self.phase!='completed':continue
                # Retain the terminal frame briefly; Pause and End interrupt this wait.
                self.condition.wait_for(lambda:self.closed or not self.auto_learning,timeout=1)
                if self.closed:return
                if not self.auto_learning:continue
                if self.teacher_busy:continue
                try:
                    seed=self.relay_learner.pair_seed if self.relay_learner.plus is not None else self.next_seed()
                    self.new_run(seed,paired=True);super().command('run')
                except Exception as error:
                    self.auto_learning=False;self.error=str(error);self.phase='error';self.paused=True
                    self.diagnostics.exception('relay_advance_failed')

    def adaptive_status(self):
        with self.condition:
            from flydoom.research_evidence import extension_evidence
            progress_root=Path('runs/self-learning-expanded') if Path('runs/self-learning-expanded/progress.json').exists() else Path('runs/self-learning')
            learner=self.relay_learner
            groups=[]
            for index in np.argsort(-abs(learner.center-1).ravel())[:12]:
                channel,group=np.unravel_index(index,learner.center.shape)
                if learner.center[channel,group]!=1:
                    groups.append({'cell_type':str(learner.bridge.types[group]),
                                   'channel':('positive visual','negative visual','recurrent','tau')[channel],
                                   'gain':float(learner.center[channel,group])})
            return {'enabled':True,'scenario':self.scenario,'scenarios':TASKS,'running':self.auto_learning,
                    'remaining_episodes':self.remaining,'trial':None if not self.relay_trial else {k:v for k,v in self.relay_trial.items() if k!='parameters'},
                    'completed_pairs':learner.updates,'active_groups':int(learner.active.sum()),
                    'gain_delta_l2':float(np.linalg.norm(learner.center-1)),
                    'changed_groups':int(np.count_nonzero(learner.center!=1)),
                    'largest_live_gain_changes':groups,
                    'last_feedback':learner.history[-1] if learner.history else None,
                    'checkpoint':str(self.checkpoint),'live_state':str(self.output/'relay-learner.json'),
                    'four_steps':self.report['four_steps'],'benchmark':(getattr(self,'selection_evidence',None) or {}).get('test_by_task',self.report['test_by_task']),
                    'selection_evidence':getattr(self,'selection_evidence',None),
                    'self_learning_progress':json.loads((progress_root/'progress.json').read_text(encoding='utf-8')) if (progress_root/'progress.json').exists() else None,
                    'research_extensions':extension_evidence(),
                    'teacher_training':self.report.get('laya_training'),
                    'reward_training':self.report.get('reward_training'),
                    'recovery_training':self.report.get('recovery_training'),
                    'sampling_evaluation':self.report.get('sampling_evaluation'),
                    'precision_training':self.report.get('precision_training'),
                    'numerical_audit':json.loads((self.checkpoint/'numerical-audit.json').read_text(encoding='utf-8')) if self.report.get('precision_training') else None,
                    'game_gate_passed':bool(self.report.get('game_gate_passed',False)),
                    'evidence':self.adaptive_evidence,
                    'biological_validation_passed':False,'promoted':False}

    def metadata(self):
        value=super().metadata();value['adaptive']=True
        if self.report.get('neural_numerics'):
            value['neural_numerics']=self.report['neural_numerics']
        if self.report.get('decision_sampling'):
            value['decision_sampling']=self.report['decision_sampling']
        value['selection_evidence']=getattr(self,'selection_evidence',None)
        value['teacher']['role']='Inherited offline Laya teacher; eight-action curriculum uses training-only engine annotations. Laya comparison supports the original six actions.'
        if self.report.get('laya_training'):
            value['teacher']['role']='Laya taught relay connections and the readout from images; basic strafe/fire curriculum. Live decisions use the eye and brain only.'
            value['laya_training']=self.report['laya_training']
        if self.report.get('reward_training'):
            value['reward_training']=self.report['reward_training']
            value['teacher']['role']='Offline initialization teacher; subsequent learning used game rewards and retained teacher examples. No live teacher input.'
        if self.report.get('recovery_training'):
            value['recovery_training']=self.report['recovery_training']
            value['teacher']['role']='Offline correction on student-visited images, following inherited reward training. No live teacher input or weight changes.'
        if self.report.get('sampling_evaluation'):
            value['sampling_evaluation']=self.report['sampling_evaluation']
            value['decision_sampling']=self.report['decision_sampling']
            value['teacher']['role']='Offline Laya teacher. Frozen live actions sample the learned reward-trained neural policy; no teacher call or weight update.'
        if self.report.get('precision_training'):
            value['precision_training']=self.report['precision_training']
            value['neural_numerics']=self.report['neural_numerics']
            value['numerical_audit']=json.loads((self.checkpoint/'numerical-audit.json').read_text(encoding='utf-8'))
            value['decision_sampling']=self.report['decision_sampling']
            value['teacher']['role']='Offline Laya supervision on corrected neural states. Frozen live play uses the eye and circuit only.'
        return value

    def resume_relay(self,path):
        if self.report.get('neural_numerics'):
            raise ValueError('Legacy relay optimization cannot resume into corrected neural arithmetic')
        path=Path(path)
        with self.condition:
            if self.busy or not self.paused or self.sequence:raise ValueError('Resume before the first live decision')
            identity=json.loads(path.with_name('relay-learner-identity.json').read_text())
            if digest(path,'sha256')!=identity['sha256']:raise ValueError('Saved relay learner changed')
            state=json.loads(path.read_text())
            for key,file in [('checkpoint_sha256',self.checkpoint/'report.json'),('decoder_sha256',self.checkpoint/'actor.npz'),('rule_sha256',Path(__file__).with_name('reward_plasticity.py'))]:
                if digest(file,'sha256')!=state[key]:raise ValueError('Relay resume identity mismatch')
            self.relay_learner.restore(state['learner'])
            # Restarting a server resumes learned gains, not a partly played trial.
            self.relay_learner.cancel();self.new_run(self.next_seed())

    def resume_live(self,folder):
        raise ValueError('Use --resume-relay for the adaptive learner; its decoder remains frozen')

    def close(self):
        super().close()
        if hasattr(self,'coordinator'):self.coordinator.join(timeout=5)
