"""Interactive retinal control and reward feedback; separate from the old LIF pilot."""

import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Condition, Thread
from time import perf_counter
from urllib.parse import urlsplit, parse_qs
import webbrowser

import numpy as np

from flydoom.data import digest
from flydoom.eye_mapping import write_json
from flydoom.live_activity import frame_png
from flydoom.movement_core import make_game, apply_action
from flydoom.retinal_policy import ACTIONS, load_policy

STATIC=Path(__file__).parent/'web/retinal'


class RetinalSession:
    def __init__(self, checkpoint, output):
        self.encoder,self.actor,self.report=load_policy(checkpoint)
        self.checkpoint=Path(checkpoint);self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.condition=Condition();self.paused=True;self.busy=False;self.closed=False;self.pending=0
        self.game=None;self.sequence=0;self.seed=74000;self.error=None;self.learning=True;self.phase='loading'
        self.new_run(self.seed)
        self.worker=Thread(target=self.loop,daemon=True);self.worker.start()

    def new_run(self,seed):
        if type(seed)!=int or not 0<=seed<2**32:raise ValueError('Require a uint32 seed')
        plan=json.loads((self.checkpoint/'plan.json').read_text())
        reserved={r['seed'] for s,rows in plan['splits'].items() if s!='reward_train' for r in rows}
        if seed in reserved:raise ValueError('This seed belongs to a reserved validation/test split')
        with self.condition:
            if self.busy or not self.paused:raise ValueError('Pause and wait for the current decision first')
            if self.game:self.game.close()
            self.game=make_game(seed);self.game.set_seed(seed);self.game.new_episode()
            self.encoder.reset();self.actor.reset();self.sequence=0;self.seed=seed;self.error=None;self.phase='ready';self.pending=0
            self.rng=np.random.default_rng(seed);self.folder=self.output/datetime.now().strftime('run-%H%M%S-%f')
            self.folder.mkdir();self.actor.save(self.folder/'actor-start.npz')
            write_json(self.folder/'identity.json',{'checkpoint':str(self.checkpoint),
                       'checkpoint_sha256':digest(self.checkpoint/'report.json','sha256'),
                       'play_source_sha256':digest(Path(__file__),'sha256'),'seed':seed,
                       'initial_actor_sha256':digest(self.folder/'actor-start.npz','sha256'),
                       'mode':'Retinal output controls the game; new-run retains this process learner weights'})
            self.latest={'sequence':0,'frame':frame_png(self.game.get_state().screen_buffer),'action':'Not decided',
                         'probabilities':[1/6]*6,'reward':0,'return':0,'kills':0,'trace':[],
                         'terms':None,'feedback':None,'after_frame':None,'encoding_ms':0,'wall_ms':0}
            write_json(self.folder/'0000.json',self.latest)
            return {'run':str(self.folder)}

    def command(self,command):
        with self.condition:
            if command=='pause':self.paused=True;self.pending=0
            elif command=='stop':self.paused=True;self.pending=0;self.phase='stopped'
            elif command in ('run','step'):
                if self.phase!='ready':raise ValueError('Create a new run after the episode ends')
                if self.busy:raise ValueError('Wait for the current decision')
                if command=='run':self.paused=False
                else:self.paused=True;self.pending+=1
            else:raise ValueError('Unknown control')
            self.condition.notify_all()

    def set_learning(self,enabled):
        with self.condition:
            if type(enabled)!=bool or self.busy or not self.paused:raise ValueError('Change learning only while paused and idle')
            self.learning=enabled;self.actor.eligibility.fill(0)

    def step(self):
        start=perf_counter();frame=self.game.get_state().screen_buffer.copy()
        raw,telemetry=self.encoder.encode(frame,telemetry=True)
        action,p,features=self.actor.decide(raw,self.rng if self.learning else None)
        before=self.actor.weights.copy();terms=self.actor.contributions(features,action)
        # Logit terms are captured before feedback changes the decoder.
        for cell in terms['cells']:
            i=int(np.searchsorted(self.encoder.ids,np.uint64(int(cell['root_id']))))
            cell['cell_type']=str(self.encoder.types[i]);cell['state']=float(self.encoder.model.delta[i])
            cell['tau_ms']=float(self.encoder.model.tau[i])
        effect=apply_action(self.game,action)
        feedback=self.actor.feedback(features,action,p,effect['reward_delta']) if self.learning else None
        after=self.game.get_state()
        sequence=self.sequence+1
        record={'sequence':sequence,'frame':frame_png(frame),'after_frame':frame_png(after.screen_buffer) if after else None,
                'action':ACTIONS[action],'probabilities':p.tolist(),'reward':effect['reward_delta'],
                'return':float(self.game.get_total_reward()),'kills':effect['kills'],'effect':effect,
                'terms':terms,'feedback':feedback,'learning_at_decision':self.learning,
                'encoding_ms':telemetry['encoding_ms'],'trace':telemetry['trace'],
                'neural_clock_ms':telemetry['neural_clock_ms'],'wall_ms':(perf_counter()-start)*1000}
        path=self.folder/f'{sequence:04d}.npz'
        np.savez_compressed(path,root_ids=self.encoder.ids,delta=self.encoder.model.delta,frame=frame,
                            decoder_root_ids=self.actor.root_ids,features=features,weights_before=before,
                            weights_after=self.actor.weights,probabilities=p,action=action,reward=effect['reward_delta'])
        record['recording_sha256']=digest(path,'sha256')
        write_json(self.folder/f'{sequence:04d}.json',record)
        self.actor.save(self.folder/'actor-latest.npz')
        with self.condition:
            self.latest=record;self.sequence=sequence
            if effect['episode_finished'] or sequence>=75:
                self.phase='completed';self.paused=True;self.pending=0
            # A concurrent End run remains authoritative after the in-flight step.
            self.busy=False;self.condition.notify_all()

    def loop(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda:self.closed or self.phase=='ready' and (not self.paused or self.pending>0))
                if self.closed:return
                if self.pending:self.pending-=1
                self.busy=True
            try:self.step()
            except Exception as error:
                with self.condition:
                    self.error=str(error);self.phase='error';self.paused=True;self.busy=False;self.pending=0
                    self.condition.notify_all()

    def state(self,sequence=None):
        with self.condition:
            if sequence is not None and (type(sequence)!=int or not 0<=sequence<=self.sequence):raise ValueError('Unknown recorded decision')
            value=self.latest if sequence is None or sequence==self.sequence else json.loads((self.folder/f'{sequence:04d}.json').read_text())
            return {**value,'phase':self.phase,'paused':self.paused,'busy':self.busy,'learning':self.learning,
                    'error':self.error,'latest_sequence':self.sequence,'run':str(self.folder),'seed':self.seed}

    def metadata(self):
        return {'mode':'Closed-loop retinal controller','actions':ACTIONS,'neurons':len(self.encoder.ids),
                'readout_cells':len(self.actor.root_ids),'checkpoint':str(self.checkpoint),
                'gains':self.encoder.gains.tolist(),'benchmark':self.report['test'],
                'controls_action':True,'online_laya':False,'whole_brain_motor_path':False,
                'scope':'T4/T5 cells feed an engineered action decoder. Full-brain descending control is not implemented.',
                'camera':self.encoder.camera,'retina_uv':self.encoder.uv[self.encoder.visible].tolist()}

    def close(self):
        with self.condition:self.closed=True;self.paused=True;self.condition.notify_all()
        self.worker.join(timeout=30)
        if self.worker.is_alive():raise RuntimeError('Retinal worker did not stop')
        if self.game:self.game.close()


def handler(session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,value,status=200,mime='application/json'):
            raw=value if isinstance(value,bytes) else json.dumps(value,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',mime+'; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            path=urlsplit(self.path)
            try:
                if path.path in ('/','/app.js','/style.css'):
                    name,mime={'/':('index.html','text/html'),'/app.js':('app.js','text/javascript'),'/style.css':('style.css','text/css')}[path.path]
                    return self.send((STATIC/name).read_bytes(),mime=mime)
                if path.path=='/api/meta':return self.send(session.metadata())
                if path.path=='/api/state':
                    value=parse_qs(path.query).get('sequence',[None])[0]
                    return self.send(session.state(int(value) if value is not None else None))
                return self.send({'error':'Unknown route'},404)
            except (ValueError,KeyError,OSError) as error:self.send({'error':str(error)},400)
        def do_POST(self):
            if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{self.server.server_port}'):
                return self.send({'error':'Use the local origin'},403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if self.headers.get('Content-Type')!='application/json' or not 0<size<=256:raise ValueError('Invalid request')
                body=json.loads(self.rfile.read(size))
                if self.path=='/api/control':session.command(body['command'])
                elif self.path=='/api/learning':session.set_learning(body['enabled'])
                elif self.path=='/api/new-run':session.new_run(body['seed'])
                else:raise ValueError('Unknown route')
                self.send({'ok':True})
            except (ValueError,KeyError,TypeError,OSError) as error:self.send({'error':str(error)},400)
    return Handler


def main(argv=None):
    # Preserve the previous entry point while serving the complete workspace.
    from flydoom.integrated_desk import main as integrated_main
    return integrated_main(argv)


if __name__=='__main__':main()
