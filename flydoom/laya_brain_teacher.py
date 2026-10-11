"""Frozen Laya teacher with explicit mapping into the eight-action student."""
import argparse
import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
from threading import Lock,Thread
from time import perf_counter

from flydoom.adaptive_policy import ACTIONS,TASKS
from flydoom.vision import VisionClient,verified_manifest


def teacher_actions(scenario):
    if scenario not in TASKS:raise ValueError('Unsupported teacher scenario')
    return ACTIONS[1:4] if scenario=='basic' else ACTIONS[1:]


def map_probabilities(probabilities,options):
    import numpy as np
    p=np.asarray(probabilities,float)
    if len(set(options))!=len(options) or any(a not in ACTIONS for a in options):raise ValueError('Invalid teacher action mapping')
    if p.shape!=(len(options),) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1):raise ValueError('Invalid teacher distribution')
    result=np.zeros(len(ACTIONS));result[[ACTIONS.index(a) for a in options]]=p;return result


class LayaBrainTeacher(VisionClient):
    def __init__(self,log,scenario='basic'):
        teacher_actions(scenario)
        Path(log).parent.mkdir(parents=True,exist_ok=True)
        self.log=open(log,'a',encoding='utf-8');self.lock=Lock();self.messages=Queue()
        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONIOENCODING':'utf-8',
             'KMP_BLOCKTIME':'0','OMP_WAIT_POLICY':'PASSIVE'}
        try:
            self.process=subprocess.Popen([sys.executable,'-u','-m','flydoom.laya_brain_teacher','--worker','--scenario',scenario],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except BaseException:self.log.close();raise
        def reader():
            for line in self.process.stdout:self.messages.put(line)
            self.messages.put(None)
        Thread(target=reader,daemon=True).start()
        try:self.metadata=self.receive(180)['ready']
        except BaseException:self.close();raise


def worker(scenario):
    import numpy as np
    model=Path('models/laya-vision');source=Path('models/laya-vision-source')
    manifest=verified_manifest(model,source);sys.path.insert(0,str(source.resolve()));options=teacher_actions(scenario)
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import laya
        from laya.games import doom_question
        from PIL import Image
        torch.set_num_threads(4)
        agent=laya.load_vlm(str(model.resolve()),device='cuda' if torch.cuda.is_available() else 'cpu',dtype='fp32')
        agent.model.eval()
        prompt=doom_question(scenario,options)
    print(json.dumps({'ready':{'repo':manifest['repo'],'revision':manifest['revision'],
          'source_commit':manifest['source_commit'],'manifest_sha256':hashlib.sha256((model/'download.json').read_bytes()).hexdigest(),
          'actions':ACTIONS,'teacher_actions':options,'question':prompt,'scenario':scenario,'weights_trained':False,
          'role':'Frozen image teacher for offline distillation; no engine annotations'}}),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get('command')=='close':break
            if request['shape']!=[240,320,3]:raise ValueError('Require native RGB observation')
            frame=np.frombuffer(base64.b64decode(request['pixels'],validate=True),np.uint8).reshape(request['shape'])
            started=perf_counter();raw={}
            with contextlib.redirect_stdout(sys.stderr),torch.inference_mode():
                result=agent.predict({'image':Image.fromarray(frame)},prompt,strict=True,n_permutations=1,_raw_logits=raw)
            logits=np.asarray(raw['action'],float)
            if logits.shape!=(len(options),) or not np.isfinite(logits).all():raise ValueError('Invalid teacher action logits')
            temperature=max(float(result['provenance']['temperatures']['action']),1e-3)
            z=logits/temperature;p=np.exp(z-z.max());p/=p.sum();p=map_probabilities(p,options)
            print(json.dumps({'result':{'probabilities':p.tolist(),'logits':logits.tolist(),'temperature':temperature,
                  'frame_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),'seconds':perf_counter()-started,
                  'actions':ACTIONS}},allow_nan=False),flush=True)
        except Exception as error:print(json.dumps({'error':str(error)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true',required=True)
    parser.add_argument('--scenario',choices=TASKS,default='basic');args=parser.parse_args();worker(args.scenario)
