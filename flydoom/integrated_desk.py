"""One research desk for anatomy, optical input, neural control and Laya evidence."""

import argparse
import base64
from collections import deque
from datetime import datetime
import hashlib
from http.server import ThreadingHTTPServer
import io
import json
import socket
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit, parse_qs
import webbrowser

import numpy as np
from PIL import Image

from flydoom import anatomy, named_anatomy
from flydoom.data import digest
from flydoom.eye_mapping import write_json
from flydoom.live_map import packed
from flydoom.retinal_mapping import load_counts, RESULT
from flydoom.retinal_play import RetinalSession, handler as base_handler
from flydoom.visual_observer import sample_frame


class IntegratedSession(RetinalSession):
    def __init__(self, checkpoint, output):
        super().__init__(checkpoint, output)
        self.full_ids,self.rows,self.contacts,_=load_counts()
        self.lookup={str(root):i for i,root in enumerate(self.full_ids)}
        self.visual_lookup={str(root):i for i,root in enumerate(self.encoder.ids)}
        self.decoder_lookup={str(root):i for i,root in enumerate(self.actor.root_ids)}
        self.visual_full_indices=np.array([self.lookup[str(root)] for root in self.encoder.ids])
        self.receptors=json.loads((RESULT/'receptors.json').read_text())
        self.receptor_lookup={r['root_id']:i for i,r in enumerate(self.receptors)}
        self.outgoing=self.encoder.model.weights.T.tocsr()
        self.asset_lock=Lock();self.teacher_busy=False;self.record_cache=None
        self.map_data=self.build_map()

    def state(self,sequence=None):
        with self.condition:
            return {**super().state(sequence),'teacher_busy':getattr(self,'teacher_busy',False)}

    def build_map(self):
        positions=np.array([[float(r['pos_'+axis]) for axis in 'xyz'] for r in self.rows])*[.004,.004,.04]
        center=(positions.min(0)+positions.max(0))/2;extent=float(np.ptp(positions,axis=0).max())
        roles=np.zeros(len(positions),np.uint8);roles[self.visual_full_indices]=3
        roles[self.visual_full_indices[self.encoder.inputs]]=1
        roles[[self.lookup[str(root)] for root in self.actor.root_ids]]=2
        return {'ids':[str(root) for root in self.full_ids], 'positions_f32':packed((positions-center)*1.7/extent,'<f4'),
                'roles_u8':packed(roles,'u1'),'edges':[], 'center_um':center.tolist(),'extent_um':extent,
                'visual_indices':self.visual_full_indices.tolist(), 'note':'Full anatomy; only the visual subcircuit is simulated in this mode'}

    def metadata(self):
        value=super().metadata()
        teacher=json.loads(Path('models/laya-vision/download.json').read_text())
        value.update(full_neurons=len(self.full_ids), hidden=0,memory_features=[],
                     teacher={'repo':teacher['repo'],'revision':teacher['revision'],'role':'Offline distillation teacher; on-demand same-frame comparison'},
                     distillation=json.loads((self.checkpoint/'distillation.json').read_text()),
                     unified=True)
        return value

    def new_run(self,seed):
        with self.condition:
            if getattr(self,'teacher_busy',False):raise ValueError('Wait for the current Laya comparison')
            value=super().new_run(seed);self.record_cache=None;return value

    def command(self,command):
        with self.condition:
            if getattr(self,'teacher_busy',False) and command in ('run','step'):raise ValueError('Wait for the current Laya comparison')
            return super().command(command)

    def record(self,sequence,run=None):
        with self.condition:
            if run is not None and run!=str(self.folder):raise ValueError('The requested run is no longer active')
            state=self.state(sequence)
            key=(str(self.folder),sequence)
            if self.record_cache is not None and self.record_cache[0]==key:return state,self.record_cache[1]
            if sequence:
                path=self.folder/f'{sequence:04d}.npz'
                if digest(path,'sha256')!=state['recording_sha256']:raise ValueError('Recorded state checksum mismatch')
                with np.load(path,allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
                np.testing.assert_array_equal(arrays['root_ids'],self.encoder.ids)
            else:
                frame=np.asarray(Image.open(io.BytesIO(base64.b64decode(state['frame'].split(',',1)[1]))).convert('RGB'))
                with np.load(self.folder/'actor-start.npz',allow_pickle=False) as initial:
                    weights=initial['weights'].copy()
                arrays={'frame':frame,'delta':np.zeros(len(self.encoder.ids)),
                        'weights_before':weights,'weights_after':weights.copy(),
                        'features':np.r_[np.zeros(len(self.actor.indices)),1.]}
            self.record_cache=(key,arrays)
            return state,arrays

    def activity(self,sequence,run=None):
        _,arrays=self.record(sequence,run)
        return {'sequence':sequence,'delta_f32':packed(arrays['delta'],'<f4'),
                'unit':'Dimensionless graded state; no state is simulated for the remaining anatomical cells'}

    def label(self,root):
        row=self.rows[self.lookup[root]]
        return {'id':root,'label':row.get('cell_type') or row.get('super_class') or root}

    def search(self,query):
        q=query.strip().lower()
        if len(q)>100:raise ValueError('Query too long')
        indices=(self.lookup[str(root)] for root in self.actor.root_ids) if not q else (
            i for i,r in enumerate(self.rows) if q in str(self.full_ids[i]) or any(q in r.get(k,'').lower() for k in ('cell_type','super_class','class')))
        result=[]
        for i in indices:
            result.append(self.label(str(self.full_ids[i])))
            if len(result)>=32:break
        return {'cells':result}

    def inspect(self,root,sequence,run=None):
        if root not in self.lookup:raise ValueError('Unknown root ID')
        _,saved=self.record(sequence,run);row=self.rows[self.lookup[root]]
        value={'id':root,'label':self.label(root)['label'],'sequence':sequence,'annotation':row,'edges':[],
               'simulated':root in self.visual_lookup,'decoder':None}
        if root not in self.visual_lookup:
            value['note']='Anatomical context only; this cell is outside the active visual circuit.'
            return value
        i=self.visual_lookup[root];state=saved['delta'];weights=self.encoder.model.weights
        released=np.maximum(self.encoder.model.baseline+state,0)-np.maximum(self.encoder.model.baseline,0)
        a,b=weights.indptr[i:i+2];sources=weights.indices[a:b];w=weights.data[a:b];products=w*released[sources]
        for j in np.argsort(-abs(products),kind='stable')[:12]:
            source=str(self.encoder.ids[sources[j]])
            value['edges'].append({'source':self.label(source),'target':self.label(root),
                                  'weight':float(w[j]),'release':float(released[sources[j]]),
                                  'contribution':float(products[j]),
                                  'contacts':int(self.contacts[self.lookup[root],self.lookup[source]])})
        value.update(state=float(state[i]),tau_ms=float(self.encoder.model.tau[i]),
                     note='Endpoint weight × release diagnostic; not a replay of the final integration input.')
        if root in self.decoder_lookup:
            j=self.decoder_lookup[root];activation=float(saved['features'][j])
            value['decoder']={'activation':activation,'weights':saved['weights_before'][:,j].tolist(),
                              'contributions':(saved['weights_before'][:,j]*activation).tolist(),
                              'weight_changes':(saved['weights_after'][:,j]-saved['weights_before'][:,j]).tolist()}
        return value

    def retinal_path(self,index):
        receptor=self.receptors[index]
        path=[receptor['root_id'],receptor['l1_root_id'],receptor['mi1_root_id']]
        start=self.visual_lookup[path[-1]];targets={self.visual_lookup[str(root)] for root in self.actor.root_ids}
        queue=deque([(start,0)]);parent={start:None};found=None
        while queue and len(parent)<4000:
            i,depth=queue.popleft()
            if i in targets:found=i;break
            if depth>=5:continue
            a,b=self.outgoing.indptr[i:i+2];partners=self.outgoing.indices[a:b];weights=self.outgoing.data[a:b]
            for j in np.argsort(-abs(weights),kind='stable')[:48]:
                target=int(partners[j])
                if weights[j] and target not in parent:parent[target]=i;queue.append((target,depth+1))
        if found is not None:
            tail=[]
            while found!=start:tail.append(str(self.encoder.ids[found]));found=parent[found]
            path.extend(reversed(tail))
        edges=[]
        for source,target in zip(path,path[1:]):
            s,t=self.visual_lookup[source],self.visual_lookup[target]
            edges.append({'source':self.label(source),'target':self.label(target),
                          'weight':float(self.encoder.model.weights[t,s]),
                          'contacts':int(self.contacts[self.lookup[target],self.lookup[source]])})
        return path,edges

    def retina(self,sequence,index=None,run=None):
        state,saved=self.record(sequence,run)
        if index is None:
            options=np.flatnonzero(self.encoder.visible)
            index=int(options[np.argmin(np.sum((self.encoder.uv[options]-.5)**2,axis=1))])
        if type(index)!=int or not 0<=index<len(self.receptors):raise ValueError('Invalid receptor index')
        contrast=sample_frame(saved['frame'],self.encoder.uv,self.encoder.visible)
        path,edges=self.retinal_path(index)
        return {'sequence':sequence,'selected_index':index,'selected':self.receptors[index],
                'input_drive':float(contrast[index]),'path':path,'edges':edges,
                'nodes':[{**self.label(root),'state':float(saved['delta'][self.visual_lookup[root]])} for root in path],
                'reaches_decoder':path[-1] in self.decoder_lookup,
                'samples':[{'index':int(i),'uv':self.encoder.uv[i].tolist(),'drive':float(contrast[i])} for i in np.flatnonzero(self.encoder.visible)],
                'note':'Provisional optical assignment. Path follows existing nonzero edges; it is not proof of exclusive causal credit.'}

    def skeleton(self,root):
        if root not in self.lookup:raise ValueError('Unknown root ID')
        with self.asset_lock:
            candidates=[Path('data/anatomy/flywire/skeletons')/(root+'.bin')]
            folder=Path('data/anatomy/flywire/integrated-skeletons');folder.mkdir(parents=True,exist_ok=True)
            target=folder/(root+'.bin');candidates.insert(0,target)
            path=next((p for p in candidates if p.exists() and p.with_suffix('.sha256').exists()),None)
            if path:
                body=path.read_bytes()
                if hashlib.sha256(body).hexdigest()!=path.with_suffix('.sha256').read_text().strip():raise ValueError('Skeleton checksum changed')
            elif (eye_path:=Path('runs/eye-mapping-v1/skeletons')/root).exists():
                entry=None
                for version in ('v2','v1'):
                    manifest=Path('runs')/('eye-mapping-'+version)/'skeleton_manifest.json'
                    if manifest.exists():entry=json.loads(manifest.read_text()).get(root)
                    if entry:break
                if entry is None:raise ValueError('Cached eye morphology has no verification manifest')
                body=eye_path.read_bytes()
                if hashlib.sha256(body).hexdigest()!=entry['sha256']:raise ValueError('Eye skeleton checksum changed')
            else:
                body=anatomy.download(anatomy.SKELETONS+'/'+root);anatomy.skeleton_arrays(body)
                target.write_bytes(body);target.with_suffix('.sha256').write_text(hashlib.sha256(body).hexdigest())
            positions,links=anatomy.skeleton_arrays(body)
            return {'id':root,'positions_f32':packed(anatomy.normalize(positions,self.map_data)[links].reshape(-1,3),'<f4'),
                    'nodes':len(positions),'segments':len(links),'sha256':hashlib.sha256(body).hexdigest(),
                    'note':'Actual v783 morphology. Existing disk caches are retained.'}

    def compare_teacher(self,sequence,run=None):
        with self.condition:
            if self.busy or not self.paused or self.teacher_busy:raise ValueError('Pause and wait for the current operation first')
            state,saved=self.record(sequence,run);frame=saved['frame'].copy();folder=self.folder
            path=folder/f'teacher-{sequence:04d}.json'
            if path.exists():return json.loads(path.read_text())
            self.teacher_busy=True
        teacher=None
        try:
            from flydoom.movement_teacher import MovementTeacher
            teacher=MovementTeacher(folder/'teacher.log');answer=teacher.predict(frame)
            value={'sequence':sequence,'run':str(folder),'teacher':answer,'identity':teacher.metadata,
                   'student_probabilities':state['probabilities'],'policy_overridden':False,
                   'note':'Same-frame Laya comparison only; no action or weight update performed.'}
            write_json(path,value);return value
        finally:
            if teacher:teacher.close()
            with self.condition:self.teacher_busy=False;self.condition.notify_all()


def handler(session):
    base=base_handler(session)
    class Handler(base):
        def do_GET(self):
            p=urlsplit(self.path);q=parse_qs(p.query)
            try:
                scripts={'/map-core.js':'live/map.js','/map-anatomy.js':'laboratory/map.js','/integrated.js':'retinal/integrated.js'}
                if p.path in scripts:return self.send((Path(__file__).parent/'web'/scripts[p.path]).read_bytes(),mime='text/javascript')
                if p.path=='/api/map':return self.send(session.map_data)
                if p.path=='/api/anatomy':return self.send(named_anatomy.surfaces(session.map_data))
                if p.path=='/api/skeleton':return self.send(session.skeleton(q['id'][0]))
                if p.path=='/api/cells':return self.send(session.search(q.get('q',[''])[0]))
                if p.path=='/api/map-activity':return self.send(session.activity(int(q['sequence'][0]),q.get('run',[None])[0]))
                if p.path=='/api/neuron':return self.send(session.inspect(q['id'][0],int(q['sequence'][0]),q.get('run',[None])[0]))
                if p.path=='/api/retina':return self.send(session.retina(int(q['sequence'][0]),int(q['index'][0]) if 'index' in q else None,q.get('run',[None])[0]))
                return super().do_GET()
            except (ValueError,KeyError,OSError) as error:self.send({'error':str(error)},400)
        def do_POST(self):
            if self.path!='/api/teacher':return super().do_POST()
            if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{self.server.server_port}'):return self.send({'error':'Use local origin'},403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if self.headers.get('Content-Type')!='application/json' or not 0<size<=1024:raise ValueError('Invalid request')
                body=json.loads(self.rfile.read(size));self.send(session.compare_teacher(body['sequence'],body.get('run')))
            except (ValueError,KeyError,TypeError,OSError,RuntimeError) as error:self.send({'error':str(error)},400)
    return Handler


class DeskServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can silently bind beside an old server.
    allow_reuse_address=False

    def server_bind(self):
        if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('runs/retinal-control-v1'))
    parser.add_argument('--port',type=int,help='Explicit port; otherwise select a free port from 8782 through 8791')
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args(argv)
    server=None;session=None
    for port in ([args.port] if args.port is not None else range(8782,8792)):
        try:
            server=DeskServer(('127.0.0.1',port),handler(None));break
        except OSError as error:
            if args.port is not None:
                parser.error(f'Cannot open port {args.port}: {error}. Close the previous server or omit --port to select a free port.')
    if server is None:raise OSError('No available local desk port between 8782 and 8791')
    try:
        session=IntegratedSession(args.checkpoint,Path('runs')/datetime.now().strftime('integrated-live-%Y%m%d-%H%M%S-%f'))
        server.RequestHandlerClass=handler(session)
        print(f'Integrated Fly Doom desk: http://127.0.0.1:{server.server_port}',flush=True)
        if not args.no_browser:webbrowser.open(f'http://127.0.0.1:{server.server_port}')
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:pass
    finally:
        server.server_close()
        if session:session.close()


if __name__=='__main__':main()
