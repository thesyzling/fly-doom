"""Persistent sensorimotor champion registry with verified, reversible promotion."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from flydoom.data import digest

ROOT=Path('runs/sensorimotor-learning')
INITIAL=Path('runs/retinal-control-v1')


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)


def record(folder):
    folder=Path(folder).resolve()
    return {'path':str(folder),'report_sha256':digest(folder/'report.json','sha256')}


def verify(item):
    folder=Path(item['path'])
    if digest(folder/'report.json','sha256')!=item['report_sha256']:raise ValueError('Registered report changed')
    from flydoom.sensorimotor import load_any
    load_any(folder)
    return folder


def state(root=ROOT):
    path=Path(root)/'registry.json'
    return json.loads(path.read_text()) if path.exists() else {'champion':record(INITIAL),'history':[],'latest':None}


def champion(root=ROOT):
    value=state(root);folder=Path(value['champion']['path'])
    if digest(folder/'report.json','sha256')!=value['champion']['report_sha256']:raise ValueError('Champion report changed')
    return folder


def register(folder,expected_parent,root=ROOT):
    candidate=record(folder);verify(candidate)
    report=json.loads((Path(folder)/'report.json').read_text())
    current=state(root)
    if current['champion']!=record(expected_parent):raise ValueError('Champion changed during candidate training')
    accepted=report['promotion']['accepted'] is True
    current['history'].append({'time':datetime.now(timezone.utc).isoformat(),'previous':current['champion'],
                               'candidate':candidate,'promoted':accepted})
    current['latest']=candidate
    if accepted:current['champion']=candidate
    atomic_json(Path(root)/'registry.json',current)
    return current


def rollback(root=ROOT):
    current=state(root)
    event=next((h for h in reversed(current['history']) if h.get('promoted') and h['candidate']==current['champion']),None)
    if event is None:raise ValueError('No promoted champion to roll back')
    verify(event['previous']);old=current['champion'];current['champion']=event['previous']
    current['history'].append({'time':datetime.now(timezone.utc).isoformat(),'rollback_from':old,'to':current['champion']})
    atomic_json(Path(root)/'registry.json',current)
    return current


class CycleLock:
    def __init__(self,root=ROOT):self.path=Path(root)/'cycle.lock'
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('x') as stream:stream.write(str(os.getpid()))
        return self
    def __exit__(self,*args):self.path.unlink(missing_ok=True)
