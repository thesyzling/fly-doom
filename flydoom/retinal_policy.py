"""Closed-loop retinal policy with an explicit, engineered T4/T5 motor decoder.

All policy features are measured-graph cell states. Raw pixels, engine positions,
teacher outputs and rewards are excluded from policy inputs. Reward can update
the decoder through a bounded stochastic policy-gradient eligibility rule.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from flydoom.data import digest
from flydoom.retinal_feedback import OUTPUT as VISUAL_CHECKPOINT, load_checkpoint
from flydoom.retinal_mapping import RESULT
from flydoom.visual_observer import camera_projection, sample_frame

TYPES = tuple(f'T{k}{letter}' for k in (4, 5) for letter in 'abcd')
ACTIONS = ('WAIT', 'MOVE_LEFT', 'MOVE_RIGHT', 'ATTACK', 'MOVE_FORWARD', 'MOVE_BACKWARD')
WINDOW_MS = 4000/35


class RetinalEncoder:
    def __init__(self, checkpoint=VISUAL_CHECKPOINT):
        self.ids, self.model, self.report = load_checkpoint(checkpoint)
        import csv
        with Path('data/processed/fafb783/neuron_annotations.tsv').open(encoding='utf-8', newline='') as stream:
            rows = {r['root_id']:r for r in csv.DictReader(stream, delimiter='\t')}
        self.types = np.array([rows[str(root)]['cell_type'] for root in self.ids])
        self.outputs = np.flatnonzero(np.isin(self.types, TYPES))
        mapping = json.loads((RESULT/'report.json').read_text())
        if digest(RESULT/'receptors.json','sha256') != mapping['output_sha256']['receptors.json']:
            raise ValueError('Receptor mapping identity changed')
        receptors = json.loads((RESULT/'receptors.json').read_text())
        roots = np.array([int(r['root_id']) for r in receptors], np.uint64)
        self.inputs = np.searchsorted(self.ids, roots)
        np.testing.assert_array_equal(self.ids[self.inputs], roots)
        self.directions = np.array([r['optical_direction'] for r in receptors])
        self.uv, self.visible, self.camera = camera_projection(self.directions, 320, 240)
        self.steps = int(np.ceil(WINDOW_MS/self.model.dt_ms))
        self.model.decay = np.exp(-(WINDOW_MS/self.steps)/self.model.tau)
        self.base_weights = self.model.weights.data.copy()
        self.edge_targets = np.repeat(np.arange(len(self.ids)), np.diff(self.model.weights.indptr))
        self.gain_masks = [self.types[self.edge_targets] == kind for kind in TYPES]
        self.gains = np.ones(8)
        self.reset()

    def reset(self):
        self.model.reset()
        self.decisions = 0

    def set_gains(self, gains):
        gains = np.asarray(gains, float)
        if gains.shape != (8,) or not np.isfinite(gains).all() or np.any(gains < .95) or np.any(gains > 1.05):
            raise ValueError('Visual task gains must stay in [0.95, 1.05]')
        self.model.weights.data[:] = self.base_weights
        for gain, mask in zip(gains, self.gain_masks):
            self.model.weights.data[mask] *= gain
        # Tonic release remains the physiology parent's baseline. Task gains act
        # on evoked transmission, an explicit engineering intervention.
        self.gains = gains.copy()

    def encode(self, frame, *, connected=True, telemetry=False):
        start = perf_counter()
        if np.asarray(frame).shape != (240,320,3):
            raise ValueError('Require native RGB 320 x 240 observations')
        contrast = sample_frame(frame, self.uv, self.visible)
        drive = np.zeros(len(self.ids), np.float32); drive[self.inputs] = contrast
        trace = []
        groups = {kind:np.flatnonzero(self.types==kind) for kind in ('R1-6','L1','L2','Mi1','C2','C3','T4a','T5a')} if telemetry else {}
        for tick in range(self.steps):
            self.model.step(drive, connected=connected)
            if telemetry and ((tick+1)%10==0 or tick==self.steps-1):
                trace.append({'ms':(tick+1)*WINDOW_MS/self.steps,
                              'values':{k:float(self.model.delta[g].mean()) for k,g in groups.items()}})
        self.decisions += 1
        return self.model.delta[self.outputs].copy(), {'encoding_ms':(perf_counter()-start)*1000,
                'neural_processing_window_ms':WINDOW_MS, 'neural_clock_ms':self.decisions*WINDOW_MS,
                'trace':trace, 'visible_receptors':int(self.visible.sum()), 'connected':connected}


def softmax(logits):
    value = np.exp(logits-np.max(logits)); return value/value.sum()


class RetinalActor:
    def __init__(self, indices, root_ids, mean, scale, weights=None):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.root_ids = np.asarray(root_ids, dtype=np.uint64)
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        size = len(self.indices)
        if size == 0 or len(self.root_ids)!=size or self.mean.shape!=(size,) or self.scale.shape!=(size,) or np.any(self.scale<=0):
            raise ValueError('Invalid neural decoder schema')
        self.weights = np.zeros((6,size+1)) if weights is None else np.asarray(weights,float).copy()
        if self.weights.shape!=(6,size+1) or not np.isfinite(self.weights).all():
            raise ValueError('Invalid decoder weights')
        self.reset()

    def reset(self):
        self.eligibility = np.zeros_like(self.weights)
        self.updates = 0

    def features(self, raw):
        raw = np.asarray(raw)
        if raw.ndim!=1 or not np.isfinite(raw).all():
            raise ValueError('Invalid neural state vector')
        return np.r_[np.tanh((raw[self.indices]-self.mean)/self.scale), 1.]

    def decide(self, raw, rng=None):
        features = self.features(raw)
        probabilities = softmax(self.weights @ features)
        action = int(np.argmax(probabilities)) if rng is None else int(rng.choice(6,p=probabilities))
        return action, probabilities, features

    def feedback(self, features, action, probabilities, reward, *, learning_rate=.002):
        """Reward-modulated score-function trace; no biological dopamine claim.

Only stochastic on-policy actions may call this update. Rewards are actual
engine rewards, scaled by 100. Eligibility carries delayed outcome credit.
"""
        if not 0 <= action < 6 or not np.isfinite(reward) or not 0 < learning_rate <= .01:
            raise ValueError('Invalid reward update')
        expected = softmax(self.weights @ features)
        if not np.allclose(expected, probabilities, atol=1e-8):
            raise ValueError('Stale or off-policy probabilities')
        score = np.eye(6)[action]-probabilities
        self.eligibility = .97*.9*self.eligibility + np.outer(score, features)
        gradient = float(np.clip(reward/100,-1,1))*self.eligibility
        norm = np.linalg.norm(gradient)
        if norm > 1:
            gradient /= norm
        change = learning_rate*gradient
        self.weights += change
        self.updates += 1
        after = softmax(self.weights @ features)
        return {'reward':float(reward), 'reward_scaled':float(np.clip(reward/100,-1,1)),
                'weight_delta_l2':float(np.linalg.norm(change)), 'updates':self.updates,
                'action_probability_before':float(probabilities[action]),
                'action_probability_after':float(after[action]),
                'scope':'Engineered decoder updated; biological gains are fixed during this live episode'}

    def contributions(self, features, action, limit=12):
        terms = self.weights[action,:-1]*features[:-1]
        order = np.argsort(-abs(terms),kind='stable')[:limit]
        return {'bias':float(self.weights[action,-1]), 'logit':float(self.weights[action]@features),
                'all_neural_terms':float(terms.sum()), 'other_terms':float(terms.sum()-terms[order].sum()),
                'cells':[{'root_id':str(self.root_ids[j]), 'weight':float(self.weights[action,j]),
                          'activation':float(features[j]), 'contribution':float(terms[j])} for j in order]}

    def save(self, path):
        np.savez_compressed(path, indices=self.indices, root_ids=self.root_ids, mean=self.mean,
                            scale=self.scale, weights=self.weights)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if data['root_ids'].dtype!=np.uint64:
                raise ValueError('Decoder IDs must remain exact uint64 values')
            return cls(data['indices'],data['root_ids'],data['mean'],data['scale'],data['weights'])


def load_policy(folder):
    folder = Path(folder)
    report = json.loads((folder/'report.json').read_text())
    if report.get('status')!='completed' or report.get('schema')!='retinal_closed_loop_v1':
        raise ValueError('Require a completed retinal control experiment')
    for name, sha in report['output_sha256'].items():
        if digest(folder/name,'sha256')!=sha:
            raise ValueError('Retinal policy identity mismatch: '+name)
    for name, sha in report['source_sha256'].items():
        if digest(Path(__file__).parent/name,'sha256')!=sha:
            raise ValueError('Retinal policy implementation changed: '+name)
    encoder = RetinalEncoder()
    if digest(VISUAL_CHECKPOINT/'report.json','sha256')!=report['visual_parent_sha256']:
        raise ValueError('Visual parent changed')
    actor = RetinalActor.load(folder/'actor.npz')
    np.testing.assert_array_equal(encoder.ids[encoder.outputs[actor.indices]],actor.root_ids)
    encoder.set_gains(report['selected_gains'])
    return encoder, actor, report
