"""Corrected causal relay replay for reward learning on existing anatomical edges."""
import numpy as np
from PIL import Image
from pathlib import Path

from flydoom.bridge_training import Bridge
from flydoom.data import digest
from flydoom.retinal_policy import WINDOW_MS
from flydoom.stable_vision import StableGradedVision, evoked_release
from flydoom.visual_observer import sample_frame


class StableBridge(Bridge):
    def __init__(self, encoder, visual_ids):
        if not isinstance(encoder.model, StableGradedVision):
            raise ValueError('Reward replay requires corrected neural arithmetic')
        super().__init__(encoder, visual_ids)

    def replay(self, drive, resets, parameters):
        p = self.validate(parameters)
        if drive.shape[1:] != (self.encoder.steps, 2, len(self.nodes)) or len(resets) != len(drive):
            raise ValueError('Causal relay drive shape mismatch')
        if not len(drive) or not resets[0]:
            raise ValueError('Replay must begin at an episode reset')
        g = p[:, self.groups].astype(np.float32)
        decay = np.exp(-(WINDOW_MS/self.encoder.steps)/(self.tau*g[3])).astype(np.float32)
        state = np.zeros(len(self.nodes), np.float32)
        output = np.empty((len(drive), len(self.outputs)), np.float32)
        for i, frame in enumerate(drive):
            if resets[i]:
                state.fill(0)
            for incoming in frame:
                release = evoked_release(self.baseline, state)
                coupling = g[0]*incoming[0]+g[1]*incoming[1]+g[2]*(self.internal@release)
                state = decay*state+(1-decay)*coupling
            output[i] = state[self.outputs]
        if not np.isfinite(output).all():
            raise FloatingPointError('Non-finite corrected relay replay')
        return output


def record_frame(bridge, frame, destination):
    """Cache base-graph incoming drive while the full current graph advances."""
    e = bridge.encoder
    contrast = sample_frame(frame, e.uv, e.visible)
    for tick in range(e.steps):
        release = evoked_release(e.model.baseline, e.model.delta)
        destination[tick, 0] = bridge.positive@release[bridge.visual]
        destination[tick, 1] = bridge.negative@release[bridge.visual]
        e.advance(contrast, WINDOW_MS/e.steps)
    return e.model.delta[e.outputs].copy()


def verify_replay(bridge, drive, resets, parameters, reference, actor):
    replayed = bridge.replay(drive, resets, parameters)
    # The old 2e-7 absolute tolerance was larger than many entire neuron signals.
    np.testing.assert_allclose(replayed, reference, rtol=3e-5, atol=3e-12)
    actual = np.array([actor.features(r) for r in replayed])
    expected = np.array([actor.features(r) for r in reference])
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    return {'max_absolute_error': float(np.max(abs(replayed-reference))),
            'max_feature_error': float(np.max(abs(actual-expected))),
            'frames': len(reference), 'arithmetic': 'stable_evoked_release_v1'}


def cache_rehearsal(bridge, actor, rows, path):
    if not rows or any(r['split'] != 'train' for r in rows):
        raise ValueError('Rehearsal accepts TRAIN images only')
    bridge.apply(bridge.neutral())
    drive = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
        shape=(len(rows), bridge.encoder.steps, 2, len(bridge.nodes)))
    resets = np.array([r['step'] == 0 for r in rows])
    reference = []
    for i, row in enumerate(rows):
        if resets[i]:
            bridge.encoder.reset()
        image = Path(row['path'])
        if digest(image, 'sha256') != row['png_sha256']:
            raise ValueError('Rehearsal image identity changed')
        with Image.open(image) as im:
            frame = np.asarray(im.convert('RGB'))
        reference.append(record_frame(bridge, frame, drive[i]))
        if (i+1) % 32 == 0:
            print(f'Corrected teacher rehearsal {i+1}/{len(rows)}', flush=True)
    drive.flush()
    audit = verify_replay(bridge, drive, resets, bridge.neutral(), np.array(reference), actor)
    audit['cache_sha256'] = digest(Path(path), 'sha256')
    return drive, resets, audit
