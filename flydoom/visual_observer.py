"""Synchronized, stateful retinal observation alongside the existing Doom policy.

The graded circuit never supplies motor features here. Camera alignment and
contrast gain are explicit engineering assumptions, not calibrated fly optics.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from flydoom.data import digest
from flydoom.retinal_feedback import load_checkpoint
from flydoom.retinal_mapping import RESULT


def camera_projection(directions, width, height, hfov_degrees=90.):
    directions = np.asarray(directions, float)
    if directions.ndim != 2 or directions.shape[1] != 3 or not np.isfinite(directions).all():
        raise ValueError('Expected finite optical unit vectors')
    if width < 2 or height < 2 or not 1 < hfov_degrees < 179:
        raise ValueError('Invalid camera extent')
    forward = directions.mean(axis=0)
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError('Optical mean direction is undefined')
    forward /= norm
    up = np.array([0., 0., 1.])
    if abs(forward @ up) > .95:
        up = np.array([0., 1., 0.])
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    depth = directions @ forward
    tangent = np.tan(np.deg2rad(hfov_degrees)/2)
    x = (directions @ right) / np.maximum(depth, 1e-8) / tangent
    y = (directions @ up) / np.maximum(depth, 1e-8) / (tangent*height/width)
    visible = (depth > 0) & (abs(x) <= 1) & (abs(y) <= 1)
    uv = np.column_stack([(x+1)/2, (1-y)/2])
    # Invisible receptors receive neutral gray, never clamped border pixels.
    return uv, visible, {'forward': forward.tolist(), 'up': up.tolist(), 'right': right.tolist(),
                         'horizontal_fov_degrees': hfov_degrees,
                         'alignment': 'Mean mapped optical direction faces camera; world Z defines up; provisional'}


def sample_frame(frame, uv, visible):
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError('Expected RGB uint8 observation')
    gray = frame.astype(float).mean(axis=2) / 255
    height, width = gray.shape
    pixels = np.full(len(uv), .5)
    p = uv[visible] * [width-1, height-1]
    x, y = p[:, 0], p[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
    dx, dy = x-x0, y-y0
    pixels[visible] = ((1-dx)*(1-dy)*gray[y0,x0] + dx*(1-dy)*gray[y0,x1]
                       + (1-dx)*dy*gray[y1,x0] + dx*dy*gray[y1,x1])
    return .1*(2*pixels-1)


class VisualObserver:
    def __init__(self, folder, full_ids, full_rows):
        self.folder = Path(folder)
        self.ids, self.model, self.report = load_checkpoint(self.folder)
        full_lookup = {str(root): i for i, root in enumerate(full_ids)}
        self.rows = [full_rows[full_lookup[str(root)]] for root in self.ids]
        self.types = np.array([r['cell_type'] for r in self.rows])
        mapping = json.loads((RESULT/'report.json').read_text())
        if digest(RESULT/'receptors.json', 'sha256') != mapping['output_sha256']['receptors.json']:
            raise ValueError('Retinal mapping changed')
        receptors = json.loads((RESULT/'receptors.json').read_text())
        roots = np.array([int(r['root_id']) for r in receptors], dtype=np.uint64)
        self.inputs = np.searchsorted(self.ids, roots)
        if not np.array_equal(self.ids[self.inputs], roots):
            raise ValueError('Observer receptor identities mismatch')
        self.directions = np.array([r['optical_direction'] for r in receptors])
        self.groups = {k:np.flatnonzero(self.types==k) for k in ('R1-6','L1','L2','C2','C3','Mi1','T4a','T5a')}
        self.identity = {'enabled': True, 'mode': 'shadow / observation only',
                         'checkpoint': str(folder), 'checkpoint_sha256': digest(self.folder/'report.json','sha256'),
                         'observer_source_sha256': digest(Path(__file__), 'sha256'),
                         'retinal_mapping_sha256': digest(RESULT/'report.json', 'sha256'),
                         'neurons': len(self.ids), 'mapped_receptors': len(self.inputs),
                         'controls_action': False, 'laya_online': False,
                         'independent_physiology_validation': False,
                         'units': 'Dimensionless evoked graded state; not mV or spikes',
                         'frame_protocol': 'Hold each policy input frame for applied game tics / 35 Hz; no intermediate-frame sampling',
                         'camera_note': 'Provisional 90-degree pinhole stimulus; not a calibrated match to Doom optics'}
        self.reset()

    def reset(self):
        self.model.reset()
        self.elapsed_ms = 0.

    def observe(self, frame, game_tics, sequence, output):
        start = perf_counter()
        if type(game_tics) is not int or not 0 <= game_tics <= 4:
            raise ValueError('Expected zero to four game tics')
        uv, visible, camera = camera_projection(self.directions, frame.shape[1], frame.shape[0])
        contrast = sample_frame(frame, uv, visible)
        duration = game_tics*1000/35
        steps = int(np.ceil(duration/self.model.dt_ms))
        drive = np.zeros(len(self.ids), np.float32); drive[self.inputs] = contrast
        trace = []
        previous = self.model.delta.copy()
        if steps:
            trace.append({'time_ms': 0., 'values': {k:float(self.model.delta[g].mean()) for k,g in self.groups.items()}})
            # Exactly match simulated interval; never round cumulative game time.
            dt = duration/steps
            self.model.decay = np.exp(-dt/self.model.tau)
            stride = max(1, steps//20)
            for tick in range(steps):
                previous = self.model.delta.copy()
                self.model.step(drive)
                if (tick+1) % stride == 0 or tick == steps-1:
                    trace.append({'time_ms': (tick+1)*dt,
                                  'values': {k:float(self.model.delta[g].mean()) for k,g in self.groups.items()}})
        self.elapsed_ms += duration
        released = np.maximum(self.model.baseline+previous, 0)-np.maximum(self.model.baseline, 0)
        cells = []
        for kind, group in self.groups.items():
            i = int(group[np.argmax(abs(self.model.delta[group]))])
            start_i, stop_i = self.model.weights.indptr[i:i+2]
            sources = self.model.weights.indices[start_i:stop_i]
            weights = self.model.weights.data[start_i:stop_i]
            products = weights*released[sources]
            order = np.argsort(-abs(products), kind='stable')[:5]
            cells.append({'root_id':str(self.ids[i]), 'cell_type':kind, 'delta':float(self.model.delta[i]),
                          'tau_ms':float(self.model.tau[i]), 'input_drive':float(drive[i]),
                          'total_synaptic_drive':float(products.sum()),
                          'edges':[{'source':str(self.ids[sources[j]]), 'source_type':str(self.types[sources[j]]),
                                    'target':str(self.ids[i]), 'weight':float(weights[j]),
                                    'presynaptic_release':float(released[sources[j]]),
                                    'contribution':float(products[j])} for j in order]})
        output = Path(output)
        output.mkdir(exist_ok=True)
        path = output/f'{sequence:04d}.npz'
        if path.exists():
            raise ValueError('Observer recording already exists')
        np.savez_compressed(path, root_ids=self.ids, delta=self.model.delta, previous_delta=previous,
                            receptor_root_ids=self.ids[self.inputs], input_contrast=contrast,
                            visible=visible, frame=frame, duration_ms=duration)
        wall_ms = (perf_counter()-start)*1000
        return {'sequence':sequence, 'identity':self.identity, 'simulated_ms':self.elapsed_ms,
                'interval_ms':duration, 'observer_wall_ms':wall_ms,
                'observer_meets_interval_budget': bool(duration and wall_ms <= duration),
                'visible_receptors':int(visible.sum()), 'camera':camera, 'trace':trace, 'cells':cells,
                'input_samples':[{'uv':uv[i].tolist(), 'contrast':float(contrast[i])} for i in np.flatnonzero(visible)],
                'recording':str(path), 'recording_sha256':digest(path,'sha256'),
                'contribution_time': 'Presynaptic release immediately before final integration step'}
