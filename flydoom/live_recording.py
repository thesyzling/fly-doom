"""Lossless live records and genuine engine frames without changing actions."""

from hashlib import sha256
from contextlib import contextmanager
import io
import os
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from flydoom.movement_core import apply_action


@contextmanager
def responsive_clock(timer=None):
    """Scope Windows' timer request to engine IPC, always releasing it."""
    if timer is None and os.name=='nt':
        import ctypes
        timer=ctypes.windll.winmm
    active=timer is not None and timer.timeBeginPeriod(1)==0
    try:yield
    finally:
        if active:timer.timeEndPeriod(1)


def write_arrays(path, **arrays):
    """Create a standard NPZ in memory; hash its exact bytes before one write."""
    buffer=io.BytesIO()
    with ZipFile(buffer,'w',compression=ZIP_DEFLATED,compresslevel=1) as archive:
        for name,array in arrays.items():
            with archive.open(name+'.npy','w',force_zip64=True) as stream:
                np.lib.format.write_array(stream,np.asanyarray(array),allow_pickle=False)
    data=buffer.getvalue()
    checksum=sha256(data).hexdigest()
    path.write_bytes(data)
    return checksum


def observed_action(game, action):
    """Observe each genuine engine tic while using the pinned action function."""
    frames=[]
    class Observer:
        def __getattr__(self,name):return getattr(game,name)
        def make_action(self,buttons,tics):
            result=game.make_action(buttons,tics)
            state=game.get_state()
            if state is not None:frames.append(state.screen_buffer.copy())
            return result
    with responsive_clock():effect=apply_action(Observer(),action)
    return effect,frames
