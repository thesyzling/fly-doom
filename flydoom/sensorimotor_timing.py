"""Conditional population timing and input-adaptation fit to pinned flash data."""
import numpy as np
from scipy.optimize import least_squares
from flydoom.retinal_policy import WINDOW_MS
from flydoom.timing_sources import load_split, REVISION
from flydoom.timing_training import observe, fit_scales, metrics

TYPES=('R1-6','L1','L2','C2','C3')


def flash(encoder, times, dt=1000/120/10):
    previous=encoder.model.decay.copy()
    encoder.model.decay=np.exp(-dt/encoder.model.tau)
    groups=[np.flatnonzero(encoder.types==name) for name in ('L1','L2')]
    timeline=np.arange(int(np.ceil(max(times)/dt))+1)*dt
    result=[]
    try:
        for sign in (-1,1):
            encoder.reset();values=np.zeros((len(timeline),2))
            for tick,t in enumerate(timeline[:-1]):
                pulse=sign*.1*np.clip((20-t)/dt,0,1)
                encoder.advance(np.full(len(encoder.inputs),pulse,np.float32),dt)
                values[tick+1]=[encoder.model.delta[g].mean() for g in groups]
            result.append(observe(timeline,values,times,3.).T)
        return np.stack(result,axis=1)
    finally:
        encoder.model.decay=previous;encoder.reset()


def fit(encoder, max_nfev=20):
    times,target=load_split('train');scale=np.maximum(np.sqrt(np.mean(target**2,axis=(1,2))),1e-8)
    start=np.array([float(np.median(encoder.model.tau[encoder.types==k])) for k in TYPES]+
                   [encoder.adaptation_gain,encoder.adaptation_tau_ms,encoder.dark_gain])
    initial_raw=flash(encoder,times);initial_gain,initial=fit_scales(initial_raw,target)
    calls=0
    def apply(x):encoder.set_timing(dict(zip(TYPES,x[:5])),x[5],x[6],x[7])
    def residual(x):
        nonlocal calls
        apply(x);_,prediction=fit_scales(flash(encoder,times),target);calls+=1
        if calls%20==0:print(f'Timing simulation {calls}',flush=True)
        return ((prediction-target)/scale[:,None,None]).ravel()
    # A nonzero starting adaptation avoids a zero finite-difference direction.
    trial=start.copy();trial[5]=max(.2,trial[5])
    lower=[2]*5+[0,5,.25];upper=[150]*5+[1.5,250,4]
    result=least_squares(residual,np.clip(trial,np.array(lower)+1e-4,np.array(upper)-1e-4),
                         bounds=(lower,upper),diff_step=.01,max_nfev=max_nfev,ftol=1e-5,xtol=1e-5)
    apply(result.x);gain,prediction=fit_scales(flash(encoder,times),target)
    accepted=np.mean(((prediction-target)/scale[:,None,None])**2)<=np.mean(((initial-target)/scale[:,None,None])**2)
    if not accepted:apply(start);gain=initial_gain;prediction=initial
    selected=result.x if accepted else start
    test_times,test_target=load_split('test')
    test_prediction=flash(encoder,test_times)*gain[:,None,None]
    test_scale=np.maximum(np.sqrt(np.mean(test_target**2,axis=(1,2))),1e-8)
    test=metrics(test_prediction,test_target,test_times,test_scale)
    fine=flash(encoder,times,dt=1000/120/20)*gain[:,None,None]
    return {'source_revision':REVISION,'trained_types':TYPES,'initial':metrics(initial,target,times,scale),
            'final':metrics(prediction,target,times,scale),'development_low_luminance':test,
            'tau_ms':dict(zip(TYPES,selected[:5].tolist())),'adaptation_gain':float(selected[5]),
            'adaptation_tau_ms':float(selected[6]),'dark_gain':float(selected[7]),
            'selected_by_training_only':bool(accepted),'optimizer_success':bool(result.success),'simulations':calls,
            'fine_dt_relative_rms':float(np.sqrt(np.mean((fine-prediction)**2))/max(np.sqrt(np.mean(prediction**2)),1e-12)),
            'waveform_threshold_passed':test['nrmse']<=.5,
            'scope':'Conditional fitted dynamics and engineering input adaptation; low luminance was used in earlier development, not an independent biological test. No intrinsic tau identification.',
            'time_ms':times.tolist(),'measured':target.tolist(),'predicted':prediction.tolist()}
