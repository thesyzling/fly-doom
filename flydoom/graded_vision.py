"""Experimental graded-release dynamics on a measured visual subgraph.

States and release are dimensionless, not measured millivolts or firing rates.
The numerical contraction bound is an engineering choice, not a fit to biology.
"""

import numpy as np
from scipy import sparse


def type_scaled_weights(counts, signs, types, bound=.6):
    """Retain contact ratios and signs with one initial gain per target type."""
    counts = sparse.csr_matrix(counts, dtype=np.float32)
    signs, types = np.asarray(signs), np.asarray(types)
    n = counts.shape[0]
    if counts.shape != (n, n) or signs.shape != (n,) or types.shape != (n,):
        raise ValueError("Inconsistent graph, sign or cell-type dimensions")
    if not 0 < bound < 1 or not np.isfinite(counts.data).all() or np.any(counts.data < 0):
        raise ValueError("Expected nonnegative counts and a contraction bound in (0, 1)")
    if not np.isin(signs, [-1, 0, 1]).all():
        raise ValueError("Invalid provisional transmitter signs")
    weights = counts.copy()
    weights.data *= signs[weights.indices]
    mass = np.asarray(abs(weights).sum(axis=1)).ravel()
    gains = {kind: float(bound / max(1., mass[types == kind].max())) for kind in sorted(set(types))}
    row_gain = np.array([gains[k] for k in types], dtype=np.float32)
    weights.data *= np.repeat(row_gain, np.diff(weights.indptr))
    return weights, gains


class GradedVision:
    """Leaky voltage with rectified release and a converged tonic baseline.

dx/dt = (-x + W * (relu(v0+x)-relu(v0)) + drive) / tau.
v0 is the fixed point of v0 = bias + W*relu(v0). Connected=False
removes only evoked transmission while preserving the same tonic baseline.
"""

    def __init__(self, weights, tau_ms, *, dt_ms=1., bias=1.):
        self.weights = sparse.csr_matrix(weights, dtype=np.float32)
        n = self.weights.shape[0]
        self.tau = np.asarray(tau_ms, dtype=np.float32)
        if n == 0 or self.weights.shape != (n, n) or self.tau.shape != (n,):
            raise ValueError("Expected a nonempty square graph and one time constant per cell")
        if not np.isfinite(self.weights.data).all() or not np.isfinite(self.tau).all() or np.any(self.tau <= 0):
            raise ValueError("Dynamics must be finite with positive time constants")
        if not np.isfinite(dt_ms) or dt_ms <= 0 or not np.isfinite(bias):
            raise ValueError("Invalid time step or tonic bias")
        self.row_bound = float(np.asarray(abs(self.weights).sum(axis=1)).max())
        if self.row_bound >= 1:
            raise ValueError("Pilot requires a strict contraction bound")
        self.dt_ms = float(dt_ms)
        self.decay = np.exp(-dt_ms / self.tau)
        baseline = np.full(n, bias, dtype=np.float32)
        for _ in range(1000):
            updated = bias + self.weights @ np.maximum(baseline, 0)
            if np.max(np.abs(updated - baseline)) < 1e-6:
                baseline = updated
                break
            baseline = updated
        else:
            raise ValueError("Tonic baseline did not converge")
        self.baseline = baseline
        self.baseline_residual = float(np.max(np.abs(baseline - bias - self.weights @ np.maximum(baseline, 0))))
        self.delta = np.zeros(n, dtype=np.float32)
        self.blocked_release = np.zeros(n, dtype=bool)

    def reset(self):
        self.delta.fill(0)

    def step(self, drive, *, connected=True):
        drive = np.asarray(drive, dtype=np.float32)
        if drive.shape != self.delta.shape or not np.isfinite(drive).all():
            raise ValueError("Expected one finite drive per cell")
        released = np.maximum(self.baseline + self.delta, 0) - np.maximum(self.baseline, 0)
        released[self.blocked_release] = 0
        coupling = self.weights @ released if connected else 0
        self.delta[:] = self.decay * self.delta + (1 - self.decay) * (drive + coupling)
        if not np.isfinite(self.delta).all():
            raise FloatingPointError("Non-finite graded activity")
        return self.delta
