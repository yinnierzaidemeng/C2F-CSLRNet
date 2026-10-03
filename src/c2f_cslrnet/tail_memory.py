# SPDX-License-Identifier: MIT
# Project research components; see docs/IMPLEMENTATION.md.
"""Frozen label-free tail-memory scoring; no global router or metrics."""
from __future__ import annotations
from collections import deque
from typing import Dict
import numpy as np
from scipy.signal import lfilter

def fit_exponential_tail_calibration(
    train_residuals: np.ndarray,
    tail_quantile: float = 0.98,
) -> Dict[str, np.ndarray]:
    """Fit a semi-parametric per-channel survival model on normal residuals only.

    The empirical survival function is retained up to a high training quantile.
    Above that threshold an exponential mean-excess continuation is used. This
    prevents all residuals beyond the largest training observation from receiving
    the same score, which is a common ranking failure of a finite empirical CDF.
    """
    values = np.asarray(train_residuals, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("train_residuals must be finite [samples, channels]")
    if not 0.5 <= float(tail_quantile) < 1.0:
        raise ValueError("tail_quantile must be in [0.5,1)")
    threshold = np.quantile(values, float(tail_quantile), axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale_floor = np.maximum((q75 - q25) * 0.05, 1e-8)
    mean_excess = np.empty(values.shape[1], dtype=np.float64)
    exceed_probability = np.empty(values.shape[1], dtype=np.float64)
    exceed_count = np.empty(values.shape[1], dtype=np.int64)
    for channel in range(values.shape[1]):
        excess = values[:, channel] - threshold[channel]
        excess = excess[excess > 0.0]
        exceed_count[channel] = len(excess)
        mean_excess[channel] = max(
            float(excess.mean()) if len(excess) else 0.0,
            float(scale_floor[channel]),
        )
        exceed_probability[channel] = (len(excess) + 1.0) / (len(values) + 1.0)
    return {
        "sorted_reference": np.sort(values, axis=0),
        "threshold": threshold,
        "mean_excess": mean_excess,
        "exceed_probability": exceed_probability,
        "exceed_count": exceed_count,
        "tail_quantile": np.asarray(float(tail_quantile), dtype=np.float64),
    }


def exponential_tail_evidence(
    residuals: np.ndarray,
    calibration: Dict[str, np.ndarray],
) -> np.ndarray:
    """Convert residuals to empirical evidence with severity-preserving tails."""
    values = np.asarray(residuals, dtype=np.float64)
    reference = np.asarray(calibration["sorted_reference"], dtype=np.float64)
    if values.ndim != 2 or reference.ndim != 2 or values.shape[1] != reference.shape[1]:
        raise ValueError("residuals and calibration have incompatible channel dimensions")
    if not np.isfinite(values).all():
        raise ValueError("residuals contains NaN or Inf")
    count = len(reference)
    evidence = np.empty_like(values)
    threshold = np.asarray(calibration["threshold"], dtype=np.float64)
    mean_excess = np.asarray(calibration["mean_excess"], dtype=np.float64)
    exceed_probability = np.asarray(calibration["exceed_probability"], dtype=np.float64)
    for channel in range(values.shape[1]):
        left = np.searchsorted(reference[:, channel], values[:, channel], side="left")
        survival = (count - left + 1.0) / (count + 1.0)
        channel_evidence = -np.log10(np.maximum(survival, 1e-300))
        mask = values[:, channel] > threshold[channel]
        if np.any(mask):
            continued = -np.log10(max(float(exceed_probability[channel]), 1e-300))
            continued += (
                values[mask, channel] - threshold[channel]
            ) / (max(float(mean_excess[channel]), 1e-12) * np.log(10.0))
            channel_evidence[mask] = continued
        evidence[:, channel] = np.minimum(channel_evidence, 300.0)
    return evidence


def causal_rolling_max(
    values: np.ndarray,
    window: int,
    prefix: np.ndarray = None,
) -> np.ndarray:
    """Trailing maximum with optional normal-training history and O(n) cost."""
    current = np.asarray(values, dtype=np.float64).reshape(-1)
    if int(window) <= 0:
        raise ValueError("window must be positive")
    if int(window) == 1:
        return current.copy()
    history = (
        np.asarray(prefix, dtype=np.float64).reshape(-1)[-(int(window) - 1) :]
        if prefix is not None
        else np.empty(0, dtype=np.float64)
    )
    combined = np.concatenate([history, current])
    output = np.empty_like(combined)
    candidates = deque()
    for index, value in enumerate(combined):
        while candidates and candidates[0] <= index - int(window):
            candidates.popleft()
        while candidates and combined[candidates[-1]] <= value:
            candidates.pop()
        candidates.append(index)
        output[index] = combined[candidates[0]]
    return output[len(history) :]


def topk(values,k):
    k=min(int(k),values.shape[1])
    if k<=0:return values.mean(axis=1)
    return np.partition(values,values.shape[1]-k,axis=1)[:,-k:].mean(axis=1)


def empirical(reference,values):
    sorted_=np.sort(reference,axis=0)
    n=len(sorted_); out=np.empty_like(values,dtype=float)
    for c in range(values.shape[1]):
        rank=np.searchsorted(sorted_[:,c],values[:,c],side='left')
        out[:,c]=-np.log10((n-rank+1)/(n+1))
    return out


def scalar_pair(train,test,q=.98,kind='exp'):
    if kind=='ecdf':return empirical(train[:,None],train[:,None])[:,0],empirical(train[:,None],test[:,None])[:,0]
    fitted=fit_exponential_tail_calibration(train[:,None],q)
    return exponential_tail_evidence(train[:,None],fitted)[:,0],exponential_tail_evidence(test[:,None],fitted)[:,0]


def temporal_pair(train,test,window,operator):
    if operator=='max':
        return causal_rolling_max(train,window),causal_rolling_max(test,window,prefix=train)
    if operator=='mean':
        def mean(x,prefix):
            history=prefix[-(window-1):] if window>1 else np.empty(0)
            combined=np.r_[history,x]
            cs=np.r_[0.,np.cumsum(combined)]; i=np.arange(len(combined)); starts=np.maximum(0,i-window+1)
            result=(cs[i+1]-cs[starts])/(i-starts+1)
            return result[len(history):]
        return mean(train,np.empty(0)),mean(test,train)
    if operator=='ewma':
        alpha=2/(window+1)
        tr,state=lfilter([alpha],[1,-(1-alpha)],train,zi=[(1-alpha)*train[0]])
        te,_=lfilter([alpha],[1,-(1-alpha)],test,zi=state)
        return tr,te
    raise ValueError(operator)


class ScoreBank:
    """Caches scalar aggregates only; full channel arrays are transient."""

    def __init__(self, train, test):
        self.train = np.asarray(train, dtype=float)
        self.test = np.asarray(test, dtype=float)
        assert self.train.ndim == 2 and self.test.shape[1] == self.train.shape[1]
        assert len(self.train) > 0 and np.isfinite(self.train).all() and np.isfinite(self.test).all()
        self.cache = {}

    def channel(self, kind, k, q=0.98, scale='mad'):
        key = (kind, k, q, scale)
        if key in self.cache:
            return self.cache[key]
        tr, te = (self.train, self.test)
        if kind == 'mse':
            return (np.square(tr).mean(axis=1), np.square(te).mean(axis=1))
        if kind in ['z', 'robust']:
            if kind == 'z':
                center = tr.mean(axis=0)
                spread = tr.std(axis=0)
            else:
                center = np.median(tr, axis=0)
                spread = 1.4826 * np.median(abs(tr - center), axis=0) if scale == 'mad' else (np.quantile(tr, 0.75, axis=0) - np.quantile(tr, 0.25, axis=0)) / 1.349
            spread = np.maximum(spread, 1e-06)
            tr_e = np.maximum((tr - center) / spread, 0)
            te_e = np.maximum((te - center) / spread, 0)
        elif kind == 'ecdf':
            tr_e = empirical(tr, tr)
            te_e = empirical(tr, te)
        elif kind == 'exp':
            fit = fit_exponential_tail_calibration(tr, q)
            tr_e = exponential_tail_evidence(tr, fit)
            te_e = exponential_tail_evidence(te, fit)
        else:
            raise ValueError(kind)
        for kk in [0, 1, 2, 4, 8]:
            self.cache[kind, kk, q, scale] = (topk(tr_e, kk), topk(te_e, kk))
        return self.cache[key]

    def point(self, q, k, kind='exp'):
        key = ('point', q, k, kind)
        if key not in self.cache:
            self.cache[key] = scalar_pair(*self.channel(kind, k, q), q, kind)
        return self.cache[key]

    def single(self, q, k, w, variant='full'):
        key = ('single', q, k, w, variant)
        if key in self.cache:
            return self.cache[key]
        kind = 'ecdf' if variant == 'ecdf_all' else 'exp'
        point = self.channel(kind, k, q) if variant == 'no_point_recalibration' else self.point(q, k, kind)
        maximum = temporal_pair(*point, w, 'max')
        m = maximum if variant == 'no_max_recalibration' else scalar_pair(*maximum, q, kind)
        if variant == 'point_only':
            result = point
        elif variant == 'recent_max_only':
            result = m
        else:
            result = tuple((np.sqrt(np.maximum(a * b, 0)) for a, b in zip(point, m)))
        self.cache[key] = result
        return result

    def dual(self):
        a = scalar_pair(*self.single(0.98, 2, 65), 0.98)
        b = scalar_pair(*self.single(0.9, 2, 129), 0.98)
        return tuple((0.1 * x + 0.9 * y for x, y in zip(a, b)))


def branch_arrays(ref,test):
    Bank,scalar=ScoreBank,scalar_pair;b=Bank(ref,test)
    ar,a=scalar(*b.single(.98,2,65),.98);br,bb=scalar(*b.single(.90,2,129),.98)
    expected_r,expected=b.dual()
    assert np.array_equal(.1*ar+.9*br,expected_r) and np.array_equal(.1*a+.9*bb,expected)
    return ar,br,a,bb
