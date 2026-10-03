# SPDX-License-Identifier: MIT
# Project research components; see docs/IMPLEMENTATION.md.
"""Past-only branch histories and equal-file feature moments."""
import numpy as np
from scipy.signal import lfilter

SETTINGS = {'feature_std_floor': 0.001}
FEATURE_NAMES = ('prev_a', 'prev_b', 'mean17_a', 'mean17_b', 'std17_a', 'std17_b', 'mean65_a', 'mean65_b', 'std65_a', 'std65_b', 'disagreement17', 'disagreement65')

def rolling(x,w):
    return lfilter(np.ones(w),[1.],x)/np.minimum(np.arange(len(x))+1,w)


def history_features(a,b):
    a=np.log1p(a);b=np.log1p(b);out=[a,b]
    for w in [17,65]:
        am,bm=rolling(a,w),rolling(b,w)
        out += [am,bm,np.sqrt(np.maximum(rolling(a*a,w)-am*am,0)),np.sqrt(np.maximum(rolling(b*b,w)-bm*bm,0))]
    out += [rolling(np.abs(a-b),17),rolling(np.abs(a-b),65)]
    z=np.column_stack(out)
    return np.vstack([np.zeros((1,len(out))),z[:-1]]).astype(np.float32)


def moments(units):
    first=[];second=[]
    for u in units:
        x=u['arrays']['features'];i=np.linspace(0,len(x)-1,min(4096,len(x))).astype(int);xx=x[i].astype(float)
        first.append(xx.mean(axis=0));second.append((xx*xx).mean(axis=0))
    mean=np.mean(first,axis=0);std=np.sqrt(np.maximum(np.mean(second,axis=0)-mean*mean,0))
    return mean.astype(np.float32),np.maximum(std,SETTINGS['feature_std_floor']).astype(np.float32)
