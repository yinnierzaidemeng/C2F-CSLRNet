# SPDX-License-Identifier: MIT
# Project research components; see docs/IMPLEMENTATION.md.
"""Bounded gate definition and frozen FP32 block inference, without training orchestration."""
import numpy as np
import torch

def gate_class(torch):
    class Gate(torch.nn.Module):
        def __init__(self,b0=0.):
            super().__init__();self.register_buffer('anchor_logit',torch.tensor(b0,dtype=torch.float32))
            self.net=torch.nn.Sequential(torch.nn.Linear(12,8),torch.nn.Tanh(),torch.nn.Linear(8,1))
            torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
        def forward(self,x):return torch.sigmoid(self.anchor_logit+torch.tanh(self.net(x).reshape(-1)))
    return Gate


def predict(torch,model,x,mean,std,device='cpu'):
    out=[]
    with torch.inference_mode():
        for i in range(0,len(x),8192):out.append(model(torch.as_tensor(np.clip((x[i:i+8192]-mean)/std,-8,8),device=device)).cpu().numpy())
    return np.concatenate(out).astype(float)

BoundedHistoryGate = gate_class(torch)

def predict_gate(model, features, feature_mean, feature_std, device="cpu"):
    """Thin binding to the frozen predictor; model must already be in eval mode."""
    return predict(torch, model, features, feature_mean, feature_std, device)
