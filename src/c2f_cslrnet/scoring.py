# SPDX-License-Identifier: MIT
"""Thin composition of frozen branches, history features and bounded gate.

This adapter has received static review only. It has not been run to establish
numerical equivalence of this extracted package with the archived experiments.
"""
import numpy as np

from .gate import predict_gate
from .history import history_features
from .tail_memory import branch_arrays


def score_residuals(reference_residuals, target_residuals, gate, feature_mean, feature_std, device="cpu"):
    """Score precomputed absolute standardized main-head residuals.

    reference_residuals and target_residuals have shape [time, channels]. The
    reference is the designated calibration prefix, with no label cleaning.
    The supplied gate, moments and anchor must already be fitted and frozen.
    Calling this is inference; no fitting or checkpoint loading is performed.
    """
    ar, br, a, b = branch_arrays(reference_residuals, target_residuals)
    features = history_features(np.r_[ar, a], np.r_[br, b])
    target_features = features[len(ar):]
    alpha = predict_gate(gate, target_features, feature_mean, feature_std, device)
    return {"scores": alpha * a + (1 - alpha) * b, "alpha": alpha,
            "a": a, "b": b, "reference_a": ar, "reference_b": br}
