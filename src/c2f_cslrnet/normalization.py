# SPDX-License-Identifier: MIT
# Project research components; see docs/IMPLEMENTATION.md.
"""Exact-constant FP32 correction on fitting rows only."""
from dataclasses import dataclass
import numpy as np

VERSION = 'exact-fit-constant-fp32-v1-20260928'

@dataclass
class ExactConstantStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values):
        values = np.asarray(values)
        if values.dtype != np.float32 or values.ndim != 2 or len(values) == 0:
            raise ValueError('Require the original nonempty FP32 fit feature matrix')
        if not np.isfinite(values).all():
            raise ValueError('Require finite preprocessed fit values')
        # Keep the historical reduction, threshold, cast, and memory layout.
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        mean = mean.astype(np.float32)
        std = std.astype(np.float32)
        constant = np.all(values == values[0:1], axis=0)
        mean[constant] = values[0, constant]
        std[constant] = np.float32(1.0)
        return cls(mean=mean, std=std)

    def transform(self, values):
        return ((values - self.mean) / self.std).astype(np.float32)
