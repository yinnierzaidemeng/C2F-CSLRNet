# SPDX-License-Identifier: MIT
# Project research components; see docs/IMPLEMENTATION.md.
"""Training-prefix spectral reliability; caller supplies the permitted endpoint."""
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

def _estimate_channel_spectral_reliability(source_file: Path, train_end: int) -> List[float]:
    """仅用训练前缀估计各通道稳定周期的可靠度。

    该先验不读取测试段，更不读取标签；它只用于将无明显周期的通道的频域支路降权。
    可靠度综合了非直流谱峰相对中位谱能量的显著性，以及该峰在总频带能量中的集中度。
    """
    table = pd.read_csv(source_file)
    values = table.drop(columns=["Label"], errors="ignore").select_dtypes(include=[np.number]).iloc[:train_end]
    values = values.ffill().fillna(0.0).to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        return []
    segment = values[-min(len(values), 4096) :]
    center = segment.mean(axis=0, keepdims=True)
    scale = segment.std(axis=0, keepdims=True) + 1e-8
    normalized = (segment - center) / scale
    # 在多个独立训练块上寻找同一个频带，避免将一次性趋势/工况切换误判为周期。
    block_len = 256 if len(normalized) >= 256 else len(normalized)
    block_count = max(1, min(16, len(normalized) // block_len))
    normalized = normalized[-block_count * block_len :].reshape(block_count, block_len, -1)
    ramp = np.linspace(0.0, 1.0, block_len, dtype=np.float64)[None, :, None]
    normalized = normalized - (
        normalized[:, 0:1, :] + ramp * (normalized[:, -1:, :] - normalized[:, 0:1, :])
    )
    power = np.abs(np.fft.rfft(normalized, axis=1)) ** 2
    frequency = np.fft.rfftfreq(block_len)
    # 周期 4 至 64 个采样点与 W=64 内的块间频域建模相匹配；零频率被明确排除。
    valid = (frequency >= 1.0 / 64.0) & (frequency <= 1.0 / 4.0)
    if not np.any(valid):
        return [0.0] * values.shape[1]
    band = power[:, valid, :]
    mean_band = band.mean(axis=0)
    reliability = np.zeros(values.shape[1], dtype=np.float64)
    for channel in range(values.shape[1]):
        spectrum = mean_band[:, channel]
        if spectrum.sum() <= 1e-10:
            continue
        peak_index = int(np.argmax(spectrum))
        peak = float(spectrum[peak_index])
        median = float(np.median(spectrum)) + 1e-12
        concentration = peak / (float(spectrum.sum()) + 1e-12)
        block_peak = band[:, peak_index, channel]
        local_max = band[:, :, channel].argmax(axis=1)
        consistency = float(np.mean(np.abs(local_max - peak_index) <= 1))
        stability = 1.0 / (1.0 + float(block_peak.std() / (block_peak.mean() + 1e-12)))
        prominence = (np.log1p(peak / median) - np.log(4.0)) / (np.log(40.0) - np.log(4.0))
        reliability[channel] = (
            np.clip(prominence, 0.0, 1.0)
            * np.clip(concentration * 10.0, 0.0, 1.0)
            * consistency
            * stability
        )
    return reliability.astype(np.float32).tolist()

estimate_channel_spectral_reliability = _estimate_channel_spectral_reliability
