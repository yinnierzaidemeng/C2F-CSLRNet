# SPDX-License-Identifier: MIT
# Project research components; see docs/IMPLEMENTATION.md.
"""Frozen component bodies; unused global-routing architectures are excluded."""
import math
import torch
import torch.nn.functional as F
from torch import nn


def stable_softmax(logits: torch.Tensor, dim: int) -> torch.Tensor:
    """以 FP32 执行 softmax，兼顾 AMP 速度与数值稳定性。

    注意力分数出现极端值或非有限值会在这里被截断，避免一次异常 batch 直接让训练
    产生 NaN；输出再转换回原精度以节省显存。
    """
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    logits = logits.clamp(min=-1e4, max=1e4)
    return torch.softmax(logits.float(), dim=dim).to(dtype=logits.dtype)


def patch_count(length: int, patch_len: int, stride: int) -> int:
    """计算末尾复制填充后，一个序列可得到的 patch 数量。"""
    if length <= patch_len:
        return 1
    return int(math.ceil((length - patch_len) / stride)) + 1


def unfold_patches(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """把 ``[批, 长度]`` 序列切成 ``[批, patch数, patch长度]``。

    末尾不足一个 patch 时重复最后一个时间点，而不是丢弃窗口末端信息。该行为也让
    配置中的窗口长度不必恰好被 patch 长度和步长整除。
    """
    length = x.size(1)
    if length < patch_len:
        pad_len = patch_len - length
    else:
        remainder = (length - patch_len) % stride
        pad_len = 0 if remainder == 0 else stride - remainder
    if pad_len > 0:
        pad_value = x[:, -1:].expand(-1, pad_len)
        x = torch.cat([x, pad_value], dim=1)
    return x.unfold(dimension=1, size=patch_len, step=stride)


def fold_patches_average(
    patches: torch.Tensor, original_length: int, patch_len: int, stride: int
) -> torch.Tensor:
    """将重叠 patch 平均折叠回 ``[批, 原始长度]``。

    频域支路需要把经频谱门控后的各个时间块还原为周期项，才能从原信号中扣除它并
    将真正的残差交给 TCN。末尾复制填充的规则与 :func:`unfold_patches` 完全一致。
    """
    batch, patch_num, _ = patches.shape
    padded_length = (patch_num - 1) * stride + patch_len
    values = patches.new_zeros(batch, padded_length)
    counts = patches.new_zeros(batch, padded_length)
    for index in range(patch_num):
        start = index * stride
        values[:, start : start + patch_len] += patches[:, index, :]
        counts[:, start : start + patch_len] += 1
    return (values / counts.clamp_min(1.0))[:, :original_length]


def causal_moving_average(x: torch.Tensor, window: int) -> torch.Tensor:
    """对 ``[批, 时间]`` 进行左侧复制填充的因果移动平均。"""
    if window <= 1:
        return x
    values = F.pad(x.unsqueeze(1), (window - 1, 0), mode="replicate")
    return F.avg_pool1d(values, kernel_size=window, stride=1).squeeze(1)


class ProbSparseSelfAttention(nn.Module):
    """Informer 风格的 ProbSparse 自注意力。

    它先对每个 query 从抽取的 key 中估计稀疏度，只为最重要的少量 query 计算完整
    注意力；未选中的 query 使用 value 的全局均值上下文，从而减少长序列开销。训练时
    使用随机抽样，评估时使用确定性的均匀抽样，保证同一输入重复推理得到相同结果。
    """

    def __init__(self, d_model: int, n_heads: int, factor: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.factor = max(1, int(factor))
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对 ``[批, token数, d_model]`` token 序列提取稀疏全局依赖。"""
        batch, length, _ = x.shape
        if length == 1:
            return self.output(self.value(x))

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        query = split_heads(self.query(x))
        key = split_heads(self.key(x))
        value = split_heads(self.value(x))
        sample_size = min(length, max(1, self.factor * int(math.ceil(math.log(length)))))
        top_queries = sample_size

        # W=64 的各分支通常只有 3--7 个 token，此时 ProbSparse 的采样数会等于
        # token 总数，并且 top-k 也会选中全部 query。原实现仍执行高级索引、
        # topk、gather 与 scatter，数学上已经是完整注意力，却会产生大量小 CUDA
        # kernel 和 CPU launch 开销。评估阶段直接计算等价的完整注意力；训练阶段
        # 继续保留随机采样路径，避免改变既有训练语义。
        if not self.training and sample_size == length:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            weights = self.dropout(stable_softmax(scores, dim=-1))
            context = torch.matmul(weights, value)
            context = context.transpose(1, 2).contiguous().view(batch, length, -1)
            return self.output(context)

        # 训练阶段随机估计 query 稀疏度；评估阶段使用循环均匀采样，避免 eval() 下
        # 仍因 torch.randint 而产生不稳定的预测/异常分数。
        if self.training:
            sampled_key_indices = torch.randint(length, (length, sample_size), device=x.device)
        else:
            query_offsets = torch.arange(length, device=x.device).unsqueeze(1)
            sample_offsets = torch.div(
                torch.arange(sample_size, device=x.device) * length,
                sample_size,
                rounding_mode="floor",
            ).unsqueeze(0)
            sampled_key_indices = (query_offsets + sample_offsets).remainder(length)
        sampled_keys = key[:, :, sampled_key_indices, :]
        sampled_scores = (query.unsqueeze(-2) * sampled_keys).sum(dim=-1)
        sparsity = sampled_scores.max(dim=-1).values - sampled_scores.mean(dim=-1)
        top_indices = sparsity.topk(top_queries, dim=-1, sorted=False).indices

        gather_indices = top_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        top_query = query.gather(dim=2, index=gather_indices)
        scores = torch.matmul(top_query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.dropout(stable_softmax(scores, dim=-1))
        updates = torch.matmul(weights, value)

        # 未入选 query 采用全局 value 均值上下文，这是 Informer 的近似核心。
        context = value.mean(dim=2, keepdim=True).expand(-1, -1, length, -1).clone()
        context.scatter_(dim=2, index=gather_indices, src=updates)
        context = context.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.output(context)


class MultiheadSelfAttention(nn.Module):
    """PyTorch 完整多头自注意力，用于与 ProbSparse 的加速实验对照。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # 通道独立建模会把“样本数 × 传感器数”展平为 batch；按块调用以免超出 CUDA 网格限制。
        self.max_attention_batch = 8192

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完整计算所有 token 对之间的注意力，必要时沿 batch 维分块。"""
        if x.size(0) <= self.max_attention_batch:
            output, _ = self.attention(x, x, x, need_weights=False)
            return output

        outputs = []
        for chunk in x.split(self.max_attention_batch, dim=0):
            output, _ = self.attention(chunk, chunk, chunk, need_weights=False)
            outputs.append(output)
        return torch.cat(outputs, dim=0)


class InformerEncoderLayer(nn.Module):
    """包含注意力残差块和前馈残差块的单层编码器。"""

    def __init__(self, d_model: int, n_heads: int, factor: int, dropout: float, attention_type: str) -> None:
        super().__init__()
        if attention_type == "probsparse":
            self.attn = ProbSparseSelfAttention(d_model, n_heads, factor, dropout)
        elif attention_type == "multihead":
            self.attn = MultiheadSelfAttention(d_model, n_heads, dropout)
        else:
            raise ValueError("attention_type must be 'probsparse' or 'multihead'")
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """依次完成注意力融合、归一化、前馈映射和第二次残差归一化。"""
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class InformerEncoder(nn.Module):
    """堆叠多层 Informer 编码器以逐步提炼长程 token 表示。"""

    def __init__(
        self, d_model: int, n_heads: int, layers: int, factor: int, dropout: float, attention_type: str
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [InformerEncoderLayer(d_model, n_heads, factor, dropout, attention_type) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """让 token 依次通过所有编码层。"""
        for layer in self.layers:
            x = layer(x)
        return x


class Chomp1d(nn.Module):
    """移除因果卷积右侧填充，确保每个位置不窥视未来时间点。"""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """裁剪卷积结果末尾的填充长度。"""
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """两层膨胀因果卷积构成的 TCN 残差块。"""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对 ``[批, token数, d_model]`` 做短期局部时序建模。"""
        # Conv1d 以 [批, 通道, 长度] 接收输入，因此进入和离开卷积时各转置一次。
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y)


class TCNEncoder(nn.Module):
    """按 1、2、4... 膨胀率堆叠 TCN 块，扩大短期分支的感受野。"""

    def __init__(self, d_model: int, levels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TemporalBlock(d_model, kernel_size, dilation=2**i, dropout=dropout)
                for i in range(levels)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """依次应用所有膨胀卷积残差块。"""
        for block in self.blocks:
            x = block(x)
        return x


class CausalTrendPeriodicDecomposer(nn.Module):
    """用可学习尺度选择提取因果趋势，周期项由后续频域支路重建。

    输入的每个传感器都独立计算三种因果平滑结果；一个共享的小型门控网络根据局部
    波动强度和一阶差分强度选择合适的趋势尺度。这样不会把测试窗口终点以后的信息
    反向泄漏到当前预测中，也不会为不同通道数引入额外参数。
    """

    def __init__(self, windows=(7, 15, 31)) -> None:
        super().__init__()
        self.windows = tuple(int(window) for window in windows)
        self.scale_gate = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, len(self.windows)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else x.new_zeros(x.size(0), 1)
        stats = torch.stack(
            [x.var(dim=1, unbiased=False), diff.abs().mean(dim=1)], dim=-1
        )
        weights = stable_softmax(self.scale_gate(stats), dim=-1)
        smooth = torch.stack(
            [causal_moving_average(x, window) for window in self.windows], dim=-1
        )
        return (smooth * weights.unsqueeze(1)).sum(dim=-1)


class AttentionPatchBranch(nn.Module):
    """趋势块编码器，可选择逐变量或尺度内全局的块注意力。"""

    def __init__(
        self,
        window_size: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        informer_layers: int,
        factor: int,
        dropout: float,
        attention_type: str,
        use_local_attention: bool = True,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_count(window_size, patch_len, stride), d_model))
        self.informer = (
            InformerEncoder(d_model, n_heads, informer_layers, factor, dropout, attention_type)
            if use_local_attention
            else None
        )
        self.summary_norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, return_tokens: bool = False):
        patches = unfold_patches(x, self.patch_len, self.stride)
        tokens = self.patch_proj(patches) + self.pos_embed[:, : patches.size(1), :]
        encoded = self.informer(tokens) if self.informer is not None else tokens
        summary = 0.5 * encoded[:, -1, :] + 0.5 * encoded.mean(dim=1)
        summary = self.summary_norm(summary)
        return (summary, encoded) if return_tokens else summary


class ResidualTCNPatchBranch(nn.Module):
    """只用 TCN 处理趋势和周期都无法解释的短时残差。"""

    def __init__(
        self,
        window_size: int,
        patch_len: int,
        stride: int,
        d_model: int,
        tcn_levels: int,
        conv_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_count(window_size, patch_len, stride), d_model))
        self.tcn = TCNEncoder(d_model, tcn_levels, conv_kernel_size, dropout)
        self.summary_norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, return_tokens: bool = False):
        patches = unfold_patches(x, self.patch_len, self.stride)
        tokens = self.patch_proj(patches) + self.pos_embed[:, : patches.size(1), :]
        encoded = self.tcn(tokens)
        summary = 0.5 * encoded[:, -1, :] + 0.5 * encoded.mean(dim=1)
        summary = self.summary_norm(summary)
        return (summary, encoded) if return_tokens else summary


class SpectralPatchBranch(nn.Module):
    """对去趋势块进行频谱软门控，并在块间建模频谱的时间演化。"""

    def __init__(
        self,
        window_size: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        informer_layers: int,
        factor: int,
        dropout: float,
        attention_type: str,
        use_local_attention: bool = True,
        use_dynamic_reliability: bool = False,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.freq_bins = patch_len // 2 + 1
        self.use_dynamic_reliability = bool(use_dynamic_reliability)
        self.spectral_gate = nn.Sequential(
            nn.Linear(self.freq_bins, self.freq_bins),
            nn.Sigmoid(),
        )
        # 使用幅值、单位实部、单位虚部表达频谱。避免 atan2(0, 0) 在恒定通道的
        # 零频谱 bin 上产生未定义梯度，从而污染整次训练的参数。
        self.patch_proj = nn.Linear(self.freq_bins * 3, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_count(window_size, patch_len, stride), d_model))
        self.informer = (
            InformerEncoder(d_model, n_heads, informer_layers, factor, dropout, attention_type)
            if use_local_attention
            else None
        )
        self.summary_norm = nn.LayerNorm(d_model)
        self.register_buffer("non_dc_mask", torch.cat([torch.zeros(1), torch.ones(self.freq_bins - 1)]))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        reliability: torch.Tensor = None,
        return_tokens: bool = False,
        return_dynamic_reliability: bool = False,
    ):
        """返回周期摘要以及重建后的周期项。

        ``reliability`` 是每个通道由训练前缀频谱计算的 ``[批]`` 静态可靠度。其为
        0 时本支路对融合和周期重建均不施加影响，避免为无周期工况制造伪周期特征。
        """
        patches = unfold_patches(x, self.patch_len, self.stride)
        # FFT 在 AMP 下统一以 FP32 完成，随后投影层自行遵循外层精度设置。
        spectrum = torch.fft.rfft(patches.float(), dim=-1, norm="ortho")
        raw_magnitude = spectrum.abs()
        magnitude = torch.log1p(raw_magnitude)
        non_dc_mask = self.non_dc_mask.view(1, 1, -1)
        gate = self.spectral_gate(magnitude) * non_dc_mask
        dynamic_reliability = None
        if self.use_dynamic_reliability:
            # 每个输入窗口、每个频域块单独估计“窄带周期性”。峰值集中且谱熵低时
            # 可信度高；宽带或近似白噪声时仅保留较弱的频域通路。该统计只来自当前
            # 预测窗口，因此不会将目标时刻或测试标签泄漏进模型。
            non_dc_magnitude = raw_magnitude * non_dc_mask
            non_dc_sum = non_dc_magnitude.sum(dim=-1).clamp_min(1e-6)
            peak_ratio = non_dc_magnitude.amax(dim=-1) / non_dc_sum
            bins = max(1, self.freq_bins - 1)
            if bins == 1:
                dynamic_reliability = torch.full_like(peak_ratio, 0.35)
            else:
                uniform_ratio = 1.0 / float(bins)
                concentration = ((peak_ratio - uniform_ratio) / (1.0 - uniform_ratio + 1e-6)).clamp(0.0, 1.0)
                probabilities = non_dc_magnitude / non_dc_sum.unsqueeze(-1)
                spectral_entropy = -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
                spectral_entropy = spectral_entropy / math.log(float(bins))
                sharpness = (1.0 - spectral_entropy).clamp(0.0, 1.0)
                dynamic_reliability = (0.35 + 0.65 * (0.5 * concentration + 0.5 * sharpness)).clamp(0.35, 1.0)
            gate = gate * dynamic_reliability.unsqueeze(-1)
        if reliability is not None:
            gate = gate * reliability.reshape(-1, 1, 1).clamp(0.0, 1.0)
        filtered = spectrum * gate.to(dtype=spectrum.real.dtype)
        periodic_patches = torch.fft.irfft(filtered, n=self.patch_len, dim=-1, norm="ortho").to(dtype=x.dtype)
        periodic = fold_patches_average(periodic_patches, x.size(1), self.patch_len, self.stride)
        safe_magnitude = raw_magnitude.clamp_min(1e-6)
        nonzero = raw_magnitude > 1e-6
        unit_real = torch.where(nonzero, spectrum.real / safe_magnitude, torch.zeros_like(spectrum.real))
        unit_imag = torch.where(nonzero, spectrum.imag / safe_magnitude, torch.zeros_like(spectrum.imag))
        spectral_features = torch.cat(
            [magnitude * gate, unit_real * gate, unit_imag * gate], dim=-1
        ).to(dtype=x.dtype)
        tokens = self.patch_proj(spectral_features)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        encoded = self.informer(tokens) if self.informer is not None else tokens
        summary = 0.5 * encoded[:, -1, :] + 0.5 * encoded.mean(dim=1)
        summary = self.summary_norm(summary)
        if reliability is not None:
            summary = summary * reliability.reshape(-1, 1).to(dtype=summary.dtype)
        if return_tokens:
            if return_dynamic_reliability:
                return summary, periodic, encoded, dynamic_reliability
            return summary, periodic, encoded
        return summary, periodic


class ThreeWayGate(nn.Module):
    """为趋势、周期和局部残差表示学习逐通道的三路融合权重。"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model * 3, 3)

    def forward(self, trend: torch.Tensor, periodic: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        weights = stable_softmax(self.gate(torch.cat([trend, periodic, residual], dim=-1)), dim=-1)
        return (
            weights[..., 0:1] * trend
            + weights[..., 1:2] * periodic
            + weights[..., 2:3] * residual
        )


class CoarseToFineLagCrossChannelAttention(nn.Module):
    """粗到细的稀疏变量—时滞关系模块。

    第一阶段在 Patch Token 上定位 ``Top-k (source variable, coarse delay block)``；
    第二阶段仅在这些候选块附近，从原始成分信号中抽取**精确结束于** ``t-τ`` 的局部
    片段作为 Key/Value。因而 ``τ=1``、``τ=2`` 等原始采样点延迟不再被映射到同一
    Patch Token，关系图可以被解释为采样点级的预测依赖，而非粗糙的 patch 索引。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        topk: int,
        dropout: float,
        window_size: int,
        patch_len: int,
        stride: int,
        coarse_topk: int = 2,
        fine_lag_radius: int = 8,
        fine_lag_window: int = 8,
        use_fine_correlation_prior: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if coarse_topk <= 0 or fine_lag_radius < 0 or fine_lag_window <= 0:
            raise ValueError("invalid coarse-to-fine lag configuration")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.topk = max(1, int(topk))
        self.coarse_topk = max(1, int(coarse_topk))
        self.fine_lag_radius = int(fine_lag_radius)
        self.fine_lag_window = int(fine_lag_window)
        self.use_fine_correlation_prior = bool(use_fine_correlation_prior)

        # 块级关系只用于候选检索，保持计算随 C×K×P 增长而不是 C²×T。
        self.variable_query = nn.Linear(d_model, d_model)
        self.variable_key = nn.Linear(d_model, d_model)
        self.coarse_query = nn.Linear(d_model, d_model)
        self.coarse_key = nn.Linear(d_model, d_model)
        # 细粒度阶段将原始采样片段编码成关系 Key/Value。
        self.fine_query = nn.Linear(d_model, d_model)
        self.fine_patch_proj = nn.Linear(self.fine_lag_window, d_model)
        self.fine_key = nn.Linear(d_model, d_model)
        self.fine_value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        offsets = torch.arange(-self.fine_lag_radius, self.fine_lag_radius + 1, dtype=torch.long)
        self.register_buffer("fine_offsets", offsets, persistent=False)
        self.fine_lag_embedding = nn.Parameter(torch.zeros(len(offsets), d_model))
        # 以 softplus 约束为非负：NCC 仅作为“局部波形越对齐、lag 越可信”的先验，
        # 不强迫模型忽略学习式 attention。
        self.fine_correlation_logit = (
            nn.Parameter(torch.tensor(0.0)) if self.use_fine_correlation_prior else None
        )

        token_count = patch_count(window_size, patch_len, stride)
        starts = torch.arange(token_count, dtype=torch.long) * int(stride)
        # 最后一个 patch 可能包含末端复制填充；其物理结束位置仍是窗口最后一个观测。
        ends = (starts + int(patch_len) - 1).clamp(max=int(window_size) - 1)
        patch_lags = int(window_size) - 1 - ends
        self.register_buffer("patch_lags", patch_lags, persistent=False)
        nn.init.trunc_normal_(self.fine_lag_embedding, std=0.02)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, count, _ = values.shape
        return values.view(batch, count, self.n_heads, self.head_dim).transpose(1, 2)

    def _sample_precise_history(
        self,
        component: torch.Tensor,
        source_indices: torch.Tensor,
        exact_lags: torch.Tensor,
    ) -> torch.Tensor:
        """提取以 ``t-τ`` 为末端的原始采样片段。

        ``source_indices`` 为 ``[B, target C, coarse K]``，``exact_lags`` 为
        ``[B, target C, coarse K, fine L]``。左侧越界时复制序列首点，保证极短历史
        下也严格只使用过去观测。
        """
        batch, channels, length = component.shape
        _, targets, candidates = source_indices.shape
        fine_count = exact_lags.size(-1)
        source_values = component.unsqueeze(1).expand(-1, targets, -1, -1)
        source_values = source_values.gather(
            2, source_indices.unsqueeze(-1).expand(-1, -1, -1, length)
        )
        local_offsets = torch.arange(
            -(self.fine_lag_window - 1), 1, device=component.device, dtype=torch.long
        )
        ends = (length - 1 - exact_lags).unsqueeze(-1)
        positions = (ends + local_offsets).clamp(min=0, max=length - 1)
        expanded = source_values.unsqueeze(3).expand(-1, -1, -1, fine_count, -1)
        return expanded.gather(-1, positions)

    def forward(
        self,
        summary: torch.Tensor,
        tokens: torch.Tensor,
        component: torch.Tensor,
        return_relation_maps: bool = False,
    ):
        """返回关系上下文；可选返回稀疏 ``source-lag`` 概率。

        ``summary``: ``[B,C,D]``，``tokens``: ``[B,C,P,D]``，``component``:
        ``[B,C,T]``。所有细粒度片段均以输入窗口末端及其历史为界，不读取预测目标。
        """
        batch, channels, dim = summary.shape
        if component.shape[:2] != (batch, channels):
            raise ValueError("component signal must match summary batch and channel dimensions")
        if channels <= 1:
            zero = sum((parameter.sum() * 0.0) for parameter in self.parameters())
            context = summary.new_zeros(batch, channels, dim) + zero
            empty = {
                "source_indices": torch.empty(batch, channels, 0, dtype=torch.long, device=summary.device),
                "coarse_lags": torch.empty(batch, channels, 0, dtype=torch.long, device=summary.device),
                "fine_lags": torch.empty(batch, channels, 0, 0, dtype=torch.long, device=summary.device),
                "probabilities": summary.new_empty(batch, channels, 0, 0),
            }
            return (context, empty) if return_relation_maps else context

        # 1) 块级：先选择少量候选源变量。
        variable_query = self.variable_query(summary)
        variable_key = self.variable_key(summary)
        variable_scores = torch.matmul(variable_query, variable_key.transpose(-2, -1)) / math.sqrt(dim)
        diagonal = torch.eye(channels, device=summary.device, dtype=torch.bool).unsqueeze(0)
        variable_scores = variable_scores.masked_fill(diagonal, -1e4)
        source_topk = min(self.topk, channels - 1)
        source_scores, source_indices = variable_scores.topk(source_topk, dim=-1)

        # 2) 块级：在已选源变量的全部历史块中定位延迟范围。
        coarse_query = self.coarse_query(summary)
        coarse_key = self.coarse_key(tokens)
        all_keys = coarse_key.unsqueeze(1).expand(-1, channels, -1, -1, -1)
        selected_keys = all_keys.gather(
            2,
            source_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, tokens.size(2), dim
            ),
        )
        coarse_scores = (coarse_query.unsqueeze(2).unsqueeze(3) * selected_keys).sum(dim=-1) / math.sqrt(dim)
        coarse_scores = coarse_scores + source_scores.unsqueeze(-1)
        coarse_scores = coarse_scores.flatten(2, 3)
        candidate_count = min(self.coarse_topk, coarse_scores.size(-1))
        coarse_top_scores, coarse_indices = coarse_scores.topk(candidate_count, dim=-1)
        source_slot = torch.div(coarse_indices, tokens.size(2), rounding_mode="floor")
        patch_indices = coarse_indices.remainder(tokens.size(2))
        coarse_source_indices = source_indices.gather(2, source_slot)
        coarse_lags = self.patch_lags.to(device=summary.device).index_select(
            0, patch_indices.reshape(-1)
        ).view_as(patch_indices)

        # 3) 块内：围绕候选块的原始采样点延迟进行细粒度精确搜索。
        proposed_lags = coarse_lags.unsqueeze(-1) + self.fine_offsets.view(1, 1, 1, -1)
        fine_valid = (proposed_lags >= 0) & (proposed_lags < component.size(-1))
        fine_lags = proposed_lags.clamp(min=0, max=component.size(-1) - 1)
        local_history = self._sample_precise_history(component, coarse_source_indices, fine_lags)
        local_features = self.fine_patch_proj(local_history.to(dtype=summary.dtype))
        local_features = local_features + self.fine_lag_embedding.view(1, 1, 1, -1, dim)

        query = self._split_heads(self.fine_query(summary))
        fine_key = self.fine_key(local_features).view(
            batch, channels, candidate_count, len(self.fine_offsets), self.n_heads, self.head_dim
        ).permute(0, 4, 1, 2, 3, 5)
        fine_value = self.fine_value(local_features).view(
            batch, channels, candidate_count, len(self.fine_offsets), self.n_heads, self.head_dim
        ).permute(0, 4, 1, 2, 3, 5)
        fine_scores = (query.unsqueeze(-2).unsqueeze(-2) * fine_key).sum(dim=-1) / math.sqrt(self.head_dim)
        fine_scores = fine_scores + coarse_top_scores.unsqueeze(1).unsqueeze(-1)
        if self.fine_correlation_logit is not None:
            target_history = component[:, :, -self.fine_lag_window :]
            target_history = target_history - target_history.mean(dim=-1, keepdim=True)
            target_history = target_history / target_history.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            source_history = local_history - local_history.mean(dim=-1, keepdim=True)
            source_history = source_history / source_history.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            ncc = (source_history * target_history.unsqueeze(2).unsqueeze(2)).mean(dim=-1)
            fine_scores = fine_scores + F.softplus(self.fine_correlation_logit) * ncc.unsqueeze(1)
        # 边界外 offset 不应因 clamp 后与 τ=0/末端重复而获得额外概率质量。
        fine_scores = fine_scores.masked_fill(~fine_valid.unsqueeze(1), -1e4)
        fine_weights = stable_softmax(fine_scores.flatten(-2), dim=-1).view_as(fine_scores)
        dropped_weights = self.dropout(fine_weights)
        context = (dropped_weights.unsqueeze(-1) * fine_value).sum(dim=(-3, -2))
        context = context.transpose(1, 2).contiguous().view(batch, channels, dim)
        context = self.norm(self.output(context))

        if not return_relation_maps:
            return context
        # 关系图使用未 dropout 的、跨头平均概率，便于评测阶段稳定导出。
        relation_map = {
            "source_indices": coarse_source_indices,
            "coarse_lags": coarse_lags,
            "fine_lags": fine_lags,
            "probabilities": fine_weights.mean(dim=1),
        }
        return context, relation_map


class MultiScaleCoarseToFineLagRelationAttention(nn.Module):
    """趋势、周期、残差分别进行粗到细精确变量—时滞检索。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        topk: int,
        dropout: float,
        window_size: int,
        trend_patch_len: int,
        trend_stride: int,
        periodic_patch_len: int,
        periodic_stride: int,
        residual_patch_len: int,
        residual_stride: int,
        coarse_topk: int,
        fine_lag_radius: int,
        fine_lag_window: int,
        use_fine_correlation_prior: bool = False,
    ) -> None:
        super().__init__()
        branch_args = dict(
            d_model=d_model,
            n_heads=n_heads,
            topk=topk,
            dropout=dropout,
            window_size=window_size,
            coarse_topk=coarse_topk,
            fine_lag_radius=fine_lag_radius,
            fine_lag_window=fine_lag_window,
            use_fine_correlation_prior=use_fine_correlation_prior,
        )
        self.trend = CoarseToFineLagCrossChannelAttention(
            patch_len=trend_patch_len, stride=trend_stride, **branch_args
        )
        self.periodic = CoarseToFineLagCrossChannelAttention(
            patch_len=periodic_patch_len, stride=periodic_stride, **branch_args
        )
        self.residual = CoarseToFineLagCrossChannelAttention(
            patch_len=residual_patch_len, stride=residual_stride, **branch_args
        )
        self.scale_gate = ThreeWayGate(d_model)

    def forward(
        self,
        trend_summary: torch.Tensor,
        trend_tokens: torch.Tensor,
        trend_signal: torch.Tensor,
        periodic_summary: torch.Tensor,
        periodic_tokens: torch.Tensor,
        periodic_signal: torch.Tensor,
        residual_summary: torch.Tensor,
        residual_tokens: torch.Tensor,
        residual_signal: torch.Tensor,
        return_relation_maps: bool = False,
        return_branch_contexts: bool = False,
    ):
        trend_result = self.trend(
            trend_summary, trend_tokens, trend_signal, return_relation_maps=return_relation_maps
        )
        periodic_result = self.periodic(
            periodic_summary, periodic_tokens, periodic_signal, return_relation_maps=return_relation_maps
        )
        residual_result = self.residual(
            residual_summary, residual_tokens, residual_signal, return_relation_maps=return_relation_maps
        )
        if return_relation_maps:
            trend_context, trend_map = trend_result
            periodic_context, periodic_map = periodic_result
            residual_context, residual_map = residual_result
            maps = {"trend": trend_map, "periodic": periodic_map, "residual": residual_map}
        else:
            trend_context, periodic_context, residual_context = trend_result, periodic_result, residual_result
            maps = None
        context = self.scale_gate(trend_context, periodic_context, residual_context)
        branches = {
            "trend": trend_context,
            "periodic": periodic_context,
            "residual": residual_context,
        }
        if return_relation_maps and return_branch_contexts:
            return context, maps, branches
        if return_relation_maps:
            return context, maps
        if return_branch_contexts:
            return context, branches
        return context


class SharedRelationValueEncoder(nn.Module):
    """在固定 source—lag 概率图下，为单个成分聚合局部历史值。"""

    def __init__(self, d_model: int, fine_lag_window: int) -> None:
        super().__init__()
        self.patch_proj = nn.Linear(fine_lag_window, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        component: torch.Tensor,
        relation_map: dict,
        sampler: CoarseToFineLagCrossChannelAttention,
    ) -> torch.Tensor:
        sources = relation_map["source_indices"]
        if sources.size(2) == 0:
            zero = sum((parameter.sum() * 0.0) for parameter in self.parameters())
            return component.new_zeros(component.size(0), component.size(1), self.output.out_features) + zero
        local_history = sampler._sample_precise_history(component, sources, relation_map["fine_lags"])
        values = self.value(self.patch_proj(local_history.to(dtype=component.dtype)))
        weights = relation_map["probabilities"].to(dtype=values.dtype)
        context = (weights.unsqueeze(-1) * values).sum(dim=(2, 3))
        return self.norm(self.output(context))


class SharedCoarseToFineLagRelationAttention(nn.Module):
    """共享 source—lag 图的零假设模型 ``M0``。

    三个成分仍使用各自局部信号作为 Value，从而保留异构编码；但候选源变量、候选
    延迟块以及块内精确 lag 概率完全由同一个 selector 从原始混合信号中产生。于是
    ``A^T=A^P=A^R``，可与独立图模型形成严格的核心假设对照。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        topk: int,
        dropout: float,
        window_size: int,
        selector_patch_len: int,
        selector_stride: int,
        coarse_topk: int,
        fine_lag_radius: int,
        fine_lag_window: int,
        use_fine_correlation_prior: bool = False,
    ) -> None:
        super().__init__()
        self.selector_patch_len = int(selector_patch_len)
        self.selector_stride = int(selector_stride)
        self.summary_fuse = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.selector_patch_proj = nn.Linear(self.selector_patch_len, d_model)
        self.selector_pos = nn.Parameter(
            torch.zeros(1, patch_count(window_size, self.selector_patch_len, self.selector_stride), d_model)
        )
        self.selector = CoarseToFineLagCrossChannelAttention(
            d_model=d_model,
            n_heads=n_heads,
            topk=topk,
            dropout=dropout,
            window_size=window_size,
            patch_len=self.selector_patch_len,
            stride=self.selector_stride,
            coarse_topk=coarse_topk,
            fine_lag_radius=fine_lag_radius,
            fine_lag_window=fine_lag_window,
            use_fine_correlation_prior=use_fine_correlation_prior,
        )
        self.trend_value = SharedRelationValueEncoder(d_model, fine_lag_window)
        self.periodic_value = SharedRelationValueEncoder(d_model, fine_lag_window)
        self.residual_value = SharedRelationValueEncoder(d_model, fine_lag_window)
        self.scale_gate = ThreeWayGate(d_model)
        nn.init.trunc_normal_(self.selector_pos, std=0.02)

    def forward(
        self,
        trend_summary: torch.Tensor,
        trend_signal: torch.Tensor,
        periodic_summary: torch.Tensor,
        periodic_signal: torch.Tensor,
        residual_summary: torch.Tensor,
        residual_signal: torch.Tensor,
        return_relation_maps: bool = False,
        return_branch_contexts: bool = False,
    ):
        batch, channels, dim = trend_summary.shape
        if channels <= 1:
            zero = sum((parameter.sum() * 0.0) for parameter in self.parameters())
            context = trend_summary.new_zeros(batch, channels, dim) + zero
            empty = {
                "source_indices": torch.empty(batch, channels, 0, dtype=torch.long, device=trend_summary.device),
                "coarse_lags": torch.empty(batch, channels, 0, dtype=torch.long, device=trend_summary.device),
                "fine_lags": torch.empty(batch, channels, 0, 0, dtype=torch.long, device=trend_summary.device),
                "probabilities": trend_summary.new_empty(batch, channels, 0, 0),
            }
            branches = {"trend": context, "periodic": context, "residual": context}
            if return_relation_maps and return_branch_contexts:
                return context, {"shared": empty}, branches
            if return_relation_maps:
                return context, {"shared": empty}
            if return_branch_contexts:
                return context, branches
            return context

        # 三成分之和等于原始输入窗口；共享 selector 因此只选择一张图，而不偏向某一成分。
        shared_signal = trend_signal + periodic_signal + residual_signal
        shared_summary = self.summary_fuse(
            torch.cat([trend_summary, periodic_summary, residual_summary], dim=-1)
        )
        patches = unfold_patches(
            shared_signal.reshape(batch * channels, shared_signal.size(-1)),
            self.selector_patch_len,
            self.selector_stride,
        )
        shared_tokens = self.selector_patch_proj(patches)
        shared_tokens = shared_tokens + self.selector_pos[:, : shared_tokens.size(1), :]
        shared_tokens = shared_tokens.view(batch, channels, shared_tokens.size(1), -1)
        _, relation_map = self.selector(
            shared_summary, shared_tokens, shared_signal, return_relation_maps=True
        )
        trend_context = self.trend_value(trend_signal, relation_map, self.selector)
        periodic_context = self.periodic_value(periodic_signal, relation_map, self.selector)
        residual_context = self.residual_value(residual_signal, relation_map, self.selector)
        context = self.scale_gate(trend_context, periodic_context, residual_context)
        # ``selector`` 在 M0 中只产生共享关系概率；其原有 fine Value/output 层不
        # 应改变三成分的独立值聚合，但仍需以零系数连入图中，避免 DDP 将其误判为
        # 未使用参数。M1（独立图）则正常使用这些层生成每个成分的关系上下文。
        selector_value_zero = sum(
            (parameter.sum() * 0.0)
            for module in (self.selector.fine_value, self.selector.output, self.selector.norm)
            for parameter in module.parameters()
        )
        context = context + selector_value_zero
        branches = {
            "trend": trend_context,
            "periodic": periodic_context,
            "residual": residual_context,
        }
        if return_relation_maps and return_branch_contexts:
            return context, {"shared": relation_map}, branches
        if return_relation_maps:
            return context, {"shared": relation_map}
        if return_branch_contexts:
            return context, branches
        return context
