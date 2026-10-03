# SPDX-License-Identifier: MIT
"""Composition of the frozen C2F active paths (full, shared, radius 0).

The component bodies are unchanged in components.py. This small composition
adapter removes inactive research architectures and file/device orchestration.
It has been statically inspected, not numerically equivalence-tested.
"""
from __future__ import annotations

import inspect

import torch
from torch import nn

from .components import (
    AttentionPatchBranch,
    CausalTrendPeriodicDecomposer,
    MultiScaleCoarseToFineLagRelationAttention,
    ResidualTCNPatchBranch,
    SharedCoarseToFineLagRelationAttention,
    SpectralPatchBranch,
    ThreeWayGate,
)


class C2FCSLRNet(nn.Module):
    """One-step channel-shared predictor with component-conditioned retrieval.

    Inputs are [batch, window, channels]. Supply the spectral reliability from
    this file's permitted estimation prefix; it is a nonpersistent buffer.
    Defaults below come from the frozen 64-point configuration, not upstream
    Informer defaults. ``variant`` is full, shared, or radius0.
    """

    def __init__(
        self, *, channel_spectral_reliability, variant="full", window_size=64,
        d_model=64, n_heads=4, informer_layers=3, factor=5, tcn_levels=3,
        conv_kernel_size=4, dropout=0.1, attention_type="probsparse",
        trend_patch_len=32, trend_stride=16, periodic_patch_len=24,
        periodic_stride=12, residual_patch_len=16, residual_stride=8,
        cross_channel_topk=8, coarse_lag_topk=2, fine_lag_radius=8,
        fine_lag_window=8, relation_loss_weight=0.2, relation_score_weight=0.0,
    ):
        super().__init__()
        if variant not in {"full", "shared", "radius0"}:
            raise ValueError("variant must be full, shared, or radius0")
        if attention_type != "probsparse":
            raise ValueError("This release exposes the frozen ProbSparse configuration")
        if variant == "radius0":
            fine_lag_radius = 0
        self.variant = variant
        self.relation_loss_weight = float(relation_loss_weight)
        self.relation_score_weight = float(relation_score_weight)
        attention_branch_kwargs = dict(
            window_size=window_size, d_model=d_model, n_heads=n_heads,
            informer_layers=informer_layers, factor=factor, dropout=dropout,
            attention_type=attention_type,
        )
        # Preserve the frozen constructor's parameter-producing module order.
        self.decomposer = CausalTrendPeriodicDecomposer()
        self.trend = AttentionPatchBranch(
            patch_len=trend_patch_len, stride=trend_stride,
            use_local_attention=True, **attention_branch_kwargs,
        )
        self.periodic = SpectralPatchBranch(
            patch_len=periodic_patch_len, stride=periodic_stride,
            use_local_attention=True, use_dynamic_reliability=False,
            **attention_branch_kwargs,
        )
        self.residual = ResidualTCNPatchBranch(
            window_size=window_size, patch_len=residual_patch_len,
            stride=residual_stride, d_model=d_model, tcn_levels=tcn_levels,
            conv_kernel_size=conv_kernel_size, dropout=dropout,
        )
        self.component_gate = ThreeWayGate(d_model)
        relation_kwargs = dict(
            d_model=d_model, n_heads=n_heads, topk=cross_channel_topk,
            dropout=dropout, window_size=window_size,
            coarse_topk=coarse_lag_topk, fine_lag_radius=fine_lag_radius,
            fine_lag_window=fine_lag_window, use_fine_correlation_prior=False,
        )
        if variant == "shared":
            self.cross_channel = SharedCoarseToFineLagRelationAttention(
                selector_patch_len=residual_patch_len,
                selector_stride=residual_stride, **relation_kwargs,
            )
        else:
            self.cross_channel = MultiScaleCoarseToFineLagRelationAttention(
                trend_patch_len=trend_patch_len, trend_stride=trend_stride,
                periodic_patch_len=periodic_patch_len, periodic_stride=periodic_stride,
                residual_patch_len=residual_patch_len, residual_stride=residual_stride,
                **relation_kwargs,
            )
        self.relation_head = nn.Linear(d_model, 1)
        self.register_buffer(
            "channel_spectral_reliability",
            torch.as_tensor(channel_spectral_reliability, dtype=torch.float32),
            persistent=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1),
        )

    def forward(self, x, return_aux=False, return_relation_maps=False):
        batch, length, channels = x.shape
        x_ch = x.permute(0, 2, 1).reshape(batch * channels, length)
        if self.channel_spectral_reliability.numel() != channels:
            raise ValueError("Supply one spectral reliability value per input channel")
        reliability = self.channel_spectral_reliability.to(device=x.device, dtype=x.dtype)
        reliability = reliability.unsqueeze(0).expand(batch, -1).reshape(-1)
        trend_signal = self.decomposer(x_ch)
        detrended = x_ch - trend_signal
        return_tokens = self.variant != "shared"
        trend_result = self.trend(trend_signal, return_tokens=return_tokens)
        periodic_result = self.periodic(
            detrended, reliability, return_tokens=return_tokens,
            return_dynamic_reliability=False,
        )
        if return_tokens:
            trend_features, trend_tokens = trend_result
            periodic_features, periodic_signal, periodic_tokens = periodic_result
        else:
            trend_features = trend_result
            periodic_features, periodic_signal = periodic_result
        residual_signal = detrended - periodic_signal
        residual_result = self.residual(residual_signal, return_tokens=return_tokens)
        if return_tokens:
            residual_features, residual_tokens = residual_result
        else:
            residual_features = residual_result
        fused = self.component_gate(trend_features, periodic_features, residual_features)
        fused = fused.view(batch, channels, -1)
        if self.variant == "shared":
            relation_result = self.cross_channel(
                trend_features.view(batch, channels, -1), trend_signal.view(batch, channels, length),
                periodic_features.view(batch, channels, -1), periodic_signal.view(batch, channels, length),
                residual_features.view(batch, channels, -1), residual_signal.view(batch, channels, length),
                return_relation_maps=return_relation_maps,
            )
        else:
            relation_result = self.cross_channel(
                trend_features.view(batch, channels, -1),
                trend_tokens.view(batch, channels, trend_tokens.size(1), -1),
                trend_signal.view(batch, channels, length),
                periodic_features.view(batch, channels, -1),
                periodic_tokens.view(batch, channels, periodic_tokens.size(1), -1),
                periodic_signal.view(batch, channels, length),
                residual_features.view(batch, channels, -1),
                residual_tokens.view(batch, channels, residual_tokens.size(1), -1),
                residual_signal.view(batch, channels, length),
                return_relation_maps=return_relation_maps,
            )
        relation_maps = None
        if return_relation_maps:
            relation_context, relation_maps = relation_result
        else:
            relation_context = relation_result
        relation_prediction = self.relation_head(relation_context).squeeze(-1)
        fused = fused + relation_context
        pred = self.head(fused.reshape(batch * channels, -1)).view(batch, channels)
        if return_aux:
            return pred, {"relation_prediction": relation_prediction, "relation_maps": relation_maps}
        return pred


def build_predictor(configuration, *, channel_spectral_reliability=None):
    """Construct an exposed variant from a published example configuration.

    This does not fit normalization, compute the spectral prior, load a
    checkpoint, or recreate the original trainer's initialization protocol.
    """
    model = dict(configuration["model"])
    required = {"use_component_branches": True, "use_cross_channel_attention": True}
    disabled = (
        "use_global_variable_block_attention", "use_recent_block_retention",
        "use_variable_role_encoding", "use_dynamic_periodic_reliability",
        "use_multiscale_lag_attention", "use_adaptive_hybrid_c2f_lag_attention",
        "use_fine_correlation_prior",
    )
    for key, value in required.items():
        if model.get(key) is not value:
            raise ValueError("Unsupported configuration: " + key)
    if any(model.get(key, False) for key in disabled):
        raise ValueError("Configuration enables an excluded research architecture")
    variant = configuration["variant"]
    if model.get("use_shared_coarse_to_fine_lag_attention", False) != (variant == "shared"):
        raise ValueError("Shared selector switch disagrees with variant")
    if model.get("use_coarse_to_fine_lag_attention", False) != (variant != "shared"):
        raise ValueError("Component selector switch disagrees with variant")
    if variant == "radius0" and model["fine_lag_radius"] != 0:
        raise ValueError("Materialize the radius0 override before construction")
    accepted = set(inspect.signature(C2FCSLRNet).parameters)
    kwargs = {key: value for key, value in model.items() if key in accepted}
    kwargs.update(window_size=configuration["data"]["window_size"], variant=variant)
    if channel_spectral_reliability is not None:
        kwargs["channel_spectral_reliability"] = channel_spectral_reliability
    return C2FCSLRNet(**kwargs)
