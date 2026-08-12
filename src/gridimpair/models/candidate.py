from __future__ import annotations

import torch
from torch import nn

from gridimpair.contracts import ModelOutput
from gridimpair.models.tcn import CausalConv1d, CausalResidualBlock, ChannelLayerNorm


class _MultiScaleEncoder(nn.Module):
    def __init__(self, channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        branch = max(4, hidden // 3)
        self.projection = nn.Conv1d(channels * 2, hidden, 1)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    CausalConv1d(
                        hidden,
                        hidden,
                        kernel,
                        groups=hidden,
                        bias=False,
                    ),
                    nn.Conv1d(hidden, branch, 1, bias=False),
                    nn.GELU(),
                )
                for kernel in (3, 5, 9)
            ]
        )
        self.merge = nn.Conv1d(branch * 3, hidden, 1)
        self.residual = nn.Sequential(
            CausalResidualBlock(hidden, dilation=1, dropout=dropout, depthwise=True),
            CausalResidualBlock(hidden, dilation=2, dropout=dropout, depthwise=True),
            CausalResidualBlock(hidden, dilation=4, dropout=dropout, depthwise=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        difference = torch.diff(inputs, dim=-1, prepend=inputs[..., :1])
        projected = self.projection(torch.cat([inputs, difference], dim=1))
        multi_scale = self.merge(torch.cat([branch(projected) for branch in self.branches], dim=1))
        return self.residual(projected + multi_scale)


class CausalARResidualMIL(nn.Module):
    """Dual-view causal AR-residual model with no ID or onset metadata."""

    def __init__(
        self, channels: int, hidden: int = 48, *, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.raw_encoder = _MultiScaleEncoder(channels, hidden, dropout)
        self.residual_encoder = _MultiScaleEncoder(channels, hidden, dropout)
        self.gate = nn.Conv1d(hidden * 2, hidden, 1)
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, 1),
            ChannelLayerNorm(hidden),
            nn.GELU(),
        )
        self.instance_head = nn.Conv1d(hidden, 1, 1)
        self.attention_head = nn.Sequential(
            nn.Conv1d(hidden, max(4, hidden // 2), 1),
            nn.Tanh(),
            nn.Conv1d(max(4, hidden // 2), 1, 1),
        )

    def forward(
        self,
        raw: torch.Tensor,
        residual: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        if raw.shape != residual.shape:
            raise ValueError("raw and residual views must have identical shapes")
        if raw.ndim != 3:
            raise ValueError("views must have shape (batch, channels, time)")
        raw_features = self.raw_encoder(raw)
        residual_features = self.residual_encoder(residual)
        joined = torch.cat([raw_features, residual_features], dim=1)
        gate = torch.sigmoid(self.gate(joined))
        blended = gate * raw_features + (1 - gate) * residual_features
        features = blended + self.fusion(joined)
        instances = self.instance_head(features).squeeze(1)
        attention_logits = self.attention_head(features).squeeze(1)
        if mask is not None:
            if mask.shape != instances.shape:
                raise ValueError("mask must have shape (batch, time)")
            attention_logits = attention_logits.masked_fill(~mask.bool(), -torch.inf)
        attention = torch.nan_to_num(torch.softmax(attention_logits, dim=-1))
        bag = (attention * instances).sum(dim=-1)
        return ModelOutput(bag_logit=bag, instance_logits=instances, attention=attention)


# Backward-compatible import only. The residual view is autoregressive, not a
# power-flow residual, so new provenance uses the scientifically honest name.
PhysicsResidualMIL = CausalARResidualMIL
