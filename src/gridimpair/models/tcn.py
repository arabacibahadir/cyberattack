from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from gridimpair.contracts import ModelOutput


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.left_padding = (self.kernel_size[0] - 1) * self.dilation[0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(inputs, (self.left_padding, 0)))


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each time point."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.normalization(inputs.transpose(1, 2)).transpose(1, 2)


class CausalResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        groups = channels if depthwise else 1
        self.network = nn.Sequential(
            CausalConv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.Conv1d(channels, channels, 1, bias=False) if depthwise else nn.Identity(),
            ChannelLayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.network(inputs)


def mil_pool(
    features: torch.Tensor,
    instance_logits: torch.Tensor,
    *,
    mode: str,
    attention_head: nn.Module | None,
    mask: torch.Tensor | None,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if mask is not None:
        if mask.shape != instance_logits.shape:
            raise ValueError("mask must have shape (batch, time)")
        valid = mask.to(dtype=torch.bool)
    else:
        valid = torch.ones_like(instance_logits, dtype=torch.bool)
    if mode == "mean":
        weights = valid.to(instance_logits.dtype)
        pooled = (instance_logits * weights).sum(-1) / weights.sum(-1).clamp_min(1)
        return pooled, None
    if mode == "topk":
        masked = instance_logits.masked_fill(~valid, -torch.inf)
        k = min(top_k, instance_logits.shape[-1])
        return torch.topk(masked, k=k, dim=-1).values.mean(-1), None
    if mode == "attention":
        if attention_head is None:
            raise RuntimeError("attention pooling requires an attention head")
        attention_logits = attention_head(features).squeeze(1).masked_fill(~valid, -torch.inf)
        attention = torch.softmax(attention_logits, dim=-1)
        attention = torch.nan_to_num(attention)
        return (attention * instance_logits).sum(-1), attention
    raise ValueError(f"unknown MIL pooling mode: {mode}")


class StandardCausalTCN(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: int = 48,
        *,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        pooling: str = "attention",
        top_k: int = 4,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(channels, hidden, 1)
        self.blocks = nn.Sequential(
            *[
                CausalResidualBlock(
                    hidden,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
                for level in range(levels)
            ]
        )
        self.instance_head = nn.Conv1d(hidden, 1, 1)
        self.attention_head = nn.Conv1d(hidden, 1, 1)
        self.pooling = pooling
        self.top_k = top_k

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("input must have shape (batch, channels, time)")
        return self.blocks(self.input_projection(inputs))

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> ModelOutput:
        features = self.encode(inputs)
        instances = self.instance_head(features).squeeze(1)
        bag, attention = mil_pool(
            features,
            instances,
            mode=self.pooling,
            attention_head=self.attention_head,
            mask=mask,
            top_k=self.top_k,
        )
        return ModelOutput(bag_logit=bag, instance_logits=instances, attention=attention)
