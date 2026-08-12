from __future__ import annotations

import torch
from torch import nn

from gridimpair.contracts import ModelOutput
from gridimpair.models.tcn import CausalResidualBlock


class TailMIL(nn.Module):
    """Local causal adaptation of temporal/variable TAIL-MIL aggregation.

    Variable attention is evaluated independently at every time point. The
    resulting signal is fused with a causal temporal encoder; no exact episode
    boundary or scenario identifier is consumed.
    """

    def __init__(
        self, channels: int, hidden: int = 48, *, levels: int = 3, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.variable_attention = nn.Conv1d(channels, channels, 1)
        self.input_projection = nn.Conv1d(channels + 1, hidden, 1)
        self.temporal = nn.Sequential(
            *[
                CausalResidualBlock(
                    hidden, dilation=2**level, dropout=dropout, depthwise=True
                )
                for level in range(levels)
            ]
        )
        self.instance_head = nn.Conv1d(hidden, 1, 1)
        self.temporal_attention = nn.Conv1d(hidden, 1, 1)

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> ModelOutput:
        variable_weights = torch.softmax(self.variable_attention(inputs), dim=1)
        variable_summary = (variable_weights * inputs).sum(dim=1, keepdim=True)
        features = self.temporal(
            self.input_projection(torch.cat([inputs, variable_summary], dim=1))
        )
        instances = self.instance_head(features).squeeze(1)
        attention_logits = self.temporal_attention(features).squeeze(1)
        if mask is not None:
            attention_logits = attention_logits.masked_fill(~mask.bool(), -torch.inf)
        attention = torch.nan_to_num(torch.softmax(attention_logits, dim=-1))
        bag = (attention * instances).sum(dim=-1)
        return ModelOutput(bag_logit=bag, instance_logits=instances, attention=attention)
