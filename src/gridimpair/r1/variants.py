"""Architecture ablation variants of CyberGuard-DER (CausalARResidualMIL).

Each flag removes one component while keeping every other layer identical:

* ``use_difference``  - concatenate the first difference before projection
* ``use_residual``    - add the frozen AR-residual view and its encoder
* ``use_gate``        - gated fusion (otherwise the two views are averaged)
* ``use_attention``   - attention-MIL pooling (otherwise mean pooling)

With all flags on, the module has exactly the same layers and parameter count
as :class:`gridimpair.models.candidate.CausalARResidualMIL`.
"""
from __future__ import annotations

import torch
from torch import nn

from gridimpair.contracts import ModelOutput
from gridimpair.models.tcn import CausalConv1d, CausalResidualBlock, ChannelLayerNorm

VARIANTS: dict[str, dict[str, bool]] = {
    "v0_base_causal_cnn": dict(use_difference=False, use_residual=False, use_gate=False, use_attention=False),
    "v1_plus_difference": dict(use_difference=True, use_residual=False, use_gate=False, use_attention=False),
    "v2_plus_ar_residual": dict(use_difference=True, use_residual=True, use_gate=False, use_attention=False),
    "v3_plus_gated_fusion": dict(use_difference=True, use_residual=True, use_gate=True, use_attention=False),
    "v4_plus_attention_mil": dict(use_difference=True, use_residual=True, use_gate=True, use_attention=True),
}


class _Encoder(nn.Module):
    def __init__(self, channels: int, hidden: int, dropout: float, use_difference: bool) -> None:
        super().__init__()
        self.use_difference = use_difference
        branch = max(4, hidden // 3)
        self.projection = nn.Conv1d(channels * (2 if use_difference else 1), hidden, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(
                CausalConv1d(hidden, hidden, kernel, groups=hidden, bias=False),
                nn.Conv1d(hidden, branch, 1, bias=False),
                nn.GELU(),
            )
            for kernel in (3, 5, 9)
        ])
        self.merge = nn.Conv1d(branch * 3, hidden, 1)
        self.residual = nn.Sequential(*[
            CausalResidualBlock(hidden, dilation=d, dropout=dropout, depthwise=True) for d in (1, 2, 4)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_difference:
            inputs = torch.cat([inputs, torch.diff(inputs, dim=-1, prepend=inputs[..., :1])], dim=1)
        projected = self.projection(inputs)
        multi_scale = self.merge(torch.cat([b(projected) for b in self.branches], dim=1))
        return self.residual(projected + multi_scale)


class AblationMIL(nn.Module):
    def __init__(self, channels: int, hidden: int = 48, *, dropout: float = 0.1, use_difference: bool = True,
                 use_residual: bool = True, use_gate: bool = True, use_attention: bool = True) -> None:
        super().__init__()
        self.use_residual, self.use_gate, self.use_attention = use_residual, use_gate, use_attention
        self.raw_encoder = _Encoder(channels, hidden, dropout, use_difference)
        if use_residual:
            self.residual_encoder = _Encoder(channels, hidden, dropout, use_difference)
            if use_gate:
                self.gate = nn.Conv1d(hidden * 2, hidden, 1)
                self.fusion = nn.Sequential(nn.Conv1d(hidden * 2, hidden, 1), ChannelLayerNorm(hidden), nn.GELU())
        self.instance_head = nn.Conv1d(hidden, 1, 1)
        if use_attention:
            self.attention_head = nn.Sequential(
                nn.Conv1d(hidden, max(4, hidden // 2), 1), nn.Tanh(), nn.Conv1d(max(4, hidden // 2), 1, 1)
            )

    def forward(self, raw: torch.Tensor, residual: torch.Tensor, mask: torch.Tensor | None = None) -> ModelOutput:
        features = self.raw_encoder(raw)
        if self.use_residual:
            residual_features = self.residual_encoder(residual)
            if self.use_gate:
                joined = torch.cat([features, residual_features], dim=1)
                gate = torch.sigmoid(self.gate(joined))
                features = gate * features + (1 - gate) * residual_features + self.fusion(joined)
            else:
                features = 0.5 * (features + residual_features)
        instances = self.instance_head(features).squeeze(1)
        if self.use_attention:
            logits = self.attention_head(features).squeeze(1)
            if mask is not None:
                logits = logits.masked_fill(~mask.bool(), -torch.inf)
            attention = torch.nan_to_num(torch.softmax(logits, dim=-1))
        else:
            attention = torch.full_like(instances, 1.0 / instances.shape[-1])
        bag = (attention * instances).sum(dim=-1)
        return ModelOutput(bag_logit=bag, instance_logits=instances, attention=attention)


def count_macs(model: nn.Module, channels: int, steps: int = 900, two_views: bool = True) -> int:
    total = [0]

    def hook(module, _inputs, output):
        total[0] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv1d)]
    with torch.no_grad():
        model.eval()
        x = torch.zeros(1, channels, steps)
        model(x, x) if two_views else model(x)
    for h in handles:
        h.remove()
    return total[0]
