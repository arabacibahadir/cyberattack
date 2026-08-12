from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class TrialConfig:
    trial_id: int
    values: dict[str, Any]


def make_lhs_trials(
    space: Mapping[str, Mapping[str, Any]], n: int, seed: int
) -> list[TrialConfig]:
    if n < 1:
        raise ValueError("n must be positive")
    names = sorted(space)
    if not names:
        raise ValueError("HPO space cannot be empty")
    sample = qmc.LatinHypercube(d=len(names), seed=seed).random(n=n)
    trials: list[TrialConfig] = []
    for trial_index, unit_values in enumerate(sample):
        values: dict[str, Any] = {}
        for name, unit in zip(names, unit_values, strict=True):
            dimension = space[name]
            kind = dimension.get("type")
            if kind == "float":
                low, high = float(dimension["low"]), float(dimension["high"])
                values[name] = low + unit * (high - low)
            elif kind == "log_float":
                low, high = float(dimension["low"]), float(dimension["high"])
                values[name] = math.exp(math.log(low) + unit * (math.log(high) - math.log(low)))
            elif kind == "int":
                low, high = int(dimension["low"]), int(dimension["high"])
                values[name] = min(high, low + int(unit * (high - low + 1)))
            elif kind == "choice":
                choices = list(dimension["values"])
                if not choices:
                    raise ValueError(f"choice dimension {name!r} is empty")
                values[name] = choices[min(len(choices) - 1, int(unit * len(choices)))]
            else:
                raise ValueError(f"unknown HPO dimension type: {kind}")
        trials.append(TrialConfig(trial_id=trial_index, values=values))
    return trials
