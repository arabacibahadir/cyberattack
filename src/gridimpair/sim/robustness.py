from __future__ import annotations

from dataclasses import dataclass, replace

from gridimpair.contracts import ScenarioConfig
from gridimpair.sim.scenario import build_pair


@dataclass(frozen=True)
class RobustnessScenario:
    config: ScenarioConfig
    impairment_kind: str

    def __post_init__(self) -> None:
        if self.impairment_kind not in {"burst", "jitter"}:
            raise ValueError("robustness impairment must be burst or jitter")


def generate_robustness_matrix(
    *, base_seed: int = 42026, replicates: int = 3
) -> list[RobustnessScenario]:
    """Held-out 3 regimes × 2 unseen shifts × 3 paired replicates."""
    if replicates < 1:
        raise ValueError("robustness replicates must be positive")
    scenarios: list[RobustnessScenario] = []
    for regime_index, regime in enumerate(
        ("balanced", "high_load_low_der", "low_load_high_der")
    ):
        for kind_index, kind in enumerate(("burst", "jitter")):
            for replicate in range(replicates):
                pair_id = f"robust-r{regime_index}-{kind}-rep{replicate}"
                seed = (
                    base_seed
                    + regime_index * 10_000
                    + kind_index * 1_000
                    + replicate * 10
                )
                pair = build_pair(
                    ScenarioConfig(
                        scenario_id=f"{pair_id}-b",
                        pair_id=pair_id,
                        split="robustness",
                        regime=regime,
                        delay_steps=0,
                        drop_probability=0.0,
                        endpoint_fraction=1.0,
                        status="impaired",
                        profile_seed=seed + 1,
                        schedule_seed=seed + 2,
                        impairment_seed=seed + 3,
                    )
                )
                scenarios.extend(
                    [
                        RobustnessScenario(
                            replace(pair.clean, scenario_id=f"{pair_id}-a"),
                            kind,
                        ),
                        RobustnessScenario(pair.impaired, kind),
                    ]
                )
    return scenarios
