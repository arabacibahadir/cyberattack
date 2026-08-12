from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from gridimpair.runner import _pilot_gate
from gridimpair.sim.scenario import generate_q1_v2_pilot_matrix


def test_pilot_gate_requires_convergence_finite_values_and_pair_signal(tmp_path) -> None:
    base = generate_q1_v2_pilot_matrix(base_seed=7)[:2]
    scenarios = [
        replace(base[0], scenario_id="pair-a", pair_id="pair"),
        replace(base[1], scenario_id="pair-b", pair_id="pair"),
    ]
    columns = [
        "vm_min",
        "vm_max",
        "obs_vm_p50",
        "obs_loading_p95",
        "delivery_fraction",
        "message_age_max",
    ]
    for index, scenario in enumerate(scenarios):
        target = tmp_path / scenario.scenario_id
        target.mkdir()
        trace = np.ones((3, len(columns)), dtype=float)
        trace[:, columns.index("delivery_fraction")] -= index * 0.5
        np.save(target / "trace.npy", trace)
        (target / "trace_columns.json").write_text(
            json.dumps({"columns": columns}), encoding="utf-8"
        )
        (target / "summary.json").write_text(
            json.dumps({"steps": 3, "converged_steps": 3}),
            encoding="utf-8",
        )

    result = _pilot_gate(tmp_path, scenarios)

    assert result["decision"] == "GO"
    assert result["detectable_pair_fraction"] == 1.0
