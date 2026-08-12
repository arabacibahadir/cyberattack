from __future__ import annotations

from gridimpair.sim.robustness import generate_robustness_matrix


def test_robustness_matrix_is_paired_and_out_of_distribution() -> None:
    matrix = generate_robustness_matrix(base_seed=10)
    assert len(matrix) == 36
    assert {item.impairment_kind for item in matrix} == {"burst", "jitter"}
    pairs: dict[str, set[str]] = {}
    for item in matrix:
        pairs.setdefault(item.config.pair_id, set()).add(item.config.status)
        assert item.config.split == "robustness"
    assert all(statuses == {"clean", "impaired"} for statuses in pairs.values())
