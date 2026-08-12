from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridimpair.evaluation.statistics import (
    gap_closure,
    holm_correction,
    paired_cluster_bootstrap,
    sign_flip_test,
)


def test_gap_closure_handles_headroom_and_no_headroom() -> None:
    valid = gap_closure(baseline=0.4, candidate=0.7, oracle=1.0)
    assert valid.status == "ok"
    assert valid.value == pytest.approx(0.5)
    blocked = gap_closure(baseline=1.0, candidate=1.0, oracle=1.0)
    assert blocked.status == "no_oracle_headroom"
    assert blocked.value is None


def test_paired_bootstrap_is_seed_reproducible_and_stratified() -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["a", "b", "c", "d"],
            "stratum": ["smooth-low", "smooth-low", "ramp-high", "ramp-high"],
            "candidate": [0.8, 0.7, 0.6, 0.9],
            "baseline": [0.5, 0.6, 0.4, 0.7],
        }
    )
    first = paired_cluster_bootstrap(
        frame,
        candidate_column="candidate",
        baseline_column="baseline",
        n_resamples=200,
        seed=17,
    )
    second = paired_cluster_bootstrap(
        frame,
        candidate_column="candidate",
        baseline_column="baseline",
        n_resamples=200,
        seed=17,
    )
    assert first == second
    assert first.observed == np.mean([0.3, 0.1, 0.2, 0.2])


def test_sign_flip_and_holm_are_deterministic() -> None:
    first = sign_flip_test([1.0, 2.0, 3.0], n_permutations=256, seed=9)
    second = sign_flip_test([1.0, 2.0, 3.0], n_permutations=256, seed=9)
    assert first == second
    corrected = holm_correction({"candidate-vs-tcn": 0.01, "candidate-vs-tail": 0.04})
    assert corrected["candidate-vs-tcn"] == 0.02
    assert corrected["candidate-vs-tail"] == 0.04
