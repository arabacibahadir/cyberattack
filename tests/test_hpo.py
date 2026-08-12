from __future__ import annotations

from gridimpair.contracts import stable_hash
from gridimpair.train.hpo import make_lhs_trials


SPACE = {
    "learning_rate": {"type": "log_float", "low": 1e-4, "high": 1e-2},
    "hidden": {"type": "int", "low": 8, "high": 32},
    "pooling": {"type": "choice", "values": ["mean", "attention"]},
}


def test_lhs_trials_are_deterministic_and_inside_space() -> None:
    first = make_lhs_trials(SPACE, n=12, seed=2026)
    second = make_lhs_trials(SPACE, n=12, seed=2026)
    assert stable_hash(first) == stable_hash(second)
    assert len(first) == 12
    assert all(1e-4 <= trial.values["learning_rate"] <= 1e-2 for trial in first)
    assert all(8 <= trial.values["hidden"] <= 32 for trial in first)
    assert {trial.values["pooling"] for trial in first} <= {"mean", "attention"}


def test_lhs_rejects_empty_or_unknown_dimension() -> None:
    try:
        make_lhs_trials({"x": {"type": "mystery"}}, n=2, seed=1)
    except ValueError as exc:
        assert "mystery" in str(exc)
    else:
        raise AssertionError("unknown dimension type was accepted")
