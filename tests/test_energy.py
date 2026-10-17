import math

from gridimpair import energy


def test_delta_reports_nonnegative_cpu_and_wall() -> None:
    start = energy.snapshot()
    sum(i * i for i in range(20_000))
    cost = energy.delta(start)
    assert cost["wall_seconds"] >= 0
    assert cost["cpu_seconds"] >= 0
    assert cost["cpu_energy_j_est"] >= 0
    gpu = cost["gpu_energy_j"]
    assert math.isnan(gpu) or gpu >= 0
