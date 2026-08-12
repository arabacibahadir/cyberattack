from __future__ import annotations

from collections.abc import Sequence


POLICY_NAMES = (
    "P0",
    "P1",
    "P2",
    "P3",
    "baseline",
    "candidate",
    "shadow",
    "watchdog",
)


def policy_alarm(
    policy_name: str,
    logical_time: int,
    *,
    message_age: float,
    impairment_intervals: Sequence[tuple[int, int]],
    frozen_alarm_schedule: Sequence[bool] | None = None,
) -> bool:
    if policy_name not in POLICY_NAMES:
        raise ValueError(f"unknown closed-loop policy {policy_name!r}")
    if logical_time < 0:
        raise ValueError("logical_time must be non-negative")
    if policy_name in {"P0", "P1", "shadow"}:
        return False
    if policy_name in {"P2", "watchdog"}:
        return message_age > 2
    if policy_name == "P3":
        return any(start <= logical_time < stop for start, stop in impairment_intervals)
    if frozen_alarm_schedule is None:
        raise ValueError(f"{policy_name} requires a frozen alarm schedule")
    if logical_time >= len(frozen_alarm_schedule):
        raise ValueError("frozen alarm schedule is shorter than the simulation")
    return bool(frozen_alarm_schedule[logical_time])


def local_mode_enabled(policy_name: str, *, detector_alarm: bool) -> bool:
    if policy_name not in POLICY_NAMES:
        raise ValueError(f"unknown closed-loop policy {policy_name!r}")
    if policy_name == "P1":
        return True
    if policy_name in {"P0", "shadow"}:
        return False
    return bool(detector_alarm)
