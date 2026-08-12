from __future__ import annotations

import pytest

from gridimpair.sim.controller import SafeModeController
from gridimpair.sim.policies import local_mode_enabled, policy_alarm


def test_fixed_closed_loop_policy_semantics() -> None:
    intervals = [(3, 5)]
    assert not policy_alarm("P1", 3, message_age=9, impairment_intervals=intervals)
    assert policy_alarm("P2", 3, message_age=3, impairment_intervals=intervals)
    assert policy_alarm("P3", 3, message_age=0, impairment_intervals=intervals)
    assert not local_mode_enabled("P0", detector_alarm=False)
    assert local_mode_enabled("P1", detector_alarm=False)
    assert not local_mode_enabled("candidate", detector_alarm=False)
    assert local_mode_enabled("candidate", detector_alarm=True)


def test_detector_policy_requires_frozen_causal_schedule() -> None:
    with pytest.raises(ValueError, match="frozen"):
        policy_alarm(
            "candidate",
            0,
            message_age=0,
            impairment_intervals=[],
        )
    schedule = [False, True, False]
    assert policy_alarm(
        "candidate",
        1,
        message_age=0,
        impairment_intervals=[],
        frozen_alarm_schedule=schedule,
    )


def test_alarm_uses_local_voltage_support_without_zeroing_all_der_power() -> None:
    controller = SafeModeController()
    nominal = controller.act(
        {
            "vm_pu": (0.99, 1.01),
            "loading_percent": (80.0,),
            "remote_p_mw": (0.4, 0.3),
            "local_mode": True,
        },
        None,
    )
    assert nominal.p_mw == (0.4, 0.3)
    assert nominal.q_mvar == ()
    stressed = controller.act(
        {
            "vm_pu": (1.0, 1.10),
            "loading_percent": (80.0,),
            "remote_p_mw": (0.4, 0.3),
            "local_mode": True,
        },
        None,
    )
    assert 0 < stressed.p_mw[0] < 0.4
    assert 0 < stressed.p_mw[1] < 0.3
    assert all(value < 0 for value in stressed.q_mvar)
