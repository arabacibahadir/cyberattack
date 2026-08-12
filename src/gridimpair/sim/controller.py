from __future__ import annotations

from typing import Any, Mapping

from gridimpair.contracts import ControlAction, PowerState


class SafeModeController:
    def __init__(
        self,
        *,
        volt_var_threshold: float = 1.05,
        curtail_threshold: float = 1.09,
        shedding_loading_threshold: float = 120.0,
    ) -> None:
        self.volt_var_threshold = volt_var_threshold
        self.curtail_threshold = curtail_threshold
        self.shedding_loading_threshold = shedding_loading_threshold

    def act(
        self, observation: Mapping[str, Any], state: PowerState | None
    ) -> ControlAction:
        voltages = tuple(float(value) for value in observation.get("vm_pu", (1.0,)))
        loading = tuple(
            float(value) for value in observation.get("loading_percent", (0.0,))
        )
        maximum_voltage = max(voltages, default=1.0)
        maximum_loading = max(loading, default=0.0)
        remote_p = tuple(
            float(value) for value in observation.get("remote_p_mw", ())
        )
        local_mode = bool(observation.get("local_mode", False))
        q = ()
        p = remote_p
        shed = 0.0
        if local_mode and maximum_voltage > self.volt_var_threshold:
            q_value = -min(0.5, 4.0 * (maximum_voltage - 1.0))
            q = (q_value,) * max(1, len(remote_p))
        elif local_mode and min(voltages, default=1.0) < 2.0 - self.volt_var_threshold:
            minimum_voltage = min(voltages, default=1.0)
            q_value = min(0.5, 4.0 * (1.0 - minimum_voltage))
            q = (q_value,) * max(1, len(remote_p))
        if local_mode and maximum_voltage > self.curtail_threshold:
            curtailment = max(
                0.0, 1.0 - 3.0 * (maximum_voltage - self.curtail_threshold)
            )
            p = (
                tuple(value * curtailment for value in remote_p)
                if remote_p
                else (curtailment,)
            )
        minimum_voltage = min(voltages, default=1.0)
        if local_mode and (
            minimum_voltage < 0.90
            or maximum_loading > self.shedding_loading_threshold
        ):
            voltage_need = max(0.0, 0.90 - minimum_voltage)
            loading_need = max(
                0.0,
                (maximum_loading - self.shedding_loading_threshold) / 200,
            )
            shed = min(0.25, max(2.0 * voltage_need, loading_need))
        return ControlAction(
            p_mw=p,
            q_mvar=q,
            shed_fraction=shed,
            safe_mode=local_mode,
        )
