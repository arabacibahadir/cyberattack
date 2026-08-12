from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from gridimpair.contracts import ControlAction
from gridimpair.sim.controller import SafeModeController
from gridimpair.sim.plant import PowerFlowPlant
from gridimpair.sim.profiles import make_profile
from gridimpair.sim.schedule import freeze_remote_schedule


def test_controller_prioritizes_volt_var_before_curtailment_and_shedding() -> None:
    controller = SafeModeController()
    moderate = controller.act(
        {
            "vm_pu": [1.07],
            "loading_percent": [50],
            "remote_p_mw": [0.2],
            "local_mode": True,
        },
        None,
    )
    severe = controller.act(
        {
            "vm_pu": [0.88, 1.12],
            "loading_percent": [130],
            "remote_p_mw": [0.2],
            "local_mode": True,
        },
        None,
    )
    assert moderate.q_mvar and moderate.shed_fraction == 0.0
    assert severe.p_mw and severe.shed_fraction > 0.0


def test_controller_preserves_delivered_remote_setpoints_when_safe() -> None:
    action = SafeModeController().act(
        {
            "vm_pu": [1.0],
            "loading_percent": [50],
            "remote_p_mw": [0.2, 0.3],
        },
        None,
    )
    assert action.p_mw == (0.2, 0.3)
    assert not action.safe_mode


def test_three_step_recycled_powerflow_matches_fresh_solver() -> None:
    pytest.importorskip("pandapower")
    profile = make_profile("smooth", seed=11, steps=3)
    fresh = PowerFlowPlant(recycle=False)
    recycled = PowerFlowPlant(recycle=True)
    action = ControlAction()
    fresh_states = [fresh.step(action, profile.at(i)) for i in range(3)]
    recycled_states = [recycled.step(action, profile.at(i)) for i in range(3)]
    np.testing.assert_allclose(
        [s.vm_pu for s in fresh_states],
        [s.vm_pu for s in recycled_states],
        atol=1e-7,
        rtol=0,
    )
    np.testing.assert_allclose(
        [s.loading_percent for s in fresh_states],
        [s.loading_percent for s in recycled_states],
        atol=1e-5,
        rtol=0,
    )


def test_frozen_schedule_includes_bess_endpoint() -> None:
    network = SimpleNamespace(
        sgen=pd.DataFrame({"p_mw": [0.4, 0.2]}),
        storage=pd.DataFrame({"p_mw": [0.0], "max_e_mwh": [1.0]}),
    )
    schedule = freeze_remote_schedule(
        make_profile("balanced", seed=2, steps=8), network, seed=3
    )
    assert schedule.endpoint_ids == ("sgen-0", "sgen-1", "storage-0")
    storage_schedule = schedule.p_setpoint[:, -1]
    assert storage_schedule.min() < 0 < storage_schedule.max()


def test_profile_uses_physical_time_and_has_bounded_one_step_ramps() -> None:
    profile = make_profile(
        "balanced",
        seed=2,
        steps=900,
        step_seconds=1.0,
    )
    assert np.max(np.abs(np.diff(profile.load_scale))) < 0.02
    assert np.max(np.abs(np.diff(profile.pv_scale))) < 0.02
    assert abs(profile.pv_scale[-1] - profile.pv_scale[0]) < 0.20


def test_schedule_seed_controls_reproducible_active_dispatch() -> None:
    network = SimpleNamespace(
        sgen=pd.DataFrame(
            {"p_mw": [0.4, 0.2], "sn_mva": [0.5, 0.25]}
        ),
        storage=pd.DataFrame({"p_mw": [0.0], "max_e_mwh": [1.0]}),
    )
    profile = make_profile(
        "balanced", seed=2, steps=120, step_seconds=1.0
    )
    first = freeze_remote_schedule(profile, network, seed=3)
    replay = freeze_remote_schedule(profile, network, seed=3)
    different = freeze_remote_schedule(profile, network, seed=4)
    np.testing.assert_array_equal(first.p_setpoint, replay.p_setpoint)
    assert not np.array_equal(first.p_setpoint, different.p_setpoint)


def test_plant_clips_reactive_power_to_each_inverter_capability() -> None:
    plant = PowerFlowPlant(recycle=False)
    sgen_count = len(plant.net.sgen)
    action = ControlAction(
        p_mw=tuple(0.0 for _ in range(sgen_count)),
        q_mvar=tuple(99.0 for _ in range(sgen_count)),
    )
    plant.step(action, {"load_scale": 1.0, "pv_scale": 1.0, "wind_scale": 1.0})
    apparent = plant.net.sgen.sn_mva.to_numpy(float)
    assert np.all(np.abs(plant.net.sgen.q_mvar.to_numpy(float)) <= apparent + 1e-12)


def test_storage_soc_uses_configured_physical_step_seconds() -> None:
    one_second = PowerFlowPlant(recycle=False, step_seconds=1.0)
    one_minute = PowerFlowPlant(recycle=False, step_seconds=60.0)
    sgen_count = len(one_second.net.sgen)
    storage_count = len(one_second.net.storage)
    action = ControlAction(
        p_mw=(
            *tuple(0.0 for _ in range(sgen_count)),
            *tuple(0.01 for _ in range(storage_count)),
        )
    )
    initial = one_second._storage_soc.copy()
    state_second = one_second.step(
        action, {"load_scale": 1.0, "pv_scale": 1.0, "wind_scale": 1.0}
    )
    state_minute = one_minute.step(
        action, {"load_scale": 1.0, "pv_scale": 1.0, "wind_scale": 1.0}
    )
    delta_second = np.asarray(state_second.soc) - initial
    delta_minute = np.asarray(state_minute.soc) - initial
    np.testing.assert_allclose(delta_minute, 60 * delta_second, rtol=1e-8, atol=1e-10)


def test_ieee33_variant_contains_der_and_solves_one_step() -> None:
    plant = PowerFlowPlant(network_name="ieee33_der", recycle=False)
    assert len(plant.net.sgen) >= 6
    assert len(plant.net.storage) >= 1
    state = plant.step(
        ControlAction(),
        {"load_scale": 1.0, "pv_scale": 1.0, "wind_scale": 1.0},
    )
    assert state.converged
    assert len(state.vm_pu) == 33
