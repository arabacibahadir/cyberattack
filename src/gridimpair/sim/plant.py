from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from gridimpair.contracts import ControlAction, PowerState


def _load_pandapower():
    try:
        import pandapower as pp
        from pandapower.networks import case33bw, create_cigre_network_mv
    except ImportError as exc:
        raise RuntimeError(
            "pandapower is required for PowerFlowPlant; run setup.ps1 first"
        ) from exc
    return pp, create_cigre_network_mv, case33bw


def _make_ieee33_der(pp: Any, case33bw: Any) -> Any:
    net = case33bw()
    for bus, capacity in zip(
        (5, 10, 16, 22, 27, 31),
        (0.30, 0.35, 0.40, 0.35, 0.45, 0.50),
        strict=True,
    ):
        pp.create_sgen(
            net,
            bus=bus,
            p_mw=0.6 * capacity,
            q_mvar=0.0,
            sn_mva=capacity,
            name=f"DER-{bus}",
            type="PV",
        )
    for bus in (14, 30):
        pp.create_storage(
            net,
            bus=bus,
            p_mw=0.0,
            max_e_mwh=0.5,
            soc_percent=50.0,
            min_p_mw=-0.20,
            max_p_mw=0.20,
            sn_mva=0.25,
            name=f"BESS-{bus}",
        )
    return net


class PowerFlowPlant:
    def __init__(
        self,
        *,
        network_name: str = "cigre_mv",
        recycle: bool = False,
        storage_duration_hours: float = 2.0,
        step_seconds: float = 1.0,
    ) -> None:
        self._pp, create_network, case33bw = _load_pandapower()
        if network_name == "cigre_mv":
            self.net = create_network(with_der="all")
        elif network_name == "ieee33_der":
            self.net = _make_ieee33_der(self._pp, case33bw)
        else:
            raise ValueError("network_name must be cigre_mv or ieee33_der")
        if storage_duration_hours <= 0:
            raise ValueError("storage_duration_hours must be positive")
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        self.network_name = network_name
        self.step_seconds = float(step_seconds)
        self.recycle = recycle
        self.logical_time = 0
        self._solved_once = False
        self._load_p = self.net.load.p_mw.to_numpy(copy=True)
        self._load_q = self.net.load.q_mvar.to_numpy(copy=True)
        self._sgen_p = self.net.sgen.p_mw.to_numpy(copy=True)
        self._sgen_q = self.net.sgen.q_mvar.to_numpy(copy=True)
        configured_sgen_sn = (
            self.net.sgen.sn_mva.fillna(0.0).to_numpy(dtype=float, copy=True)
            if "sn_mva" in self.net.sgen
            else np.zeros(len(self.net.sgen), dtype=float)
        )
        self._sgen_sn = np.maximum.reduce(
            (
                configured_sgen_sn,
                np.abs(self._sgen_p) * 1.10,
                np.full(len(self.net.sgen), 0.01),
            )
        )
        if len(self.net.sgen):
            self.net.sgen.loc[:, "sn_mva"] = self._sgen_sn
        self._storage_p = (
            self.net.storage.p_mw.to_numpy(copy=True)
            if len(self.net.storage)
            else np.empty(0)
        )
        if len(self.net.storage):
            configured_energy = self.net.storage.max_e_mwh.to_numpy(
                dtype=float, copy=True
            )
            derived_energy = np.maximum(
                np.abs(self._storage_p) * storage_duration_hours, 0.1
            )
            configured_energy = np.where(
                np.isfinite(configured_energy) & (configured_energy > 0),
                configured_energy,
                derived_energy,
            )
            self.net.storage.loc[:, "max_e_mwh"] = configured_energy
            self.net.storage.loc[:, "soc_percent"] = (
                self.net.storage.soc_percent.fillna(50.0)
            )
        self._storage_soc = (
            self.net.storage.soc_percent.fillna(50.0).to_numpy(copy=True)
            if len(self.net.storage)
            else np.empty(0)
        )
        self._storage_energy = (
            np.maximum(
                self.net.storage.max_e_mwh.to_numpy(dtype=float, copy=True),
                1e-6,
            )
            if len(self.net.storage)
            else np.empty(0)
        )

    def step(
        self, action: ControlAction, exogenous: Mapping[str, float]
    ) -> PowerState:
        load_scale = float(exogenous.get("load_scale", 1.0))
        pv_scale = float(exogenous.get("pv_scale", 1.0))
        wind_scale = float(exogenous.get("wind_scale", 1.0))
        self.net.load.loc[:, "p_mw"] = self._load_p * load_scale * (1 - action.shed_fraction)
        self.net.load.loc[:, "q_mvar"] = self._load_q * load_scale * (1 - action.shed_fraction)
        if len(self.net.sgen):
            scales = np.full(len(self.net.sgen), wind_scale)
            scales[: max(1, len(scales) // 2)] = pv_scale
            p = self._sgen_p * scales
            if action.p_mw:
                count = min(len(p), len(action.p_mw))
                if len(action.p_mw) == 1 and len(p) != 1:
                    p *= float(action.p_mw[0])
                else:
                    p[:count] = np.asarray(action.p_mw[:count], dtype=float)
            p = np.clip(p, 0.0, self._sgen_sn)
            q = self._sgen_q.copy()
            if action.q_mvar:
                count = min(len(q), len(action.q_mvar))
                if len(action.q_mvar) == 1:
                    q += float(action.q_mvar[0]) / max(1, len(q))
                else:
                    q[:count] += np.asarray(action.q_mvar[:count], dtype=float)
            q_limit = np.sqrt(np.maximum(self._sgen_sn**2 - p**2, 0.0))
            q = np.clip(q, -q_limit, q_limit)
            if "min_q_mvar" in self.net.sgen:
                minimum_q = self.net.sgen.min_q_mvar.to_numpy(dtype=float)
                q = np.maximum(q, np.where(np.isfinite(minimum_q), minimum_q, -q_limit))
            if "max_q_mvar" in self.net.sgen:
                maximum_q = self.net.sgen.max_q_mvar.to_numpy(dtype=float)
                q = np.minimum(q, np.where(np.isfinite(maximum_q), maximum_q, q_limit))
            self.net.sgen.loc[:, "p_mw"] = p
            self.net.sgen.loc[:, "q_mvar"] = q
        if len(self.net.storage):
            storage_p = self._storage_p.copy()
            offset = len(self.net.sgen)
            if len(action.p_mw) > offset:
                count = min(
                    len(storage_p), len(action.p_mw) - offset
                )
                storage_p[:count] = np.asarray(
                    action.p_mw[offset : offset + count], dtype=float
                )
            proposed_soc = self._storage_soc + (
                storage_p
                / self._storage_energy
                * (self.step_seconds / 3600)
                * 100
            )
            lower_block = (proposed_soc <= 0) & (storage_p < 0)
            upper_block = (proposed_soc >= 100) & (storage_p > 0)
            storage_p[lower_block | upper_block] = 0.0
            self._storage_soc = np.clip(
                self._storage_soc
                + storage_p
                / self._storage_energy
                * (self.step_seconds / 3600)
                * 100,
                0,
                100,
            )
            self.net.storage.loc[:, "p_mw"] = storage_p
            self.net.storage.loc[:, "soc_percent"] = self._storage_soc
        run_kwargs: dict[str, Any] = {
            "algorithm": "nr",
            "init": "results" if self._solved_once else "auto",
            "numba": False,
        }
        if self.recycle and self._solved_once:
            run_kwargs["recycle"] = {"bus_pq": True, "trafo": True, "gen": True}
        self._pp.runpp(self.net, **run_kwargs)
        self._solved_once = True
        loading_parts = []
        for table_name in ("res_line", "res_trafo"):
            table = getattr(self.net, table_name)
            if len(table) and "loading_percent" in table:
                loading_parts.extend(table.loading_percent.to_numpy(float).tolist())
        state = PowerState(
            logical_time=self.logical_time,
            vm_pu=tuple(self.net.res_bus.vm_pu.to_numpy(float).tolist()),
            loading_percent=tuple(loading_parts),
            p_mw=tuple(self.net.res_bus.p_mw.to_numpy(float).tolist()),
            q_mvar=tuple(self.net.res_bus.q_mvar.to_numpy(float).tolist()),
            i_ka=tuple(self.net.res_line.i_ka.to_numpy(float).tolist()),
            load_p_mw=float(self.net.load.p_mw.sum()),
            generation_p_mw=float(self.net.sgen.p_mw.sum()),
            soc=tuple(self._storage_soc.tolist()),
            converged=bool(self.net.converged),
        )
        self.logical_time += 1
        return state
