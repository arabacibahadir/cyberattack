from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from gridimpair.contracts import stable_hash


def hash_unlock_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClosedLoopConfig:
    on_policy_retraining: bool = False
    test_time_adaptation: bool = False
    namespace: str = "primary"

    def __post_init__(self) -> None:
        if self.on_policy_retraining:
            raise ValueError("on-policy retraining is forbidden")
        if self.test_time_adaptation:
            raise ValueError("test-time adaptation is forbidden")
        if self.namespace not in {"primary", "robustness"}:
            raise ValueError("namespace must be primary or robustness")


@dataclass(frozen=True)
class ElectricalEndpointSummary:
    voltage_violation_seconds: float
    overload_duration_seconds: float
    overload_integral_percent_seconds: float
    curtailment_mwh: float
    unserved_energy_mwh: float
    safe_mode_fraction: float


def summarize_electrical_trace(
    frame: pd.DataFrame,
    *,
    step_seconds: float,
    voltage_lower: float = 0.95,
    voltage_upper: float = 1.05,
    loading_limit_percent: float = 100.0,
) -> ElectricalEndpointSummary:
    required = {
        "vm_min",
        "vm_max",
        "loading_max",
        "curtailed_mw",
        "shed_mw",
        "safe_mode",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"missing electrical trace columns: {sorted(missing)}")
    if frame.empty or step_seconds <= 0:
        raise ValueError("trace must be non-empty and step_seconds positive")
    violation = (frame["vm_min"] < voltage_lower) | (
        frame["vm_max"] > voltage_upper
    )
    overload_excess = (frame["loading_max"] - loading_limit_percent).clip(
        lower=0
    )
    step_hours = step_seconds / 3600
    return ElectricalEndpointSummary(
        voltage_violation_seconds=float(violation.sum() * step_seconds),
        overload_duration_seconds=float(
            (overload_excess > 0).sum() * step_seconds
        ),
        overload_integral_percent_seconds=float(
            overload_excess.sum() * step_seconds
        ),
        curtailment_mwh=float(frame["curtailed_mw"].clip(lower=0).sum() * step_hours),
        unserved_energy_mwh=float(frame["shed_mw"].clip(lower=0).sum() * step_hours),
        safe_mode_fraction=float(frame["safe_mode"].astype(bool).mean()),
    )


def paired_endpoint_delta(
    clean: pd.DataFrame,
    impaired: pd.DataFrame,
    *,
    step_seconds: float,
) -> dict[str, float]:
    clean_summary = asdict(
        summarize_electrical_trace(clean, step_seconds=step_seconds)
    )
    impaired_summary = asdict(
        summarize_electrical_trace(impaired, step_seconds=step_seconds)
    )
    return {
        name: float(impaired_summary[name] - clean_summary[name])
        for name in clean_summary
    }


def policy_regret_table(
    absolute_endpoints: pd.DataFrame,
    *,
    reference_policy: str,
) -> pd.DataFrame:
    required = {"pair_id", "status", "policy", "endpoint", "value"}
    if missing := required - set(absolute_endpoints.columns):
        raise ValueError(f"missing policy-regret columns: {sorted(missing)}")
    keys = ["pair_id", "status", "endpoint"]
    if absolute_endpoints.duplicated([*keys, "policy"]).any():
        raise ValueError("policy-regret input contains duplicate policy rows")
    reference = absolute_endpoints.loc[
        absolute_endpoints["policy"] == reference_policy,
        [*keys, "value"],
    ].rename(columns={"value": "reference_value"})
    if reference.empty:
        raise ValueError("reference policy is absent")
    merged = absolute_endpoints.merge(
        reference,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    if merged["reference_value"].isna().any():
        raise ValueError("reference policy is incomplete")
    merged["regret"] = merged["value"] - merged["reference_value"]
    return merged


class LockedManifestReader:
    def __init__(self, manifest_path: Path, guard_path: Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.guard_path = Path(guard_path)
        self.consumed_path = self.guard_path.with_suffix(
            self.guard_path.suffix + ".consumed"
        )
        self._unlocked = False

    def unlock(self, token: str, *, config_hash: str, split_hash: str) -> None:
        if self.consumed_path.exists():
            raise PermissionError("unlock token has already been consumed")
        specification = json.loads(self.guard_path.read_text(encoding="utf-8"))
        if hash_unlock_token(token) != specification["token_hash"]:
            raise PermissionError("invalid unlock token")
        if (
            config_hash != specification["config_hash"]
            or split_hash != specification["split_hash"]
        ):
            raise PermissionError("frozen config/split hash mismatch")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if stable_hash(manifest) != specification["manifest_hash"]:
            raise PermissionError("locked manifest hash mismatch")
        try:
            descriptor = os.open(
                self.consumed_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise PermissionError("unlock token has already been consumed") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"config_hash": config_hash, "split_hash": split_hash}))
        self._unlocked = True

    def read(self) -> dict[str, Any]:
        if not self._unlocked:
            raise PermissionError("locked manifest requires unlock")
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))


def validate_paired_results(
    frame: pd.DataFrame,
    *,
    expected_statuses: tuple[str, str] = ("clean", "impaired"),
) -> None:
    required = {"pair_id", "scenario_id", "status", "model"}
    if missing := required - set(frame.columns):
        raise ValueError(f"missing result columns: {sorted(missing)}")
    if frame.duplicated(["scenario_id", "model"]).any():
        raise ValueError("duplicate scenario/model result")
    expected = set(expected_statuses)
    for (pair_id, model), group in frame.groupby(["pair_id", "model"]):
        statuses = set(group["status"])
        if statuses != expected:
            raise ValueError(
                f"pair {pair_id!r}, model {model!r} has statuses {sorted(statuses)}"
            )


def select_finalist_modes(
    validation_results: pd.DataFrame,
    *,
    candidate_name: str,
    fixed_policies: tuple[str, ...] = ("P0", "P1", "P2", "P3"),
) -> list[str]:
    required = {"model", "validation_score"}
    if missing := required - set(validation_results.columns):
        raise ValueError(f"missing finalist columns: {sorted(missing)}")
    baselines = validation_results.loc[
        ~validation_results["model"].isin([candidate_name, *fixed_policies])
    ]
    if baselines.empty:
        raise ValueError("at least one non-proposed baseline is required")
    strongest = str(
        baselines.sort_values("validation_score", ascending=False).iloc[0]["model"]
    )
    return [*fixed_policies, strongest, candidate_name]
