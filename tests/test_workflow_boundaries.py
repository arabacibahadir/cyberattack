from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gridimpair.config import TrainingConfig
from gridimpair.workflow import (
    CYBER_CHANNELS,
    HYBRID_CHANNELS,
    PHYSICAL_CHANNELS,
    _archive_training_run,
    _execution_training_plan,
    _family_base_name,
    _family_feature_tier,
    _prepare_feature_tiers,
    _requires_locked_access,
    _select_named_channels,
    _training_run_hash,
    discover_training_records,
)


def _write_trace(root: Path, scenario_id: str, split: str, status: str) -> None:
    run = root / scenario_id
    run.mkdir(parents=True)
    np.save(run / "trace.npy", np.ones((4, 3), dtype=np.float32))
    (run / "public_manifest.json").write_text(
        json.dumps(
            {
                "scenario_id": scenario_id,
                "pair_id": scenario_id,
                "split": split,
                "status": status,
            }
        ),
        encoding="utf-8",
    )


def test_training_discovery_never_opens_locked_trace(tmp_path: Path) -> None:
    _write_trace(tmp_path, "development", "development", "clean")
    _write_trace(tmp_path, "locked", "locked_test", "restricted")
    records = discover_training_records(tmp_path)
    assert [record.scenario_id for record in records] == ["development"]
    assert records[0].label == 0


def test_limit_creates_bounded_smoke_training_plan() -> None:
    config, seeds, trials = _execution_training_plan(
        TrainingConfig(max_epochs=80, min_epochs=12, patience=8),
        configured_seeds=[2026, 2027, 2028],
        configured_trials=24,
        limit=2,
    )
    assert config.max_epochs == 2
    assert config.min_epochs == 1
    assert config.patience == 1
    assert seeds == (2026,)
    assert trials == 2


def test_candidate_tensor_selects_only_preregistered_physical_channels() -> None:
    columns = ["message_age_max", *PHYSICAL_CHANNELS, "delivery_fraction"]
    values = np.arange(2 * len(columns) * 4).reshape(2, len(columns), 4)
    selected = _select_named_channels(values, columns, PHYSICAL_CHANNELS)
    assert selected.shape == (2, len(PHYSICAL_CHANNELS), 4)
    assert np.array_equal(selected, values[:, 1 : 1 + len(PHYSICAL_CHANNELS)])
    assert "message_age_max" not in PHYSICAL_CHANNELS
    assert "safe_mode" not in PHYSICAL_CHANNELS


def test_observability_tiers_are_explicit_and_nonleaking() -> None:
    assert set(PHYSICAL_CHANNELS).isdisjoint(CYBER_CHANNELS)
    assert HYBRID_CHANNELS == (*PHYSICAL_CHANNELS, *CYBER_CHANNELS)
    assert "message_age_max" in CYBER_CHANNELS
    assert "delivery_fraction" in CYBER_CHANNELS
    assert "obs_vm_p50" in PHYSICAL_CHANNELS
    assert "obs_loading_p95" in PHYSICAL_CHANNELS


def test_neural_family_names_map_to_architecture_and_observability_tier() -> None:
    configured = {
        "candidate": "hybrid",
        "candidate_physical": "physical",
        "tcn_cyber": "cyber",
        "tcn_hybrid": "hybrid",
    }
    assert _family_base_name("candidate_physical") == "candidate"
    assert _family_base_name("tcn_cyber") == "tcn"
    assert _family_feature_tier("candidate", configured) == "hybrid"
    assert _family_feature_tier("candidate_physical", configured) == "physical"
    assert _family_feature_tier("tcn_cyber", configured) == "cyber"


def test_feature_tiers_normalize_on_development_and_fit_residuals_on_clean_only() -> None:
    columns = [*PHYSICAL_CHANNELS, *CYBER_CHANNELS]
    raw = np.arange(4 * len(columns) * 6, dtype=np.float32).reshape(
        4, len(columns), 6
    )
    tiers, normalizations, residuals, ridges = _prepare_feature_tiers(
        raw,
        columns,
        development_mask=np.asarray([True, True, False, False]),
        labels=np.asarray([0, 1, 0, 1]),
    )
    assert tiers["physical"].shape == (4, len(PHYSICAL_CHANNELS), 6)
    assert tiers["cyber"].shape == (4, len(CYBER_CHANNELS), 6)
    assert tiers["hybrid"].shape == (4, len(HYBRID_CHANNELS), 6)
    assert len(normalizations["hybrid"]["mean"]) == len(HYBRID_CHANNELS)
    assert residuals["physical"].shape == tiers["physical"].shape
    assert ridges["physical"].training_rows_ == 2


def test_training_run_hash_changes_with_input_artifact() -> None:
    first = _training_run_hash(
        training_config={"max_epochs": 2},
        input_artifacts=[{"scenario_id": "s1", "artifact_hash": "a"}],
        limit=2,
        implementation_hash="code",
    )
    second = _training_run_hash(
        training_config={"max_epochs": 2},
        input_artifacts=[{"scenario_id": "s1", "artifact_hash": "b"}],
        limit=2,
        implementation_hash="code",
    )
    assert first != second


def test_incompatible_completed_training_run_is_archived(tmp_path: Path) -> None:
    run_dir = tmp_path / "training"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "best.pt").write_bytes(b"smoke")
    archive = _archive_training_run(run_dir, "oldhash")
    assert (archive / "best.pt").read_bytes() == b"smoke"
    assert run_dir.is_dir()
    assert not (run_dir / "best.pt").exists()


def test_limited_smoke_with_no_locked_records_is_open_only() -> None:
    assert not _requires_locked_access(
        locked_record_count=0, limit=2
    )
    assert _requires_locked_access(
        locked_record_count=2, limit=2
    )
    assert _requires_locked_access(
        locked_record_count=0, limit=None
    )
