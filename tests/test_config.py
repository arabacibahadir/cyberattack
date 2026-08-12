from __future__ import annotations

from pathlib import Path

import pytest

from gridimpair.config import load_config


def _write_config(root: Path, workers: int = 4) -> Path:
    config = root / "config.toml"
    config.write_text(
        "\n".join(
            [
                "[paths]",
                'data_root = "dataset"',
                'artifact_root = "artifacts"',
                "",
                "[runtime]",
                f"simulation_workers = {workers}",
                'precision = "auto"',
                "",
                "[simulation]",
                "steps = 900",
                "",
                "[training]",
                "max_epochs = 80",
                "min_epochs = 12",
                "patience = 8",
            ]
        ),
        encoding="utf-8",
    )
    return config


def test_config_resolves_paths_relative_to_config_file(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    assert config.paths.data_root == (tmp_path / "dataset").resolve()
    assert config.paths.artifact_root == (tmp_path / "artifacts").resolve()
    assert config.runtime.simulation_workers == 4


def test_worker_count_cannot_exceed_local_limit(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match="simulation_workers must be between 1 and 4"
    ):
        load_config(_write_config(tmp_path, workers=5))


def test_invalid_precision_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace('precision = "auto"', 'precision = "fp8"')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="precision"):
        load_config(path)


def test_orchestration_keys_remain_in_raw_config_without_breaking_dataclasses(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('\nhpo_space = "hpo_space.json"\nhpo_trials = 24\n')
    config = load_config(path)
    assert config.training.max_epochs == 80
    assert config.raw["training"]["hpo_trials"] == 24


def test_simulation_rejects_unknown_network_and_nonpositive_step_seconds(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "steps = 900",
        'steps = 900\nnetwork = "unknown"\nstep_seconds = 0',
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="network"):
        load_config(path)
