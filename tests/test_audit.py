from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gridimpair.data.audit import audit_source_dataset


def test_audit_reports_quality_without_loading_unrelated_files(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "audit"
    data.mkdir()
    pd.DataFrame(
        {
            "time": [0, 1, 2, 3],
            "constant": [7.0, 7.0, 7.0, 7.0],
            "signal": [0.0, 1.0, np.nan, np.inf],
        }
    ).to_csv(data / "bag_a.csv", index=False)
    (data / "note.txt").write_text("not part of the dataset", encoding="utf-8")

    summary = audit_source_dataset(data, output)

    assert summary.csv_count == 1
    assert summary.total_nan == 1
    assert summary.total_inf == 1
    assert summary.constant_channels == 1
    manifest = json.loads((output / "source_manifest.json").read_text(encoding="utf-8"))
    assert [item["path"] for item in manifest["files"]] == ["bag_a.csv"]
    assert len(manifest["files"][0]["sha256"]) == 64
    missingness = pd.read_csv(output / "channel_missingness.csv")
    assert missingness.loc[missingness["channel"] == "signal", "nan_count"].item() == 1
    intervals = pd.read_csv(output / "time_grid_intervals.csv")
    assert intervals["interval"].tolist() == [1.0, 1.0, 1.0]


def test_audit_detects_irregular_time_grid(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"time": [0, 1, 3], "signal": [1, 2, 3]}).to_csv(
        data / "irregular.csv", index=False
    )
    summary = audit_source_dataset(data, tmp_path / "out")
    assert summary.irregular_time_files == 1


def test_audit_computes_real_random_vs_grouped_leakage_diagnostic(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(3)
    for file_id, attacked, offset in (
        ("3(a)", 0, 0.0),
        ("3(b)", 0, 2.0),
        ("3(f)", 1, 1.0),
        ("3(g)", 1, 3.0),
    ):
        signal = rng.normal(attacked * 0.5 + offset, 0.1, 32)
        pd.DataFrame(
            {
                "Time": np.arange(32),
                "ILOAD": signal,
                "PBset": np.full(32, offset),
            }
        ).to_csv(data / f"{file_id}.csv", index=False)
    audit_source_dataset(data, tmp_path / "out", window=8, stride=4)
    leakage = pd.read_csv(tmp_path / "out" / "leakage_diagnostics.csv")
    assert set(leakage["split_strategy"]) == {
        "random_window",
        "grouped_file",
    }
    assert leakage["auprc"].notna().all()
    assert (tmp_path / "out" / "univariate_channel_diagnostics.csv").exists()
    assert (tmp_path / "out" / "fingerprint_diagnostics.csv").exists()
    embedding = pd.read_csv(tmp_path / "out" / "fingerprint_embedding.csv")
    assert {"component_1", "component_2", "file_id"} <= set(embedding)
