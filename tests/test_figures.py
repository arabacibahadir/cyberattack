from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from gridimpair.reporting.figures import render_figure_manifest
from gridimpair.reporting.sources import (
    _absolute_endpoints_source,
    prepare_figure_sources,
)


def test_figure_renderer_writes_three_formats_source_and_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame({"epoch": [1, 2, 3], "loss": [0.8, 0.5, 0.3]}).to_csv(
        source, index=False
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "id": "training-curve",
                        "title": "Training curve",
                        "kind": "line",
                        "source": str(source),
                        "x": "epoch",
                        "y": "loss",
                        "section": "smoke",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    outputs = render_figure_manifest(manifest, tmp_path / "figures")
    output = outputs[0]
    assert output.svg.exists() and output.pdf.exists() and output.png.exists()
    assert output.source_table.exists() and output.provenance.exists()
    with Image.open(output.png) as image:
        dpi = image.info["dpi"]
    assert dpi[0] == pytest.approx(600, rel=0.01)


def test_figure_manifest_rejects_point_adjusted_metric(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame({"x": [1], "point_adjusted_f1": [1.0]}).to_csv(source, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "id": "bad",
                        "title": "Bad metric",
                        "kind": "bar",
                        "source": str(source),
                        "x": "x",
                        "y": "point_adjusted_f1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="point-adjusted"):
        render_figure_manifest(manifest, tmp_path / "out")


def test_prepare_sources_uses_real_artifacts_and_skips_unavailable(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    audit = artifacts / "audit"
    training = artifacts / "training" / "final" / "candidate" / "seed-2026"
    results = artifacts / "results"
    audit.mkdir(parents=True)
    training.mkdir(parents=True)
    results.mkdir(parents=True)
    pd.DataFrame(
        {
            "split_strategy": ["random_window", "grouped_file"],
            "auprc": [0.9, 0.7],
        }
    ).to_csv(audit / "leakage_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "epoch": [1, 2],
            "train_loss": [0.8, 0.6],
            "validation_loss": [0.9, 0.7],
            "learning_rate": [1e-3, 0.0],
            "is_best": [1, 1],
        }
    ).to_csv(training / "metrics.csv", index=False)

    source_dir = artifacts / "figure_sources"
    generated_manifest = artifacts / "figures" / "render_manifest.json"
    status = prepare_figure_sources(
        artifact_root=artifacts,
        configured_manifest=Path("configs/figure_manifest.json"),
        source_dir=source_dir,
        generated_manifest=generated_manifest,
        smoke=True,
    )

    assert (source_dir / "fig02-split-leakage.csv").exists()
    curves = pd.read_csv(source_dir / "figs17-training-curves.csv")
    assert curves["model"].tolist() == ["candidate", "candidate"]
    assert curves["run_phase"].tolist() == ["final", "final"]
    assert curves["seed_count"].tolist() == [1, 1]
    assert status.loc[status["id"] == "fig10-closed-loop", "status"].item() == "missing"
    rendered = json.loads(generated_manifest.read_text(encoding="utf-8"))
    rendered_ids = {entry["id"] for entry in rendered["figures"]}
    assert "fig02-split-leakage" in rendered_ids
    assert "figs17-training-curves" in rendered_ids
    assert "fig10-closed-loop" not in rendered_ids


def test_absolute_endpoint_source_aggregates_per_run_rows(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame(
        {
            "pair_id": ["p1", "p2", "p1", "p2"],
            "status": ["clean", "clean", "impaired", "impaired"],
            "policy": ["P0"] * 4,
            "endpoint": ["voltage_violation_seconds"] * 4,
            "value": [1.0, 3.0, 5.0, 7.0],
        }
    ).to_csv(results / "closed_loop_absolute_endpoints.csv", index=False)

    frame = _absolute_endpoints_source(tmp_path)

    assert frame is not None
    assert frame.to_dict("records") == [
        {"policy": "P0", "status": "clean", "endpoint_value": 2.0},
        {"policy": "P0", "status": "impaired", "endpoint_value": 6.0},
    ]
