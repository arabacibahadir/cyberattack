from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gridimpair.reporting.tables import export_result_table


TABLE_SOURCES = {
    "table01-open-bag-metrics": "results/open_metrics.csv",
    "table02-locked-bag-metrics": "results/locked_metrics.csv",
    "table03-open-temporal-metrics": "results/temporal_open_metrics.csv",
    "table04-locked-temporal-metrics": "results/temporal_locked_metrics.csv",
    "table05-paired-statistics": "results/paired_statistics_locked.csv",
    "table06-closed-loop-endpoints": "results/closed_loop_endpoints.csv",
    "table07-robustness": "results/robustness_metrics.csv",
    "table08-inference-latency": "results/inference_latency.csv",
}


def prepare_result_tables(
    artifact_root: Path, output_dir: Path
) -> list[dict[str, str]]:
    artifact_root = Path(artifact_root)
    output_dir = Path(output_dir)
    records: list[dict[str, str]] = []
    for table_id, relative in TABLE_SOURCES.items():
        source = artifact_root / relative
        if not source.exists():
            continue
        frame = pd.read_csv(source)
        if frame.empty:
            continue
        outputs = export_result_table(frame, output_dir / table_id)
        records.append({"table_id": table_id, "source": relative, **outputs})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table_manifest.json").write_text(
        json.dumps(records, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return records
