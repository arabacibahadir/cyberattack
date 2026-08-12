from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from gridimpair.contracts import stable_hash


def export_result_table(
    frame: pd.DataFrame,
    output_stem: Path,
    *,
    sort_by: Sequence[str] = (),
    float_format: str = "%.4f",
) -> dict[str, str]:
    if frame.empty:
        raise ValueError("result table cannot be empty")
    table = frame.sort_values(list(sort_by)) if sort_by else frame.copy()
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_stem.with_suffix(".csv")
    tex_path = output_stem.with_suffix(".tex")
    provenance_path = output_stem.with_suffix(".provenance.json")
    table.to_csv(csv_path, index=False, float_format=float_format)
    table.to_latex(tex_path, index=False, float_format=lambda value: float_format % value)
    hashable = (
        table.astype(object)
        .replace([np.inf, -np.inf], None)
        .where(pd.notna(table), None)
    )
    source_hash = stable_hash(hashable.to_dict(orient="records"))
    provenance_path.write_text(
        json.dumps(
            {
                "source_hash": source_hash,
                "rows": len(table),
                "columns": list(table.columns),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "csv": str(csv_path.resolve()),
        "latex": str(tex_path.resolve()),
        "provenance": str(provenance_path.resolve()),
        "source_hash": source_hash,
    }
