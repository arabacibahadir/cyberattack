from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gridimpair.data.windows import make_grouped_windows


def test_bag_never_crosses_split_boundary(tmp_path: Path) -> None:
    files = []
    for bag in ("bag_a", "bag_b"):
        path = tmp_path / f"{bag}.csv"
        pd.DataFrame({"x": range(20)}).to_csv(path, index=False)
        files.append(path)
    manifest = {"bag_a": "development", "bag_b": "locked_test"}
    index = make_grouped_windows(files, window=8, stride=4, split_manifest=manifest)
    assert index.groupby("bag_id")["split"].nunique().max() == 1
    assert set(index.columns) == {"file_id", "bag_id", "start", "stop", "split"}
    assert index["stop"].max() == 20


def test_conflicting_file_and_bag_split_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bag_a.csv"
    pd.DataFrame({"x": range(8)}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="multiple splits"):
        make_grouped_windows(
            [path, path],
            window=4,
            stride=2,
            split_manifest={"bag_a": ["development", "locked_test"]},
        )
