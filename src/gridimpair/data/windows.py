from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


WINDOW_COLUMNS = ["file_id", "bag_id", "start", "stop", "split"]


def _resolve_split(value: str | Sequence[str], bag_id: str) -> str:
    if isinstance(value, str):
        return value
    unique = set(value)
    if len(unique) != 1:
        raise ValueError(f"bag {bag_id!r} is assigned to multiple splits")
    return next(iter(unique))


def make_grouped_windows(
    files: Sequence[Path],
    window: int,
    stride: int,
    split_manifest: Mapping[str, str | Sequence[str]],
) -> pd.DataFrame:
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be positive")
    rows: list[dict[str, object]] = []
    observed: dict[str, str] = {}
    for source in files:
        path = Path(source)
        bag_id = path.stem
        if bag_id not in split_manifest:
            raise KeyError(f"no split assignment for bag {bag_id!r}")
        split = _resolve_split(split_manifest[bag_id], bag_id)
        previous = observed.setdefault(bag_id, split)
        if previous != split:
            raise ValueError(f"bag {bag_id!r} is assigned to multiple splits")
        row_count = len(pd.read_csv(path, usecols=[0]))
        if row_count < window:
            continue
        starts = list(range(0, row_count - window + 1, stride))
        if starts and starts[-1] != row_count - window:
            starts.append(row_count - window)
        for start in starts:
            rows.append(
                {
                    "file_id": path.stem,
                    "bag_id": bag_id,
                    "start": start,
                    "stop": start + window,
                    "split": split,
                }
            )
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)
