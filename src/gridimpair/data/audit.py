from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from gridimpair.contracts import stable_hash


@dataclass(frozen=True)
class AuditSummary:
    csv_count: int
    total_rows: int
    total_nan: int
    total_inf: int
    duplicate_rows: int
    constant_channels: int
    quasi_constant_channels: int
    irregular_time_files: int
    manifest_hash: str


def compare_window_split_leakage(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = 2026,
    test_size: float = 0.25,
) -> pd.DataFrame:
    """Compare a diagnostic random-window split with the valid grouped split."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int).reshape(-1)
    group_ids = np.asarray(groups).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(group_ids):
        raise ValueError("features, labels, and groups must share their first dimension")
    del test_size
    splitters = {
        "random_window": StratifiedKFold(
            n_splits=2, shuffle=True, random_state=seed
        ).split(x, y),
        "grouped_file": StratifiedGroupKFold(
            n_splits=2, shuffle=True, random_state=seed
        ).split(x, y, groups=group_ids),
    }
    rows = []
    for strategy, splitter in splitters.items():
        fold_truth: list[np.ndarray] = []
        fold_scores: list[np.ndarray] = []
        overlaps: list[int] = []
        valid = True
        for train, test in splitter:
            if np.unique(y[train]).size < 2 or np.unique(y[test]).size < 2:
                valid = False
                break
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(random_state=seed, max_iter=2_000),
            )
            model.fit(x[train], y[train])
            fold_truth.append(y[test])
            fold_scores.append(model.predict_proba(x[test])[:, 1])
            overlaps.append(
                len(set(group_ids[train]) & set(group_ids[test]))
            )
        if valid:
            score = float(
                average_precision_score(
                    np.concatenate(fold_truth), np.concatenate(fold_scores)
                )
            )
            status = "diagnostic_only"
        else:
            score = float("nan")
            status = "insufficient_class_coverage"
        rows.append(
            {
                "split_strategy": strategy,
                "auprc": score,
                "group_overlap": max(overlaps, default=0),
                "status": status,
            }
        )
    frame = pd.DataFrame(rows)
    random_score = frame.loc[
        frame["split_strategy"] == "random_window", "auprc"
    ].iloc[0]
    grouped_score = frame.loc[
        frame["split_strategy"] == "grouped_file", "auprc"
    ].iloc[0]
    frame["optimism_gap"] = random_score - grouped_score
    return frame


def univariate_best_channel_diagnostic(
    cases: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = 2026,
) -> pd.DataFrame:
    """Rank channel summary fingerprints without treating them as model inputs."""

    values = np.asarray(cases, dtype=float)
    if values.ndim != 3:
        raise ValueError("cases must have shape (cases, channels, time)")
    rows = []
    for channel in range(values.shape[1]):
        signal = values[:, channel, :]
        features = np.c_[
            signal.mean(axis=1),
            signal.std(axis=1),
            signal.min(axis=1),
            signal.max(axis=1),
        ]
        comparison = compare_window_split_leakage(
            features, labels, groups, seed=seed
        )
        grouped = comparison.loc[
            comparison["split_strategy"] == "grouped_file"
        ].iloc[0]
        rows.append(
            {
                "channel_index": channel,
                "grouped_auprc": grouped["auprc"],
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "grouped_auprc", ascending=False, na_position="last"
    )


def _source_label(file_id: str) -> int | None:
    match = re.fullmatch(r"3\(([a-o])\)", file_id.lower())
    if not match:
        return None
    letter = match.group(1)
    if "a" <= letter <= "e":
        return 0
    if "f" <= letter <= "n":
        return 1
    return None


def _window_features(
    frames: list[tuple[str, int, pd.DataFrame]],
    *,
    window: int,
    stride: int,
    drop_setpoints: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    common_columns = set.intersection(
        *[
            set(frame.select_dtypes(include=[np.number]).columns)
            for _, _, frame in frames
        ]
    )
    columns = sorted(
        column
        for column in common_columns
        if str(column).strip().lower()
        not in {"time", "timestamp", "t", "seconds"}
        and (
            not drop_setpoints
            or "set" not in str(column).lower()
        )
    )
    if not columns:
        raise ValueError("no common numeric model channels were found")
    feature_rows: list[np.ndarray] = []
    case_rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for file_id, label, frame in frames:
        values = frame[columns].to_numpy(dtype=float).T
        if values.shape[1] < window:
            continue
        starts = list(range(0, values.shape[1] - window + 1, stride))
        if starts[-1] != values.shape[1] - window:
            starts.append(values.shape[1] - window)
        for start in starts:
            case = values[:, start : start + window]
            summary = np.concatenate(
                [
                    case.mean(axis=1),
                    case.std(axis=1),
                    case.min(axis=1),
                    case.max(axis=1),
                    case[:, -1] - case[:, 0],
                ]
            )
            feature_rows.append(summary)
            case_rows.append(case)
            labels.append(label)
            groups.append(file_id)
    if not feature_rows:
        raise ValueError("window settings produced no diagnostic windows")
    return (
        np.stack(feature_rows),
        np.asarray(labels, dtype=int),
        np.asarray(groups),
        np.stack(case_rows),
        columns,
    )


def _fingerprint_diagnostic(
    features: np.ndarray, groups: np.ndarray, *, seed: int
) -> pd.DataFrame:
    encoded, file_ids = pd.factorize(groups, sort=True)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.25, random_state=seed
    )
    train, test = next(splitter.split(features, encoded))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=seed, max_iter=2_000
        ),
    )
    model.fit(features[train], encoded[train])
    return pd.DataFrame(
        [
            {
                "task": "recover_file_id_from_window_summary",
                "random_window_accuracy": float(
                    accuracy_score(encoded[test], model.predict(features[test]))
                ),
                "chance_accuracy": 1.0 / len(file_ids),
                "file_count": len(file_ids),
                "diagnostic_only": True,
            }
        ]
    )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _time_grid_is_irregular(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size < 3:
        return False
    differences = np.diff(numeric)
    return bool(
        np.any(differences <= 0)
        or not np.allclose(differences, differences[0], rtol=1e-6, atol=1e-12)
    )


def audit_source_dataset(
    root: Path,
    output_dir: Path,
    *,
    window: int = 256,
    stride: int = 64,
    seed: int = 2026,
) -> AuditSummary:
    if window < 2 or stride < 1:
        raise ValueError("window must be at least 2 and stride positive")
    root = Path(root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".jpg", ".jpeg", ".pdf"}
    )
    manifest_files: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, Any]] = []
    missingness_rows: list[dict[str, Any]] = []
    time_interval_rows: list[dict[str, Any]] = []
    labeled_frames: list[tuple[str, int, pd.DataFrame]] = []
    totals = {
        "rows": 0,
        "nan": 0,
        "inf": 0,
        "duplicates": 0,
        "constant": 0,
        "quasi": 0,
        "irregular": 0,
    }

    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest = _sha256_file(path)
        item: dict[str, Any] = {
            "path": relative,
            "kind": path.suffix.lower().lstrip("."),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            source_label = _source_label(path.stem)
            if source_label is not None:
                labeled_frames.append((path.stem, source_label, frame))
            item["columns"] = [str(column) for column in frame.columns]
            item["dtypes"] = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
            numeric = frame.select_dtypes(include=[np.number])
            array = numeric.to_numpy(dtype=float, copy=True)
            nan_count = int(np.isnan(array).sum())
            inf_count = int(np.isinf(array).sum())
            duplicate_count = int(frame.duplicated().sum())
            time_column = next(
                (
                    name
                    for name in frame.columns
                    if str(name).strip().lower() in {"time", "timestamp", "t", "seconds"}
                ),
                None,
            )
            irregular = int(
                time_column is not None and _time_grid_is_irregular(frame[time_column])
            )
            for column in frame.columns:
                values = frame[column]
                numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(
                    dtype=float
                )
                missingness_rows.append(
                    {
                        "file_id": path.stem,
                        "channel": str(column),
                        "nan_count": int(values.isna().sum()),
                        "inf_count": int(np.isinf(numeric_values).sum()),
                        "missing_fraction": float(values.isna().mean()),
                    }
                )
            if time_column is not None:
                time_values = pd.to_numeric(
                    frame[time_column], errors="coerce"
                ).to_numpy(dtype=float)
                time_values = time_values[np.isfinite(time_values)]
                for interval in np.diff(time_values):
                    time_interval_rows.append(
                        {
                            "file_id": path.stem,
                            "interval": float(interval),
                        }
                    )
            totals["rows"] += len(frame)
            totals["nan"] += nan_count
            totals["inf"] += inf_count
            totals["duplicates"] += duplicate_count
            totals["irregular"] += irregular
            quality_rows.append(
                {
                    "file_id": path.stem,
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "duplicate_rows": duplicate_count,
                    "irregular_time_grid": irregular,
                }
            )
            numeric_summaries: dict[str, dict[str, float]] = {}
            for column in numeric.columns:
                values = pd.to_numeric(numeric[column], errors="coerce").to_numpy(float)
                finite = values[np.isfinite(values)]
                unique = int(np.unique(finite).size) if finite.size else 0
                variance = float(np.var(finite)) if finite.size else float("nan")
                constant = int(finite.size > 0 and unique <= 1)
                quasi = int(
                    finite.size > 0
                    and not constant
                    and np.max(np.bincount(pd.factorize(finite)[0]))
                    / finite.size
                    >= 0.995
                )
                totals["constant"] += constant
                totals["quasi"] += quasi
                variance_rows.append(
                    {
                        "file_id": path.stem,
                        "channel": str(column),
                        "variance": variance,
                        "unique_values": unique,
                        "constant": constant,
                        "quasi_constant": quasi,
                    }
                )
                numeric_summaries[str(column)] = {
                    "mean": float(np.mean(finite)) if finite.size else None,
                    "std": float(np.std(finite)) if finite.size else None,
                }
            fingerprint_rows.append(
                {
                    "file_id": path.stem,
                    "schema_hash": stable_hash(item["dtypes"]),
                    "fingerprint_hash": stable_hash(numeric_summaries),
                    "diagnostic_only": True,
                    "random_window_score": np.nan,
                    "grouped_split_score": np.nan,
                    "optimism_gap": np.nan,
                }
            )
        manifest_files.append(item)

    manifest = {"root_name": root.name, "files": manifest_files}
    manifest_hash = stable_hash(manifest)
    manifest["manifest_hash"] = manifest_hash
    _atomic_json(output_dir / "source_manifest.json", manifest)
    pd.DataFrame(quality_rows).to_csv(output_dir / "data_quality.csv", index=False)
    pd.DataFrame(variance_rows).to_csv(output_dir / "channel_variance.csv", index=False)
    pd.DataFrame(fingerprint_rows).to_csv(
        output_dir / "file_fingerprints.csv", index=False
    )
    pd.DataFrame(missingness_rows).to_csv(
        output_dir / "channel_missingness.csv", index=False
    )
    pd.DataFrame(time_interval_rows).to_csv(
        output_dir / "time_grid_intervals.csv", index=False
    )
    if len(labeled_frames) >= 4:
        features, labels, groups, cases, channel_names = _window_features(
            labeled_frames, window=window, stride=stride
        )
        leakage = compare_window_split_leakage(
            features, labels, groups, seed=seed
        )
        leakage["window"] = window
        leakage["stride"] = stride
        leakage["feature_set"] = "all_non_time_channels"
        leakage.to_csv(
            output_dir / "leakage_diagnostics.csv", index=False
        )
        univariate = univariate_best_channel_diagnostic(
            cases, labels, groups, seed=seed
        )
        univariate["channel"] = [
            channel_names[index]
            for index in univariate["channel_index"].astype(int)
        ]
        univariate.to_csv(
            output_dir / "univariate_channel_diagnostics.csv", index=False
        )
        _fingerprint_diagnostic(features, groups, seed=seed).to_csv(
            output_dir / "fingerprint_diagnostics.csv", index=False
        )
        standardized = StandardScaler().fit_transform(features)
        components = PCA(n_components=2, random_state=seed).fit_transform(
            standardized
        )
        pd.DataFrame(
            {
                "component_1": components[:, 0],
                "component_2": components[:, 1],
                "file_id": groups,
                "label": labels,
                "diagnostic_only": True,
            }
        ).to_csv(output_dir / "fingerprint_embedding.csv", index=False)
        (
            no_setpoint_features,
            no_setpoint_labels,
            no_setpoint_groups,
            _,
            _,
        ) = _window_features(
            labeled_frames,
            window=window,
            stride=stride,
            drop_setpoints=True,
        )
        feature_set_rows = []
        for feature_set, feature_values, feature_labels, feature_groups in (
            (
                "all_non_time_channels",
                features,
                labels,
                groups,
            ),
            (
                "without_time_and_setpoints",
                no_setpoint_features,
                no_setpoint_labels,
                no_setpoint_groups,
            ),
        ):
            result = compare_window_split_leakage(
                feature_values,
                feature_labels,
                feature_groups,
                seed=seed,
            )
            result["feature_set"] = feature_set
            feature_set_rows.append(result)
        pd.concat(feature_set_rows, ignore_index=True).to_csv(
            output_dir / "feature_set_diagnostics.csv", index=False
        )
    else:
        pd.DataFrame(
            [
                {
                    "split_strategy": "random_window",
                    "auprc": np.nan,
                    "group_overlap": np.nan,
                    "status": "insufficient_labeled_bags",
                    "optimism_gap": np.nan,
                    "window": window,
                    "stride": stride,
                    "feature_set": "all_non_time_channels",
                },
                {
                    "split_strategy": "grouped_file",
                    "auprc": np.nan,
                    "group_overlap": 0,
                    "status": "insufficient_labeled_bags",
                    "optimism_gap": np.nan,
                    "window": window,
                    "stride": stride,
                    "feature_set": "all_non_time_channels",
                },
            ]
        ).to_csv(output_dir / "leakage_diagnostics.csv", index=False)
        pd.DataFrame(
            columns=[
                "channel_index",
                "grouped_auprc",
                "diagnostic_only",
                "channel",
            ]
        ).to_csv(
            output_dir / "univariate_channel_diagnostics.csv", index=False
        )
        pd.DataFrame(
            columns=[
                "task",
                "random_window_accuracy",
                "chance_accuracy",
                "file_count",
                "diagnostic_only",
            ]
        ).to_csv(
            output_dir / "fingerprint_diagnostics.csv", index=False
        )
        pd.DataFrame(
            columns=[
                "component_1",
                "component_2",
                "file_id",
                "label",
                "diagnostic_only",
            ]
        ).to_csv(output_dir / "fingerprint_embedding.csv", index=False)
    summary = AuditSummary(
        csv_count=sum(path.suffix.lower() == ".csv" for path in paths),
        total_rows=totals["rows"],
        total_nan=totals["nan"],
        total_inf=totals["inf"],
        duplicate_rows=totals["duplicates"],
        constant_channels=totals["constant"],
        quasi_constant_channels=totals["quasi"],
        irregular_time_files=totals["irregular"],
        manifest_hash=manifest_hash,
    )
    _atomic_json(output_dir / "summary.json", asdict(summary))
    return summary
