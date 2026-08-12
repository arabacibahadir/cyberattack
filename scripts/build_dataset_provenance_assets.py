from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "artifacts_q1_v2" / "audit"
DEFAULT_OUTPUT = ROOT / "manuscript" / "dataset_evidence"

COLORS = {
    "normal": "#0072B2",
    "DoS": "#E69F00",
    "DDoS": "#D55E00",
    "hybrid": "#CC79A7",
    "neutral": "#4D4D4D",
    "grouped": "#009E73",
    "random": "#D55E00",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def source_class(file_id: str) -> str:
    match = re.fullmatch(r"3\(([a-o])\)", file_id.lower())
    if not match:
        return "unmapped"
    letter = match.group(1)
    if "a" <= letter <= "e":
        return "normal"
    if "f" <= letter <= "g":
        return "DoS"
    if "h" <= letter <= "n":
        return "DDoS"
    return "hybrid"


def find_dataset_root(root: Path) -> Path:
    candidates = sorted(root.glob("Multilayer Cyberattacks Identification*"))
    candidates = [p for p in candidates if p.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one dataset directory, found {len(candidates)}")
    return candidates[0]


def verify_manifest(dataset_root: Path, audit_root: Path) -> dict[str, Any]:
    manifest_path = audit_root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        item["path"]: item["sha256"]
        for item in manifest["files"]
        if item.get("kind") == "csv"
    }
    csv_paths = sorted(dataset_root.glob("*.csv"))
    if len(csv_paths) != 15 or set(expected) != {p.name for p in csv_paths}:
        raise RuntimeError("local CSV inventory does not match the frozen q1-v2 manifest")
    mismatches = []
    for path in csv_paths:
        observed = sha256_file(path)
        if observed != expected[path.name]:
            mismatches.append((path.name, expected[path.name], observed))
    if mismatches:
        raise RuntimeError(f"CSV checksum mismatch: {mismatches}")
    return manifest


def channel_role(column: str) -> tuple[str, str, str]:
    if column == "Time":
        return (
            "simulation coordinate",
            "Regular numeric coordinate; not interpreted as wall-clock acquisition time",
            "coordinate unit not independently established",
        )
    token = column.split("|")[-1]
    if "ILOAD" in token:
        role = "current-labelled load field"
    elif "APLOAD" in token:
        role = "APLOAD-labelled load field"
    elif "AQLOAD" in token:
        role = "AQLOAD-labelled load field"
    elif "BPLOAD" in token:
        role = "BPLOAD-labelled load field"
    elif token.endswith("set"):
        role = "setpoint-labelled field"
    else:
        role = "dataset-provided numeric field"
    return role, f"Header token: {token}; semantics kept conservative", "not documented"


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.tick_params(labelsize=8)


def save_figure(
    fig: plt.Figure,
    figure_id: str,
    source: pd.DataFrame,
    output: Path,
    inputs: list[Path],
    description: str,
) -> dict[str, Any]:
    source_path = output / f"{figure_id}-source.csv"
    source.to_csv(source_path, index=False)
    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if extension == "png" else {}
        fig.savefig(output / f"{figure_id}.{extension}", bbox_inches="tight", **kwargs)
    word_dir = output / "word"
    word_dir.mkdir(exist_ok=True)
    fig.savefig(word_dir / f"{figure_id}.png", dpi=340, bbox_inches="tight")
    plt.close(fig)
    provenance = {
        "figure_id": figure_id,
        "description": description,
        "source_table": source_path.name,
        "source_hash": sha256_file(source_path),
        "input_files": [str(p.relative_to(ROOT)).replace("\\", "/") for p in inputs],
        "input_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in inputs},
        "formats": {
            "png": f"{figure_id}.png",
            "pdf": f"{figure_id}.pdf",
            "svg": f"{figure_id}.svg",
            "word_png": f"word/{figure_id}.png",
        },
        "png_dpi": 600,
        "word_png_dpi": 340,
    }
    atomic_json(output / f"{figure_id}-provenance.json", provenance)
    return provenance


def build_assets(dataset_root: Path, audit_root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = verify_manifest(dataset_root, audit_root)
    csv_paths = sorted(dataset_root.glob("*.csv"))
    frames = {p.stem: pd.read_csv(p) for p in csv_paths}
    columns = list(next(iter(frames.values())).columns)
    if any(list(frame.columns) != columns for frame in frames.values()):
        raise RuntimeError("CSV schemas are not identical")

    inventory_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    for path in csv_paths:
        file_id = path.stem
        frame = frames[file_id]
        numeric = frame.select_dtypes(include=[np.number])
        values = numeric.to_numpy(dtype=float)
        differences = np.diff(frame["Time"].to_numpy(dtype=float))
        constant_count = 0
        for column in columns[1:]:
            variance = float(np.var(frame[column].to_numpy(dtype=float)))
            constant = int(frame[column].nunique(dropna=False) <= 1)
            constant_count += constant
            parts = column.split("|")
            concise_channel = "/".join(parts[-2:]) if len(parts) >= 2 else column
            variance_rows.append(
                {
                    "file_id": file_id,
                    "source_class": source_class(file_id),
                    "channel": column,
                    "short_channel": concise_channel,
                    "variance": variance,
                    "constant": constant,
                }
            )
        interval_rows.extend(
            {
                "file_id": file_id,
                "interval_index": i,
                "interval": float(value),
            }
            for i, value in enumerate(differences)
        )
        inventory_rows.append(
            {
                "file_id": file_id,
                "source_class": source_class(file_id),
                "rows": len(frame),
                "columns": len(frame.columns),
                "time_min": float(frame["Time"].min()),
                "time_max": float(frame["Time"].max()),
                "interval": float(np.median(differences)),
                "nan_count": int(np.isnan(values).sum()),
                "inf_count": int(np.isinf(values).sum()),
                "duplicate_rows": int(frame.duplicated().sum()),
                "irregular_time_grid": int(
                    np.any(differences <= 0)
                    or not np.allclose(differences, differences[0], rtol=1e-6, atol=1e-12)
                ),
                "constant_non_time_channels": constant_count,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "checksum_prefix": sha256_file(path)[:12],
            }
        )
    inventory = pd.DataFrame(inventory_rows)
    variance = pd.DataFrame(variance_rows)
    intervals = pd.DataFrame(interval_rows)

    # Tables
    provenance = pd.DataFrame(
        [
            ["Dataset creators", "Muhammad Faheem; Mahmoud Ahmad Al-Khasawneh", "Mendeley Data Version 1"],
            ["Dataset title", "Multilayer Cyberattacks Identification and Classification Using Machine Learning in Internet of Blockchain (IoBC)-Based Energy Networks", "Mendeley Data Version 1"],
            ["Dataset DOI", "10.17632/zc9z7m7gcd.1", "Public persistent identifier"],
            ["Associated article", "Data in Brief 54 (2024) 110461; DOI 10.1016/j.dib.2024.110461", "Source article"],
            ["Version and publication date", "Version 1; 12 February 2024", "Mendeley landing record"],
            ["Licence", "Creative Commons Attribution 4.0 International (CC BY 4.0)", "Reuse requires attribution"],
            ["Our role", "Secondary-data user; no original data collection", "This study"],
        ],
        columns=["item", "verified_value", "evidence_basis"],
    )
    claims = pd.DataFrame(
        [
            ["Sensor/node count", "250 nodes in the Specifications Table narrative; 200 wireless sensors in Table 3", "CSV does not encode node identifiers", "Report both source statements; do not reconcile by assumption"],
            ["Temporal acquisition", "15-minute acquisition interval; 150-s simulation in Table 3", "Each CSV spans 0–0.4998 with constant increment 0.00015", "Treat Time as a simulation coordinate only"],
            ["Measured variables", "Environmental, operational, electrical and network measurements are described", "Delivered CSV schema contains load-subsystem current/power/setpoint-labelled fields only", "Describe the delivered subset, not the broader claimed sensor universe"],
            ["Communication medium", "5G plus optical fibre in Data Description; 5G plus Ethernet in Table 3", "No link-layer field exists in the CSVs", "Attribute both descriptions to the source; do not infer the path used per file"],
            ["Attack interval", "Approximately 0.081–0.3 s is discussed for examples", "No row-level label or onset/offset column is delivered", "Do not use the narrative interval as machine-readable temporal ground truth"],
            ["Units", "Several physical quantities are discussed narratively", "CSV headers do not provide a complete unit dictionary", "Do not convert or assign physical units"],
        ],
        columns=["topic", "source_report", "locally_observed_evidence", "treatment_in_this_study"],
    )
    variable_dictionary = pd.DataFrame(
        [
            {
                "column_index": i,
                "column": column,
                "short_name": column.split("|")[-1],
                "conservative_role": channel_role(column)[0],
                "interpretation_note": channel_role(column)[1],
                "unit": channel_role(column)[2],
                "audit_use": "coordinate excluded from model features" if column == "Time" else "numeric audit channel",
            }
            for i, column in enumerate(columns)
        ]
    )
    audit_summary = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    leakage = pd.read_csv(audit_root / "leakage_diagnostics.csv")
    fingerprint = pd.read_csv(audit_root / "fingerprint_diagnostics.csv")
    univariate = pd.read_csv(audit_root / "univariate_channel_diagnostics.csv")
    feature_sets = pd.read_csv(audit_root / "feature_set_diagnostics.csv")
    audit_results = pd.DataFrame(
        [
            ["CSV files", 15, "files", "All files included in integrity audit"],
            ["Rows", 49995, "rows", "3,333 rows per file"],
            ["NaN values", audit_summary["total_nan"], "cells", "Deterministic audit"],
            ["Infinite values", audit_summary["total_inf"], "cells", "Deterministic audit"],
            ["Exact duplicate rows", audit_summary["duplicate_rows"], "rows", "Within-file comparison"],
            ["Irregular time-grid files", audit_summary["irregular_time_files"], "files", "Monotonicity and constant-step check"],
            ["Constant file-channel combinations", audit_summary["constant_channels"], "combinations", "15 files × 19 numeric columns; Time is non-constant"],
            ["Diagnostic windows", 700, "windows", "14 files × 50 windows; hybrid file excluded"],
            ["Random-window AUPRC", float(leakage.loc[leakage.split_strategy == "random_window", "auprc"].iloc[0]), "AUPRC", "Diagnostic only"],
            ["Grouped-file AUPRC", float(leakage.loc[leakage.split_strategy == "grouped_file", "auprc"].iloc[0]), "AUPRC", "Diagnostic only"],
            ["Optimism gap", float(leakage.optimism_gap.iloc[0]), "AUPRC", "Random minus grouped"],
            ["File-ID recovery accuracy", float(fingerprint.random_window_accuracy.iloc[0]), "accuracy", "14-class diagnostic"],
            ["File-ID chance accuracy", float(fingerprint.chance_accuracy.iloc[0]), "accuracy", "1/14"],
            ["Best univariate grouped AUPRC", float(univariate.grouped_auprc.max()), "AUPRC", "Diagnostic only"],
        ],
        columns=["measure", "value", "unit", "interpretation"],
    )
    use_matrix = pd.DataFrame(
        [
            ["File inventory and SHA-256 verification", "Used", "All 15 CSVs and 15 companion JPGs", "Provenance and integrity"],
            ["Schema/missingness/finite-value audit", "Used", "All 49,995 rows", "Data-quality characterization"],
            ["Constant-channel and variance diagnostics", "Used", "File-channel combinations", "Redundancy/fingerprint risk"],
            ["Grouped-window leakage diagnostic", "Used", "3(a)–3(n); 700 windows", "Exploratory only; file is the independent unit"],
            ["Hybrid file 3(o)", "Descriptive only", "Integrity, composition and representative trace", "Excluded from binary window diagnostic because it is one mixed-condition bag"],
            ["Primary q1-v2 model training", "Not used", "No source CSV input", "Training uses paired pandapower traces"],
            ["Calibration and temporal thresholding", "Not used", "No source CSV input", "Uses validation partitions of the paired simulation"],
            ["Locked-v2 evaluation", "Not used", "No source CSV input", "Uses IEEE33-DER paired simulation traces"],
            ["Closed-loop policy evaluation", "Not used", "No source CSV input", "Uses frozen alarm schedules and paired simulations"],
        ],
        columns=["operation", "status", "data_scope", "scientific_role"],
    )
    traceability = pd.DataFrame(
        [
            ["Source ownership and licence", "PDF metadata; Mendeley Version 1 landing record", "Manual verified-source extraction", "§1; Table 1"],
            ["CSV inventory and checksums", "artifacts_q1_v2/audit/source_manifest.json; raw CSVs", "sha256_file; verify_manifest", "§2; Tables 3 and A1"],
            ["Integrity and constants", "raw CSVs; artifacts_q1_v2/audit/summary.json", "pandas deterministic audit; gridimpair.data.audit.audit_source_dataset", "§3; Figures 4–5; Table 5"],
            ["Leakage comparison", "artifacts_q1_v2/audit/leakage_diagnostics.csv", "gridimpair.data.audit.compare_window_split_leakage", "§4; Figure 7"],
            ["File fingerprint", "fingerprint_embedding.csv; fingerprint_diagnostics.csv", "gridimpair.data.audit._fingerprint_diagnostic", "§4; Figure 6"],
            ["Univariate channels", "univariate_channel_diagnostics.csv", "gridimpair.data.audit.univariate_best_channel_diagnostic", "§4; Figure 8"],
            ["Boundary from primary evidence", "src/gridimpair/runner.py; src/gridimpair/workflow.py", "audit stage vs discover_training_records", "§5; Table 6"],
        ],
        columns=["claim_family", "source_artifact", "computing_component", "document_location"],
    )

    table_map = {
        "table01-provenance.csv": provenance,
        "table02-source-claims-vs-files.csv": claims,
        "table03-file-inventory.csv": inventory.drop(columns=["sha256"]),
        "table04-variable-dictionary.csv": variable_dictionary,
        "table05-audit-results.csv": audit_results,
        "table06-use-matrix.csv": use_matrix,
        "tableA1-full-checksums.csv": inventory[["file_id", "source_class", "bytes", "sha256"]],
        "tableA2-code-traceability.csv": traceability,
    }
    table_manifest = []
    for name, frame in table_map.items():
        path = output / name
        frame.to_csv(path, index=False)
        table_manifest.append({"file": name, "rows": len(frame), "columns": len(frame.columns), "sha256": sha256_file(path)})

    figure_manifest: list[dict[str, Any]] = []

    # Figure 1: lineage and explicit boundary.
    lineage = pd.DataFrame(
        [
            [1, "Third-party Mendeley Data V1", "15 CSV + 15 JPG; CC BY 4.0", "external source"],
            [2, "Local immutable inventory", "SHA-256, schema and row counts", "secondary-data audit"],
            [3, "Exploratory diagnostics", "grouped windows, fingerprints and split optimism", "diagnostic only"],
            [4, "Primary q1-v2 evidence", "independent paired pandapower traces", "training/evaluation source"],
        ],
        columns=["stage", "label", "detail", "role"],
    )
    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    ax.axis("off")
    xs = [0.025, 0.265, 0.505, 0.765]
    fills = ["#D9EAF7", "#D9EAF7", "#FCE4D6", "#E2F0D9"]
    for i, row in lineage.iterrows():
        box = FancyBboxPatch((xs[i], 0.34), 0.185, 0.42, boxstyle="round,pad=0.015", facecolor=fills[i], edgecolor="#404040", linewidth=1.2, transform=ax.transAxes)
        ax.add_patch(box)
        label = textwrap.fill(str(row["label"]), width=22)
        detail = textwrap.fill(str(row["detail"]), width=24)
        ax.text(xs[i] + 0.0925, 0.65, label, ha="center", va="center", fontsize=8.1, fontweight="bold", transform=ax.transAxes, linespacing=1.15)
        ax.text(xs[i] + 0.0925, 0.45, detail, ha="center", va="center", fontsize=7.2, transform=ax.transAxes, linespacing=1.15)
        if i < 2:
            ax.annotate("", xy=(xs[i + 1] - 0.008, 0.55), xytext=(xs[i] + 0.193, 0.55), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 1.3})
    ax.plot([0.735, 0.735], [0.24, 0.82], color="#C00000", linewidth=2, transform=ax.transAxes)
    ax.annotate("audit-informed design only", xy=(0.765, 0.55), xytext=(0.67, 0.88), xycoords=ax.transAxes, ha="center", fontsize=7.8, arrowprops={"arrowstyle": "->", "lw": 1.1, "linestyle": "--"})
    ax.text(0.5, 0.10, "Boundary: source CSVs were not transferred into q1-v2 model fitting, calibration, locked testing, or control evaluation.", ha="center", color="#C00000", fontsize=8.3, fontweight="bold", transform=ax.transAxes)
    figure_manifest.append(save_figure(fig, "fig01-data-lineage", lineage, output, [audit_root / "source_manifest.json", ROOT / "src/gridimpair/runner.py", ROOT / "src/gridimpair/workflow.py"], "Third-party data lineage and boundary from primary q1-v2 evidence"))

    # Figure 2: corpus composition.
    composition = inventory.groupby("source_class", sort=False).agg(file_count=("file_id", "size"), row_count=("rows", "sum")).reset_index()
    order = ["normal", "DoS", "DDoS", "hybrid"]
    composition["source_class"] = pd.Categorical(composition["source_class"], order, ordered=True)
    composition = composition.sort_values("source_class")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, metric, ylabel in zip(axes, ["file_count", "row_count"], ["Files", "Rows"]):
        values = composition[metric]
        colors = [COLORS[str(c)] for c in composition.source_class]
        bars = ax.bar(composition.source_class.astype(str), values, color=colors, edgecolor="#333333", linewidth=0.5)
        ax.bar_label(bars, labels=[f"{int(v):,}" for v in values], padding=3, fontsize=8)
        ax.set_ylabel(ylabel)
        style_axes(ax)
    fig.suptitle("Delivered corpus composition by source-assigned file category", fontsize=11, fontweight="bold")
    figure_manifest.append(save_figure(fig, "fig02-corpus-composition", composition, output, csv_paths, "File and row composition by source-assigned category"))

    # Figure 3: representative trajectories.
    representative = {"normal": "3(a)", "DoS": "3(f)", "DDoS": "3(h)", "hybrid": "3(o)"}
    trace_channels = [
        "Subsystem #1|Loads|LOAD5|APLOAD5",
        "Subsystem #1|Loads|LOAD5|BPLOAD5",
        "Subsystem #1|Loads|LOAD6|APLOAD6",
        "Subsystem #1|Loads|LOAD7|APLOAD7",
    ]
    trajectory_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), sharex=True)
    for ax, (category, file_id) in zip(axes.flat, representative.items()):
        frame = frames[file_id]
        for channel in trace_channels:
            short = channel.split("|")[-1]
            ax.plot(frame["Time"], frame[channel], linewidth=0.8, label=short)
            trajectory_rows.extend({"file_id": file_id, "source_class": category, "time": t, "channel": short, "value": value} for t, value in zip(frame["Time"], frame[channel]))
        ax.set_title(f"{category}: {file_id}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Reported Time coordinate")
        ax.set_ylabel("Reported numeric value\n(unit not documented)")
        style_axes(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Representative file-level trajectories; panels are not row-level attack annotations", y=0.985, fontsize=11, fontweight="bold")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    trajectory_source = pd.DataFrame(trajectory_rows)
    figure_manifest.append(save_figure(fig, "fig03-representative-trajectories", trajectory_source, output, [dataset_root / f"{v}.csv" for v in representative.values()], "Representative raw trajectories for four source-assigned categories"))

    # Figure 4: time-grid regularity.
    interval_summary = intervals.groupby("file_id").interval.agg(["count", "min", "median", "max"]).reset_index()
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    x = np.arange(len(interval_summary))
    scale = 1e-16
    median_residual = (interval_summary["median"] - 0.00015) / scale
    min_residual = (interval_summary["min"] - 0.00015) / scale
    max_residual = (interval_summary["max"] - 0.00015) / scale
    ax.scatter(x, median_residual, color="#0072B2", s=30, label="median residual", zorder=3)
    ax.vlines(x, min_residual, max_residual, color="#666666", linewidth=1.0, label="min–max residual")
    ax.set_xticks(x, interval_summary.file_id, rotation=45, ha="right")
    ax.set_ylabel("Numerical residual from 0.00015\n(in units of 10^-16)")
    ax.text(0.98, 0.07, "49,980 intervals; residuals reflect floating-point representation only", transform=ax.transAxes, ha="right", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Regularity of the delivered Time coordinate", fontsize=11, fontweight="bold")
    figure_manifest.append(save_figure(fig, "fig04-time-grid", intervals, output, csv_paths, "All within-file time-coordinate intervals"))

    # Figure 5: constant-channel heatmap.
    matrix = variance.pivot(index="file_id", columns="short_channel", values="constant").loc[sorted(frames)]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=matplotlib.colors.ListedColormap(["#FFFFFF", "#C00000"]), vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index, fontsize=8)
    ax.set_xlabel("Non-Time channel")
    ax.set_ylabel("Source file")
    ax.set_title("Constant file-channel combinations (red = constant)", fontsize=11, fontweight="bold")
    ax.set_xticks(np.arange(-.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="#D9D9D9", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    figure_manifest.append(save_figure(fig, "fig05-constant-channel-heatmap", variance, output, csv_paths, "Binary constant-channel matrix for 15 files and 18 non-Time channels"))

    # Figure 6: PCA fingerprint embedding from frozen audit.
    embedding_path = audit_root / "fingerprint_embedding.csv"
    embedding = pd.read_csv(embedding_path)
    embedding["source_class"] = embedding.file_id.map(source_class)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for category in ["normal", "DoS", "DDoS"]:
        subset = embedding[embedding.source_class == category]
        ax.scatter(subset.component_1, subset.component_2, s=16, alpha=0.65, label=category, color=COLORS[category], edgecolors="none")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.legend(frameon=False)
    style_axes(ax)
    ax.set_title("Window-summary fingerprint structure (700 windows; 14 files)", fontsize=11, fontweight="bold")
    figure_manifest.append(save_figure(fig, "fig06-file-fingerprint-pca", embedding, output, [embedding_path], "PCA projection of grouped-window summary fingerprints"))

    # Figure 7: split leakage.
    fig, ax = plt.subplots(figsize=(6.7, 4.2))
    plot_leakage = leakage.copy()
    colors = [COLORS["random"] if s == "random_window" else COLORS["grouped"] for s in plot_leakage.split_strategy]
    bars = ax.bar(["Random window", "Grouped file"], plot_leakage.auprc, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in plot_leakage.auprc], padding=3, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Diagnostic AUPRC")
    ax.text(0.5, 0.09, f"Optimism gap = {plot_leakage.optimism_gap.iloc[0]:.4f}\nGroup overlap: 14 vs 0", transform=ax.transAxes, ha="center", fontsize=8.5)
    style_axes(ax)
    ax.set_title("Random-window optimism relative to file-grouped evaluation", fontsize=11, fontweight="bold")
    figure_manifest.append(save_figure(fig, "fig07-split-leakage", plot_leakage, output, [audit_root / "leakage_diagnostics.csv"], "Random-window and grouped-file diagnostic AUPRC"))

    # Figure 8: univariate grouped AUPRC.
    uni = univariate.sort_values("grouped_auprc", ascending=True).copy()
    uni["short_channel"] = uni.channel.map(lambda value: f"{value.split('|')[-2]}/{value.split('|')[-1]}")
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.barh(uni.short_channel, uni.grouped_auprc, color="#56B4E9", edgecolor="#333333", linewidth=0.4)
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=1.0, label="balanced prevalence reference")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Grouped-file diagnostic AUPRC")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax)
    ax.set_title("Single-channel window-summary diagnostics", fontsize=11, fontweight="bold")
    figure_manifest.append(save_figure(fig, "fig08-univariate-channels", uni, output, [audit_root / "univariate_channel_diagnostics.csv"], "Leakage-aware grouped single-channel diagnostic ranking"))

    assertions = {
        "csv_count": len(csv_paths),
        "total_rows": int(inventory.rows.sum()),
        "companion_jpg_count": len(list(dataset_root.glob("*.jpg"))),
        "schema_columns": len(columns),
        "time_intervals": len(intervals),
        "constant_file_channel_combinations": int(variance.constant.sum()),
        "window_count": len(pd.read_csv(embedding_path)),
        "random_auprc": float(leakage.loc[leakage.split_strategy == "random_window", "auprc"].iloc[0]),
        "grouped_auprc": float(leakage.loc[leakage.split_strategy == "grouped_file", "auprc"].iloc[0]),
        "optimism_gap": float(leakage.optimism_gap.iloc[0]),
        "file_id_accuracy": float(fingerprint.random_window_accuracy.iloc[0]),
        "file_id_chance": float(fingerprint.chance_accuracy.iloc[0]),
        "best_univariate_auprc": float(univariate.grouped_auprc.max()),
        "feature_set_scores_identical": bool(
            np.allclose(
                feature_sets.loc[feature_sets.feature_set == "all_non_time_channels", "auprc"].to_numpy(),
                feature_sets.loc[feature_sets.feature_set == "without_time_and_setpoints", "auprc"].to_numpy(),
            )
        ),
        "manifest_hash": manifest["manifest_hash"],
    }
    expected = {
        "csv_count": 15,
        "total_rows": 49995,
        "companion_jpg_count": 15,
        "schema_columns": 19,
        "time_intervals": 49980,
        "constant_file_channel_combinations": 143,
        "window_count": 700,
    }
    for key, value in expected.items():
        if assertions[key] != value:
            raise RuntimeError(f"assertion failed for {key}: {assertions[key]} != {value}")
    if not assertions["feature_set_scores_identical"]:
        raise RuntimeError("feature-set diagnostic scores are no longer identical")

    final_manifest = {
        "status": "completed",
        "purpose": "secondary-dataset provenance and diagnostic reporting only",
        "dataset_root_name": dataset_root.name,
        "audit_root": str(audit_root.relative_to(ROOT)).replace("\\", "/"),
        "figure_count": len(figure_manifest),
        "table_count": len(table_manifest),
        "figures": figure_manifest,
        "tables": table_manifest,
        "assertions": assertions,
        "prohibited_uses": [
            "primary q1-v2 model fitting",
            "calibration",
            "temporal threshold fitting",
            "locked-v2 evaluation",
            "closed-loop policy evaluation",
        ],
    }
    atomic_json(output / "manifest.json", final_manifest)
    print(json.dumps({"output": str(output), "figures": len(figure_manifest), "tables": len(table_manifest), "assertions": assertions}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build secondary-dataset provenance assets without running q1-v2 experiments")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset_root = args.data_root or find_dataset_root(ROOT)
    build_assets(dataset_root.resolve(), args.audit_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
