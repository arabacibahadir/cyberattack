from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
Q1_SOURCES = ROOT / "artifacts_q1_v2" / "figure_sources"
Q1_MANIFEST = ROOT / "artifacts_q1_v2" / "figures" / "render_manifest.json"
DATASET_SOURCES = ROOT / "manuscript" / "dataset_evidence"
OUTPUT = ROOT / "manuscript" / "figures_redesigned_q1_v3"

INK = "#26323D"
MUTED = "#68737D"
GRID = "#D9DEE3"
BLUE = "#2F6B9A"
ORANGE = "#D47C36"
TEAL = "#4C9278"
PURPLE = "#8064A2"
GOLD = "#B08D3E"
OPEN_BLUE = "#DCE8F1"
OPEN_ORANGE = "#F5E3D6"
OPEN_TEAL = "#DDECE6"
OPEN_PURPLE = "#E9E2F0"
NEUTRAL = "#AAB2B9"
PALETTE = [BLUE, ORANGE, TEAL, PURPLE, GOLD]

MODEL_NAMES = {
    "candidate": "Candidate-H",
    "candidate_physical": "Candidate-P",
    "Proposed-H": "Candidate-H",
    "Proposed-P": "Candidate-P",
    "message_age_watchdog": "AoI watchdog",
    "ridge_residual": "AR residual",
    "tcn_cyber": "TCN-C",
    "tcn_hybrid": "TCN-H",
    "tcn_physical": "TCN-P",
    "multirocket_hydra_cyber": "MR-Hydra-C",
    "multirocket_hydra_hybrid": "MR-Hydra-H",
    "multirocket_hydra_physical": "MR-Hydra-P",
    "xgboost_cyber": "XGB-C",
    "xgboost_hybrid": "XGB-H",
    "xgboost_physical": "XGB-P",
    "mil_attention_physical": "Attention MIL-P",
    "mil_mean_physical": "Mean MIL-P",
    "mil_topk_physical": "Top-k MIL-P",
    "tail_mil_physical": "TAIL-MIL-P",
    "ewma_cusum": "EWMA-CUSUM",
    "flatline": "Flatline",
    "derivative": "Derivative",
}


@dataclass
class ChartContract:
    figure_id: str
    figure_label: str
    title: str
    subtitle: str
    question: str
    takeaway: str
    family: str
    variant: str
    source: str
    rows: int
    palette_policy: str
    non_color_encoding: str
    claim_boundary: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def label(value: object) -> str:
    text = MODEL_NAMES.get(str(value), str(value))
    return text.replace("_", " ")


def short_channel(value: str) -> str:
    parts = str(value).split("|")
    return "/".join(parts[-2:]) if len(parts) > 1 else str(value)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def figure_shell(title: str, subtitle: str, figsize: tuple[float, float] = (10.8, 6.4)) -> plt.Figure:
    fig = plt.figure(figsize=figsize)
    fig.text(0.055, 0.955, title, fontsize=15, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.055, 0.908, subtitle, fontsize=9.5, ha="left", va="top", color=MUTED)
    return fig


def add_footer(fig: plt.Figure, source: Path | str, boundary: str) -> None:
    source_name = Path(source).name if not isinstance(source, str) or "/" in source or "\\" in source else source
    note = f"Source: {source_name}  |  Claim boundary: {boundary}"
    fig.text(0.055, 0.025, textwrap.fill(note, width=165), fontsize=7.2, color=MUTED, ha="left", va="bottom")


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def save_figure(fig: plt.Figure, figure_id: str, output: Path) -> dict[str, str | int]:
    output.mkdir(parents=True, exist_ok=True)
    word = output / "word"
    word.mkdir(exist_ok=True)
    paths = {
        "png": output / f"{figure_id}.png",
        "svg": output / f"{figure_id}.svg",
        "pdf": output / f"{figure_id}.pdf",
        "word_png": word / f"{figure_id}.png",
    }
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths["word_png"], dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return {
        "id": figure_id,
        "png": paths["png"].name,
        "svg": paths["svg"].name,
        "pdf": paths["pdf"].name,
        "word_png": f"word/{paths['word_png'].name}",
        "png_sha256": sha256_file(paths["png"]),
        "word_png_sha256": sha256_file(paths["word_png"]),
    }


def axis_for(fig: plt.Figure, rect: tuple[float, float, float, float] = (0.10, 0.14, 0.86, 0.68)) -> plt.Axes:
    return fig.add_axes(rect)


def horizontal_bar(
    data: pd.DataFrame,
    category: str,
    value: str,
    title: str,
    subtitle: str,
    source: Path,
    boundary: str,
    reference: float | None = None,
    xlim: tuple[float, float] | None = None,
    ci: tuple[str, str] | None = None,
) -> plt.Figure:
    frame = data.copy().sort_values(value, ascending=True)
    fig = figure_shell(title, subtitle)
    ax = axis_for(fig, (0.27, 0.14, 0.68, 0.68))
    y = np.arange(len(frame))
    values = frame[value].to_numpy(float)
    ax.barh(y, values, color=OPEN_BLUE, edgecolor=BLUE, linewidth=1.0, height=0.64)
    if ci:
        low = frame[ci[0]].to_numpy(float)
        high = frame[ci[1]].to_numpy(float)
        ax.errorbar(values, y, xerr=np.vstack([values - low, high - values]), fmt="o", color=INK, ecolor=INK, capsize=3, markersize=4, zorder=3)
    else:
        ax.scatter(values, y, s=22, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yticks(y, [label(v) for v in frame[category]])
    if reference is not None:
        ax.axvline(reference, color=INK, linestyle="--", linewidth=1.0)
        ax.text(reference, len(frame) - 0.25, f"reference {reference:g}", fontsize=7.5, ha="right", va="bottom", color=MUTED)
    if xlim:
        ax.set_xlim(*xlim)
    span = (ax.get_xlim()[1] - ax.get_xlim()[0]) or 1
    for yi, val in zip(y, values):
        ax.text(val + 0.012 * span, yi, f"{val:.4f}" if abs(val) < 2 else f"{val:.2f}", va="center", fontsize=7.8)
    ax.set_xlabel(value.replace("_", " "))
    style_axis(ax, "x")
    add_footer(fig, source, boundary)
    return fig


def dot_ci(
    data: pd.DataFrame,
    category: str,
    value: str,
    low: str,
    high: str,
    title: str,
    subtitle: str,
    source: Path,
    boundary: str,
    xlim: tuple[float, float] | None = None,
    zero: bool = False,
) -> plt.Figure:
    frame = data.copy().sort_values(value, ascending=True)
    fig = figure_shell(title, subtitle)
    ax = axis_for(fig, (0.28, 0.14, 0.67, 0.68))
    y = np.arange(len(frame))
    values = frame[value].to_numpy(float)
    lower = frame[low].to_numpy(float)
    upper = frame[high].to_numpy(float)
    ax.errorbar(values, y, xerr=np.vstack([values - lower, upper - values]), fmt="o", color=BLUE, ecolor=INK, capsize=3, markersize=6, linewidth=1.2)
    if zero:
        ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(y, [label(v) for v in frame[category]])
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(value.replace("_", " "))
    style_axis(ax, "x")
    add_footer(fig, source, boundary)
    return fig


def heatmap(
    data: pd.DataFrame,
    row: str,
    column: str,
    value: str,
    title: str,
    subtitle: str,
    source: Path,
    boundary: str,
    value_format: str = ".3f",
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    matrix = data.pivot(index=row, columns=column, values=value)
    fig = figure_shell(title, subtitle)
    left = 0.28 if len(matrix.index) > 8 else 0.18
    ax = axis_for(fig, (left, 0.14, 0.72 if left > 0.2 else 0.77, 0.68))
    image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(matrix.columns)), [label(v) for v in matrix.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), [label(v) for v in matrix.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                norm = (float(val) - (vmin if vmin is not None else np.nanmin(matrix))) / max(1e-12, (vmax if vmax is not None else np.nanmax(matrix)) - (vmin if vmin is not None else np.nanmin(matrix)))
                ax.text(j, i, format(float(val), value_format), ha="center", va="center", fontsize=7.1, color="white" if norm > 0.55 else INK)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    add_footer(fig, source, boundary)
    return fig


def small_multiples(
    data: pd.DataFrame,
    group: str,
    draw: Callable[[plt.Axes, pd.DataFrame, int], None],
    title: str,
    subtitle: str,
    source: Path,
    boundary: str,
    columns: int = 3,
    sharex: bool = True,
    sharey: bool = True,
) -> plt.Figure:
    groups = list(dict.fromkeys(data[group].tolist()))
    rows = math.ceil(len(groups) / columns)
    fig = figure_shell(title, subtitle, figsize=(10.8, max(6.4, 2.15 * rows + 1.4)))
    grid = fig.subplots(rows, columns, squeeze=False, sharex=sharex, sharey=sharey)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.82, hspace=0.46, wspace=0.28)
    for index, (name, ax) in enumerate(zip(groups, grid.flat)):
        subset = data[data[group] == name].copy()
        draw(ax, subset, index)
        ax.set_title(label(name), fontsize=9, fontweight="bold", loc="left")
    for ax in grid.flat[len(groups) :]:
        ax.axis("off")
    add_footer(fig, source, boundary)
    return fig


def draw_diagram(
    title: str,
    subtitle: str,
    nodes: list[dict],
    edges: list[dict],
    boundary: str,
) -> plt.Figure:
    fig = figure_shell(title, subtitle)
    ax = axis_for(fig, (0.04, 0.12, 0.92, 0.72))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    palette = {"source": OPEN_BLUE, "audit": OPEN_ORANGE, "model": OPEN_TEAL, "locked": OPEN_PURPLE, "control": "#EFE7D5", "neutral": "#F1F3F5"}
    for node in nodes:
        x, y, w, h = node["box"]
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.014", facecolor=palette.get(node.get("kind", "neutral"), "#F1F3F5"), edgecolor=INK, linewidth=1.0)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.61, textwrap.fill(node["title"], 20), ha="center", va="center", fontsize=9, fontweight="bold")
        if node.get("detail"):
            ax.text(x + w / 2, y + h * 0.26, textwrap.fill(node["detail"], 27), ha="center", va="center", fontsize=7.1, color=MUTED)
    by_id = {node["id"]: node for node in nodes}
    for edge in edges:
        a = by_id[edge["from"]]["box"]
        b = by_id[edge["to"]]["box"]
        start = (a[0] + a[2], a[1] + a[3] / 2) if edge.get("orientation", "h") == "h" else (a[0] + a[2] / 2, a[1])
        end = (b[0], b[1] + b[3] / 2) if edge.get("orientation", "h") == "h" else (b[0] + b[2] / 2, b[1] + b[3])
        start = edge.get("start_xy", start)
        end = edge.get("end_xy", end)
        if edge.get("route"):
            points = [start, *edge["route"], end]
            for p0, p1 in zip(points[:-2], points[1:-1]):
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=edge.get("color", INK), linewidth=1.2, linestyle=edge.get("style", "-"))
            arrow = FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=edge.get("color", INK), linestyle=edge.get("style", "-"))
            ax.add_patch(arrow)
        else:
            arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=edge.get("color", INK), linestyle=edge.get("style", "-"), connectionstyle=edge.get("connection", "arc3,rad=0"))
            ax.add_patch(arrow)
        if edge.get("label"):
            mx, my = edge.get("label_xy", ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2))
            ax.text(mx, my + edge.get("label_dy", 0.025), edge["label"], fontsize=7.2, ha="center", va="bottom", color=MUTED, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
    add_footer(fig, "study protocol and frozen q1-v2 code paths", boundary)
    return fig


def contract(
    figure_id: str,
    figure_label: str,
    title: str,
    subtitle: str,
    question: str,
    takeaway: str,
    family: str,
    variant: str,
    source: Path | str,
    rows: int,
    boundary: str,
    palette: str = "hard two-root cap",
    non_color: str = "direct labels, ordering and marker/line-style distinctions",
) -> ChartContract:
    return ChartContract(figure_id, figure_label, title, subtitle, question, takeaway, family, variant, str(source), rows, palette, non_color, boundary)


def diagram_definitions() -> list[tuple[ChartContract, plt.Figure, str]]:
    diagrams: list[tuple[ChartContract, plt.Figure, str]] = []
    nodes1 = [
        {"id": "third", "box": (0.02, 0.60, 0.17, 0.22), "title": "Third-party dataset", "detail": "15 CSV + 15 JPG; CC BY 4.0", "kind": "source"},
        {"id": "audit", "box": (0.25, 0.60, 0.18, 0.22), "title": "Secondary-data audit", "detail": "integrity, leakage, fingerprints", "kind": "audit"},
        {"id": "sim", "box": (0.02, 0.17, 0.17, 0.22), "title": "Independent simulation", "detail": "paired clean/impaired traces", "kind": "source"},
        {"id": "fit", "box": (0.25, 0.17, 0.18, 0.22), "title": "Development and fitting", "detail": "models, calibration, thresholds", "kind": "model"},
        {"id": "locked", "box": (0.51, 0.17, 0.18, 0.22), "title": "Locked evaluation", "detail": "network shift and robustness", "kind": "locked"},
        {"id": "control", "box": (0.77, 0.17, 0.18, 0.22), "title": "Closed-loop endpoints", "detail": "paired electrical outcomes", "kind": "control"},
    ]
    edges1 = [
        {"from": "third", "to": "audit", "label": "read-only audit"},
        {"from": "audit", "to": "fit", "label": "design warning only", "orientation": "v", "style": "--", "color": ORANGE, "connection": "arc3,rad=-0.15"},
        {"from": "sim", "to": "fit", "label": "trace.npy"},
        {"from": "fit", "to": "locked", "label": "frozen models"},
        {"from": "locked", "to": "control", "label": "frozen alarms"},
    ]
    title = "Evidence layers and non-transfer boundary"
    subtitle = "Third-party source data inform audit design only; all confirmatory q1-v2 evidence follows the independent simulation path"
    fig = draw_diagram(title, subtitle, nodes1, edges1, "No source CSV, coefficient, scaler, label or threshold enters the primary q1-v2 path.")
    c = contract("diag01-evidence-flow", "Diagram 1", title, subtitle, "How are evidence layers separated?", "The source dataset and confirmatory simulation remain empirically disjoint.", "Structure", "directed evidence-flow diagram", "frozen protocol", 6, "Architectural provenance; not a quantitative result.")
    mermaid1 = """flowchart LR
  A[Third-party dataset] --> B[Secondary-data audit]
  C[Independent paired simulation] --> D[Development and fitting]
  B -. design warning only .-> D
  D --> E[Locked evaluation]
  E --> F[Closed-loop endpoints]
"""
    diagrams.append((c, fig, mermaid1))

    nodes2 = [
        {"id": "profile", "box": (0.01, 0.67, 0.15, 0.18), "title": "Profiles + frozen dispatch", "detail": "load/DER regimes", "kind": "source"},
        {"id": "plant", "box": (0.23, 0.67, 0.15, 0.18), "title": "AC network plant", "detail": "CIGRE MV / IEEE33-DER", "kind": "control"},
        {"id": "ctrl", "box": (0.45, 0.67, 0.15, 0.18), "title": "Controller", "detail": "Volt/VAR, curtailment, shedding", "kind": "control"},
        {"id": "cmd", "box": (0.68, 0.67, 0.15, 0.18), "title": "Command event engine", "detail": "delay, loss, jitter, sequence", "kind": "audit"},
        {"id": "der", "box": (0.85, 0.67, 0.13, 0.18), "title": "DER endpoints", "detail": "delivered commands", "kind": "source"},
        {"id": "tel", "box": (0.68, 0.22, 0.15, 0.18), "title": "Telemetry event engine", "detail": "causal delivery + AoI", "kind": "audit"},
        {"id": "obs", "box": (0.45, 0.22, 0.15, 0.18), "title": "Observability views", "detail": "physical / cyber / hybrid", "kind": "model"},
        {"id": "det", "box": (0.23, 0.22, 0.15, 0.18), "title": "Frozen detector", "detail": "bag + temporal scores", "kind": "model"},
        {"id": "policy", "box": (0.01, 0.22, 0.15, 0.18), "title": "Alarm policy", "detail": "P0–P3 and learned alarms", "kind": "locked"},
    ]
    edges2 = [
        {"from": "profile", "to": "plant", "label": "setpoints"}, {"from": "plant", "to": "ctrl", "label": "state"},
        {"from": "ctrl", "to": "cmd", "label": "commands"}, {"from": "cmd", "to": "der", "label": "actual delivery"},
        {"from": "der", "to": "tel", "orientation": "v", "label": "return telemetry"},
        {"from": "tel", "to": "obs", "label": "delivered only", "connection": "arc3,rad=0"},
        {"from": "obs", "to": "det", "label": "causal tensors", "connection": "arc3,rad=0"},
        {"from": "det", "to": "policy", "label": "alarms", "connection": "arc3,rad=0"},
        {"from": "policy", "to": "ctrl", "orientation": "v", "label": "mode selection", "route": [(0.18, 0.08), (0.40, 0.08), (0.40, 0.58), (0.45, 0.58)], "end_xy": (0.45, 0.67), "label_xy": (0.29, 0.08), "label_dy": 0.012},
    ]
    title = "Cyber-physical detection and control architecture"
    subtitle = "Command and telemetry impairments are event-driven; every detector and controller read is restricted to messages already delivered"
    fig = draw_diagram(title, subtitle, nodes2, edges2, "Logical-time architecture only; no wall-clock or real-time execution claim.")
    c = contract("diag02-system-architecture", "Diagram 2", title, subtitle, "Where do impairments and detectors sit in the loop?", "Both command and return telemetry paths are impaired while causal delivery gates all reads.", "Structure", "cyber-physical architecture", "src/gridimpair", 9, "Architecture schematic; component placement does not quantify performance.")
    mermaid2 = """flowchart LR
  P[Profiles + frozen dispatch] --> N[AC network plant] --> C[Controller]
  C --> CE[Command event engine] --> D[DER endpoints]
  D --> TE[Telemetry event engine] --> O[Observability views] --> M[Frozen detector] --> A[Alarm policy]
  A --> C
"""
    diagrams.append((c, fig, mermaid2))

    nodes3 = [
        {"id": "freeze", "box": (0.01, 0.62, 0.14, 0.20), "title": "Protocol freeze", "detail": "configs, hashes, split manifest", "kind": "audit"},
        {"id": "pilot", "box": (0.20, 0.62, 0.12, 0.20), "title": "Pilot gate", "detail": "48 runs", "kind": "source"},
        {"id": "pairs", "box": (0.37, 0.62, 0.14, 0.20), "title": "Paired scenarios", "detail": "1,008 clean/impaired runs", "kind": "source"},
        {"id": "dev", "box": (0.57, 0.62, 0.13, 0.20), "title": "Development", "detail": "fit + HPO", "kind": "model"},
        {"id": "vala", "box": (0.75, 0.62, 0.10, 0.20), "title": "Validation-A", "detail": "selection", "kind": "model"},
        {"id": "valb", "box": (0.88, 0.62, 0.10, 0.20), "title": "Validation-B", "detail": "calibration", "kind": "model"},
        {"id": "freeze2", "box": (0.67, 0.18, 0.16, 0.20), "title": "Frozen decision system", "detail": "ensemble, calibrator, threshold, dwell", "kind": "locked"},
        {"id": "locked", "box": (0.43, 0.18, 0.16, 0.20), "title": "One-time locked access", "detail": "288 runs; network shift", "kind": "locked"},
        {"id": "report", "box": (0.17, 0.18, 0.18, 0.20), "title": "Paired reporting", "detail": "bag, temporal, robustness, control", "kind": "control"},
    ]
    edges3 = [
        {"from": "freeze", "to": "pilot"}, {"from": "pilot", "to": "pairs"}, {"from": "pairs", "to": "dev"},
        {"from": "dev", "to": "vala"}, {"from": "vala", "to": "valb"},
        {"from": "valb", "to": "freeze2", "orientation": "v"}, {"from": "freeze2", "to": "locked", "connection": "arc3,rad=0"},
        {"from": "locked", "to": "report", "connection": "arc3,rad=0"},
    ]
    title = "Reproducible training-to-locked reporting flow"
    subtitle = "Selection, calibration and temporal decision rules are frozen before the one-time locked network-shift evaluation"
    fig = draw_diagram(title, subtitle, nodes3, edges3, "Split-level flow; epochs and windows are not independent locked observations.")
    c = contract("diag03-locked-dataflow", "Diagram 3", title, subtitle, "How is locked access isolated from model development?", "Every fitted component is frozen before locked access and reporting uses scenario pairs.", "Progression", "protocol flowchart", "artifacts_q1_v2 protocol manifests", 9, "Protocol schematic; counts describe the executed design, not statistical power by themselves.")
    mermaid3 = """flowchart LR
  F[Protocol freeze] --> P[Pilot gate] --> S[Paired scenarios]
  S --> D[Development] --> A[Validation-A] --> B[Validation-B]
  B --> R[Frozen decision system] --> L[One-time locked access] --> O[Paired reporting]
"""
    diagrams.append((c, fig, mermaid3))
    return diagrams


def render_q1(entry: dict, index: int) -> tuple[ChartContract, plt.Figure]:
    figure_id = entry["id"]
    source = Path(entry["source"])
    data = pd.read_csv(source)
    main = entry["section"] == "main"
    figure_label = f"Main Figure {index}" if main else f"Supplementary Figure {index}"
    title = entry["title"]
    boundary = "Descriptive or prespecified diagnostic; interpret with the study protocol and stated statistical unit."
    subtitle = f"Frozen q1-v2 artifact | N = {len(data):,} plotted rows"
    family, variant = "Comparison & Ranking", entry["kind"]

    if figure_id == "fig01-data-quality":
        fig = figure_shell(title, "15 source CSVs; issue count combines schema, non-finite, duplicate-row and irregular-grid failures")
        ax = axis_for(fig)
        x = np.arange(len(data))
        ax.scatter(x, data.issue_count, s=70, facecolor=OPEN_BLUE, edgecolor=BLUE, linewidth=1.2)
        ax.set_xticks(x, data.file_id, rotation=45, ha="right")
        ax.set_ylim(-0.15, 1.0)
        ax.set_ylabel("Detected issue count")
        ax.text(0.99, 0.88, "0 issues in every file", transform=ax.transAxes, ha="right", fontsize=11, fontweight="bold", color=BLUE)
        style_axis(ax)
        add_footer(fig, source, "Passing these checks does not imply informative or non-redundant channels.")
        subtitle = "15 source CSVs; deterministic structural checks"
        family, variant = "Uncertainty & Benchmark", "zero-event dot strip"
    elif figure_id == "fig02-split-leakage":
        d = data.copy(); d["split_strategy"] = d.split_strategy.map({"random_window": "Random window", "grouped_file": "Grouped file"})
        fig = horizontal_bar(d, "split_strategy", "score", title, "700 overlapping windows; file overlap = 14 vs 0", source, "Exploratory source-data diagnostic, not final model performance.", reference=0.5, xlim=(0, 1.02))
        subtitle = "700 overlapping windows; AUPRC; file overlap = 14 vs 0"
    elif figure_id == "fig03-causal-timeline":
        fig = figure_shell(title, "First 120 event-ledger rows; production, intended-delivery and actual-delivery lanes")
        ax = axis_for(fig)
        colors = {"produced": BLUE, "intended": ORANGE, "delivered": TEAL}
        markers = {"produced": "o", "intended": "s", "delivered": "^"}
        for event, subset in data.groupby("event"):
            ax.scatter(subset.logical_time + subset.sequence * 0.003, subset.lane, s=26, color=colors[event], marker=markers[event], label=event, alpha=0.78)
        ax.set_yticks(sorted(data.lane.unique()), ["production", "intended", "actual delivery"])
        ax.set_xlabel("Logical time with sequence jitter for visibility")
        ax.set_ylabel("Event lane")
        ax.legend(frameon=False, ncol=3, loc="upper left")
        style_axis(ax, "x")
        add_footer(fig, source, "Ledger chronology verifies causality; it is not a delay-severity distribution.")
        subtitle = "120 ledger rows; logical time"
        family, variant = "Sequence", "aligned event-lane timeline"
    elif figure_id == "fig04-scenario-matrix":
        fig = figure_shell(title, "Executed main design: 504 clean + 504 impaired runs = 504 matched scenario pairs")
        ax = axis_for(fig, (0.16, 0.24, 0.75, 0.48))
        clean = int(data.loc[data.condition == "clean", "run_count"].iloc[0]); impaired = int(data.loc[data.condition == "impaired", "run_count"].iloc[0])
        ax.barh([0], [-clean], color=OPEN_BLUE, edgecolor=BLUE, height=0.45, label="clean")
        ax.barh([0], [impaired], color=OPEN_ORANGE, edgecolor=ORANGE, height=0.45, label="impaired")
        ax.axvline(0, color=INK, linewidth=1)
        ax.text(-clean / 2, 0, f"Clean\n{clean}", ha="center", va="center", fontweight="bold")
        ax.text(impaired / 2, 0, f"Impaired\n{impaired}", ha="center", va="center", fontweight="bold")
        ax.set_xlim(-560, 560); ax.set_yticks([]); ax.set_xlabel("Run count (mirrored to show pairing)")
        ax.set_xticks([-500, -250, 0, 250, 500], ["500", "250", "0", "250", "500"])
        style_axis(ax, "x")
        add_footer(fig, source, "Bars show paired-status balance; factorial coverage is defined by the executed protocol.")
        subtitle = "1,008 runs; 504 matched pairs"
        family, variant = "Composition", "mirrored paired-balance bar"
    elif figure_id in {"fig05-detectability-floor", "figs05-best-channel", "figs20-false-alarms"}:
        category = entry["x"]; value = entry["y"]
        ref = 0.5 if value in {"auprc", "score"} else None
        xlim = (0, 1.03) if value in {"auprc", "score"} else None
        sub = "Locked-v2 bag AUPRC; balanced prevalence reference = 0.50" if figure_id == "fig05-detectability-floor" else ("Grouped-file single-channel diagnostic AUPRC" if figure_id == "figs05-best-channel" else "Frozen temporal decisions; false alarm episodes per run")
        fig = horizontal_bar(data.assign(**{category: data[category].map(label if category == "model" else short_channel)}), category, value, title, sub, source, boundary, reference=ref, xlim=xlim)
        subtitle = sub
        family, variant = "Comparison & Ranking", "sorted horizontal bar with direct values"
    elif figure_id in {"fig06-open-loop", "fig09-localization"}:
        fig = dot_ci(data, "model", entry["y"], "ci_lower", "ci_upper", title, "Point estimate with stored paired-cluster bootstrap interval", source, boundary, xlim=(0, 1.03 if figure_id == "fig06-open-loop" else 0.52))
        subtitle = "Locked-v2 point estimates with paired-cluster bootstrap intervals"
        family, variant = "Uncertainty & Benchmark", "forest dot-and-interval"
    elif figure_id == "fig07-calibration":
        data = data[data.model != "ideal"].copy()
        def draw(ax, subset, idx):
            ax.plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=0.8)
            sizes = 18 + 130 * subset["count"].to_numpy(float) / max(1, data["count"].max())
            ax.scatter(subset.predicted_probability, subset.observed_frequency, s=sizes, facecolor=OPEN_BLUE, edgecolor=BLUE, linewidth=0.8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); style_axis(ax)
        fig = small_multiples(data, "model", draw, title, "Locked-v2 reliability; marker area encodes bin count; dashed line is ideal calibration", source, "Descriptive reliability plot; no calibration model is refit on locked data.", columns=4)
        subtitle = "Locked-v2 reliability; marker area = bin count"
        family, variant = "Uncertainty & Benchmark", "faceted reliability diagram"
    elif figure_id == "fig08-risk-coverage":
        def draw(ax, subset, idx):
            ax.plot(subset.coverage, subset.risk, color=BLUE, linewidth=1.2)
            ax.set_xlim(0, 1); ax.set_ylim(-0.025, 1); style_axis(ax)
        fig = small_multiples(data, "model", draw, title, "Locked-v2 selective risk; shared 0–1 axes across model facets", source, "Selective-risk diagnostic; it does not establish closed-loop benefit.", columns=3)
        subtitle = "Locked-v2; 288 ordered predictions per model"
        family, variant = "Trend", "shared-scale small multiples"
    elif figure_id in {"fig10-closed-loop", "figs26-policy-regret"}:
        if figure_id == "fig10-closed-loop":
            d = data.sort_values("endpoint_delta")
            fig = figure_shell(title, "Impaired minus clean voltage-violation duration; negative values favor the impaired-policy realization")
            ax = axis_for(fig, (0.20, 0.16, 0.74, 0.66)); y=np.arange(len(d)); v=d.endpoint_delta.to_numpy(float)
            ax.hlines(y, 0, v, color=NEUTRAL, linewidth=2); ax.scatter(v,y,c=[BLUE if x<0 else ORANGE for x in v],s=55,zorder=3)
            ax.axvline(0,color=INK,linewidth=.9); ax.set_yticks(y,d.policy); ax.set_xlabel("Voltage-violation duration delta (s/run)"); style_axis(ax,"x")
            for yi,x in zip(y,v): ax.text(x + (0.35 if x>=0 else -0.35), yi, f"{x:+.2f}", ha="left" if x>=0 else "right", va="center", fontsize=8)
            add_footer(fig, source, "Policy-controller endpoint effect; not learned-detector superiority.")
            subtitle = "N = 144 matched scenario pairs; seconds per run"
            family, variant = "Decomposition & Progression", "zero-anchored signed lollipop"
        else:
            d=data.copy(); summary=d.groupby("policy").regret.agg(["min","max","mean"]).reset_index().sort_values("mean")
            fig=figure_shell(title,"Two stored regret contrasts per policy; range and mean shown without assigning an undocumented status label")
            ax=axis_for(fig,(0.20,0.16,0.74,0.66)); y=np.arange(len(summary))
            ax.hlines(y,summary["min"],summary["max"],color=INK,linewidth=1.5); ax.scatter(summary["min"],y,facecolor="white",edgecolor=BLUE,s=45); ax.scatter(summary["max"],y,color=ORANGE,s=45); ax.scatter(summary["mean"],y,color=INK,marker="|",s=100)
            ax.axvline(0,color=INK,linewidth=.9); ax.set_yticks(y,summary.policy); ax.set_xlabel("Regret versus P0 (s; negative indicates improvement)"); style_axis(ax,"x")
            ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markeredgecolor=BLUE,markerfacecolor='white',label='stored minimum'),Line2D([0],[0],marker='o',color='none',markerfacecolor=ORANGE,markeredgecolor=ORANGE,label='stored maximum'),Line2D([0],[0],marker='|',color=INK,label='mean')],frameon=False,ncol=3,loc='lower left')
            add_footer(fig,source,"The source table lacks a status field; the two contrasts are not relabelled by assumption.")
            subtitle="Two stored contrasts per policy; seconds"
            family,variant="Distribution","range-and-mean plot"
    elif figure_id == "fig12-robustness":
        pivot=data.pivot(index="model",columns="shift",values="score").sort_values("burst")
        fig=figure_shell(title,"Separate 120-run robustness set; VUS-PR under burst-loss and bounded-jitter shifts")
        ax=axis_for(fig,(0.23,0.16,0.71,0.66)); y=np.arange(len(pivot))
        ax.hlines(y,pivot.min(axis=1),pivot.max(axis=1),color=NEUTRAL,linewidth=2)
        ax.scatter(pivot["burst"],y,color=BLUE,s=55,label="burst"); ax.scatter(pivot["jitter"],y,facecolor="white",edgecolor=ORANGE,s=55,label="jitter")
        ax.set_yticks(y,[label(v) for v in pivot.index]); ax.set_xlim(0,1); ax.set_xlabel("VUS-PR"); style_axis(ax,"x"); ax.legend(frameon=False,ncol=2,loc="lower right")
        add_footer(fig,source,"Shift-specific robustness; do not infer a universal ordering beyond these conditions.")
        subtitle="120-run robustness set; VUS-PR"
        family,variant="Comparison & Ranking","dumbbell comparison"
    elif figure_id == "figs01-schema":
        fig=figure_shell(title,"All 15 source files share one ordered 19-column schema hash")
        ax=axis_for(fig,(0.16,0.28,0.74,0.38)); ax.barh([0],[15],color=OPEN_BLUE,edgecolor=BLUE,height=.45); ax.text(7.5,0,"15 / 15 files",ha="center",va="center",fontweight="bold",fontsize=13); ax.set_xlim(0,15); ax.set_yticks([]); ax.set_xlabel("Files with matching ordered schema"); style_axis(ax,"x")
        add_footer(fig,source,"Schema agreement does not establish semantic equivalence or informative channels.")
        subtitle="15 files; one ordered schema hash"
        family,variant="Uncertainty & Benchmark","completion bar"
    elif figure_id == "figs02-missingness":
        d=data.copy(); d["channel"]=d.channel.map(short_channel)
        fig=horizontal_bar(d,"channel","missing_fraction",title,"49,995 rows × 19 columns; all observed missing fractions are zero",source,"Zero explicit missingness does not exclude constant or duplicated dynamics.",xlim=(0,0.05))
        subtitle="49,995 rows; deterministic missing-value audit"
        family,variant="Uncertainty & Benchmark","zero-value horizontal dot plot"
    elif figure_id == "figs03-channel-variance":
        values=data.variance.to_numpy(float); transformed=np.log10(values+1e-12)
        fig=figure_shell(title,"285 file-channel combinations; log10(variance + 10⁻¹²) preserves the zero-variance mass")
        ax=axis_for(fig); ax.hist(transformed,bins=30,color=OPEN_BLUE,edgecolor=BLUE,linewidth=.7); ax.set_xlabel("log10(variance + 10⁻¹²)"); ax.set_ylabel("File-channel combinations"); style_axis(ax)
        ax.text(.98,.88,f"Zero variance: {(values==0).sum()} combinations",transform=ax.transAxes,ha="right",fontweight="bold",color=BLUE)
        add_footer(fig,source,"Variance is a redundancy diagnostic, not a sensor-failure label.")
        subtitle="15 files × 19 columns = 285 combinations"
        family,variant="Distribution","log-transformed histogram"
    elif figure_id == "figs04-fingerprints":
        fig=figure_shell(title,"700 overlapping windows; color = normal/attack bag label; independent grouping unit = source file")
        ax=axis_for(fig); colors=np.where(data.label.astype(int)==0,BLUE,ORANGE); ax.scatter(data.component_1,data.component_2,c=colors,s=18,alpha=.55,edgecolors="none")
        ax.set_xlabel("PCA component 1"); ax.set_ylabel("PCA component 2"); style_axis(ax)
        ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=BLUE,label='normal'),Line2D([0],[0],marker='o',color='none',markerfacecolor=ORANGE,label='attack-family bag')],frameon=False)
        add_footer(fig,source,"PCA is exploratory and does not establish causal attack signatures.")
        subtitle="700 windows from 14 source files; PCA visualization only"
        family,variant="Relationship","PCA scatter"
    elif figure_id == "figs06-time-grid":
        residual=(data.interval.to_numpy(float)-0.00015)/1e-16
        fig=figure_shell(title,"49,980 intervals; residuals from nominal 0.00015 coordinate units are floating-point scale")
        ax=axis_for(fig); ax.hist(residual,bins=np.linspace(residual.min()-.05,residual.max()+.05,18),color=OPEN_BLUE,edgecolor=BLUE); ax.set_xlabel("Residual from 0.00015 (10⁻¹⁶ coordinate units)"); ax.set_ylabel("Intervals"); style_axis(ax)
        add_footer(fig,source,"The regular coordinate is not converted to wall-clock seconds.")
        subtitle="49,980 consecutive intervals"
        family,variant="Distribution","floating-point residual histogram"
    elif figure_id == "figs08-clean-voltage":
        value=float(data.vm_pu.iloc[0])
        fig=figure_shell(title,"900 logical 1-s steps; all stored values equal 1.0300 pu within floating-point precision")
        ax=axis_for(fig,(0.16,0.27,0.74,0.40)); ax.scatter([value],[0],s=130,facecolor=OPEN_BLUE,edgecolor=BLUE,linewidth=1.4)
        ax.set_xlim(value-.004,value+.004); ax.set_ylim(-0.5,0.5); ax.set_yticks([]); ax.set_xlabel("Voltage magnitude (pu; focused range around the stored value)"); ax.text(.5,.68,f"min = median = max = {value:.4f} pu",transform=ax.transAxes,ha="center",fontweight="bold",color=BLUE); style_axis(ax,"x")
        add_footer(fig,source,"Constant clean-run sanity check; not a dynamic voltage-response result or grid-code certification.")
        subtitle="900 logical 1-s steps; constant stored value"
        family,variant="Uncertainty & Benchmark","constant-value status dot"
    elif figure_id in {"figs07-profile-regimes","figs09-clean-loading"}:
        fig=figure_shell(title, "900 logical 1-s steps" if figure_id != "figs07-profile-regimes" else "Three seed-reproducible operating regimes; 900 logical 1-s steps each")
        ax=axis_for(fig)
        if "regime" in data.columns:
            for i,(name,sub) in enumerate(data.groupby("regime")): ax.plot(sub[entry["x"]],sub[entry["y"]],color=PALETTE[i],linewidth=1.0,label=label(name))
            ax.legend(frameon=False,ncol=3,loc="upper center")
        else:
            ax.plot(data[entry["x"]],data[entry["y"]],color=BLUE,linewidth=1.2)
            if figure_id=="figs09-clean-loading": ax.axhline(100,color=INK,linestyle="--",linewidth=.9,label="100% reference"); ax.legend(frameon=False)
        ax.set_xlabel(entry["x"].replace("_"," ")); ax.set_ylabel(entry["y"].replace("_"," ")); style_axis(ax)
        add_footer(fig,source,"Synthetic profile/plant sanity check; not a measured population distribution or grid-code certification.")
        subtitle="900 logical 1-s steps" + (" per regime" if "regime" in data.columns else "")
        family,variant="Trend","highlighted line" if "regime" not in data.columns else "multi-series line"
    elif figure_id == "figs10-event-replay":
        fig=figure_shell(title,"First 500 deterministic event sequences; delivery time in logical steps")
        ax=axis_for(fig); ax.scatter(data.sequence,data.delivery_time,s=11,facecolor=OPEN_BLUE,edgecolor=BLUE,linewidth=.4,alpha=.8); ax.set_xlabel("Event sequence"); ax.set_ylabel("Delivery time (logical step)"); style_axis(ax)
        add_footer(fig,source,"Deterministic replay evidence; not a latency benchmark.")
        subtitle="500 event sequences"
        family,variant="Relationship","sequence scatter"
    elif figure_id in {"figs11-delay-severity","figs12-drop-severity","figs14-regime-heterogeneity"}:
        column=entry["x"]; fig=heatmap(data,"model",column,"score",title,"Shared 0–1 scale; each cell is the stored subgroup score",source,"Subgroup point estimates are not a causal dose-response analysis.",vmin=0,vmax=1)
        subtitle=f"{len(data)} model-condition cells; shared 0–1 score scale"
        family,variant="Matrix & Cohort","annotated heatmap"
    elif figure_id == "figs13-endpoint-fraction":
        pivot=data.pivot(index="model",columns="endpoint_fraction",values="score").sort_values(0.25)
        fig=figure_shell(title,"Stored score at 0.25 and 1.00 affected-endpoint fractions")
        ax=axis_for(fig,(0.29,0.14,0.66,0.68)); y=np.arange(len(pivot)); ax.hlines(y,pivot[.25],pivot[1.0],color=NEUTRAL,linewidth=1.8); ax.scatter(pivot[.25],y,facecolor="white",edgecolor=BLUE,s=42,label="0.25"); ax.scatter(pivot[1.0],y,color=ORANGE,s=42,label="1.00")
        ax.set_yticks(y,[label(v) for v in pivot.index]); ax.set_xlim(0,1); ax.set_xlabel("Stored subgroup score"); style_axis(ax,"x"); ax.legend(frameon=False,ncol=2,loc="lower right")
        add_footer(fig,source,"Two tested fractions do not define a continuous scaling law.")
        subtitle="14 model families × 2 endpoint fractions"
        family,variant="Comparison & Ranking","dumbbell slope comparison"
    elif figure_id == "figs15-seed-variance":
        fig=figure_shell(title,"Three final seeds per model; logarithmic loss axis exposes both near-zero and chance-level solutions")
        ax=axis_for(fig,(0.28,0.14,0.67,0.68)); models=list(dict.fromkeys(data.model)); ybase=np.arange(len(models))
        for i,seed in enumerate(sorted(data.seed.unique())):
            vals=data[data.seed==seed].set_index("model").reindex(models).validation_loss
            ax.scatter(vals,ybase+(i-1)*.15,s=38,color=PALETTE[i],label=str(seed),alpha=.9)
        ax.set_xscale("log"); ax.set_yticks(ybase,[label(v) for v in models]); ax.set_xlabel("Best validation loss (log scale)"); style_axis(ax,"x"); ax.legend(frameon=False,ncol=3)
        add_footer(fig,source,"Seeds characterize optimization variability; they are not independent scenario samples.")
        subtitle="9 model families × seeds 2026, 2027 and 2028"
        family,variant="Distribution","grouped dot plot on log scale"
    elif figure_id == "figs16-hpo-response":
        def draw(ax,subset,idx):
            ax.scatter(subset.trial_id,subset.validation_loss,s=20,facecolor=OPEN_BLUE,edgecolor=BLUE,linewidth=.5)
            positive=subset.validation_loss[subset.validation_loss>0]
            if len(positive) and positive.max()/positive.min()>50: ax.set_yscale("log")
            else: ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel("trial"); style_axis(ax)
        fig=small_multiples(data,"model",draw,title,"Latin-hypercube trials; validation loss on logarithmic scale",source,"Development-only HPO trials are not independent test replications.",columns=3,sharex=True,sharey=False)
        subtitle="Six tuned families; 16 trials per family where executed"
        family,variant="Relationship","faceted HPO response scatter"
    elif figure_id == "figs17-training-curves":
        def draw(ax,subset,idx):
            subset=subset.sort_values("epoch"); ax.plot(subset.epoch,subset.validation_loss,color=BLUE,linewidth=1.1)
            lower=np.clip(subset.validation_lower.to_numpy(float),1e-7,None); upper=np.clip(subset.validation_upper.to_numpy(float),1e-7,None)
            ax.fill_between(subset.epoch,lower,upper,color=OPEN_BLUE,alpha=.7,linewidth=0)
            positive=subset.validation_loss[subset.validation_loss>0]
            if len(positive) and positive.max()/positive.min()>50: ax.set_yscale("log")
            else: ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel("epoch"); style_axis(ax)
        fig=small_multiples(data,"model",draw,title,"Validation loss by epoch; bands encode stored seed variability",source,"Epochs and seeds are optimization observations, not independent scenarios.",columns=3,sharex=False,sharey=False)
        subtitle="Final training phase; logarithmic validation-loss axes"
        family,variant="Trend","faceted learning curves with uncertainty bands"
    elif figure_id == "figs18-attention":
        fig=figure_shell(title,"One locked scenario; attention weights and binary impairment interval")
        ax=axis_for(fig); ax.plot(data.logical_time,data.attention,color=BLUE,linewidth=1.1,label="attention")
        mask=data.label.to_numpy(float)>0; ax.fill_between(data.logical_time,0,data.attention.max(),where=mask,color=OPEN_ORANGE,alpha=.6,label="source interval")
        ax.set_xlabel("Logical time (s)"); ax.set_ylabel("Attention weight"); style_axis(ax); ax.legend(frameon=False)
        add_footer(fig,source,"Attention is a model-internal weighting trace, not a causal explanation.")
        subtitle="900 logical 1-s steps; scenario q1v2-r0-c00-rep10-b"
        family,variant="Trend","line with interval overlay"
    elif figure_id == "figs19-delay-distribution":
        def draw(ax,subset,idx):
            values=np.sort(subset.delay.to_numpy(float)); y=np.arange(1,len(values)+1)/len(values); ax.step(values,y,where="post",color=BLUE,linewidth=1.1); ax.set_xlim(left=0); ax.set_ylim(0,1); style_axis(ax)
        fig=small_multiples(data,"model",draw,title,"Empirical cumulative distribution of detected-event onset delay",source,"Delay must be interpreted jointly with recall, IoU and false alarms/run.",columns=3,sharex=False,sharey=True)
        subtitle=f"{len(data):,} detected-event delay observations; seconds"
        family,variant="Distribution","faceted ECDF"
    elif figure_id == "figs21-bootstrap":
        fig=dot_ci(data,"baseline","effect","ci_lower","ci_upper",title,"Candidate-minus-baseline bag-discrimination margin; 97.5% paired bootstrap intervals",source,"Bag-discrimination margins are not closed-loop endpoint effects.",zero=True)
        subtitle="19 paired comparisons; 10,000-resample stratified bootstrap"
        family,variant="Uncertainty & Benchmark","forest plot"
    elif figure_id == "figs22-compute":
        fig=figure_shell(title,"Stored GPU-hours versus locked bag AUPRC; each point is one neural model family")
        ax=axis_for(fig); ax.scatter(data.gpu_hours,data.score,s=55,facecolor=OPEN_BLUE,edgecolor=BLUE)
        for _,r in data.iterrows(): ax.annotate(label(r.model),(r.gpu_hours,r.score),xytext=(4,4),textcoords="offset points",fontsize=7)
        ax.set_xlabel("Training GPU-hours"); ax.set_ylabel("Locked bag AUPRC"); ax.set_xlim(left=0); ax.set_ylim(.47,1.02); style_axis(ax)
        add_footer(fig,source,"Hardware- and implementation-specific trade-off; not a complexity bound.")
        subtitle="9 neural models; local workstation measurements"
        family,variant="Relationship","labelled compute-performance scatter"
    elif figure_id == "figs23-latency":
        fig=figure_shell(title,"CPU batch-1 and CUDA batch-64 measurements are separated because they are not directly rank-comparable")
        axes=fig.subplots(1,2); fig.subplots_adjust(left=.18,right=.98,bottom=.14,top=.80,wspace=.42)
        for ax,(device,subset),color in zip(axes,data.groupby("device"),[BLUE,ORANGE]):
            subset=subset.sort_values("latency_ms"); y=np.arange(len(subset)); ax.barh(y,subset.latency_ms,color=OPEN_BLUE if device=="cpu" else OPEN_ORANGE,edgecolor=color); ax.set_yticks(y,[label(v) for v in subset.model]); ax.set_xscale("log"); ax.set_xlabel("Latency (ms; log scale)"); ax.set_title(f"{device.upper()} | batch {subset.batch_size.iloc[0]}",loc="left",fontweight="bold"); style_axis(ax,"x")
        add_footer(fig,source,"Different devices and batch sizes preclude a single global ranking or deadline-compliance claim.")
        subtitle="20 stored benchmarks; device and batch size retained"
        family,variant="Comparison & Ranking","device-faceted log bars"
    elif figure_id == "figs24-failure-cases":
        fig=figure_shell(title,"Candidate absolute bag-score error versus scenario activity")
        ax=axis_for(fig); markers={0.25:"o",1.0:"s"}; regimes=list(dict.fromkeys(data.regime));
        for i,regime in enumerate(regimes):
            for fraction,mark in markers.items():
                sub=data[(data.regime==regime)&(data.endpoint_fraction==fraction)]; ax.scatter(sub.activity,sub.error,s=28,color=PALETTE[i],marker=mark,alpha=.65,label=f"{label(regime)} | f={fraction:g}")
        ax.set_xlabel("Scenario activity index"); ax.set_ylabel("Absolute bag-score error"); style_axis(ax); ax.legend(frameon=False,ncol=2,fontsize=7)
        add_footer(fig,source,"Diagnostic map; condition-specific failure probability requires denominators and uncertainty.")
        subtitle="144 locked scenario identifiers; marker shape = affected endpoint fraction"
        family,variant="Relationship","grouped scatter with shape encoding"
    elif figure_id == "figs25-observability":
        d=data.copy(); d["model"]=d.model.map(label); tiers=["physical","cyber","hybrid"]
        fig=figure_shell(title,"Locked bag AUPRC stratified by observability tier")
        axes=fig.subplots(1,3,sharex=True); fig.subplots_adjust(left=.09,right=.98,bottom=.14,top=.80,wspace=.45)
        for ax,tier,color in zip(axes,tiers,[BLUE,ORANGE,TEAL]):
            sub=d[d.tier==tier].sort_values("score"); y=np.arange(len(sub)); ax.scatter(sub.score,y,s=50,color=color); ax.set_yticks(y,sub.model); ax.set_xlim(.47,1.02); ax.axvline(.5,color=INK,linestyle="--",linewidth=.8); ax.set_title(tier.capitalize(),fontweight="bold"); ax.set_xlabel("AUPRC"); style_axis(ax,"x")
        add_footer(fig,source,"Observability result; it does not support novel-architecture superiority.")
        subtitle="15 locked model/tier estimates; N = 288 runs"
        family,variant="Comparison & Ranking","tier-faceted dot plot"
    elif figure_id == "figs27-absolute-endpoints":
        pivot=data.pivot(index="policy",columns="status",values="endpoint_value").sort_values("clean")
        fig=figure_shell(title,"Absolute voltage-violation duration by policy and paired status")
        ax=axis_for(fig,(.20,.16,.74,.66)); y=np.arange(len(pivot)); ax.hlines(y,pivot.clean,pivot.impaired,color=NEUTRAL,linewidth=2); ax.scatter(pivot.clean,y,facecolor="white",edgecolor=BLUE,s=50,label="clean"); ax.scatter(pivot.impaired,y,color=ORANGE,s=50,label="impaired")
        ax.set_yticks(y,pivot.index); ax.set_xlabel("Voltage-violation duration (s/run; focused scale)"); style_axis(ax,"x"); ax.legend(frameon=False,ncol=2)
        add_footer(fig,source,"Focused scale exposes paired changes; exact absolute levels remain labelled by the axis.")
        subtitle="144 locked pairs; seconds per run"
        family,variant="Comparison & Ranking","paired-status dumbbell"
    else:
        # Conservative fallback for a manifest entry not covered above.
        x=entry.get("x"); y=entry.get("y"); kind=entry["kind"]
        if kind=="bar" and x and y:
            fig=horizontal_bar(data,x,y,title,subtitle,source,boundary)
        elif kind=="histogram" and x:
            fig=figure_shell(title,subtitle); ax=axis_for(fig); ax.hist(data[x].dropna(),bins=30,color=OPEN_BLUE,edgecolor=BLUE); ax.set_xlabel(x.replace("_"," ")); ax.set_ylabel("Count"); style_axis(ax); add_footer(fig,source,boundary)
        elif kind=="scatter" and x and y:
            fig=figure_shell(title,subtitle); ax=axis_for(fig); ax.scatter(data[x],data[y],s=25,facecolor=OPEN_BLUE,edgecolor=BLUE); ax.set_xlabel(x.replace("_"," ")); ax.set_ylabel(y.replace("_"," ")); style_axis(ax); add_footer(fig,source,boundary)
        else:
            fig=figure_shell(title,subtitle); ax=axis_for(fig); ax.plot(data[x],data[y],color=BLUE); ax.set_xlabel(x.replace("_"," ")); ax.set_ylabel(y.replace("_"," ")); style_axis(ax); add_footer(fig,source,boundary)

    c = contract(figure_id, figure_label, title, subtitle, f"What pattern does {title.lower()} show?", "The redesigned chart exposes the stored comparison without changing source values.", family, variant, source.relative_to(ROOT), len(data), boundary)
    return c, fig


def render_dataset(figure_id: str, index: int) -> tuple[ChartContract, plt.Figure]:
    source=DATASET_SOURCES/f"{figure_id}-source.csv"; data=pd.read_csv(source)
    label_id=f"Dataset Figure {index}"
    boundary="Secondary-data diagnostic only; no primary q1-v2 model fitting or event-localization claim."
    if figure_id=="fig01-data-lineage":
        nodes=[]
        xs=[.02,.27,.52,.77]; kinds=["source","audit","audit","locked"]
        for i,r in data.iterrows(): nodes.append({"id":str(i),"box":(xs[i],.38,.19,.32),"title":r.label,"detail":r.detail,"kind":kinds[i]})
        edges=[{"from":"0","to":"1","label":"inventory"},{"from":"1","to":"2","label":"diagnostics"},{"from":"2","to":"3","label":"no data transfer","style":"--","color":ORANGE}]
        title="Third-party data lineage and analytical boundary"; subtitle="Source CSVs are audited but never transferred into q1-v2 fitting, calibration, locked testing or control"
        fig=draw_diagram(title,subtitle,nodes,edges,"The dashed edge denotes design influence only, not a model-data path.")
        family,variant="Structure","data-lineage diagram"
    elif figure_id=="fig02-corpus-composition":
        title="Delivered corpus composition by source-assigned family"; subtitle="Equal 3,333-row bags; file count and row count shown in separate panels"
        fig=figure_shell(title,subtitle); axes=fig.subplots(1,2); fig.subplots_adjust(left=.10,right=.97,bottom=.18,top=.80,wspace=.34)
        for ax,value,xlabel in zip(axes,["file_count","row_count"],["Files","Rows"]):
            d=data.sort_values(value); y=np.arange(len(d)); ax.barh(y,d[value],color=OPEN_BLUE,edgecolor=BLUE); ax.set_yticks(y,d.source_class); ax.set_xlabel(xlabel); style_axis(ax,"x")
            for yi,v in zip(y,d[value]): ax.text(v,yi,f" {int(v):,}",va="center",fontsize=8)
        add_footer(fig,source,"Counts are package composition, not independent experimental replications.")
        family,variant="Composition","two-panel horizontal bars"
    elif figure_id=="fig03-representative-trajectories":
        title="Representative source-bag trajectories"; subtitle="One file per source-assigned family; units are not documented"
        fig=figure_shell(title,subtitle,figsize=(10.8,7.0)); axes=fig.subplots(2,2,sharex=True); fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.82,hspace=.40,wspace=.25)
        for ax,(family,sub) in zip(axes.flat,data.groupby("source_class",sort=False)):
            for i,(channel,ch) in enumerate(sub.groupby("channel")): ax.plot(ch.time,ch.value,color=PALETTE[i],linewidth=.8,label=channel)
            ax.set_title(f"{family}: {sub.file_id.iloc[0]}",loc="left",fontweight="bold"); ax.set_xlabel("Reported Time coordinate"); ax.set_ylabel("Reported numeric value"); style_axis(ax)
        axes.flat[0].legend(frameon=False,ncol=2,fontsize=7)
        add_footer(fig,source,"Panels are file-level morphology, not row-level attack annotations.")
        family,variant="Trend","shared-layout small multiples"
    elif figure_id=="fig04-time-grid":
        title="Regularity of the delivered Time coordinate"; subtitle="49,980 within-file intervals; nominal step = 0.00015 coordinate units"
        residual=(data.interval.to_numpy(float)-.00015)/1e-16; summary=pd.DataFrame({"file_id":data.file_id,"residual":residual}).groupby("file_id").residual.agg(["min","median","max"])
        fig=figure_shell(title,subtitle); ax=axis_for(fig); x=np.arange(len(summary)); ax.vlines(x,summary["min"],summary["max"],color=NEUTRAL); ax.scatter(x,summary["median"],color=BLUE,s=45); ax.set_xticks(x,summary.index,rotation=45,ha="right"); ax.set_ylabel("Residual from 0.00015 (10⁻¹⁶ units)"); style_axis(ax)
        add_footer(fig,source,"Floating-point residuals do not establish a wall-clock acquisition interval.")
        family,variant="Uncertainty & Benchmark","file-level range-and-median plot"
    elif figure_id=="fig05-constant-channel-heatmap":
        title="Constant file-channel combinations"; subtitle="15 source files × 18 non-Time channels; filled cells denote zero within-file variance"
        matrix=data.pivot(index="file_id",columns="short_channel",values="constant").astype(float)
        fig=figure_shell(title,subtitle); ax=axis_for(fig,(.10,.20,.86,.60)); ax.imshow(matrix,cmap=matplotlib.colors.ListedColormap(["white",BLUE]),vmin=0,vmax=1,aspect="auto"); ax.set_xticks(np.arange(len(matrix.columns)),matrix.columns,rotation=55,ha="right",fontsize=7); ax.set_yticks(np.arange(len(matrix.index)),matrix.index); ax.set_xlabel("Non-Time channel"); ax.set_ylabel("Source file")
        ax.set_xticks(np.arange(-.5,len(matrix.columns),1),minor=True); ax.set_yticks(np.arange(-.5,len(matrix.index),1),minor=True); ax.grid(which="minor",color=GRID,linewidth=.45); ax.tick_params(which="minor",bottom=False,left=False)
        add_footer(fig,source,"Constant fields indicate redundancy or frozen values, not verified sensor failure.")
        family,variant="Matrix & Cohort","binary heatmap"
    elif figure_id=="fig06-file-fingerprint-pca":
        title="Window-summary file-fingerprint structure"; subtitle="700 windows from 14 files; PCA used for visualization only"
        fig=figure_shell(title,subtitle); ax=axis_for(fig)
        for i,(fam,sub) in enumerate(data.groupby("source_class")): ax.scatter(sub.component_1,sub.component_2,s=18,color=PALETTE[i],alpha=.58,label=fam)
        ax.set_xlabel("PCA component 1"); ax.set_ylabel("PCA component 2"); style_axis(ax); ax.legend(frameon=False)
        add_footer(fig,source,"Windows are not independent; the file is the grouping unit.")
        family,variant="Relationship","PCA scatter"
    elif figure_id=="fig07-split-leakage":
        title="Random-window optimism relative to grouped-file evaluation"; subtitle="700 windows; AUPRC and file overlap shown together"
        d=data.copy(); d["name"]=d.split_strategy.map({"random_window":"Random window","grouped_file":"Grouped file"})
        fig=figure_shell(title,subtitle); ax=axis_for(fig,(.18,.20,.70,.56)); x=np.arange(2); ax.bar(x,d.auprc,color=[OPEN_ORANGE,OPEN_BLUE],edgecolor=[ORANGE,BLUE],width=.55); ax.set_xticks(x,d.name); ax.set_ylim(.88,1.0); ax.set_ylabel("Diagnostic AUPRC (focused scale)"); style_axis(ax)
        for xi,(_,r) in zip(x,d.iterrows()): ax.text(xi,r.auprc+.003,f"{r.auprc:.4f}\nfile overlap = {int(r.group_overlap)}",ha="center",fontsize=9)
        add_footer(fig,source,"Focused scale is labelled; diagnostic scores are not primary model results.")
        family,variant="Uncertainty & Benchmark","annotated paired bars"
    elif figure_id=="fig08-univariate-channels":
        title="Leakage-aware single-channel diagnostic ranking"; subtitle="Grouped-file AUPRC; balanced prevalence reference = 0.50"
        d=data.copy(); d["short_channel"]=d.channel.map(short_channel)
        fig=horizontal_bar(d,"short_channel","grouped_auprc",title,subtitle,source,"Channel ranking reflects file/family morphology, not causal event localization.",reference=.5,xlim=(0,1))
        family,variant="Comparison & Ranking","sorted horizontal bars"
    else:
        raise KeyError(figure_id)
    c=contract(figure_id,label_id,title,subtitle,f"What does {title.lower()} show?","The redesigned chart preserves the frozen audit values while clarifying the grouping unit.",family,variant,source.relative_to(ROOT),len(data),boundary)
    return c,fig


def build(output: Path) -> None:
    configure_style()
    output.mkdir(parents=True,exist_ok=True)
    contracts: list[ChartContract]=[]; rendered=[]; order=[]

    for c,fig,mermaid in diagram_definitions():
        rendered.append(save_figure(fig,c.figure_id,output)); contracts.append(c); order.append(c.figure_id)
        (output/f"{c.figure_id}.mmd").write_text(mermaid,encoding="utf-8")

    manifest=json.loads(Q1_MANIFEST.read_text(encoding="utf-8"))["figures"]
    main_i=supp_i=0
    for entry in manifest:
        if entry["section"]=="main": main_i+=1; idx=main_i
        else: supp_i+=1; idx=supp_i
        c,fig=render_q1(entry,idx); rendered.append(save_figure(fig,c.figure_id,output)); contracts.append(c); order.append(c.figure_id)

    dataset_ids=["fig01-data-lineage","fig02-corpus-composition","fig03-representative-trajectories","fig04-time-grid","fig05-constant-channel-heatmap","fig06-file-fingerprint-pca","fig07-split-leakage","fig08-univariate-channels"]
    for idx,figure_id in enumerate(dataset_ids,1):
        c,fig=render_dataset(figure_id,idx)
        atlas_id=f"dataset-{figure_id}"
        c.figure_id=atlas_id
        rendered.append(save_figure(fig,atlas_id,output)); contracts.append(c); order.append(atlas_id)

    chart_map=pd.DataFrame([asdict(c) for c in contracts]); chart_map.to_csv(output/"chart_map.csv",index=False)
    (output/"chart_map.json").write_text(json.dumps([asdict(c) for c in contracts],indent=2,ensure_ascii=False),encoding="utf-8")
    result={"figure_count":len(rendered),"order":order,"figures":rendered,"chart_map_sha256":sha256_file(output/"chart_map.csv"),"source_roots":[str(Q1_SOURCES.relative_to(ROOT)),str(DATASET_SOURCES.relative_to(ROOT))],"formats":["600-dpi PNG","SVG","PDF","300-dpi Word PNG"],"palette":"declared blue/orange/teal/purple/gold roots with direct labels and non-color distinctions"}
    (output/"manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"output":str(output),"figure_count":len(rendered),"main":main_i,"supplementary":supp_i,"dataset":len(dataset_ids),"diagrams":3},indent=2))


def main() -> None:
    parser=argparse.ArgumentParser(description="Regenerate a figures-only Q1 atlas without modifying prior documents.")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args(); build(args.output.resolve())


if __name__=="__main__":
    main()
