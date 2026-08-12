from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manuscript" / "condensed_methods_assets"
WORD = OUTPUT / "word"

BLUE = "#D9EAF7"
GREEN = "#E2F0D9"
ORANGE = "#FCE4D6"
GREY = "#E7E6E6"
RED = "#C00000"
EDGE = "#404040"


def setup() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORD.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    fill: str = BLUE,
    title_size: float = 9.0,
    body_size: float = 7.7,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012",
        facecolor=fill,
        edgecolor=EDGE,
        linewidth=1.15,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.70,
        title,
        ha="center",
        va="center",
        weight="bold",
        fontsize=title_size,
        transform=ax.transAxes,
        linespacing=1.15,
    )
    ax.text(
        x + w / 2,
        y + h * 0.35,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        transform=ax.transAxes,
        linespacing=1.18,
    )


def arrow(ax, start: tuple[float, float], stop: tuple[float, float], *, dashed: bool = False) -> None:
    ax.annotate(
        "",
        xy=stop,
        xytext=start,
        xycoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "->",
            "lw": 1.35,
            "color": EDGE,
            "linestyle": "--" if dashed else "-",
        },
    )


def save(fig, name: str, description: str, source_sections: list[str]) -> dict[str, object]:
    fig.savefig(OUTPUT / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(WORD / f"{name}.png", dpi=340, bbox_inches="tight")
    plt.close(fig)
    provenance = {
        "id": name,
        "description": description,
        "content_source": "Only the two user-specified DOCX manuscripts; no external content added",
        "source_sections": source_sections,
        "formats": ["600-dpi PNG", "PDF", "SVG", "340-dpi Word PNG"],
    }
    (OUTPUT / f"{name}-provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return provenance


def figure_evidence_boundary() -> dict[str, object]:
    fig, ax = plt.subplots(figsize=(11.2, 4.2))
    ax.axis("off")
    box(
        ax,
        0.02,
        0.38,
        0.20,
        0.38,
        "Third-party dataset",
        "Mendeley Data V1\n15 CSV + 15 JPG\n49,995 rows; CC BY 4.0",
        fill=BLUE,
    )
    box(
        ax,
        0.275,
        0.38,
        0.20,
        0.38,
        "Secondary audit",
        "checksums and schema\nconstant-channel checks\nfile-grouped leakage audit",
        fill=ORANGE,
    )
    box(
        ax,
        0.545,
        0.38,
        0.18,
        0.38,
        "Primary simulation",
        "paired clean/impaired\npandapower traces\nmatched scenario seeds",
        fill=GREEN,
    )
    box(
        ax,
        0.78,
        0.38,
        0.19,
        0.38,
        "Confirmatory evidence",
        "fitting and calibration\nlocked network shift\nclosed-loop evaluation",
        fill=GREEN,
    )
    arrow(ax, (0.22, 0.57), (0.275, 0.57))
    arrow(ax, (0.475, 0.57), (0.545, 0.57), dashed=True)
    arrow(ax, (0.725, 0.57), (0.78, 0.57))
    ax.plot([0.51, 0.51], [0.27, 0.84], color=RED, lw=2.2, transform=ax.transAxes)
    ax.text(
        0.51,
        0.18,
        "No source-CSV row, window, label, scaler, residual model, threshold, or coefficient crossed this boundary.",
        ha="center",
        va="center",
        color=RED,
        weight="bold",
        fontsize=8.5,
        transform=ax.transAxes,
    )
    ax.text(
        0.51,
        0.91,
        "Audit findings informed protocol design only",
        ha="center",
        va="center",
        fontsize=8.2,
        transform=ax.transAxes,
    )
    return save(
        fig,
        "fig01-evidence-boundary",
        "Separation between the third-party audit and primary q1-v2 evidence",
        [
            "Dataset document Sections 1, 2 and 5",
            "Materials/Methods document Section 1.1",
        ],
    )


def figure_protocol_flow() -> dict[str, object]:
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.axis("off")
    ax.text(0.5, 0.95, "Paired-counterfactual protocol and frozen information flow", ha="center", weight="bold", fontsize=12, transform=ax.transAxes)
    box(ax, 0.02, 0.59, 0.16, 0.25, "Pilot gate", "48 runs\n24 matched pairs\nfeasibility only", fill=GREY)
    box(ax, 0.22, 0.59, 0.16, 0.25, "Development", "432 runs\nreplicates 0–5\nfit + standardize", fill=BLUE)
    box(ax, 0.42, 0.59, 0.16, 0.25, "Validation-A", "144 runs\nreplicates 6–7\nmodel/HPO selection", fill=BLUE)
    box(ax, 0.62, 0.59, 0.16, 0.25, "Validation-B", "144 runs\nreplicates 8–9\ncalibration + freeze", fill=ORANGE)
    box(ax, 0.82, 0.59, 0.16, 0.25, "Locked-v2", "288 runs\n144 matched pairs\nreplicates 10–13", fill=GREEN)
    for x1, x2 in [(0.18, 0.22), (0.38, 0.42), (0.58, 0.62), (0.78, 0.82)]:
        arrow(ax, (x1, 0.715), (x2, 0.715))
    ax.text(0.50, 0.49, "CIGRE MV-DER: development and validation", ha="center", fontsize=8.6, transform=ax.transAxes)
    ax.text(0.90, 0.49, "IEEE33-DER\nnetwork shift", ha="center", fontsize=8.6, transform=ax.transAxes)
    ax.plot([0.80, 0.80], [0.45, 0.88], color=RED, lw=1.8, transform=ax.transAxes)
    box(
        ax,
        0.11,
        0.10,
        0.24,
        0.25,
        "Matched scenario pair",
        "same seed, profile, schedule,\nendpoint set and topology\nclean versus impaired",
        fill=GREEN,
    )
    box(
        ax,
        0.39,
        0.10,
        0.24,
        0.25,
        "Primary factorial design",
        "3 regimes × 12 conditions ×\n14 replicates × 2 statuses\n= 1,008 runs (504 pairs)",
        fill=ORANGE,
    )
    box(
        ax,
        0.67,
        0.10,
        0.24,
        0.25,
        "Run structure",
        "900 logical 1-s steps\ntwo prespecified episodes\nscenario pair is the unit",
        fill=BLUE,
    )
    arrow(ax, (0.35, 0.225), (0.39, 0.225))
    arrow(ax, (0.63, 0.225), (0.67, 0.225))
    return save(
        fig,
        "fig02-protocol-flow",
        "Executed q1-v2 protocol, split roles and matched-pair unit",
        ["Materials/Methods document Sections 1.1 and 1.3; Table 1"],
    )


def figure_causal_engine() -> dict[str, object]:
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.axis("off")
    ax.text(0.5, 0.94, "Causal communication impairment and controller loop", ha="center", weight="bold", fontsize=12, transform=ax.transAxes)
    box(ax, 0.02, 0.55, 0.17, 0.25, "Message production", "logical production time\nendpoint + sequence\ncommand/telemetry payload", fill=BLUE)
    box(ax, 0.23, 0.55, 0.17, 0.25, "Impairment engine", "fixed delay or jitter\nBernoulli/burst drop\nseeded heap ordering", fill=ORANGE)
    box(ax, 0.44, 0.55, 0.17, 0.25, "Delivery gate", "actual delivery time\nno pre-delivery access\nstale sequence rejected", fill=GREEN)
    box(ax, 0.65, 0.55, 0.15, 0.25, "Observability", "physical summaries\nmessage age/delivery\ncausal residual", fill=BLUE)
    box(ax, 0.84, 0.55, 0.14, 0.25, "Alarm/policy", "frozen detector\nor P0–P3 policy\nlocal/remote mode", fill=GREEN, body_size=7.3)
    for x1, x2 in [(0.19, 0.23), (0.40, 0.44), (0.61, 0.65), (0.80, 0.84)]:
        arrow(ax, (x1, 0.675), (x2, 0.675))
    box(ax, 0.35, 0.11, 0.30, 0.23, "pandapower plant and simplified controller", "Volt/VAR at <0.95 or >1.05 p.u.; curtailment >1.09 p.u.;\nconditional shedding at <0.90 p.u. or loading >120%; cap 25%", fill=GREY, body_size=7.5)
    arrow(ax, (0.91, 0.55), (0.65, 0.31))
    arrow(ax, (0.35, 0.225), (0.11, 0.55))
    ax.text(0.50, 0.03, "Deterministic replay under a fixed seed; logical-time simulation, not wall-clock real-time execution.", ha="center", color=RED, weight="bold", fontsize=8.3, transform=ax.transAxes)
    return save(
        fig,
        "fig03-causal-engine",
        "Causal event-engine, observability and controller information flow",
        ["Materials/Methods document Section 1.4"],
    )


def figure_model_pipeline() -> dict[str, object]:
    fig, ax = plt.subplots(figsize=(11.2, 5.1))
    ax.axis("off")
    ax.text(0.5, 0.95, "Observability-stratified detector and causal AR-residual MIL pipeline", ha="center", weight="bold", fontsize=12, transform=ax.transAxes)
    box(
        ax,
        0.02,
        0.38,
        0.18,
        0.34,
        "Input contract",
        "physical: 15 network summaries\ncyber: age/delivery/stale count\nhybrid: physical + cyber",
        fill=GREY,
    )
    box(ax, 0.25, 0.59, 0.19, 0.25, "Raw branch", "standardized raw values\n+ first differences\ncausal multiscale TCN", fill=BLUE)
    box(ax, 0.25, 0.25, 0.19, 0.25, "Residual branch", "frozen lag-4 Ridge residual\nclean-development fit only\ncausal multiscale TCN", fill=ORANGE)
    box(ax, 0.50, 0.42, 0.17, 0.28, "Gated fusion", "kernels 3, 5 and 9\ndilations 1, 2 and 4\ndepthwise-separable blocks", fill=GREEN)
    box(ax, 0.72, 0.42, 0.12, 0.28, "Instance head", "time-indexed\nlogits z(t)", fill=BLUE)
    box(ax, 0.88, 0.42, 0.10, 0.28, "MIL pooling", "attention a(t)\nbag logit", fill=GREEN, body_size=7.2)
    arrow(ax, (0.20, 0.58), (0.25, 0.715))
    arrow(ax, (0.20, 0.52), (0.25, 0.375))
    arrow(ax, (0.44, 0.715), (0.50, 0.60))
    arrow(ax, (0.44, 0.375), (0.50, 0.52))
    arrow(ax, (0.67, 0.56), (0.72, 0.56))
    arrow(ax, (0.84, 0.56), (0.88, 0.56))
    ax.text(0.50, 0.14, "Forbidden inputs: file/scenario identity, exact onset metadata and future observations", ha="center", color=RED, weight="bold", fontsize=8.7, transform=ax.transAxes)
    ax.text(0.50, 0.06, "Baselines were evaluated within the same physical, cyber or hybrid observability contracts.", ha="center", fontsize=8.3, transform=ax.transAxes)
    return save(
        fig,
        "fig04-model-pipeline",
        "Observability tiers and candidate causal AR-residual MIL architecture",
        ["Materials/Methods document Sections 1.5 and 1.6"],
    )


def main() -> None:
    setup()
    items = [
        figure_evidence_boundary(),
        figure_protocol_flow(),
        figure_causal_engine(),
        figure_model_pipeline(),
    ]
    manifest = {
        "status": "complete",
        "figure_count": len(items),
        "scope": "Dataset provenance and Materials and Methods only",
        "source_documents": [
            "manuscript/SCI_Q1_Dataset_Provenance_and_Use_Q1_v2.docx",
            "manuscript/SCI_Q1_Materials_and_Methods_Results_Q1_v2.docx",
        ],
        "figures": items,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "figures": len(items)}, indent=2))


if __name__ == "__main__":
    main()
