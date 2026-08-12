from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridimpair.contracts import stable_hash


PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


@dataclass(frozen=True)
class FigureOutput:
    figure_id: str
    svg: Path
    pdf: Path
    png: Path
    source_table: Path
    provenance: Path


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_entry(entry: Mapping[str, Any]) -> None:
    serialized = json.dumps(entry, sort_keys=True).lower().replace("_", "-")
    if "point-adjusted" in serialized:
        raise ValueError("point-adjusted metrics are forbidden")
    required = {"id", "title", "kind", "source"}
    if missing := required - set(entry):
        raise ValueError(f"figure entry is missing: {sorted(missing)}")
    if entry["kind"] not in {"line", "bar", "scatter", "histogram"}:
        raise ValueError(f"unsupported figure kind: {entry['kind']}")


def _draw(axis: plt.Axes, frame: pd.DataFrame, entry: Mapping[str, Any]) -> None:
    kind = entry["kind"]
    x = entry.get("x")
    y = entry.get("y")
    group = entry.get("group")
    grouped = (
        list(frame.groupby(group, sort=True)) if group else [(None, frame)]
    )
    if kind == "bar" and group:
        categories = list(dict.fromkeys(frame[x].astype(str)))
        positions = np.arange(len(categories), dtype=float)
        width = 0.8 / max(1, len(grouped))
        for index, (label, subset) in enumerate(grouped):
            mapped = subset.assign(_category=subset[x].astype(str)).set_index(
                "_category"
            )[y]
            heights = [float(mapped.get(category, np.nan)) for category in categories]
            offset = (index - (len(grouped) - 1) / 2) * width
            axis.bar(
                positions + offset,
                heights,
                width=width,
                color=PALETTE[index % len(PALETTE)],
                label=str(label),
            )
        axis.set_xticks(positions, categories)
        grouped = []
    for index, (label, subset) in enumerate(grouped):
        color = PALETTE[index % len(PALETTE)]
        legend_label = str(label) if label is not None else None
        if kind == "line":
            axis.plot(
                subset[x],
                subset[y],
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3.5,
                label=legend_label,
            )
            if entry.get("y_lower") and entry.get("y_upper"):
                axis.fill_between(
                    subset[x],
                    subset[entry["y_lower"]],
                    subset[entry["y_upper"]],
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
        elif kind == "bar":
            error = None
            if entry.get("y_lower") and entry.get("y_upper"):
                error = np.vstack(
                    [
                        subset[y] - subset[entry["y_lower"]],
                        subset[entry["y_upper"]] - subset[y],
                    ]
                )
            axis.bar(
                subset[x].astype(str),
                subset[y],
                color=color,
                yerr=error,
                capsize=3 if error is not None else 0,
            )
        elif kind == "scatter":
            axis.scatter(
                subset[x],
                subset[y],
                color=color,
                s=28,
                alpha=0.85,
                label=legend_label,
            )
        elif kind == "histogram":
            axis.hist(
                subset[x],
                bins=int(entry.get("bins", 20)),
                color=color,
                alpha=0.85,
                label=legend_label,
            )
    axis.set_title(str(entry["title"]))
    if x:
        axis.set_xlabel(str(entry.get("x_label", x)))
    if y:
        axis.set_ylabel(str(entry.get("y_label", y)))
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    if kind == "bar" and frame[x].nunique() >= 6:
        axis.tick_params(axis="x", labelrotation=35)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
    if group:
        axis.legend(
            title=str(entry.get("group_label", group)),
            frameon=False,
            fontsize=8,
            title_fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
        )


def render_figure_manifest(
    manifest_path: Path, output_dir: Path
) -> list[FigureOutput]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("figures")
    if not isinstance(entries, list):
        raise ValueError("figure manifest requires a figures list")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[FigureOutput] = []
    for entry in entries:
        _validate_entry(entry)
        source = Path(entry["source"])
        if not source.is_absolute():
            source = (manifest_path.parent / source).resolve()
        frame = pd.read_csv(source)
        if frame.empty:
            raise ValueError(f"figure {entry['id']!r} has an empty source table")
        referenced = [
            name
            for name in (
                entry.get("x"),
                entry.get("y"),
                entry.get("group"),
                entry.get("y_lower"),
                entry.get("y_upper"),
            )
            if name
        ]
        if missing := set(referenced) - set(frame.columns):
            raise ValueError(f"figure source is missing columns: {sorted(missing)}")
        figure_id = str(entry["id"])
        copied_source = output_dir / f"{figure_id}-source.csv"
        frame.to_csv(copied_source, index=False)
        figure, axis = plt.subplots(figsize=tuple(entry.get("figsize", [6.4, 4.0])))
        _draw(axis, frame, entry)
        figure.tight_layout()
        svg = output_dir / f"{figure_id}.svg"
        pdf = output_dir / f"{figure_id}.pdf"
        png = output_dir / f"{figure_id}.png"
        figure.savefig(svg, format="svg", bbox_inches="tight")
        figure.savefig(pdf, format="pdf", bbox_inches="tight")
        figure.savefig(png, format="png", dpi=600, bbox_inches="tight")
        plt.close(figure)
        provenance = output_dir / f"{figure_id}-provenance.json"
        provenance.write_text(
            json.dumps(
                {
                    "figure_id": figure_id,
                    "source_table": copied_source.name,
                    "source_hash": _file_hash(copied_source),
                    "plot_config_hash": stable_hash(entry),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "formats": {"svg": svg.name, "pdf": pdf.name, "png": png.name},
                    "png_dpi": 600,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        outputs.append(
            FigureOutput(figure_id, svg, pdf, png, copied_source, provenance)
        )
    return outputs
