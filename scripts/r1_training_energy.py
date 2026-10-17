"""Dynamic (idle-subtracted) training energy, consistent with the inference benchmark.

GPU board energy logged per epoch minus the idle board power measured by the
inference benchmark times the epoch duration (clipped at zero), plus the
TDP-based CPU estimate. Writes CSVs into <artifact_root>/r1_analysis/.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts_q1_v2"
OUT = ART / "r1_analysis"
IDLE = float(pd.read_csv(OUT / "energy_inference.csv").idle_gpu_watts.iloc[0])


def dynamic(path: Path) -> dict:
    m = pd.read_csv(path)
    gpu = (m.gpu_energy_j - IDLE * m.epoch_seconds).clip(lower=0)
    return {"gpu_dynamic_j": float(gpu.sum()), "gpu_board_j": float(m.gpu_energy_j.sum()),
            "cpu_j": float(m.cpu_energy_j_est.sum()), "seconds": float(m.epoch_seconds.sum()),
            "epochs": len(m), "peak_vram_mb": float(m.gpu_peak_memory_mb.max())}


rows = []
for p in (ART / "training").rglob("metrics.csv"):
    parts = p.relative_to(ART / "training").parts
    rows.append({"phase": parts[0], "family": parts[1], **dynamic(p)})
fam = pd.DataFrame(rows).groupby("family").agg(
    gpu_dynamic_j=("gpu_dynamic_j", "sum"), gpu_board_j=("gpu_board_j", "sum"), cpu_j=("cpu_j", "sum"),
    seconds=("seconds", "sum"), peak_vram_mb=("peak_vram_mb", "max"))
fam["total_kj"] = (fam.gpu_dynamic_j + fam.cpu_j) / 1000
fam.to_csv(OUT / "energy_training_dynamic.csv")

rows = []
for v in sorted((ART / "r1_variants" / "ablation").iterdir()):
    seeds = [dynamic(p) for p in v.glob("seed-*/metrics.csv")]
    rows.append({"variant": v.name, "gpu_dynamic_kj": sum(s["gpu_dynamic_j"] for s in seeds) / 1000,
                 "cpu_kj": sum(s["cpu_j"] for s in seeds) / 1000, "seconds": sum(s["seconds"] for s in seeds)})
abl = pd.DataFrame(rows)
abl["total_kj"] = abl.gpu_dynamic_kj + abl.cpu_kj
abl.to_csv(OUT / "energy_ablation_dynamic.csv", index=False)

base = pd.read_csv(OUT / "energy_training_baselines.csv")
base["gpu_dynamic_j"] = (base.gpu_energy_j - IDLE * base.wall_seconds).clip(lower=0)
base["total_j"] = base.gpu_dynamic_j + base.cpu_energy_j_est
base.to_csv(OUT / "energy_training_baselines_dynamic.csv", index=False)

print("idle W", round(IDLE, 2))
print(fam.round(1).to_string())
print(abl.round(3).to_string(index=False))
print(base[base.phase == "fit"][["model", "wall_seconds", "cpu_energy_j_est", "gpu_dynamic_j", "total_j"]].round(2).to_string(index=False))
