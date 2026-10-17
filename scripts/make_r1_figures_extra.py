"""Figure 2 (system architecture) and supplementary figures S3-S7.

Usage: .venv\\Scripts\\python.exe scripts\\make_r1_figures_extra.py <main_fig_dir> <supp_fig_dir>
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("base", HERE / "make_r1_figures.py")
sys_argv = sys.argv[:]
sys.argv = [sys.argv[0], sys.argv[1]]
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sys.argv = sys_argv
plt, box, arrow, save, NAME, COLOR = base.plt, base.box, base.arrow, base.save, base.NAME, base.COLOR
ART, R1 = base.ART, base.R1
MAIN = Path(sys.argv[1])
SUPP = Path(sys.argv[2])
SUPP.mkdir(parents=True, exist_ok=True)
NL = chr(10)


def save_to(fig, folder: Path, name: str):
    fig.savefig(folder / f"{name}.png", bbox_inches="tight", pad_inches=0.03, dpi=600)
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_system():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 160); ax.set_ylim(0, 98); ax.axis("off")
    box(ax, 2, 64, 30, 26, NL.join(["Remote supervisor", "profile-driven", "setpoints for PV", "and storage, one", "per endpoint and step"]), "#E8EEF7", fs=6.2)
    box(ax, 46, 60, 48, 34, NL.join(["Event-driven impairment engine", "", "delay (1, 3, 5 steps), Bernoulli drop,", "burst loss, bounded jitter", "queue ordered by delivery time", "and sequence; ledger stored"]), "#FBE9D0", fs=6.5)
    box(ax, 108, 64, 50, 26, NL.join(["Receiver (logical time t)", "accept if delivery ≤ t; reject stale", "→ 5 cyber features", "→ trust counters"]), "#E8EEF7", fs=6.6)
    box(ax, 108, 30, 50, 24, NL.join(["Detectors and router", "physical, cyber, hybrid tiers", "run probability; step-level alarm"]), "#EADCF0", fs=6.6)
    box(ax, 46, 4, 48, 32, NL.join(["Plant (pandapower)", "CIGRE MV-DER (development)", "IEEE33-DER (locked test)", "quasi-static AC power flow,", "900 steps of 1 s → 15 noisy", "physical summaries"]), "#DCEBDD", fs=6.4)
    box(ax, 2, 8, 30, 30, NL.join(["Supervisory fallback", "policies P0–P3 and", "learned alarm policies", "(reads previous-step", "feeder state)"]), "#F2F2F2", fs=6.6)
    arrow(ax, 32.4, 77, 46, 77); ax.text(39, 79.5, "commands", ha="center", fontsize=6.5, style="italic")
    arrow(ax, 94.4, 77, 108, 77); ax.text(101, 79.5, "delivered", ha="center", fontsize=6.5, style="italic")
    arrow(ax, 133, 63.6, 133, 54.4)
    arrow(ax, 70, 36.4, 70, 59.6)
    ax.text(72, 49, "telemetry", fontsize=6.5, style="italic")
    arrow(ax, 94.4, 22, 133, 29.6)
    ax.text(110, 21, "physical summaries", fontsize=6.5, style="italic")
    arrow(ax, 108, 44, 17, 38.6); ax.text(40, 43.5, "alarm schedule", fontsize=6.5, style="italic")
    arrow(ax, 32.4, 18, 46, 18); ax.text(39, 20.5, "actions", ha="center", fontsize=6.5, style="italic")
    ax.plot([102, 102], [2, 96], ls="--", lw=0.8, color="#B03A2E")
    ax.text(103, 95, "causal boundary: only information delivered by t", fontsize=6.4, color="#B03A2E", va="top")
    save_to(fig, MAIN, "Figure_2")


def fig_hpo():
    fams = ["candidate", "candidate_physical", "tcn_cyber", "tcn_hybrid", "tcn_physical", "tail_mil_physical"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0))
    for ax, fam in zip(axes.ravel(), fams):
        vals = []
        for p in sorted((ART / "training" / "hpo" / fam).glob("trial-*/metrics.csv")):
            vals.append(pd.read_csv(p).validation_loss.min())
        ax.plot(range(len(vals)), vals, "o", ms=3.5, color=COLOR.get(fam, "#5D6D7E"))
        ax.set_yscale("log"); ax.set_title(NAME.get(fam, fam)); ax.set_xlabel("Latin-hypercube trial")
    axes[0, 0].set_ylabel("Best validation loss"); axes[1, 0].set_ylabel("Best validation loss")
    fig.tight_layout(); save_to(fig, SUPP, "Figure_S3")


def fig_curves():
    fams = ["candidate", "tcn_cyber", "tcn_hybrid", "candidate_physical", "tcn_physical", "tail_mil_physical"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0))
    for ax, fam in zip(axes.ravel(), fams):
        for k, p in enumerate(sorted((ART / "training" / "final" / fam).glob("seed-*/metrics.csv"))):
            m = pd.read_csv(p)
            ax.plot(m.epoch, m.train_loss, color="#2874A6", lw=0.9, label="training" if k == 0 else None)
            ax.plot(m.epoch, m.validation_loss, color="#C0392B", lw=0.9, ls="--", label="validation-A" if k == 0 else None)
        ax.set_yscale("log"); ax.set_title(NAME.get(fam, fam)); ax.set_xlabel("Epoch")
    axes[0, 0].legend(frameon=False); axes[0, 0].set_ylabel("Loss"); axes[1, 0].set_ylabel("Loss")
    fig.tight_layout(); save_to(fig, SUPP, "Figure_S4")


def fig_risk():
    pred = pd.read_csv(ART / "results" / "predictions_locked.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for m in ["candidate", "tcn_hybrid", "tcn_cyber", "xgboost_cyber", "message_age_watchdog"]:
        g = pred[pred.model == m]
        y, p = g.label.to_numpy(), g.score.to_numpy()
        err = ((p >= 0.5).astype(int) != y).astype(float)
        order = np.argsort(-np.maximum(p, 1 - p), kind="stable")
        risk = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
        cov = np.arange(1, len(err) + 1) / len(err)
        ax.plot(cov, risk, lw=1.2, color=COLOR[m], label=NAME[m])
    ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk"); ax.legend(frameon=False, fontsize=7)
    save_to(fig, SUPP, "Figure_S5")


def fig_temporal():
    t = pd.read_csv(R1 / "ci_temporal_locked.csv")
    models = ["candidate", "tcn_hybrid", "tcn_cyber", "message_age_watchdog"]
    panels = [("impaired", "interval_iou", "IoU (impaired runs)"), ("impaired", "temporal_auprc", "Temporal AUPRC (impaired runs)"),
              ("clean", "false_alarms_per_hour", "False alarms per hour (clean runs)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (den, met, lab) in zip(axes, panels):
        for i, m in enumerate(models):
            r = t[(t.model == m) & (t.denominator == den) & (t.metric == met)].iloc[0]
            ax.bar(i, r.estimate, color=COLOR[m], yerr=[[r.estimate - r.lower], [r.upper - r.estimate]], capsize=2)
        ax.set_xticks(range(len(models))); ax.set_xticklabels([NAME[m] for m in models], rotation=30, ha="right", fontsize=7)
        ax.set_title(lab, fontsize=8)
    fig.tight_layout(); save_to(fig, SUPP, "Figure_S6")


def fig_attention():
    a = pd.read_csv(ART / "results" / "attention_locked_examples.csv")
    a = a[a.model == "candidate"]
    sid = a.scenario_id.iloc[0]
    g = a[a.scenario_id == sid]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.fill_between(g.logical_time, 0, g.label * g.attention.max(), color="#F5B7B1", step="mid", label="impairment interval")
    ax.plot(g.logical_time, g.attention, color="#C0392B", lw=0.9, label="attention weight")
    ax.set_xlabel("Logical time (s)"); ax.set_ylabel("Attention"); ax.legend(frameon=False, fontsize=7)
    save_to(fig, SUPP, "Figure_S7")


if __name__ == "__main__":
    for f in (fig_system, fig_hpo, fig_curves, fig_risk, fig_temporal, fig_attention):
        f(); print(f.__name__, "ok")
