"""Publication figures for the SUSCOM manuscript, drawn from the R1 result CSVs.

Usage: .venv\\Scripts\\python.exe scripts\\make_r1_figures.py <output_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts_q1_v2"
R1 = ART / "r1_analysis"
VAR = ART / "r1_variants"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts_q1_v2" / "r1_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 600, "mathtext.fontset": "dejavusans",
})
NAME = {
    "candidate": "CyberGuard-DER", "tcn_hybrid": "TCN-H", "tcn_cyber": "TCN-C", "tcn_physical": "TCN-P",
    "xgboost_cyber": "XGB-C", "xgboost_hybrid": "XGB-H", "xgboost_physical": "XGB-P",
    "multirocket_hydra_cyber": "MR-Hydra-C", "multirocket_hydra_hybrid": "MR-Hydra-H", "multirocket_hydra_physical": "MR-Hydra-P",
    "message_age_watchdog": "Age watchdog", "candidate_physical": "CyberGuard-P", "tail_mil_physical": "TAIL-MIL-P",
    "mil_attention_physical": "Attention MIL-P", "mil_mean_physical": "Mean MIL-P", "mil_topk_physical": "Top-k MIL-P",
    "ewma_cusum": "EWMA-CUSUM", "ridge_residual": "AR residual", "flatline": "Flatline", "derivative": "Derivative",
    "cascade": "Trust-gated cascade",
}
COLOR = {"candidate": "#C0392B", "cascade": "#000000", "tcn_hybrid": "#E67E22", "tcn_cyber": "#2874A6",
         "xgboost_cyber": "#1E8449", "multirocket_hydra_cyber": "#7D3C98", "message_age_watchdog": "#7F8C8D"}
TIER = {"physical": "Physical", "cyber": "Cyber", "hybrid": "Hybrid"}


def tier_of(model: str) -> str:
    if model == "message_age_watchdog":
        return "cyber"
    if model == "candidate":
        return "hybrid"
    for t in ("physical", "cyber", "hybrid"):
        if model.endswith(t):
            return t
    return "physical"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, fs=7.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333333"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.5, text, ha="center", fontsize=6.8, style="italic")


def fig_framework():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 160); ax.set_ylim(0, 70); ax.axis("off")
    NL = chr(10)
    box(ax, 1, 25, 24, 22, NL.join(["Received messages", "(commands, telemetry;", "timestamps and", "sequence IDs)"]), "#E8EEF7", fs=6.3)
    box(ax, 32, 42, 34, 24, NL.join(["Trust monitor", "(receiver counters, no model)", "missing timestamps", "future timestamps", "sequence anomalies"]), "#FBE9D0", fs=6.3)
    box(ax, 32, 4, 34, 26, NL.join(["Receiver bookkeeping", "message age (mean, max)", "delivery fraction", "stale rejections", "telemetry age"]), "#E8EEF7", fs=6.3)
    box(ax, 74, 25, 16, 22, NL.join(["Router", "", "trusted?"]), "#F9E79F", fs=7.2, bold=True)
    box(ax, 98, 44, 34, 22, NL.join(["Trusted route: XGB-C", "(cyber features)", "0.021 J, 0.33 ms per run"]), "#DCEBDD", fs=6.3)
    box(ax, 98, 4, 34, 26, NL.join(["Untrusted route:", "CyberGuard-DER detector", "(hybrid causal MIL)", "+ severity-conditioned", "recalibration"]), "#EADCF0", fs=6.3)
    box(ax, 139, 24, 20, 24, NL.join(["Run-level", "probability", "and alarm", "(route and", "energy logged)"]), "#F2F2F2", fs=6.2)
    arrow(ax, 25.4, 41, 32, 52); arrow(ax, 25.4, 31, 32, 18)
    arrow(ax, 66.4, 54, 74, 42); arrow(ax, 66.4, 17, 74, 30)
    arrow(ax, 90.4, 41, 98, 54, "yes"); arrow(ax, 90.4, 31, 98, 18, "no")
    arrow(ax, 132.4, 55, 139, 42); arrow(ax, 132.4, 17, 139, 30)
    ax.text(80, 68.5, "Trust thresholds, route model, and recalibration are fixed on validation data only", ha="center", fontsize=7, style="italic")
    save(fig, "Figure_1")


def fig_detector():
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.set_xlim(0, 140); ax.set_ylim(0, 92); ax.axis("off")
    box(ax, 1, 60, 20, 20, "Input $x_t$\n15 physical\n+ 5 cyber\nchannels", "#E8EEF7")
    box(ax, 1, 30, 20, 20, "Frozen AR(4)\nresidual $r_t$", "#E8EEF7")
    arrow(ax, 11, 60, 11, 50.4)
    for y, title in ((58, "Raw-view encoder → $h_t^{raw}$"), (28, "Residual-view encoder → $h_t^{res}$")):
        ax.add_patch(FancyBboxPatch((25, y), 62, 24, boxstyle="round,pad=0.2,rounding_size=1.2", fc="#DCEBDD", ec="#333333", lw=0.8))
        ax.text(56, y + 20.5, title, ha="center", va="center", fontsize=8.2, weight="bold")
        box(ax, 27, y + 2, 14, 14, "$[v_t;\\Delta v_t]$\n1×1 proj.", "white")
        box(ax, 44, y + 2, 22, 14, "Multi-kernel\nDS causal\nk = 3, 5, 9", "white")
        box(ax, 68, y + 2, 17, 14, "Dilated DS\nresidual\nd = 1, 2, 4", "white")
        arrow(ax, 41, y + 9, 44, y + 9); arrow(ax, 66, y + 9, 68, y + 9)
    arrow(ax, 21.4, 70, 25, 70); arrow(ax, 21.4, 40, 25, 40)
    box(ax, 91, 36, 20, 36, "Gated\nfusion\n\n$g_t=\\sigma(W_g[\\cdot])$\n\n$h_t=g_t h^{raw}_t$\n$+(1-g_t)h^{res}_t$\n$+F([\\cdot])$", "#FBE9D0")
    arrow(ax, 87.4, 70, 91, 64); arrow(ax, 87.4, 40, 91, 46)
    box(ax, 115, 64, 24, 16, "Instance head\n$z_t$ (per step)", "#EADCF0")
    box(ax, 114, 36, 20, 20, "Attention\nMIL pooling\n$z_{bag}=\\sum_t a_t z_t$", "#EADCF0")
    arrow(ax, 111.4, 64, 115, 70); arrow(ax, 111.4, 46, 114, 46); arrow(ax, 122, 64, 122, 56.4)
    box(ax, 91, 4, 22, 22, "Run level\n3-seed mean\n+ calibration\n→ probability", "#F2F2F2")
    box(ax, 117, 4, 22, 22, "Step level\nthreshold\n+ 3-step dwell\n→ alarm trace", "#F2F2F2")
    arrow(ax, 124, 36, 104, 26.4); arrow(ax, 137, 64, 137, 26.4)
    ax.text(1, 15, "Run-level labels only (weak supervision)\nBag-level BCE loss; AdamW; cosine LR\nNo scenario ID, onset, drop flag,\nor future sample enters the model\nDS = depthwise-separable", fontsize=7.2, va="center", linespacing=1.35)
    cx = pd.read_csv(R1 / "deployed_complexity.csv").set_index("model").loc["candidate"]
    ax.text(70, 88.5, f"Deployed width H = {int(cx.hidden)}: {cx.params / 1000:.1f}k parameters · {cx.flops_M:.1f} MFLOPs per 900-step run · receptive field 23 steps",
            fontsize=7.6, style="italic", ha="center")
    save(fig, "Figure_3")


def fig_tiers():
    ci = pd.read_csv(R1 / "ci_bag_locked.csv")
    order = ["xgboost_physical", "multirocket_hydra_physical", "tcn_physical", "tail_mil_physical", "candidate_physical",
             "message_age_watchdog", "xgboost_cyber", "multirocket_hydra_cyber", "tcn_cyber",
             "xgboost_hybrid", "multirocket_hydra_hybrid", "tcn_hybrid", "candidate"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    for ax, metric, label in zip(axes, ["auprc", "macro_f1"], ["Locked AUPRC (95% CI)", "Locked macro-F1 (95% CI)"]):
        for i, m in enumerate(order):
            r = ci[(ci.model == m) & (ci.metric == metric)]
            if r.empty:
                continue
            r = r.iloc[0]
            col = {"physical": "#95A5A6", "cyber": "#2874A6", "hybrid": "#C0392B"}[tier_of(m)]
            ax.errorbar(r.estimate, i, xerr=[[r.estimate - r.lower], [r.upper - r.estimate]], fmt="o", color=col, ms=4, capsize=2, lw=1)
        ax.axvline(0.5 if metric == "auprc" else 0.3333, color="0.6", ls=":", lw=0.8)
        ax.set_xlabel(label); ax.set_xlim(0.28, 1.03)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([f"{NAME[m]}" for m in order])
    for y0, y1, t in ((-0.5, 4.5, "Physical"), (4.5, 8.5, "Cyber"), (8.5, 12.5, "Hybrid")):
        for ax in axes:
            ax.axhspan(y0, y1, color={"Physical": "#F4F6F6", "Cyber": "#EBF5FB", "Hybrid": "#FDEDEC"}[t], zorder=0)
        axes[1].text(1.025, (y0 + y1) / 2, t, rotation=270, va="center", fontsize=7.5, transform=axes[1].get_yaxis_transform())
    save(fig, "Figure_4")


def fig_calibration():
    t = pd.read_csv(R1 / "reliability_tables.csv")
    e = pd.read_csv(R1 / "calibration_errors.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for ax, m in zip(axes, ["candidate", "tcn_hybrid", "tcn_cyber"]):
        for ctx, style, lab in (("pre_shift_validation_a", "o--", "CIGRE validation-A"), ("post_shift_locked", "s-", "IEEE33-DER locked")):
            g = t[(t.model == m) & (t.context == ctx) & (t["count"] > 0)]
            ax.plot(g.mean_confidence, g.observed_frequency, style, ms=3.5, lw=1, label=lab)
        ece = e[(e.model == m) & (e.context == "post_shift_locked")].iloc[0]
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.7)
        ax.set_title(NAME[m]); ax.set_xlabel("Mean predicted probability")
        ax.text(0.03, 0.93, f"post-shift ECE {ece.ece:.3f}\n[{ece.ece_lower:.3f}, {ece.ece_upper:.3f}]", transform=ax.transAxes, fontsize=7, va="top")
    axes[0].set_ylabel("Observed impaired fraction"); axes[0].legend(frameon=False, loc="lower right")
    save(fig, "Figure_6")


def fig_degradation():
    d = pd.read_csv(R1 / "degradation_metrics.csv")
    c = pd.read_csv(R1 / "cascade_metrics.csv")
    c = c[c.policy == "cascade"].assign(model="cascade")
    kinds = [("clock_offset", "Clock offset bound (s)"), ("clock_drift", "Drift at 900 s (s)"),
             ("timestamp_dropout", "Messages without timestamp"), ("sequence_corruption", "Corrupted sequence IDs"),
             ("metadata_loss", "Missing metadata samples"), ("metadata_delay", "Metadata delay (steps)"),
             ("physical_noise", "Physical noise (× dev. SD)"), ("physical_missing", "Missing physical samples")]
    models = ["candidate", "tcn_hybrid", "tcn_cyber", "xgboost_cyber", "multirocket_hydra_cyber", "message_age_watchdog", "cascade"]
    for metric, name, ylab in (("auprc", "Figure_8", "Locked AUPRC"), ("macro_f1", "Figure_9", "Locked macro-F1")):
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.2), sharey=True)
        for ax, (kind, xl) in zip(axes.ravel(), kinds):
            for m in models:
                src = c if m == "cascade" else d
                base = src[(src.perturbation == "none") & (src.model == m)]
                g = src[(src.perturbation == kind) & (src.model == m)].sort_values("level")
                if g.empty or base.empty:
                    continue
                x = np.r_[0.0, g.level.to_numpy()]
                y = np.r_[base[metric].iloc[0], g[metric].to_numpy()]
                ax.plot(x, y, "o-" if m != "cascade" else "s--", ms=2.8, lw=1.1 if m != "cascade" else 1.4, color=COLOR[m], label=NAME[m])
            ax.set_xlabel(xl, fontsize=7.3); ax.set_ylim(0.28 if metric == "macro_f1" else 0.45, 1.03)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylab)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
        fig.tight_layout()
        save(fig, name)


def fig_pareto():
    e = pd.read_csv(R1 / "energy_inference.csv")
    e = e[(e.device == "cpu")].copy()
    e["energy"] = e.cpu_j_per_run_est
    ci = pd.read_csv(R1 / "ci_bag_locked.csv").pivot(index="model", columns="metric", values="estimate")
    cas = pd.read_csv(R1 / "cascade_metrics.csv")
    cas_none = cas[(cas.policy == "cascade") & (cas.perturbation == "none")].iloc[0]
    xgb = e[e.model == "xgboost_cyber"].energy.iloc[0]
    key = ["message_age_watchdog", "xgboost_cyber", "tcn_cyber", "multirocket_hydra_cyber", "tcn_hybrid", "candidate"]
    markers = {"message_age_watchdog": "v", "xgboost_cyber": "o", "tcn_cyber": "^", "multirocket_hydra_cyber": "P",
               "tcn_hybrid": "X", "candidate": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharex=True)
    for ax, metric, lab in zip(axes, ["auprc", "macro_f1"], ["Locked AUPRC", "Locked macro-F1"]):
        others = [(r.energy, ci.loc[r.model, metric]) for r in e.itertuples() if r.model in ci.index and r.model not in key]
        ax.scatter([x for x, _ in others], [y for _, y in others], s=12, color="#BFC9CA", zorder=2,
                   label="other detectors" if metric == "auprc" else None)
        for m in key:
            row = e[e.model == m].iloc[0]
            ax.scatter(row.energy, ci.loc[m, metric], s=34, marker=markers[m], color=COLOR[m], zorder=3,
                       edgecolor="k", lw=0.4, label=NAME[m] if metric == "auprc" else None)
        ax.scatter(xgb, cas_none[metric], s=110, marker="s", facecolor="none", edgecolor="k", lw=1.2, zorder=4,
                   label="Trust-gated cascade (trusted)" if metric == "auprc" else None)
        ax.set_xscale("log"); ax.set_xlabel("Inference energy per run (J, CPU, log scale)"); ax.set_ylabel(lab)
        ax.set_ylim(0.28 if metric == "macro_f1" else 0.45, 1.04)
    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1), fontsize=7)
    fig.tight_layout()
    save(fig, "Figure_10")


def fig_ablation():
    a = pd.read_csv(VAR / "ablation_metrics.csv")
    names = {"v0_base_causal_cnn": "Base causal CNN", "v1_plus_difference": "+ first difference",
             "v2_plus_ar_residual": "+ AR residual view", "v3_plus_gated_fusion": "+ gated fusion",
             "v4_plus_attention_mil": "+ attention MIL\n(CyberGuard-DER)"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, metric, lab in zip(axes, ["auprc", "macro_f1"], ["Locked AUPRC", "Locked macro-F1"]):
        for i, (k, n) in enumerate(names.items()):
            g = a[a.model == k]
            seeds = g[g.seed.astype(str) != "ensemble"][metric].to_numpy(float)
            ens = g[g.seed.astype(str) == "ensemble"][metric].iloc[0]
            ax.scatter(seeds, [i] * len(seeds), s=18, color="#7F8C8D", zorder=3, label="single seed" if i == 0 else None)
            ax.scatter([ens], [i], s=46, marker="D", color="#C0392B", zorder=4, label="3-seed ensemble" if i == 0 else None)
        ax.set_xlabel(lab); ax.set_xlim(0.28, 1.03)
    axes[0].set_yticks(range(len(names))); axes[0].set_yticklabels(list(names.values())); axes[0].invert_yaxis()
    axes[1].legend(frameon=False, loc="lower left")
    save(fig, "Figure_5")


def fig_online():
    o = pd.read_csv(R1 / "online_prefix_metrics.csv")
    models = ["candidate", "tcn_hybrid", "tcn_cyber", "xgboost_cyber", "message_age_watchdog"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for m in models:
        g = o[o.model == m].sort_values("fraction")
        for ax, metric in zip(axes, ["auprc", "macro_f1", "false_positive_rate"]):
            ax.plot(g.fraction * 900, g[metric], "o-", ms=3, lw=1, color=COLOR[m], label=NAME[m])
    for ax, lab in zip(axes, ["AUPRC", "Macro-F1", "False-positive rate (clean runs)"]):
        ax.axvspan(0, 181, color="0.92", zorder=0)
        ax.set_xlabel("Observed horizon (s)"); ax.set_ylabel(lab); ax.set_xlim(0, 920)
    axes[0].text(8, 0.92, "before first\nimpairment", fontsize=6.5)
    axes[2].legend(frameon=False, fontsize=6.6)
    fig.tight_layout()
    save(fig, "Figure_7")


if __name__ == "__main__":
    for f in (fig_framework, fig_detector, fig_tiers, fig_calibration, fig_degradation, fig_pareto, fig_ablation, fig_online):
        f()
        print(f.__name__, "ok")
