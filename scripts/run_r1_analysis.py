"""R1 Phase-2 analyses on the frozen q1_v2 artifacts.

Usage (from the project root):
    .venv\\Scripts\\python.exe scripts\\run_r1_analysis.py --stage all
Stages: ci, calibration, online, energy, figures, all. Outputs go to
<artifact_root>/r1_analysis/. Nothing in training/ or results/ is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridimpair.config import load_config  # noqa: E402
from gridimpair.r1 import calibration, cascade, degradation, energy_bench, frozen, online, uncertainty  # noqa: E402
from gridimpair.workflow import discover_training_records  # noqa: E402

DISPLAY = {
    "candidate": "CyberGuard-DER", "candidate_physical": "CyberGuard-P", "tcn_hybrid": "TCN-H",
    "tcn_cyber": "TCN-C", "tcn_physical": "TCN-P", "tail_mil_physical": "TAIL-MIL-P",
    "mil_mean_physical": "Mean MIL-P", "mil_topk_physical": "Top-k MIL-P", "mil_attention_physical": "Attention MIL-P",
    "message_age_watchdog": "Age watchdog", "ewma_cusum": "EWMA-CUSUM", "ridge_residual": "AR residual",
    "flatline": "Flatline", "derivative": "Derivative",
}
for tier, short in (("physical", "P"), ("cyber", "C"), ("hybrid", "H")):
    DISPLAY[f"xgboost_{tier}"] = f"XGB-{short}"
    DISPLAY[f"multirocket_hydra_{tier}"] = f"MR-Hydra-{short}"


def display(frame: pd.DataFrame, column: str = "model") -> pd.DataFrame:
    if column in frame:
        frame.insert(frame.columns.get_loc(column) + 1, "display_name", frame[column].map(DISPLAY).fillna(frame[column]))
    return frame


def stage_ci(art: Path, out: Path, n: int) -> None:
    results, sim = art / "results", art / "simulation"
    locked = pd.read_csv(results / "predictions_locked.csv")
    strata = uncertainty.pair_strata(locked, sim)
    cis, diffs = uncertainty.bag_metric_intervals(locked, strata, n=n)
    display(cis).to_csv(out / "ci_bag_locked.csv", index=False)
    display(diffs).to_csv(out / "ci_bag_locked_differences.csv", index=False)
    per_run = pd.read_csv(results / "temporal_locked_per_run.csv")
    onsets_path = results / "onset_delays_locked_per_run.csv"
    onsets = pd.read_csv(onsets_path) if onsets_path.exists() else None
    display(uncertainty.temporal_intervals(per_run, onsets, strata, n=n)).to_csv(out / "ci_temporal_locked.csv", index=False)
    closed = results / "closed_loop_per_pair.csv"
    if closed.exists():
        uncertainty.closed_loop_intervals(pd.read_csv(closed), strata, n=n).to_csv(out / "ci_closed_loop.csv", index=False)
    print("ci: done")


def stage_calibration(art: Path, out: Path, n: int) -> None:
    results, sim = art / "results", art / "simulation"
    locked = pd.read_csv(results / "predictions_locked.csv")
    open_ = pd.read_csv(art / "training" / "predictions_open.csv")
    if "status" not in open_:
        open_["status"] = np.where(open_["label"] == 1, "impaired", "clean")
    frames = []
    for split in ("validation_a", "validation_b"):
        part = open_[open_["split"] == split]
        if len(part):
            frames.append(calibration.calibration_summary(part, context=f"pre_shift_{split}",
                                                          strata=uncertainty.pair_strata(part, sim), n=n))
    frames.append(calibration.calibration_summary(locked, context="post_shift_locked",
                                                  strata=uncertainty.pair_strata(locked, sim), n=n))
    conditions = []
    for sid in locked["scenario_id"].unique():
        public = json.loads((sim / sid / "public_manifest.json").read_text(encoding="utf-8"))
        c = public.get("condition", {})
        conditions.append({"scenario_id": sid, "condition_group": f"{c.get('impairment_kind', 'na')}|d{c.get('delay_steps')}|p{c.get('drop_probability')}|f{c.get('endpoint_fraction')}"})
    pair_condition = locked.merge(pd.DataFrame(conditions), on="scenario_id")
    impaired_cond = pair_condition[pair_condition["status"] == "impaired"][["pair_id", "condition_group"]].drop_duplicates()
    by_cond = locked.merge(impaired_cond, on="pair_id")
    frames.append(calibration.calibration_summary(by_cond, context="post_shift_by_condition", group_column="condition_group", n=n))
    robust = results / "predictions_robustness.csv"
    if robust.exists():
        frames.append(calibration.calibration_summary(pd.read_csv(robust), context="robustness_set", n=n))
    display(pd.concat(frames, ignore_index=True)).to_csv(out / "calibration_errors.csv", index=False)
    tables = []
    for context, frame in (("pre_shift_validation_a", open_[open_["split"] == "validation_a"]), ("post_shift_locked", locked)):
        for model, g in frame.groupby("model"):
            t = calibration.reliability_table(g["label"].to_numpy(int), g["score"].to_numpy(float))
            tables.append(t.assign(context=context, model=model))
    display(pd.concat(tables, ignore_index=True)).to_csv(out / "reliability_tables.csv", index=False)
    hist = pd.concat([calibration.probability_histogram(open_[open_["split"] == "validation_a"], context="pre_shift_validation_a"),
                      calibration.probability_histogram(locked, context="post_shift_locked")])
    display(hist).to_csv(out / "probability_histograms.csv", index=False)
    print("calibration: done")


def stage_online(art: Path, out: Path) -> None:
    training = art / "training"
    locked = pd.read_csv(art / "results" / "predictions_locked.csv")
    inputs = frozen.load_inputs(frozen.locked_records(art, locked))
    detectors = frozen.load_detectors(training)
    frame = online.prefix_predictions(inputs, detectors, training)
    frame.to_csv(out / "online_prefix_predictions.csv", index=False)
    display(online.prefix_summary(frame)).to_csv(out / "online_prefix_metrics.csv", index=False)
    display(online.time_to_detection(frame)).to_csv(out / "online_time_to_detection.csv", index=False)
    print("online: done")


def stage_energy(art: Path, out: Path, min_seconds: float) -> None:
    training = art / "training"
    locked = pd.read_csv(art / "results" / "predictions_locked.csv")
    inputs = frozen.load_inputs(frozen.locked_records(art, locked))
    detectors = frozen.load_detectors(training)
    display(energy_bench.inference_benchmark(inputs, detectors, training, min_seconds=min_seconds)).to_csv(
        out / "energy_inference.csv", index=False)
    energy_bench.training_energy_summary(training).to_csv(out / "energy_training_runs.csv", index=False)
    energy_bench.baseline_training_cost(training).to_csv(out / "energy_training_baselines.csv", index=False)
    print("energy: done")


def stage_degradation(art: Path, out: Path, n: int) -> None:
    training, sim = art / "training", art / "simulation"
    locked = pd.read_csv(art / "results" / "predictions_locked.csv")
    inputs = frozen.load_inputs(frozen.locked_records(art, locked))
    detectors = frozen.load_detectors(training)
    key = tuple(m for m in ("candidate", "tcn_hybrid", "tcn_cyber", "xgboost_cyber", "xgboost_hybrid", "message_age_watchdog") if m in detectors)
    metrics, ci = degradation.run_degradation(inputs, detectors, training, sim, ci_models=key,
                                              strata=uncertainty.pair_strata(locked, sim), n_resamples=n)
    display(metrics).to_csv(out / "degradation_metrics.csv", index=False)
    if not ci.empty:
        display(ci).to_csv(out / "degradation_ci.csv", index=False)
    print("degradation: done")


def stage_cascade(art: Path, out: Path, n: int) -> None:
    training, sim = art / "training", art / "simulation"
    locked = pd.read_csv(art / "results" / "predictions_locked.csv")
    detectors = frozen.load_detectors(training)
    open_records = [r for r in discover_training_records(sim) if r.split in ("validation_a", "validation_b")]
    design = cascade.design_cascade(frozen.load_inputs(open_records), detectors, training, sim)
    (out / "cascade_design.json").write_text(json.dumps({"cheap_model": design.cheap_model, "learned_model": design.learned_model,
                                                         **design.validation_summary}, indent=2), encoding="utf-8")
    print("cascade design:", design.cheap_model, design.validation_summary, flush=True)
    metrics, preds = cascade.evaluate_cascade(frozen.load_inputs(frozen.locked_records(art, locked)), detectors, design, training, sim)
    metrics.to_csv(out / "cascade_metrics.csv", index=False)
    preds.to_csv(out / "cascade_predictions.csv", index=False)
    strata = uncertainty.pair_strata(locked, sim)
    parts = []
    for (kind, level), g in preds.groupby(["perturbation", "level"], sort=False):
        cis, _ = uncertainty.bag_metric_intervals(g, strata, reference="__none__", n=min(n, 1000))
        parts.append(cis.assign(perturbation=kind, level=level))
    pd.concat(parts, ignore_index=True).to_csv(out / "cascade_ci.csv", index=False)
    print("cascade: done")


def stage_cascade_lofo(art: Path, out: Path) -> None:
    """Leave-one-family-out: route calibration never sees the evaluated perturbation family."""
    training, sim = art / "training", art / "simulation"
    locked = pd.read_csv(art / "results" / "predictions_locked.csv")
    detectors = frozen.load_detectors(training, include={"candidate", "xgboost_cyber", "message_age_watchdog",
                                                         "xgboost_hybrid", "multirocket_hydra_cyber", "multirocket_hydra_hybrid"})
    open_inputs = frozen.load_inputs([r for r in discover_training_records(sim) if r.split in ("validation_a", "validation_b")])
    locked_inputs = frozen.load_inputs(frozen.locked_records(art, locked))
    frames = []
    for family in ("clock_offset", "clock_drift", "timestamp_dropout", "sequence_corruption"):
        design = cascade.design_cascade(open_inputs, detectors, training, sim, exclude_kinds=(family,))
        grid = tuple(p for p in cascade.metadata.GRID if p.kind == family)
        metrics, _ = cascade.evaluate_cascade(locked_inputs, detectors, design, training, sim, grid=grid)
        frames.append(metrics.assign(excluded_family=family, route_calibrators=",".join(sorted(design.route_calibrators))))
        print("lofo", family, design.validation_summary.get("route_calibration_runs"), flush=True)
    pd.concat(frames, ignore_index=True).to_csv(out / "cascade_lofo_metrics.csv", index=False)
    print("cascade_lofo: done")


def stage_figures(art: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9})
    focus = ["candidate", "tcn_hybrid", "tcn_cyber", "xgboost_cyber", "message_age_watchdog"]
    rel = out / "reliability_tables.csv"
    if rel.exists():
        t = pd.read_csv(rel)
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
        for ax, model in zip(axes, ["candidate", "tcn_hybrid"]):
            for context, style in (("pre_shift_validation_a", "o--"), ("post_shift_locked", "s-")):
                g = t[(t["model"] == model) & (t["context"] == context) & (t["count"] > 0)]
                ax.plot(g["mean_confidence"], g["observed_frequency"], style, label=context.replace("_", " "))
            ax.plot([0, 1], [0, 1], color="0.6", lw=0.8)
            ax.set_title(DISPLAY.get(model, model)); ax.set_xlabel("Mean predicted probability")
        axes[0].set_ylabel("Observed impaired fraction"); axes[0].legend(frameon=False)
        fig.tight_layout(); fig.savefig(out / "fig_reliability.png", dpi=300); plt.close(fig)
    onl = out / "online_prefix_metrics.csv"
    if onl.exists():
        t = pd.read_csv(onl)
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
        for model in focus:
            g = t[t["model"] == model].sort_values("fraction")
            for ax, metric in zip(axes, ["auprc", "macro_f1"]):
                ax.plot(g["fraction"] * 100, g[metric], "o-", label=DISPLAY.get(model, model))
        for ax, metric in zip(axes, ["AUPRC", "Macro-F1"]):
            ax.set_xscale("log"); ax.set_xticks([5, 10, 25, 50, 100]); ax.set_xticklabels(["5", "10", "25", "50", "100"])
            ax.set_xlabel("Observed horizon (%)"); ax.set_ylabel(metric)
        axes[1].legend(frameon=False); fig.tight_layout(); fig.savefig(out / "fig_online_prefix.png", dpi=300); plt.close(fig)
    eng, cis = out / "energy_inference.csv", out / "ci_bag_locked.csv"
    if eng.exists() and cis.exists():
        e = pd.read_csv(eng)
        e = e[(e["kind"] != "neural") | (e["device"] == "cuda")].copy()
        e["energy_j"] = e["gpu_dynamic_j_per_run"].fillna(0).clip(lower=0) + e["cpu_j_per_run_est"]
        a = pd.read_csv(cis); a = a[a["metric"] == "auprc"][["model", "estimate"]]
        m = e.merge(a, on="model")
        m = m.sort_values("energy_j")
        front, best = [], -np.inf
        for row in m.itertuples():
            if row.estimate > best:
                front.append(row.model); best = row.estimate
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for row in m.itertuples():
            on = row.model in front
            ax.scatter(row.energy_j, row.estimate, s=36 if on else 18, color="C3" if on else "C0", zorder=3)
            ax.annotate(DISPLAY.get(row.model, row.model), (row.energy_j, row.estimate), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("Inference energy per run (J, log scale)"); ax.set_ylabel("Locked AUPRC")
        ax.set_title("Accuracy–energy trade-off (red: Pareto-optimal)")
        fig.tight_layout(); fig.savefig(out / "fig_pareto_energy.png", dpi=300); plt.close(fig)
    deg = out / "degradation_metrics.csv"
    if deg.exists():
        d = pd.read_csv(deg)
        kinds = ["clock_offset", "clock_drift", "timestamp_dropout", "sequence_corruption", "metadata_loss", "metadata_delay", "physical_noise", "physical_missing"]
        xlabels = {"clock_offset": "Clock offset bound (s)", "clock_drift": "Drift at 900 s (s)", "timestamp_dropout": "Messages without timestamp (fraction)",
                   "sequence_corruption": "Corrupted sequence IDs (fraction)", "metadata_loss": "Missing metadata samples (fraction)",
                   "metadata_delay": "Metadata delay (steps)", "physical_noise": "Added noise (× development SD)", "physical_missing": "Missing physical samples (fraction)"}
        models = ["candidate", "tcn_hybrid", "tcn_cyber", "xgboost_hybrid", "message_age_watchdog", "candidate_physical"]
        base = d[d["perturbation"] == "none"]
        fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.2), sharey=True)
        for ax, kind in zip(axes.ravel(), kinds):
            for i, model in enumerate(models):
                g = d[(d["perturbation"] == kind) & (d["model"] == model)].sort_values("level")
                b = base[base["model"] == model]
                if g.empty or b.empty:
                    continue
                x = np.r_[0.0, g["level"].to_numpy()]
                y = np.r_[b["macro_f1"].iloc[0], g["macro_f1"].to_numpy()]
                ax.plot(x, y, "o-", ms=3, color=f"C{i}", label=DISPLAY.get(model, model))
            ax.set_xlabel(xlabels[kind], fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("Locked macro-F1")
        axes[0, 0].legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(out / "fig_metadata_degradation.png", dpi=300); plt.close(fig)
    print("figures: done")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["ci", "calibration", "online", "degradation", "cascade", "cascade_lofo", "energy", "figures", "all"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "q1_v2.toml"))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--min-seconds", type=float, default=5.0)
    args = parser.parse_args()
    config = load_config(Path(args.config))
    art = config.paths.artifact_root
    out = art / "r1_analysis"
    out.mkdir(parents=True, exist_ok=True)
    stages = ["ci", "calibration", "online", "degradation", "cascade", "energy", "figures"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "ci":
            stage_ci(art, out, args.resamples)
        elif stage == "calibration":
            stage_calibration(art, out, args.resamples)
        elif stage == "online":
            stage_online(art, out)
        elif stage == "degradation":
            stage_degradation(art, out, min(args.resamples, 1000))
        elif stage == "cascade":
            stage_cascade(art, out, args.resamples)
        elif stage == "cascade_lofo":
            stage_cascade_lofo(art, out)
        elif stage == "energy":
            stage_energy(art, out, args.min_seconds)
        else:
            stage_figures(art, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
