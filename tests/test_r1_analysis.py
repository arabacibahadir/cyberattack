import numpy as np
import pytest
import pandas as pd

from gridimpair.r1 import calibration, online, uncertainty


def _paired_predictions(n_pairs: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_pairs):
        pair = f"q1v2-n1-r{k % 3}-c{k % 4:02d}-rep{k:02d}"
        for status, label, centre in (("clean", 0, 0.25), ("impaired", 1, 0.75)):
            for model, noise in (("candidate", 0.10), ("weak", 0.35)):
                score = float(np.clip(centre + rng.normal(0, noise), 0, 1))
                rows.append({"scenario_id": f"{pair}-{status}", "pair_id": pair, "status": status,
                             "label": label, "model": model, "score": score})
    return pd.DataFrame(rows)


def test_bootstrap_draws_respect_strata_sizes() -> None:
    pairs = ["a", "b", "c", "d"]
    strata = {"a": "x", "b": "x", "c": "y", "d": "y"}
    for draw in uncertainty.bootstrap_pair_draws(pairs, strata, 20, 1):
        assert len(draw) == 4
        assert sum(pairs[i] in ("a", "b") for i in draw) == 2


def test_bag_intervals_contain_estimate_and_rank_models() -> None:
    frame = _paired_predictions()
    strata = {p: "all" for p in frame["pair_id"].unique()}
    cis, diffs = uncertainty.bag_metric_intervals(frame, strata, n=200)
    for row in cis.itertuples():
        assert row.lower - 1e-9 <= row.estimate <= row.upper + 1e-9
    auprc = diffs[(diffs["model"] == "weak") & (diffs["metric"] == "auprc")].iloc[0]
    assert auprc.difference > 0 and auprc.lower > 0
    assert "p_holm" in diffs


def test_temporal_intervals_use_explicit_denominators() -> None:
    rows = []
    for k in range(10):
        for status in ("clean", "impaired"):
            rows.append({"pair_id": f"p{k}", "status": status, "model": "m",
                         "event_recall": 1.0 if status == "impaired" else np.nan,
                         "vus_pr": 0.5, "temporal_auprc": 0.5,
                         "interval_iou": 0.2 if status == "impaired" else 1.0,
                         "detection_delay": 1.0 if status == "impaired" else np.nan,
                         "false_alarms_per_run": 0.5 if status == "clean" else 0.0})
    out = uncertainty.temporal_intervals(pd.DataFrame(rows), None, {}, n=50)
    iou = out[(out["metric"] == "interval_iou")].set_index("denominator")["estimate"]
    assert iou["impaired"] == pytest.approx(0.2) and iou["mixed"] == pytest.approx(0.6)
    fah = out[(out["metric"] == "false_alarms_per_hour")]["estimate"].iloc[0]
    assert fah == pytest.approx(2.0)


def test_ece_is_zero_for_perfect_bins_and_positive_for_overconfidence() -> None:
    labels = np.array([0] * 50 + [1] * 50)
    perfect = np.array([0.0] * 50 + [1.0] * 50)
    assert calibration.calibration_errors(labels, perfect)["ece"] < 1e-12
    over = np.full(100, 0.95)
    errors = calibration.calibration_errors(labels, over)
    assert abs(errors["ece"] - 0.45) < 1e-9 and abs(errors["mce"] - 0.45) < 1e-9


def test_calibration_summary_reports_bootstrap_interval() -> None:
    frame = _paired_predictions()
    strata = {p: "all" for p in frame["pair_id"].unique()}
    summary = calibration.calibration_summary(frame, context="t", strata=strata, n=100)
    assert {"ece", "mce", "ece_lower", "ece_upper"} <= set(summary)
    assert (summary["ece_lower"] <= summary["ece_upper"]).all()


def test_prefix_summary_and_time_to_detection() -> None:
    rows = []
    for fraction, p_imp in ((0.05, 0.2), (0.5, 0.8), (1.0, 0.9)):
        for k in range(4):
            rows.append({"model": "m", "fraction": fraction, "steps": int(fraction * 900), "status": "ok",
                         "scenario_id": f"i{k}", "pair_id": f"p{k}", "label": 1, "run_status": "impaired", "score": p_imp})
            rows.append({"model": "m", "fraction": fraction, "steps": int(fraction * 900), "status": "ok",
                         "scenario_id": f"c{k}", "pair_id": f"p{k}", "label": 0, "run_status": "clean", "score": 0.1})
    frame = pd.DataFrame(rows)
    summary = online.prefix_summary(frame).set_index("fraction")
    assert summary.loc[0.05, "true_positive_rate"] == 0.0
    assert summary.loc[0.5, "true_positive_rate"] == 1.0
    ttd = online.time_to_detection(frame).iloc[0]
    assert ttd.detected_by_end == 1.0 and ttd.median_first_step == 450
