"""Trust-gated, energy-aware detector cascade (TGC).

Stage 0 (receiver counters, no model): a run is *untrusted* if the receiver
accepted any message without a timestamp, any message stamped in its future,
or more stale (non-newer sequence) rejections than the maximum seen in
unperturbed validation-B runs.

Stage 1 (trusted runs): the cheapest non-neural cyber detector whose
validation-A macro-F1 is maximal, with its frozen calibration.

Stage 2 (untrusted runs): CyberGuard-DER, whose ranking proved robust to
timestamp loss and clock errors, with a route-specific calibrator fitted only
on degraded validation-B runs (perturbation seed differs from the test seed).

Every design choice uses validation data; locked runs are scored once per
perturbation with the frozen design.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridimpair.evaluation.metrics import bag_metrics
from gridimpair.r1 import frozen, metadata
from sklearn.linear_model import LogisticRegression

from gridimpair.workflow import ProbabilityCalibrator

CHEAPNESS = ("message_age_watchdog", "xgboost_cyber", "xgboost_hybrid", "multirocket_hydra_cyber", "multirocket_hydra_hybrid")
METADATA_KINDS = ("clock_offset", "clock_drift", "timestamp_dropout", "sequence_corruption", "metadata_loss", "metadata_delay")


@dataclass
class CascadeDesign:
    cheap_model: str
    learned_model: str
    stale_threshold: float
    route_calibrators: dict[str, RouteCalibrator]
    validation_summary: dict


def _physical_spec(training_dir: Path):
    pre = json.loads((training_dir / "preprocessing.json").read_text(encoding="utf-8"))
    phys = pre["tiers"]["physical"]
    return list(phys["channels"]), np.asarray(phys["normalization"]["std"], dtype=np.float32)


def degrade(inputs: frozen.FrozenInputs, perturbation: metadata.Perturbation, sim_root: Path, training_dir: Path,
            seed: int, ledgers: dict) -> tuple[frozen.FrozenInputs, list[dict]]:
    channels, std = _physical_spec(training_dir)
    raws, indicators = [], []
    for i, r in enumerate(inputs.records):
        ind: dict = {}
        raws.append(metadata.perturb_trace(inputs.raw_values[i], inputs.columns, ledgers[r.scenario_id], perturbation,
                                           metadata._rng(r.scenario_id, perturbation, seed), channels, std, indicators=ind))
        indicators.append(ind)
    return frozen.FrozenInputs(inputs.records, np.stack(raws), inputs.labels, inputs.columns), indicators


def trust_reason(ind: dict, stale_threshold: float) -> str:
    """Receiver-observable reason a run is untrusted ("" if trusted); fixed priority."""
    if ind["missing_timestamps"] > 0:
        return "missing_timestamp"
    if ind["future_timestamps"] > 0:
        return "future_timestamp"
    if ind["stale_rejections"] > stale_threshold:
        return "sequence_anomaly"
    return ""


def untrusted(indicators: list[dict], stale_threshold: float) -> np.ndarray:
    return np.asarray([trust_reason(ind, stale_threshold) != "" for ind in indicators])


SEVERITY_KEY = {"missing_timestamp": "missing_timestamps", "future_timestamp": "future_timestamps",
                "sequence_anomaly": "stale_rejections"}


@dataclass
class RouteCalibrator:
    """Logistic recalibration in (raw score, log1p observed anomaly count).

    The anomaly count is what the receiver itself observes, so conditioning on
    it is deployable; within one severity level the map is monotone in the raw
    score whenever the fitted score coefficient is positive.
    """
    kind: str
    model: object = None
    constant: float = 0.5

    def predict(self, raw: np.ndarray, severity: np.ndarray) -> np.ndarray:
        if self.kind == "constant":
            return np.full(len(raw), self.constant)
        x = np.c_[np.asarray(raw, float), np.log1p(np.asarray(severity, float))]
        return self.model.predict_proba(x)[:, 1]


def _fit_route(raw: np.ndarray, severity: np.ndarray, labels: np.ndarray) -> RouteCalibrator:
    if np.unique(labels).size < 2 or np.unique(raw).size < 2:
        return RouteCalibrator("constant", constant=float(np.clip(labels.mean(), 1e-5, 1 - 1e-5)))
    x = np.c_[raw, np.log1p(severity)]
    return RouteCalibrator("logistic_score_severity", LogisticRegression(random_state=2026, C=1e4, max_iter=2000).fit(x, labels))


def subset(inputs: frozen.FrozenInputs, mask: np.ndarray) -> frozen.FrozenInputs:
    idx = np.flatnonzero(mask)
    return frozen.FrozenInputs([inputs.records[i] for i in idx], inputs.raw_values[idx], inputs.labels[idx], inputs.columns)


def design_cascade(open_inputs: frozen.FrozenInputs, detectors: dict[str, frozen.FrozenDetector], training_dir: Path,
                   sim_root: Path, *, seed: int = 7, learned_model: str = "candidate",
                   exclude_kinds: tuple[str, ...] = ()) -> CascadeDesign:
    """``exclude_kinds`` removes perturbation families from route calibration (leave-one-family-out)."""
    splits = np.asarray([r.split for r in open_inputs.records])
    val_a, val_b = subset(open_inputs, splits == "validation_a"), subset(open_inputs, splits == "validation_b")
    ledgers = {r.scenario_id: metadata.load_ledger(sim_root, r.scenario_id) for r in val_b.records}
    # trust threshold from unperturbed validation-B runs
    _, base_ind = degrade(val_b, metadata.Perturbation("none"), sim_root, training_dir, seed, ledgers)
    stale_threshold = float(max(ind["stale_rejections"] for ind in base_ind))
    # cheapest best non-neural cyber detector on validation-A
    prep_a = frozen.prepare(val_a, training_dir)
    scores = {}
    for name in CHEAPNESS:
        if name in detectors:
            p = frozen.calibrated(detectors[name], detectors[name].score(prep_a))
            scores[name] = bag_metrics(val_a.labels, p)["macro_f1"]
    best = max(scores.values())
    cheap = next(n for n in CHEAPNESS if n in scores and scores[n] >= best - 1e-12)
    # route calibrators, one per observed trust failure, on degraded validation-B runs
    pooled: dict[str, tuple[list, list, list]] = {}
    learned = detectors[learned_model]
    for perturbation in [metadata.Perturbation("none")] + [p for p in metadata.GRID
                                                          if p.kind in METADATA_KINDS and p.kind not in exclude_kinds]:
        degraded, ind = degrade(val_b, perturbation, sim_root, training_dir, seed, ledgers)
        reasons = [trust_reason(x, stale_threshold) for x in ind]
        if not any(reasons):
            continue
        raw = learned.score(frozen.prepare(degraded, training_dir))
        for i, reason in enumerate(reasons):
            if reason:
                pooled.setdefault(reason, ([], [], []))
                pooled[reason][0].append(raw[i])
                pooled[reason][1].append(ind[i][SEVERITY_KEY[reason]])
                pooled[reason][2].append(degraded.labels[i])
    route = {reason: _fit_route(np.asarray(x, float), np.asarray(v, float), np.asarray(y, int))
             for reason, (x, v, y) in pooled.items()}
    summary = {"stale_threshold": stale_threshold, "cheap_validation_a_macro_f1": scores, "excluded_kinds": list(exclude_kinds),
               "route_calibration_runs": {k: len(v[2]) for k, v in pooled.items()},
               "route_coefficients": {k: (v.model.coef_.round(4).tolist() if v.model is not None else None) for k, v in route.items()},
               "route_calibrator_kinds": {k: v.kind for k, v in route.items()}}
    return CascadeDesign(cheap, learned_model, stale_threshold, route, summary)


def evaluate_cascade(locked_inputs: frozen.FrozenInputs, detectors: dict[str, frozen.FrozenDetector], design: CascadeDesign,
                     training_dir: Path, sim_root: Path, *, seed: int = 2026,
                     grid: tuple[metadata.Perturbation, ...] = metadata.GRID) -> tuple[pd.DataFrame, pd.DataFrame]:
    ledgers = {r.scenario_id: metadata.load_ledger(sim_root, r.scenario_id) for r in locked_inputs.records}
    cheap, learned = detectors[design.cheap_model], detectors[design.learned_model]
    rows, preds = [], []
    for perturbation in grid:
        degraded, ind = degrade(locked_inputs, perturbation, sim_root, training_dir, seed, ledgers)
        reasons = [trust_reason(x, design.stale_threshold) for x in ind]
        flag = np.asarray([r != "" for r in reasons])
        prep = frozen.prepare(degraded, training_dir)
        p_cheap = frozen.calibrated(cheap, cheap.score(prep))
        raw_learned = learned.score(prep)
        p_learned_frozen = frozen.calibrated(learned, raw_learned)
        p_learned_route = np.asarray([
            float(np.clip(design.route_calibrators[r].predict(np.asarray([v]), np.asarray([x[SEVERITY_KEY[r]]]))[0], 0, 1))
            if r in design.route_calibrators else pf
            for r, v, pf, x in zip(reasons, raw_learned, p_learned_frozen, ind)
        ])
        p_cascade = np.where(flag, p_learned_route, p_cheap)
        y = degraded.labels
        for name, p in (("cascade", p_cascade), (f"{design.cheap_model}_only", p_cheap),
                        (f"{design.learned_model}_frozen", p_learned_frozen), (f"{design.learned_model}_route_calibrated", p_learned_route)):
            rows.append({"perturbation": perturbation.kind, "level": perturbation.level, "policy": name,
                         "routed_to_learned": float(flag.mean()) if name == "cascade" else (1.0 if "candidate" in name else 0.0),
                         **bag_metrics(y, p)})
        for r, f, p in zip(degraded.records, flag, p_cascade):
            preds.append({"perturbation": perturbation.kind, "level": perturbation.level, "scenario_id": r.scenario_id,
                          "pair_id": r.pair_id, "label": r.label, "status": "impaired" if r.label else "clean",
                          "untrusted": bool(f), "model": "cascade", "score": float(p)})
        print(f"cascade: {perturbation.label} routed={flag.mean():.3f}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(preds)
