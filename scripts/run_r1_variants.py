"""R1 Phases 3-4: architecture ablation and cyber-feature removal.

All variants reuse the frozen preprocessing (normalization and AR residual
fits) and CyberGuard-DER's selected hyperparameters, train on development
runs with early stopping on validation-A (as in the frozen protocol), are
calibrated on validation-B, and are scored once on the locked-v2 runs. They
are post-hoc diagnostics: no variant or setting is selected on locked data.

Usage:  .venv\\Scripts\\python.exe scripts\\run_r1_variants.py --part all
Parts:  ablation, removal, all.   Outputs: <artifact_root>/r1_variants/
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridimpair.config import load_config  # noqa: E402
from gridimpair.evaluation.metrics import bag_metrics, temporal_metrics  # noqa: E402
from gridimpair.evaluation.operating_points import apply_alarm_policy, fit_temporal_operating_points  # noqa: E402
from gridimpair.evaluation.temporal import interval_labels  # noqa: E402
from gridimpair.models.classical import XGBoostDetector  # noqa: E402
from gridimpair.models.tcn import StandardCausalTCN  # noqa: E402
from gridimpair.r1 import frozen  # noqa: E402
from gridimpair.r1.variants import VARIANTS, AblationMIL, count_macs  # noqa: E402
from gridimpair.train.checkpoints import load_checkpoint  # noqa: E402
from gridimpair.train.engine import LoaderBundle, fit_model  # noqa: E402
from gridimpair.contracts import stable_hash  # noqa: E402
from dataclasses import asdict  # noqa: E402
from gridimpair.workflow import (  # noqa: E402
    CYBER_CHANNELS,
    HYBRID_CHANNELS,
    _fit_calibrator,
    _residual_views,
    discover_training_records,
)

SEEDS = (2026, 2027, 2028)


class Data:
    def __init__(self, art: Path, training: Path) -> None:
        open_records = discover_training_records(art / "simulation")
        locked = pd.read_csv(art / "results" / "predictions_locked.csv")
        self.open = frozen.load_inputs(open_records)
        self.locked = frozen.load_inputs(frozen.locked_records(art, locked))
        p_open, p_locked = frozen.prepare(self.open, training), frozen.prepare(self.locked, training)
        self.values = {"open": p_open.values_by_tier, "locked": p_locked.values_by_tier}
        self.residuals = {"open": p_open.residuals_by_tier, "locked": p_locked.residuals_by_tier}
        self.splits = np.asarray([r.split for r in self.open.records])
        self.labels_open = self.open.labels.astype(int)
        self.labels_locked = self.locked.labels.astype(int)
        self.development = self.splits == "development"
        self.validation_a = self.splits == "validation_a"
        self.validation_b = self.splits == "validation_b"
        sim = art / "simulation"
        steps = self.locked.raw_values.shape[-1]
        self.step_labels = np.stack([
            interval_labels(steps, json.loads((sim / "restricted" / f"{r.scenario_id}.json").read_text(encoding="utf-8")).get("communication_intervals", []))
            for r in self.locked.records
        ])


def _train(model, values, residuals, data: Data, cfg, out_dir: Path, batch_size: int, two_views: bool, device: str):
    done = out_dir / "done.json"
    tensors = [torch.from_numpy(values)] + ([torch.from_numpy(residuals)] if two_views else [])
    dataset = TensorDataset(*tensors, torch.from_numpy(data.labels_open.astype(np.float32)))
    train_idx, val_idx = np.flatnonzero(data.development).tolist(), np.flatnonzero(data.validation_a).tolist()
    loaders = LoaderBundle(
        train=DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(2026)),
        validation=DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False),
        split_hash="r1-variants",
    )
    if not done.exists():
        fit_model(model, loaders, cfg, out_dir, device_name=device, resume=True)
        done.write_text("{}", encoding="utf-8")
    load_checkpoint(out_dir / "best.pt", model=model, expected_config_hash=stable_hash(asdict(cfg)),
                    expected_split_hash="r1-variants", restore_rng=False)
    return model


@torch.no_grad()
def _score(model, values, residuals, two_views: bool, device: str):
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    bags, steps = [], []
    for s in range(0, len(values), 64):
        x = torch.from_numpy(values[s:s + 64]).to(dev)
        out = model(x, torch.from_numpy(residuals[s:s + 64]).to(dev)) if two_views else model(x)
        bags.append(torch.sigmoid(out.bag_logit).cpu().numpy())
        steps.append(torch.sigmoid(out.instance_logits).cpu().numpy())
    return np.concatenate(bags), np.concatenate(steps)


def _evaluate(name: str, open_bag, open_steps, locked_bag, locked_steps, data: Data, temporal: bool = True) -> dict:
    calibrator = _fit_calibrator(open_bag[data.validation_b], data.labels_open[data.validation_b])
    probability = np.clip(calibrator.predict(locked_bag), 0, 1)
    row = {"model": name, **bag_metrics(data.labels_locked, probability)}
    if temporal and open_steps is not None:
        point = fit_temporal_operating_points({name: open_steps}, clean_validation_mask=data.validation_b & (data.labels_open == 0),
                                              target_false_alarms_per_run=0.10, min_on_steps=3, min_off_steps=3)[name]
        per = []
        for i in range(len(locked_steps)):
            alarms = apply_alarm_policy(locked_steps[i], threshold=float(point["threshold"]), min_on_steps=3, min_off_steps=3)
            per.append({"label": data.labels_locked[i], **temporal_metrics(data.step_labels[i], locked_steps[i], threshold=None, alarms=alarms)})
        per = pd.DataFrame(per)
        imp, cln = per[per["label"] == 1], per[per["label"] == 0]
        row.update({"vus_pr_impaired": imp["vus_pr"].mean(), "iou_impaired": imp["interval_iou"].mean(),
                    "event_recall": imp["event_recall"].mean(), "detection_delay": imp["detection_delay"].mean(),
                    "false_alarms_per_run_clean": cln["false_alarms_per_run"].mean(),
                    "iou_mixed": per["interval_iou"].mean()})
    return row, probability


def _energy(out_dir: Path) -> dict:
    m = pd.read_csv(out_dir / "metrics.csv")
    return {"train_seconds": float(m["epoch_seconds"].sum()), "train_gpu_energy_j": float(m.get("gpu_energy_j", pd.Series([np.nan])).sum()),
            "train_cpu_energy_j_est": float(m.get("cpu_energy_j_est", pd.Series([np.nan])).sum()), "epochs": len(m)}


def _neural_family(name, build, channels, open_v, open_r, locked_v, locked_r, data, cfg, out_root, batch, two_views, device, prediction_rows):
    rows, ob, os_, lb, ls_ = [], [], [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = _train(build(), open_v, open_r, data, cfg, out_root / name.replace("|", "__") / f"seed-{seed}", batch, two_views, device)
        a, b = _score(model, open_v, open_r, two_views, device)
        c, e = _score(model, locked_v, locked_r, two_views, device)
        ob.append(a); os_.append(b); lb.append(c); ls_.append(e)
        row, _ = _evaluate(name, a, b, c, e, data)
        rows.append({**row, "seed": seed, **_energy(out_root / name.replace("|", "__") / f"seed-{seed}")})
    row, probability = _evaluate(name, np.mean(ob, 0), np.mean(os_, 0), np.mean(lb, 0), np.mean(ls_, 0), data)
    probe = build()
    energy_rows = pd.DataFrame(rows)
    rows.append({**row, "seed": "ensemble", "parameters": sum(p.numel() for p in probe.parameters()),
                 "macs_per_run": count_macs(probe, channels, two_views=two_views), "train_seconds": energy_rows["train_seconds"].sum(),
                 "train_gpu_energy_j": energy_rows["train_gpu_energy_j"].sum(), "train_cpu_energy_j_est": energy_rows["train_cpu_energy_j_est"].sum()})
    for r, p in zip(data.locked.records, probability):
        prediction_rows.append({"model": name, "scenario_id": r.scenario_id, "pair_id": r.pair_id, "label": r.label,
                                "status": "impaired" if r.label else "clean", "score": float(p)})
    print(f"{name}: done", flush=True)
    return rows


def run_ablation(config, data: Data, registry: dict, out: Path, device: str) -> None:
    hp = registry["candidate"]["selected_trial"]["values"]
    cfg = replace(config.training, learning_rate=float(hp.get("learning_rate", config.training.learning_rate)))
    batch = int(config.raw.get("training", {}).get("batch_size", 32))
    v, r = data.values, data.residuals
    channels = v["open"]["hybrid"].shape[1]
    rows, preds = [], []
    for name, flags in VARIANTS.items():
        build = lambda flags=flags: AblationMIL(channels, int(hp["hidden"]), dropout=float(hp["dropout"]), **flags)
        rows += _neural_family(name, build, channels, v["open"]["hybrid"], r["open"]["hybrid"], v["locked"]["hybrid"],
                               r["locked"]["hybrid"], data, cfg, out / "ablation", batch, True, device, preds)
    pd.DataFrame(rows).to_csv(out / "ablation_metrics.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "ablation_locked_predictions.csv", index=False)


def run_removal(config, data: Data, registry: dict, out: Path, device: str) -> None:
    batch = int(config.raw.get("training", {}).get("batch_size", 32))
    rows, preds = [], []
    hybrid = list(HYBRID_CHANNELS)
    fit_mask = data.development & (data.labels_open == 0)
    for removed in CYBER_CHANNELS:
        keep = [i for i, c in enumerate(hybrid) if c != removed]
        ov, lv = data.values["open"]["hybrid"][:, keep], data.values["locked"]["hybrid"][:, keep]
        orr, ridge = _residual_views(ov, fit_mask)
        lrr = np.stack([np.nan_to_num(ridge.transform(case.T).T, nan=0.0, posinf=0.0, neginf=0.0) for case in lv]).astype(np.float32)
        channels = len(keep)
        tag = f"without_{removed}"
        hp = registry["candidate"]["selected_trial"]["values"]
        cfg = replace(config.training, learning_rate=float(hp.get("learning_rate", config.training.learning_rate)))
        build = lambda: AblationMIL(channels, int(hp["hidden"]), dropout=float(hp["dropout"]))
        for row in _neural_family(f"candidate|{tag}", build, channels, ov, orr, lv, lrr, data, cfg, out / "removal", batch, True, device, preds):
            rows.append({**row, "family": "CyberGuard-DER", "removed": removed})
        hpt = registry["tcn_hybrid"]["selected_trial"]["values"]
        cfg_t = replace(config.training, learning_rate=float(hpt.get("learning_rate", config.training.learning_rate)))
        build_t = lambda: StandardCausalTCN(channels, int(hpt["hidden"]), dropout=float(hpt["dropout"]))
        for row in _neural_family(f"tcn_hybrid|{tag}", build_t, channels, ov, None, lv, None, data, cfg_t, out / "removal", batch, False, device, preds):
            rows.append({**row, "family": "TCN-H", "removed": removed})
        xgb = XGBoostDetector().fit(ov[data.development], data.labels_open[data.development], split="development")
        o_scores = np.asarray([xgb.score(c) for c in ov]); l_scores = np.asarray([xgb.score(c) for c in lv])
        row, probability = _evaluate(f"xgboost_hybrid|{tag}", o_scores, None, l_scores, None, data, temporal=False)
        rows.append({**row, "seed": "single", "family": "XGB-H", "removed": removed})
        for r, p in zip(data.locked.records, probability):
            preds.append({"model": f"xgboost_hybrid|{tag}", "scenario_id": r.scenario_id, "pair_id": r.pair_id, "label": r.label,
                          "status": "impaired" if r.label else "clean", "score": float(p)})
        pd.DataFrame(rows).to_csv(out / "removal_metrics.csv", index=False)
        pd.DataFrame(preds).to_csv(out / "removal_locked_predictions.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="all", choices=["ablation", "removal", "all"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "q1_v2.toml"))
    args = parser.parse_args()
    config = load_config(Path(args.config))
    art = config.paths.artifact_root
    training = art / "training"
    out = art / "r1_variants"
    out.mkdir(parents=True, exist_ok=True)
    registry = {e["family"]: e for e in json.loads((training / "model_registry.json").read_text(encoding="utf-8"))}
    data = Data(art, training)
    device = config.runtime.device
    if args.part in ("ablation", "all"):
        run_ablation(config, data, registry, out, device)
    if args.part in ("removal", "all"):
        run_removal(config, data, registry, out, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
