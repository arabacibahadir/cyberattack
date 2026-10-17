# GridImpair

Source-code package for the communication-availability impairment experiments.

## Contents

- `src/`: Python package
- `tests/`: automated tests
- `configs/`: experiment and model configuration
- `scripts/`: execution and figure-generation scripts
- `pyproject.toml` and `uv.lock`: dependency definitions

Generated artifacts, trained models, datasets, manuscripts, caches, and local
machine metadata are intentionally excluded.

## Test

```powershell
python -m pytest tests -q
```


## Reproduction and extended analyses

The full pipeline (simulation, training, one-time locked evaluation, closed loop,
figures) runs with:

```powershell
.\setup.ps1
.\scripts\run_r1_reproduction.ps1
```

The extended analyses read the frozen outputs in `artifacts_q1_v2/` and do not
change them:

- `src/gridimpair/energy.py`: per-epoch training energy (NVML GPU counter; CPU
  energy estimated from process CPU time and a per-core TDP budget)
- `src/gridimpair/r1/`: scenario-pair bootstrap intervals, calibration (ECE/MCE),
  prefix-horizon online evaluation, exact replay of the stored message ledgers
  under metadata and measurement degradations, the trust-gated detector cascade,
  matched-hardware inference energy benchmark, and architecture ablation variants

```powershell
.\.venv\Scripts\python.exe scripts\run_r1_analysis.py --stage all
.\.venv\Scripts\python.exe scripts\run_r1_analysis.py --stage cascade_lofo
.\.venv\Scripts\python.exe scripts\run_r1_variants.py --part all
.\.venv\Scripts\python.exe scripts\r1_training_energy.py
.\.venv\Scripts\python.exe scripts\make_r1_figures.py
.\.venv\Scripts\python.exe scripts\make_r1_figures_extra.py <figure_dir> <supplementary_figure_dir>
```

Outputs are written to `artifacts_q1_v2/r1_analysis/`, `artifacts_q1_v2/r1_variants/`,
and the given figure folders.
