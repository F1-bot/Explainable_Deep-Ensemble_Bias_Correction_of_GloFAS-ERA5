# Reproducibility guide

This document gives the exact, step-by-step recipe to reproduce the results of
the MDPI *Water* manuscript

> **Explainable Deep-Ensemble Learning for Multi-Scale Streamflow Bias
> Correction in GloFAS-ERA5 across the Snow-Influenced Transboundary Basins of
> Central Asia**

from the `sbc` code artifact. It complements the [`README.md`](README.md):
the README explains *what* the framework is; this file is the operational
*how-to-reproduce*.

There are two reproduction tracks:

| Track | Needs credentials? | Needs downloads? | Time | What it proves |
|-------|:------------------:|:----------------:|------|----------------|
| **A — Synthetic** | No | No | < 5 min | The full pipeline runs and recovers a *known* injected GloFAS-style bias |
| **B — Real data** | Yes (your own Copernicus token) | Yes (GloFAS, ERA5-Land, CA-discharge) | Hours–days (mostly download) | The published decadal / daily / PUR skill results and all figures |

Start with **Track A** to confirm your environment, then proceed to **Track B**
for the publication results.

---

## 0. Environment

```bash
git clone https://github.com/<your-org>/sbc.git
cd sbc

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"              # editable install + pytest/ruff/jupyter
```

- **Python ≥ 3.10.** Dependencies use minimum-version (`>=`) pins so the stack
  resolves cleanly on a fresh venv and on Google Colab.
- **GPU is optional.** The CPU build of PyTorch is sufficient; CUDA is used
  automatically when `torch.cuda.is_available()`.
- The global random seed (`seed: 1234` in `configs/default.yaml`) is applied via
  `sbc.utils.seed_everything`, which seeds Python, NumPy and PyTorch. Boosting
  and deep-learning results are reproducible to within the usual
  backend-/thread-level nondeterminism of XGBoost/LightGBM/CatBoost and CUDA.

Sanity-check the install:

```bash
pytest -q                            # CPU-only contract smoke tests, no credentials
```

All tests in `tests/test_smoke.py` should pass. They verify the log-residual
target round-trips, the metric suite is sane, the synthetic schema/feature
discovery holds, and the temporal / spatial / PUR splits are leakage-safe.

> If you skip the editable install, prefix every `python -m sbc.*` /
> `python scripts/*.py` command below with `PYTHONPATH=src`.

---

## Track A — Synthetic reproduction (no credentials, no downloads)

`sbc.synthetic.generate` builds an analysis-ready modelling table — identical in
schema to the assembled real-data table — from a nival-glacial water balance,
then injects the documented GloFAS snow-region error signatures (negative volume
bias, early/damped freshet, compressed variability). Because the bias is *known*,
this is both a pipeline smoke test and a controlled verification that the
framework recovers a known bias.

**A.1 — One-line check**

```python
from sbc.synthetic import generate
from sbc import schemas
from sbc.validation.metrics import evaluate

df = schemas.validate(generate(scale="decadal", seed=1234))
print(len(df), "rows;", len(schemas.feature_columns(df)), "features")
print("raw GloFAS KGE':", evaluate(df.q_obs, df.q_glofas)["kge"])
```

**A.2 — Full config-driven experiment (fast)**

```bash
python -m sbc.experiment --scale decadal --quick --source synthetic
python -m sbc.experiment --scale daily   --quick --source synthetic
```

**A.3 — Full config-driven experiment (full epochs, all models)**

```bash
python -m sbc.experiment --scale decadal --source synthetic
```

Useful flags (`python -m sbc.experiment --help`):

- `--quick` — short training, smaller nets (smoke run);
- `--models scaling,lgbm,regimeprobnet` — restrict the model roster;
- `--no-shap` — skip the SHAP attribution step;
- `--config path/to/your.yaml` — use an alternative configuration.

**Outputs** (under `results/`):

- `results/tables/per_gauge_synthetic_<scale>[_quick].{parquet,csv}` — per-gauge,
  per-split, per-model skill;
- `results/tables/summary_synthetic_<scale>[_quick].{parquet,csv}` — aggregated
  skill table (also printed to stdout);
- `results/tables/report_synthetic_<scale>[_quick].json` — machine-readable run
  report;
- `results/shap/global_importance.{parquet,csv}` (+ regime-conditional importance
  and a beeswarm PNG when SHAP succeeds).

---

## Track B — Real-data reproduction (publication results)

### B.0 — Credentials (your own)

This repository ships **no** secrets. Create your own
`configs/secrets.env` (git-ignored) with your Copernicus Personal Access Token —
see [README → Credential setup](README.md#credential-setup-copernicus-cds--ewds).
The same token works on both portals; only the URL differs:

```ini
CDS_URL=https://cds.climate.copernicus.eu/api
CDS_KEY=<your-personal-access-token>
EWDS_URL=https://ewds.climate.copernicus.eu/api
EWDS_KEY=<your-personal-access-token>
```

You must accept the licences for **ERA5-Land** (CDS) and
**cems-glofas-historical** (EWDS) on each dataset page first. Credentials may
also be exported into the process environment, which takes priority over the
file. **Never commit `configs/secrets.env`.**

### B.1 — Ground truth + static attributes (CA-discharge)

1. Download `CA-discharge.gpkg` from Zenodo record **8147591**
   ([doi:10.1038/s41597-023-02474-8](https://doi.org/10.1038/s41597-023-02474-8))
   and place it at `datasets/ca_discharge/CA-discharge.gpkg` (git-ignored).
2. Extract the processed ground-truth, gauge metadata and static attributes:

   ```bash
   python -m sbc.data.ca_discharge
   ```

   Writes to `datasets/processed/`:
   - `gauges.parquet` — gauge metadata (`code, basin, domain, lon, lat, …`);
   - `discharge_decadal.parquet`, `discharge_daily.parquet` — `code, date, q_obs`;
   - `static_attributes.parquet` — ~1090 HydroATLAS-style attributes per gauge;
   - (optionally) `basins.parquet`, `attribute_descriptions.parquet`,
     `quality_flags.parquet`, `domain_summary.json`.

### B.2 — GloFAS-ERA5 discharge (the predictor)

```bash
python scripts/download_glofas.py        # one NetCDF per hydrological year, resumable
```

This downloads GloFAS-ERA5 v4.0 daily discharge for the study bounding box
(N 44°, W 65°, S 37°, E 78°; 1979–2020) into `datasets/glofas_era5/`. It is
large and slow; the download is **resumable** (already-complete years are
skipped) and pixel extraction later uses only fully-written yearly files, so it
is safe to proceed once enough years are present.

### B.3 — ERA5-Land forcing (snow / melt / meteorology)

```bash
python scripts/download_era5_land.py     # one NetCDF per (variable, statistic), resumable
python -m sbc.data.era5_land             # aggregate to gauges/basins
```

The downloader fills `datasets/era5_land/`; the extractor writes the
basin-aggregated forcing table to
`datasets/interim/era5land_monthly_at_basins.parquet`. ERA5-Land forcing is
**optional** — the modelling table still assembles from GloFAS + static
attributes if this step is skipped (with reduced dynamic features).

### B.4 — Snap pixels, assemble tables, run the experiment

The end-to-end driver performs the remaining steps — download the GloFAS
upstream-area map for hydrologically correct gauge→river-pixel snapping, extract
the GloFAS series at every gauge, assemble the decadal and daily modelling
tables, and run the full validation-matrix experiment for each scale:

```bash
python scripts/run_real_pipeline.py                 # full publication run
python scripts/run_real_pipeline.py --quick         # fast smoke over real data
python scripts/run_real_pipeline.py --no-uparea     # skip the aux-map download
python scripts/run_real_pipeline.py --scales decadal
```

Equivalently, the assembly and experiment steps can be run on their own once the
inputs from B.1–B.3 exist:

```bash
python -m sbc.data.assemble                         # -> datasets/processed/model_table_<scale>.parquet
python -m sbc.experiment --scale decadal --source real
python -m sbc.experiment --scale daily   --source real
```

**Outputs** (under `results/`):

- `results/tables/per_gauge_real_<scale>.{parquet,csv}`,
  `results/tables/summary_real_<scale>.{parquet,csv}`,
  `results/tables/report_real_<scale>.json`;
- `results/shap/…` SHAP attribution artifacts.

### B.5 — Figures and tables

Two complementary generators:

```bash
# (a) Core skill figures + a Markdown summary directly from a result table:
python scripts/make_figures.py --tag real_decadal
#   -> results/figures/{kge_by_split,improvement_map,kge_decomposition,shap_importance}_real_decadal.png
#   -> results/tables/summary_real_decadal.md

# (b) The 14 publication figures (study area, framework, raw bias, skill-by-split,
#     improvement map, ablation, calibration, UQ, regimes, XAI, PUR attribution,
#     hydrographs, Naryn daily, decision/FDC):
python scripts/figures/fig01_study_area.py
python scripts/figures/fig04_skill_by_split.py
# … fig02 … fig14 analogously; outputs land in results/figures/ (and results/figures/publication/)
```

The extended analyses behind the supplementary tables/figures (calibration,
conformal UQ, ablations, CMAL/UQ comparison, decision skill, group-SHAP
stability, multi-scale reconciliation, PUR attribution) are produced by the
`scripts/run_*.py` drivers — e.g. `run_final_rigor.py`, `run_calibration_fixed.py`,
`run_round2_analyses.py`, `run_cmal_eval.py`, `run_v2_experiment.py`,
`run_v3_evaluation.py`, `run_consolidate.py` — which write to
`results/tables/` and `results/figures/`.

---

## Expected outputs and where they go

| Location | Contents |
|----------|----------|
| `results/tables/` | per-gauge & summary skill tables, ablations, calibration/UQ, decision skill (`.parquet` + `.csv`) and a Markdown summary |
| `results/figures/` | manuscript and diagnostic figures (`.png`), with `publication/` for the final set |
| `results/shap/` | global, regime-conditional and flagship SHAP importances + beeswarm |
| `results/models/` | trained model artifacts (`.pt`/`.pkl`/`.cbm`/…); git-ignored |
| `datasets/processed/` | assembled modelling tables and ground-truth/static parquets |
| `datasets/interim/` | intermediate GloFAS pixel series and ERA5-Land forcing |

Paths are centralised in `sbc.config.PATHS` and derived from the repository root
(or the `SBC_ROOT` environment variable), so the same commands work on Windows,
Linux and Colab without editing absolute paths.

---

## Runtime and hardware notes

- **Track A** (synthetic, `--quick`): a few minutes on a laptop CPU.
- **Track B downloads** dominate wall-clock time: GloFAS-ERA5 is many GB of
  per-year NetCDFs and depends on Copernicus queue load; budget hours to a day.
  Both downloaders are resumable, so they can be interrupted and restarted.
- **Track B modelling**: the decadal experiment runs comfortably on CPU; the
  daily Naryn case study and the deep flagship benefit from (but do not require)
  a GPU.
- **Memory**: the full static-attribute table (~1090 columns) and daily series
  are the heaviest in-memory objects; 16 GB RAM is comfortable.

---

## Determinism

- A single `seed` (default `1234`) flows from `configs/default.yaml` through
  `sbc.utils.seed_everything` into Python / NumPy / PyTorch and the synthetic
  generator.
- Tree-ensemble libraries (XGBoost / LightGBM / CatBoost) and CUDA kernels are
  not bit-for-bit deterministic across thread counts / hardware; expect
  negligible variation in the last reported digits rather than identical bytes.
- Re-running a stage overwrites its `results/` artifacts deterministically given
  the same inputs and seed.

---

## Troubleshooting

- **`FileNotFoundError: … not found. Produce it by running sbc.data.…`** — an
  upstream extractor has not been run yet. `sbc.data.assemble` names the exact
  producer module to run (B.1–B.3 above).
- **Copernicus 4xx / "licence not accepted"** — accept the dataset licence on the
  CDS / EWDS portal, and confirm your token is in `configs/secrets.env` (or the
  environment). The synthetic track needs no credentials.
- **`No complete GloFAS NetCDFs found`** — let `scripts/download_glofas.py`
  finish at least a few full years; extraction skips partially-written files by
  size, so it is safe to run alongside an in-progress download.
- **SHAP step warns and is skipped** — non-fatal; the skill tables are still
  written. Re-run with `--no-shap` to silence it.
- **`tabulate` missing for the Markdown summary** — optional; `make_figures.py`
  falls back to a plain-text table.

---

## What is intentionally NOT in this repository

- **Credentials.** `configs/secrets.env` is git-ignored; you supply your own
  Copernicus token. No key is distributed with the code.
- **Raw data.** GloFAS / ERA5-Land NetCDFs, the CA-discharge GeoPackage, and
  HydroATLAS are git-ignored and reproduced from their public sources
  (see [README → Data sources](README.md#data-sources)).
- **Heavy model artifacts.** `results/models/*` are git-ignored and regenerated
  by the experiment runs.
