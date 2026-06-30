# sbc — Snow-influenced streamflow Bias Correction

**Explainable Deep-Ensemble Learning for Multi-Scale Streamflow Bias Correction
in GloFAS-ERA5 across the Snow-Influenced Transboundary Basins of Central Asia**

Open-source code artifact accompanying the MDPI *Water* manuscript. `sbc` is an
explainable, deep-ensemble framework that learns and removes the systematic
errors of the **GloFAS-ERA5** global river-discharge reanalysis in the snow- and
glacier-fed headwaters of Central Asia, at both the operational **decadal**
(10-day) scale and the **daily** scale.

| | |
|---|---|
| **Language** | Python ≥ 3.10 |
| **License** | MIT (declared in [`pyproject.toml`](pyproject.toml)) |
| **Package** | `sbc` (`src/` layout, `pip install -e .`) |
| **Status** | Research artifact, v0.1.0 |
| **Reproduce** | [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — one synthetic command, full real-data pipeline, paper figures/tables |
| **Cite** | [`CITATION.cff`](CITATION.cff) |

> **Reproducibility in one line** (no credentials, no downloads, < 1 minute):
> ```bash
> pip install -e . && PYTHONPATH=src python -m sbc.experiment --scale decadal --quick --source synthetic
> ```
> This runs the entire pipeline on a built-in physically-grounded synthetic
> benchmark with a *known* GloFAS-style bias and writes skill tables to
> `results/tables/`. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full
> real-data reproduction.

---

## Table of contents

- [Abstract](#abstract)
- [Multi-scale experimental design](#multi-scale-experimental-design)
- [Data sources](#data-sources)
- [Installation](#installation)
- [Credential setup (Copernicus CDS / EWDS)](#credential-setup-copernicus-cds--ewds)
- [Quick start (synthetic smoke test)](#quick-start-synthetic-smoke-test)
- [Repository structure](#repository-structure)
- [Modelling approach](#modelling-approach)
- [Validation matrix and metrics](#validation-matrix-and-metrics)
- [Explainability](#explainability)
- [Configuration](#configuration)
- [Reproducing the paper](#reproducing-the-paper)
- [Testing](#testing)
- [Citation & acknowledgement](#citation--acknowledgement)
- [License](#license)

---

## Abstract

Global hydrological reanalyses such as GloFAS-ERA5 provide spatially complete
discharge estimates but carry large, structured errors in mountainous,
snow-influenced catchments: a systematic negative volume bias, an early-shifted
and damped snowmelt freshet, and compressed flow variability. `sbc` reframes
bias correction as a **log-residual learning** problem — models predict the
multiplicative residual `log(q_obs + EPS) − log(q_glofas + EPS)`, which is added
back to the reanalysis to reconstruct corrected discharge. A diverse ensemble
(an entity-aware LSTM, a gradient-boosting trio, and a regime-aware
*probabilistic* flagship) is combined through stacking, tuned with Optuna, and
interrogated with SHAP so that every correction is traceable to physically
meaningful snow, glacier, temperature and precipitation drivers. The framework
is evaluated with a leakage-safe validation matrix (temporal holdout,
leave-one-basin-out, and prediction-in-ungauged-regions) over the decadal gauge
network of the Syr Darya / Chu / Talas systems, a high-resolution daily case
study in the Naryn headwaters, and an independent transfer test on the
hydrologically distinct Amu Darya tributaries.

The framework's contribution is the *synthesis*: an explainable, physically
constrained, multi-scale, uncertainty-aware bias-correction pipeline for a
data-sparse cryospheric region, rather than any single new primitive. Reported
skill is benchmarked against strong statistical and ML baselines; see the
manuscript and the generated tables in `results/tables/` for the numbers.

---

## Multi-scale experimental design

The study is deliberately multi-scale so that operational decadal forecasting,
event-scale daily dynamics, and out-of-domain generalisation are each tested:

| Tier | Scale | Domain | Purpose |
|------|-------|--------|---------|
| **Primary** | Decadal (10-day) | Gauge network across 7 core basins | Main bias-correction experiment at the operational planning scale |
| **Case study** | Daily | Naryn headwaters (gauges `16055`, `16068`) | High-resolution snowmelt-freshet dynamics & peak timing |
| **Transfer test** | Decadal / daily | Amu Darya tributaries | Independent generalisation / prediction-in-ungauged-regions (PUR) |

The decadal scale is treated as **primary** because the decadal observation
record is far denser than the daily record in this region; the daily experiment
is a focused Naryn case study rather than a basin-wide claim.

- **Core domain** (`domain = "core"`): `SYR_DARYA`, `NARYN`, `CHU`, `TALAS`,
  `CHIRCHIK`, `QASHQADARYA`, `AKHANGARAN`.
- **Transfer domain** (`domain = "transfer"`): `PYANDZH`, `VAKSH`,
  `KOFARNIKHAN`, `ZERAFSHAN`, `SURKHANDARYA` (Amu Darya system, held out).

The study window is the GloFAS overlap period **1979–2020**; a gauge needs at
least **8** valid post-1979 years to enter the study. The geographic bounding
box is N 44°, W 65°, S 37°, E 78°. The authoritative definitions live in
[`src/sbc/data/domain.py`](src/sbc/data/domain.py).

---

## Data sources

All datasets are openly available; large raw files are git-ignored and
re-downloadable via `scripts/` (Copernicus products require a free account and
per-dataset licence acceptance — see [Credential setup](#credential-setup-copernicus-cds--ewds)).

| Dataset | Role | Product / access | DOI / record |
|---------|------|------------------|--------------|
| **GloFAS-ERA5 historical** | Predictor to be bias-corrected (river discharge, v4.0, 0.05°, LISFLOOD forced by ERA5/HTESSEL) | Copernicus **EWDS** `cems-glofas-historical` | [10.24381/cds.a4fdd6b9](https://doi.org/10.24381/cds.a4fdd6b9) |
| **ERA5-Land** | Dynamic snow/meteorological forcing (SWE, snowmelt, snowfall, T2m, precipitation, soil moisture) | Copernicus **CDS** `reanalysis-era5-land` (0.1°; daily-statistics path, with a monthly-means fallback broadcast onto the target periods) | [10.24381/cds.68d2bb30](https://doi.org/10.24381/cds.68d2bb30) |
| **CA-discharge** | Ground-truth observed discharge (decadal / daily / monthly) for Central Asia, plus ~1090 HydroATLAS-style static catchment attributes | Zenodo record **8147591** (`CA-discharge.gpkg`) | [10.1038/s41597-023-02474-8](https://doi.org/10.1038/s41597-023-02474-8) |
| **HydroATLAS** | Upstream source of the static catchment attributes (terrain, climate, glacier, land cover) bundled in CA-discharge | [HydroSHEDS / HydroATLAS](https://www.hydrosheds.org/hydroatlas) | Linke et al., 2019, *Sci. Data* 6:283 |

> **CA-discharge:** Marti, Siegfried, Yakovlev, Karger, Ragettli et al. (2023),
> *Scientific Data* **10**:579. The consolidated GeoPackage holds gauge
> metadata, contributing-area polygons, ~1090 HydroATLAS-style static
> attributes, long-format discharge series, and QC flags. The static attributes
> used as model features are read directly from this file (no separate
> HydroATLAS download is required for the default pipeline).

No raw data ships with this repository. The 24 MB consolidated
`CA-discharge.gpkg` and all gridded NetCDF/GRIB products are git-ignored (see
[`.gitignore`](.gitignore)) and reproduced from the sources above.

---

## Installation

### Option A — local virtual environment

```bash
git clone https://github.com/F1-bot/Explainable_Deep-Ensemble_Bias_Correction_of_GloFAS-ERA5.git
cd sbc

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                 # editable install of the package itself
```

Requires **Python ≥ 3.10**. A CUDA GPU is optional — the CPU build of PyTorch is
sufficient for the boosting trio and the small EA-LSTM / flagship networks; CUDA
is used automatically when `torch.cuda.is_available()`.

> After `pip install -e .` the `sbc` package is importable directly. The scripts
> and `python -m sbc.*` examples below also work without an editable install if
> you prefix them with `PYTHONPATH=src`.

### Option B — Google Colab

```python
!pip install -q -r requirements.txt
import os
# Point sbc at your Drive-mounted project root (datasets/, results/, configs/):
os.environ["SBC_ROOT"] = "/content/drive/MyDrive/MDPI"
```

`sbc.config` resolves the repository root from the `SBC_ROOT` environment
variable when set (otherwise it infers it from the package location), so the
identical code runs unchanged on Windows, Linux and Colab.

---

## Credential setup (Copernicus CDS / EWDS)

> **You must supply your own Copernicus credentials.** This repository contains
> **no** secrets. The only file that would hold a token — `configs/secrets.env`
> — is git-ignored and must never be committed or shared.

GloFAS and ERA5-Land downloads use a single ECMWF Personal Access Token; only
the endpoint URL differs between the two portals. Create your own
`configs/secrets.env` from this template (replace the placeholders with your
token):

```ini
# CDS  (ERA5-Land):              https://cds.climate.copernicus.eu/profile
CDS_URL=https://cds.climate.copernicus.eu/api
CDS_KEY=<your-personal-access-token>

# EWDS (cems-glofas-historical): https://ewds.climate.copernicus.eu/api
EWDS_URL=https://ewds.climate.copernicus.eu/api
EWDS_KEY=<your-personal-access-token>
```

Steps:

1. Register (free) at the [CDS](https://cds.climate.copernicus.eu/) and
   [EWDS](https://ewds.climate.copernicus.eu/) portals and copy your Personal
   Access Token from your profile page.
2. **Accept the licence** for **ERA5-Land** (CDS) and
   **cems-glofas-historical** (EWDS) on each dataset page — the API returns an
   error until you do.
3. Save the token(s) into `configs/secrets.env` as above, **or** export the same
   variables into your shell environment. Process-environment variables take
   priority over the file (`sbc.data._cds.load_secrets`).

The synthetic smoke test and all unit tests run **without any credentials** —
credentials are only needed to download the real reanalysis.

---

## Quick start (synthetic smoke test)

No credentials and no downloads are required: `sbc.synthetic` generates an
analysis-ready modelling table — identical in schema to the assembled real-data
table — from a simple nival-glacial water balance, then injects the documented
GloFAS snow-region error signatures (negative volume bias, early/damped freshet,
compressed variability). Because the injected bias is *known*, this lets you
smoke-test the whole pipeline and verify that the framework recovers a known
bias.

```python
from sbc.synthetic import generate
from sbc import schemas
from sbc.validation.metrics import evaluate

df = generate(scale="daily", seed=1234)          # tidy modelling table
df = schemas.validate(df)                         # ensures the log_residual target
features = schemas.feature_columns(df)
print(len(df), "rows;", len(features), "features")
print("raw GloFAS KGE':", evaluate(df.q_obs, df.q_glofas)["kge"])
```

Run the full config-driven experiment on the synthetic data:

```bash
PYTHONPATH=src python -m sbc.experiment --scale decadal --quick --source synthetic
```

This writes per-gauge and summary skill tables, a JSON report, and SHAP
artifacts under `results/`. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for
the real-data pipeline and paper-figure regeneration.

---

## Repository structure

```
MDPI-Q1-2026/
├── src/sbc/
│   ├── config.py            # project paths (SBC_ROOT-aware) + column constants + EPS
│   ├── schemas.py           # modelling-table schema, feature discovery, log-residual target
│   ├── synthetic.py         # physically-grounded synthetic table with a known GloFAS bias
│   ├── experiment.py        # top-level, config-driven experiment runner (python -m sbc.experiment)
│   ├── multiscale.py        # cross-scale (decadal ↔ daily) reconciliation analyses
│   ├── reconciliation.py    # multi-scale reconciliation utilities
│   ├── ablation.py          # ablation harness
│   ├── utils/               # reproducible seeding, IO helpers, logging
│   ├── data/                # dataset access layer
│   │   ├── domain.py        # study-domain (core/transfer) basin & period definitions
│   │   ├── ca_discharge.py  # ground-truth + static attributes from CA-discharge.gpkg
│   │   ├── glofas.py        # GloFAS download + per-gauge river-pixel extraction
│   │   ├── era5_land.py     # ERA5-Land download + basin aggregation
│   │   ├── assemble.py      # fuse sources -> the tidy modelling table (per scale)
│   │   └── _cds.py          # CDS/EWDS credential handling + client factory
│   ├── features/            # feature engineering & hydrological-regime classification
│   ├── models/              # base interface/registry + the model zoo (see below)
│   ├── validation/          # leakage-safe CV splits, metric suite, calibration/conformal UQ
│   ├── explain/             # SHAP / group-causal-SHAP attribution tied to snow processes
│   └── viz/                 # shared figure styling
├── scripts/                 # CLI download / pipeline / analysis / figure entry points
│   └── figures/             # fig01..fig14 publication-figure generators
├── configs/                 # default.yaml + secrets.env (git-ignored, user-supplied)
├── datasets/                # raw + interim + processed data (large raw files git-ignored)
├── results/                 # figures/  tables/  models/  shap/
├── notebooks/               # exploratory & figure-production notebooks
├── tests/                   # pytest contract smoke tests (CPU-only, no credentials)
├── docs/                    # manuscript, analytical notes, claims-evidence matrix
├── requirements.txt
├── pyproject.toml
├── REPRODUCIBILITY.md
├── CITATION.cff
└── README.md
```

---

## Modelling approach

**Target.** Every model predicts the log-space multiplicative residual
`log_residual = log(q_obs + EPS) − log(q_glofas + EPS)` (`EPS = 1e-3` m³ s⁻¹);
corrected discharge is reconstructed with
`schemas.back_transform(q_glofas, residual)`. Working in log-residual space
stabilises variance across orders-of-magnitude flow and keeps all learners on a
common, additive target so they can be stacked. The modelling table is the
single source of truth — features are discovered at run time
(`schemas.feature_columns` / `static_feature_columns` / `dynamic_feature_columns`),
never hard-coded.

**Model zoo** (`src/sbc/models/`, each implementing the `BaseCorrector`
interface and self-registering via `models.base.register`):

- **Baselines** — linear scaling and quantile mapping
  (`quantile_mapping.LinearScalingCorrector`, `QuantileMappingCorrector`).
- **Boosting trio** — XGBoost, LightGBM and CatBoost gradient-boosting
  regressors on the engineered feature table (`boosting.py`).
- **EA-LSTM** — an entity-aware LSTM that ingests static catchment attributes
  through the input gate and dynamic forcing through the sequence, capturing
  snow-store memory and freshet timing (`ea_lstm.py`).
- **Regime-aware probabilistic flagship** (`RegimeProbNet`, `regime_prob_net.py`)
  — a mixture-of-experts whose gate is supervised by the hydrological regime
  (`lambda_gate`) and which carries a SWE/temperature monotonicity penalty
  (`lambda_phys`); it predicts a full predictive distribution and is trained with
  a CRPS loss, exposing `predict_quantiles` / `sample` for uncertainty bands.
- **Stacked ensemble** (`ensemble.StackedEnsemble`) — combines the base learners
  through their out-of-fold log-residual predictions with a non-negative
  least-squares (`nnls`) meta-learner by default.

The repository also includes SOTA / probabilistic baselines, a CMAL head, a
generative residual head, transfer-LSTM, snow-process and constraint variants
under `src/sbc/models/` for the ablation and positioning analyses.

**Tuning.** Hyper-parameters are searched with **Optuna**. Per-fold HPO across
the whole model × split matrix is prohibitive, so the matrix uses strong
defaults and Optuna tuning is reported once on the temporal split
(`scripts/tune_boosting.py`); set `models.boosting.n_optuna_trials > 0` in the
config to enable per-fold tuning.

---

## Validation matrix and metrics

Three complementary, leakage-safe evaluation protocols (`src/sbc/validation/`):

1. **Temporal holdout** (`splits.temporal_split`) — train on early years, test on
   held-out later years (no temporal leakage).
2. **Leave-one-basin-out (LOBO)** (`splits.spatial_folds`) — hold out an entire
   basin to test spatial transfer within the core domain.
3. **Prediction-in-ungauged-regions (PUR)** (`splits.pur_split`) — train on the
   core domain and test on the independent Amu Darya transfer domain.

The hydrological metric suite (`src/sbc/validation/metrics.py`, pure-NumPy,
NaN-aware) reports:

- **KGE′** (Kling 2012) with its `r` / `β` / `γ` decomposition and a skill score
  against the mean-flow benchmark (`kge_prime`, `kge_skill_score`);
- **NSE**, **logNSE**, **PBIAS**, **RMSE**;
- **Flow-duration-curve signatures** FHV / FMS / FLV (Yilmaz et al., 2008);
- **Annual peak-timing error** (days) — central for snowmelt-freshet skill;
- **CRPS** (ensemble and closed-form Gaussian) for the probabilistic models.

Uncertainty quantification is assessed with reliability / PIT calibration and
(regime-)conformal coverage (`validation/calibration.py`, `conformal.py`,
`regime_conformal.py`, `uq_scores.py`), and decision relevance with the
decision-skill module (`validation/decision_skill.py`).

---

## Explainability

Model behaviour is explained with **SHAP**, with attributions linked back to
snow, glacier, temperature and precipitation drivers (`src/sbc/explain/`):
tree-SHAP global and regime-conditional importance (`shap_analysis.py`),
flagship-specific XAI (`flagship_xai.py`), SHAP stability
(`shap_stability.py`), grouped causal SHAP (`group_causal_shap.py`), and a PUR
transfer-attribution analysis (`pur_attribution.py`). This is what makes each
correction traceable to a physically meaningful driver rather than an opaque
adjustment.

---

## Configuration

Every run is fully driven by [`configs/default.yaml`](configs/default.yaml):
data source (`synthetic` | `real`) and scales, feature windows and
rain-on-snow thresholds, the model roster and flagship hyper-parameters
(`n_experts`, `lambda_gate`, `lambda_phys`, `crps_loss`), the validation matrix
(`temporal_test_frac`, `leave_one_basin_out`, PUR domains, OOF folds), the SHAP
settings, and the output directories. The global `seed` (default `1234`) is
applied through `sbc.utils.seed_everything` for reproducibility. Override the
config path with `--config`, the data source with `--source`, and the model
subset with `--models`.

Filesystem paths are centralised in `sbc.config.PATHS` and derived from the
repository root (or `SBC_ROOT`), so no absolute paths are hard-coded in the
package.

---

## Reproducing the paper

See **[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)** for the full, step-by-step
recipe:

1. install the environment;
2. (synthetic) run the one-command smoke experiment — no credentials needed;
3. (real) download GloFAS-ERA5, ERA5-Land and CA-discharge, assemble the
   modelling tables, and run the validation-matrix experiment
   (`scripts/run_real_pipeline.py`);
4. regenerate the manuscript figures and tables (`scripts/make_figures.py` and
   the `scripts/figures/fig01..fig14_*.py` generators).

---

## Testing

```bash
pip install -e ".[dev]"
pytest -q                    # CPU-only contract smoke tests, no credentials
```

The smoke tests (`tests/test_smoke.py`) check the core contracts: the
log-residual target round-trips through `back_transform`, the metric suite is
sane, the synthetic schema and feature discovery hold, the CV splits are
leakage-safe (temporal / spatial / PUR), and a baseline corrector does not
materially hurt pooled skill.

---

## Citation & acknowledgement

If you use this code, please cite the accompanying MDPI *Water* paper (see
[`CITATION.cff`](CITATION.cff); details finalised on publication) and the
underlying datasets — GloFAS-ERA5, ERA5-Land, CA-discharge and HydroATLAS —
listed under [Data sources](#data-sources).

This work was supported by grant **BR24993128**. The Chu and Talas basins, which
feed the Zhambyl agricultural region, are a focus of this grant.

---

## License

Released under the **MIT License**, declared in the `license` field of
[`pyproject.toml`](pyproject.toml) (add a top-level `LICENSE` file with the full
MIT text before the public release). The third-party datasets retain their own
licences; review and accept them on the respective portals before use.
