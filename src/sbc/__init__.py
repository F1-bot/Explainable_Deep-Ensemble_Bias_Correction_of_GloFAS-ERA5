"""sbc — Snow-influenced streamflow Bias Correction.

An explainable deep-ensemble framework for multi-scale (decadal & daily)
bias correction of GloFAS-ERA5 river-discharge reanalysis across the
snow-influenced transboundary basins of Central Asia.

Package layout
--------------
sbc.config      project paths & global study configuration
sbc.data        dataset loaders (CA-discharge, GloFAS, ERA5-Land) + assembly
sbc.features    feature engineering & hydrological-regime classification
sbc.models      baselines, boosting trio, flagship RegimeProbNet, stacking
sbc.validation  leakage-safe CV splits + the hydrological metric suite
sbc.explain     SHAP / attribution analysis tied to snow processes
"""

__version__ = "0.1.0"
__all__ = ["config"]
