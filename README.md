# Study-aware learning in soil microplastic compilations

Core numerical code for **Study-aware learning reveals environmental contrasts and calibration needs in soil microplastic compilations**.

The analyses distinguish environmental comparisons within source studies from protocol associations between studies, and evaluate how local measurements calibrate predictions for a held-out study. The analysis set contains 5,853 records from 281 publications.

## Input and environment

Download version 1 of the [Wu et al. soil microplastic compilation](https://doi.org/10.5281/zenodo.20806151), licensed under CC BY 4.0. Place its original workbook at:

```text
data/raw/A global dataset of soil microplastics (2005-2026).xlsx
```

[data/README.md](data/README.md) gives the source citation and checksums. The runner checks the input MD5 before data preparation. Use Python 3.12 and the pinned package versions:

```bash
python -m pip install -r requirements.txt
python run_all.py --list
python run_all.py
```

An optional Conda environment is provided in `environment.yml`. A complete numerical run takes several hours; grouped benchmarks, repeated semi-synthetic outcomes and bootstrap analyses account for most of the computation. Worker counts are specified in the individual scripts.

## Analyses

| Step | Numerical task |
|---|---|
| 01 | Parse the workbook and construct the modelling table and descriptor roles |
| 02 | Twenty regressors and a super learner under record, site and study validation; random study-code diagnostic |
| 03, 03b | Within-study fixed effects, record-level residualisation and permutation-unit comparison |
| 07 | Study-level R-learner and augmented inverse probability weighting estimates |
| 28 | Independent sensitivity to explicit flotation-density range endpoints |
| 10 | Omitted-variable sensitivity and observed-covariate benchmarks |
| 04 | Semi-synthetic outcomes under recorded and unrecorded study confounding |
| 05 | Study-held-out transfer, local calibration and reference-protocol harmonisation |
| 08 | Wild cluster bootstrap, study deletion, nonlinear depth response and non-detect sensitivity |
| 09 | Study-bootstrap summaries of joint protocol coefficients and harmonised quantities |
| 11 | Attribution changes after adding random study codes |
| 12 | Empirical prediction-interval coverage by region |
| 27 | Replay calibration draws and score both predictors on identical non-calibration records |
| 15 | False-discovery and family-wise adjustments for the two specified test families |

The runner follows this dependency order, regardless of argument order. In particular, step 27 follows step 12, which produces the required zero-calibration record predictions.

```bash
python run_all.py --only 01       # data preparation
python run_all.py --from 05       # resume at transfer modelling
```

`scripts/study_level.py` contains the shared study-level estimators. Numerical outputs are written to `results/`, with data-preparation outputs in `data/processed/`. Steps 27 and 28 also write compact audit summaries to `manuscript/checks/`. Random seeds are specified in the scripts; use the pinned environment when comparing numerical outputs across machines.

## Statistical interpretation

Protocol coefficients describe adjusted between-study associations. The local calibration rule updates a study-constant abundance offset and preserves the fitted within-study ordering. Zero-calibration intervals use empirical residual calibration; intervals after local calibration use a Gaussian approximation. Explicit density-range endpoint sensitivity is separate from the original nine-test multiplicity family.

## Licence and citation

Code is distributed under the MIT licence in `LICENSE`. Cite the source dataset when reusing its measurements. This repository publishes the core analysis implementation; measurement data remain available from their original repository.
