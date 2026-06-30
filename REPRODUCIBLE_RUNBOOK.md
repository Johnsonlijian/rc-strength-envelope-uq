# Reproducible Runbook

## Environment

The analysis was run with Python 3.12. Key packages in the local verification environment were:

- numpy 2.3.5
- pandas 2.3.3
- scipy 1.16.3
- scikit-learn 1.9.0
- matplotlib 3.10.9
- torch 2.12.0+cpu

Finite-element scripts use `scikit-fem`; install it from `requirements.txt` when rerunning FE analyses.

## Data Acquisition

1. Read `DATASETS_AND_LINKS.csv`.
2. Download each third-party raw dataset from its original source.
3. Place raw beam files under `data/raw/` and raw column files under `data/raw_columns/` using the filenames expected by the scripts.
4. Do not commit those raw files unless their source license explicitly permits redistribution.

This repository includes derived aggregate result tables under `data/processed/`. Cleaned row-level third-party datasets are excluded because they may reproduce substantial parts of the original third-party records.

## Suggested Run Order

From the repository root:

```bash
pip install -r requirements.txt
cd code
```

If raw data have been restored locally, run the data-processing and analysis scripts in approximate dependency order:

```bash
python a01_frp_load_eda.py
python a02_frp_model_uncertainty.py
python a03_frp_extrapolation.py
python a04_extrap_robustness.py
python a05_deepbeam_steel_extrap.py
python a07_corrected_uq_matrix.py
python a08_mechanical_fallback.py
python a09_sfrc_extrap.py
python a15_steel_validate.py
python a16_steel_extrapolation.py
python a17_gpu_uq.py
python a21_steel_reliability.py
python a22_steel_gate.py
python a23_column_pm_extrap.py
python a24_column_load.py
python a25_column_extrap.py
python a26_column_mech_validate.py
```

Finite-element outputs can be regenerated with:

```bash
cd fe
python size_effect_run.py
python parametric_fe.py
python fe_gf_robustness.py
cd ..
```

Final figures and SI tables:

```bash
python fig_build.py
python a18_mechanism_v2.py
python a21_fig.py
python a27_column_fig.py
python generate_SI_tex.py
```

## Expected Public Outputs

- `figures/fig1_envelope_cliff.*`
- `figures/fig2_uq_matrix.*`
- `figures/fig3_error_vs_size.*`
- `figures/fig4_remedy.*`
- `figures/fig6_mechanism_convergence.*`
- `figures/fig7_reliability_across_size.*`
- `figures/fig8_columns.*`
- aggregate CSV/JSON outputs under `data/processed/`

## Known Boundaries

- Full raw-data regeneration requires third-party source access.
- GPU deep-UQ scripts can run on CPU but will be slower.
- Numerical results may differ slightly across package versions and hardware.
- This package is a reproducibility release, not a redistribution mirror for raw experimental databases.
