# Ginger soft-rot reservoir model

This repository contains the Python code and machine-readable numerical outputs accompanying the manuscript:

> Francis Chinedu Oba and Paul A. Ogbiyele, “Threshold and Stability Analysis of a Two-Route Reservoir Model Motivated by Ginger Soft Rot.”

The model is an abstract open-population, two-route reservoir model motivated by *Pythium*-associated ginger soft rot. The numerical parameter values are illustrative scenario values rather than field-calibrated estimates.

## Contents

- `reproduce_figures.py` — reproduces the numerical simulations, sensitivity calculations, bifurcation results, figures, and CSV summaries.
- `requirements.txt` — Python dependencies.
- `figures/` — generated publication-resolution figures and machine-readable CSV outputs.

## Reproduction

Python 3.11 or newer is recommended. From the repository directory, run:

```bash
python -m pip install -r requirements.txt
python reproduce_figures.py
```

The script uses SciPy's Radau solver with `rtol=1e-9` and `atol=1e-11`. Generated files are written to `figures/`.

## Main outputs

- `stability_scenarios.png`
- `numerical_diagnostics.png`
- `stability_equilibria.csv`
- `sensitivity_indices.csv`
- `same_gamma_scenarios.csv`
- `bifurcation_branch.csv`
- `control_sweeps.csv`
- `illustrative_baseline_parameters.csv`

## Authors

- Francis Chinedu Oba
- Paul A. Ogbiyele

## Reproducibility note

The values used in the simulations are intended to demonstrate the analytical threshold and stability results. They should not be interpreted as estimated ginger-disease parameters or as field-management recommendations.
