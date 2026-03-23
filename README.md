# Benchmark-MACE-MDP

This repository contains the core benchmark data and plotting/comparison scripts used to reproduce the figures and summary data shown in:

https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000716/v1

## Repository layout

The repository currently contains two main benchmark folders:

- `R-3B69/`
  - Small curated benchmark set of trimers used for Raman and IR comparisons.
  - Includes the structure collection `R-3B69.xyz`.

- `IR-R-7193/`
  - Larger benchmark set of isolated molecules used for IR/Raman benchmarking.
  - Includes the structure collection `datbase_IR-R-7193_wB97MD3.xyz`.

## What is included

This repository primarily stores:

- Reference structure sets in XYZ format.
- Precomputed raw Raman and IR spectra in CSV format.
- Match-score tables such as `spectra_matchscores.csv`.
- Summary property tables such as `properties_vs_xyzref_*.csv`, `summary_rmse_vs_xyzref.csv`, and related benchmark outputs.
- Python scripts used to compare spectra, compute summary statistics, and generate paper-style figures.
