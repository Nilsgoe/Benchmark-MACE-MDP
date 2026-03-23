# Benchmark-MACE-MDP

This repository contains the core benchmark data and analysis scripts used to reproduce figures and summary data for MACE-MDP: A General Dipole and Polarizability Model for Organic Molecules and Materials.

The repository currently combines three main data collections:

- `R-3B69/` for a small trimer benchmark set used for Raman and IR comparisons.
- `IR-R-7193/` for a larger isolated-molecule benchmark set used for IR/Raman benchmarking.
- `Molecular-Crystals/` for anhamronic molecular-crystal spectra, experimental reference data, and fine-tuned MACE potentials.

## Related publications

The benchmark subsets in this repository are associated with the following publications:

- ChemRxiv MACE-MDP preprint:
    N. Gönnheimer, K. Reuter, V. Kapil, and J. T. Margraf
  - https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000716/v1
- Molecular-crystal spectra reference:
  - N. Raimbault, V. Athavale, and M. Rossi, Phys. Rev. Materials 3, 053605 (2019).
  - https://doi.org/10.1103/PhysRevMaterials.3.053605
- Fine-tuned MACE molecular-crystal potentials at the vdW-DF2 level of theory:
  - F. Della Pia, B. X. Shi, V. Kapil, A. Zen, D. Alfe, and A. Michaelides, Chem. Sci. 16, 11419-11433 (2025).
  - https://doi.org/10.1039/D5SC01325A
- The MACE-MDP model can be found under: https://github.com/Nilsgoe/MACE-MDP

## Overview

### `R-3B69/`

Small curated benchmark set of trimers used for Raman and IR comparisons.

Includes:

- `R-3B69.xyz` structure collection.
- Processed Raman and IR spectra in CSV form.
- Match-score tables such as `spectra_matchscores.csv`.
- Summary tables such as `properties_vs_xyzref_*.csv` and `summary_rmse_vs_xyzref.csv`.

### `IR-R-7193/`

Larger benchmark set of isolated molecules used for IR and Raman benchmarking.

Includes:

- `datbase_IR-R-7193_wB97MD3.xyz` structure collection.
- Processed outputs for multiple models, including `mace_off/`, `mace_omol/`, and `mace_mu/`.
- Precomputed raw spectra and match-score tables.

### `Molecular-Crystals/`

Data and scripts for anahrmonic molecular-crystal Raman comparisons,of aspirin and paracetamol polymorphs.

Includes:

- `exp_spectra/` with experimental reference spectra used for comparison.
- `fine_tuned/` with scripts for harmonic and anharmonic Raman analysis and paper plotting, including `paper_plot_anharm.py`, `get_anharm_comp_raman.py`, and related workflows.

The experimental and DFT spectra in this part of the repository should be cited with N. Raimbault et al. above. The fine-tuned MACE potentials at the vdW-DF2 level of theory are from F. Della Pia et al. 

## What is included

This repository primarily stores:

- Reference structure sets in XYZ format.
- Experimental spectra for molecular crystals.
- Precomputed raw Raman and IR spectra in CSV format.
- Match-score and comparison tables.
- Python scripts used to compare spectra, compute summary statistics, and generate paper figures.

