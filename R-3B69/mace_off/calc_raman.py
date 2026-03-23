#!/usr/bin/env python3
import os ,re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase import units
from ase.optimize import BFGS
from mace.calculators.mace import MACECalculator
from mace.calculators import mace_off

# ============================================================
# CONFIG
# ============================================================
xyz_file = "../R-3B69.xyz"      # single multi-structure XYZ file
model_base_path = "MACE-MDP.model"
model_sizes = ["small", "medium", "large"]

polar_calc = MACECalculator(model_paths=model_base_path,
                            model_type="DipolePolarizabilityMACE",
                            device="cuda", default_dtype="float64")

# Function to parse name
def parse_name_from_comment(comment):
    match = re.search(r'name="([^"]+)"', comment)
    return match.group(1) if match else "unknown"

# Lorentzian broadening
def broaden_spectrum_lorentzian(freqs, intens, fwhm=10.0, x_min=0, x_max=4000, step=1.0):
    x = np.arange(x_min, x_max + step, step, dtype=np.float64)
    gamma = fwhm / 2.0
    spec = np.zeros_like(x)
    for f0, I in zip(freqs, intens):
        spec += I * (gamma / ((x - f0)**2 + gamma**2))
    return x, spec

# Custom BFGS
class BFGSWithCount(BFGS):
    def run(self, fmax=None, steps=None):
        self.step_count = 0
        for step in range(steps if steps is not None else 1000000):
            converged = super().run(fmax=fmax, steps=1)
            self.step_count += 1
            if converged:
                break
        return converged

# Read molecules
all_atoms = read(xyz_file, index=":")  # list of Atoms objects
all_comments = []
with open(xyz_file, "r") as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        natoms = int(lines[i])
        comment = lines[i+1].strip()
        all_comments.append(comment)
        i += natoms + 2

# Run for each model size
for size in model_sizes:
    print(f"Processing model size: {size}")
    output_dir = f"raman_results_{size}_SPICE"
    os.makedirs(output_dir, exist_ok=True)
    
    off_calc = mace_off(model=size, default_dtype="float64", device="cuda")
    success_records = []

    for i, (atoms, comment) in enumerate(zip(all_atoms, all_comments)):
        try:
            name = parse_name_from_comment(comment).split("/")[-1]
            print(f"Processing {name} ({i+1}/{len(all_atoms)})")
            n_atoms = len(atoms)
            masses = atoms.get_masses()

            # Assign MACE OFF calculator
            atoms.calc = off_calc

            # Geometry optimization
            dyn = BFGSWithCount(atoms)
            dyn.run(fmax=0.002, steps=5000)

            # Hessian and frequencies
            H = off_calc.get_hessian(atoms).reshape(3*n_atoms, 3*n_atoms)
            H = 0.5 * (H + H.T)
            mw = np.repeat(masses**-0.5, 3)
            Hmw = (H * mw).T * mw
            omega2, vecs = np.linalg.eigh(Hmw)
            conv = units._hbar * units.m / np.sqrt(units._e * units._amu)
            energies = conv * np.sqrt(np.abs(omega2))
            freq = np.real(energies) / units.invcm
            modes = vecs.T.reshape(3*n_atoms, n_atoms, 3)
            modes *= masses[np.newaxis, :, np.newaxis]**-0.5

            # Raman intensities
            tot_int, iso_int, ani_int = [], [], []
            pos0 = atoms.get_positions().copy()
            beta = 1.0 / (units.kB * 298.0)
            nu0 = 1e7 / 632.8

            for i_mode, mode in enumerate(modes):
                dpos = mode * 1e-3
                atoms.set_positions(pos0 + dpos)
                alpha_p = polar_calc.get_property('polarizability', atoms)
                atoms.set_positions(pos0 - dpos)
                alpha_m = polar_calc.get_property('polarizability', atoms)
                atoms.set_positions(pos0)
                deriv = (alpha_p - alpha_m) / (2e-3)
                a_iso = np.trace(deriv)/3.0
                a_ani_sq = 0.5*((deriv[0,0]-deriv[1,1])**2+(deriv[1,1]-deriv[2,2])**2+(deriv[2,2]-deriv[0,0])**2
                                +6*(deriv[0,1]**2 + deriv[1,2]**2 + deriv[2,0]**2))
                bose = 1
                if freq[i_mode] > 5:
                    disp = 1
                    I_iso = 45.0*a_iso**2*bose*disp
                    I_ani = 7.0*a_ani_sq*bose*disp
                    I_tot = I_iso + I_ani
                else:
                    I_iso = I_ani = I_tot = 0.0
                iso_int.append(I_iso)
                ani_int.append(I_ani)
                tot_int.append(I_tot)

            tot_int = np.array(tot_int)
            iso_int = np.array(iso_int)
            ani_int = np.array(ani_int)

            # Broaden spectra
            x_sim, spec_tot = broaden_spectrum_lorentzian(freq, tot_int)
            _, spec_iso = broaden_spectrum_lorentzian(freq, iso_int)
            _, spec_ani = broaden_spectrum_lorentzian(freq, ani_int)

            # Normalize
            spec_tot /= spec_tot.max()
            spec_iso /= spec_tot.max()
            spec_ani /= spec_tot.max()

            # Save CSV
            csv_path = os.path.join(output_dir, f"{name}_raman_norm.csv")
            pd.DataFrame({
                'cm1': x_sim,
                'total': spec_tot,
                'isotropic': spec_iso,
                'anisotropic': spec_ani
            }).to_csv(csv_path, index=False)

            raw_csv_path = os.path.join(output_dir, f"{name}_raman_raw.csv")
            pd.DataFrame({
                'frequency_cm1': freq,
                'total_intensity': tot_int,
                'isotropic_intensity': iso_int,
                'anisotropic_intensity': ani_int
            }).to_csv(raw_csv_path, index=False)

            # Plot
            plt.figure(figsize=(8,6))
            plt.plot(x_sim, spec_tot, label='Total')
            plt.xlim(0, 4000)
            plt.xlabel('Raman shift (cm⁻¹)')
            plt.ylabel('Intensity (arb. units)')
            plt.title(f"{name} Raman spectrum ({size})")
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"{name}_raman.png")
            plt.savefig(plot_path, dpi=400)
            plt.close()

            success_records.append({'name': name, 'steps': dyn.step_count})

        except Exception as e:
            with open(os.path.join(output_dir,'failed_molecules.txt'),'a') as f:
                f.write(f"Molecule: {name}\n{type(e).__name__}: {e}\n")
            print(f"Error in {name}, recorded.")

    # Save summary per model size
    pd.DataFrame(success_records).to_csv(os.path.join(output_dir,f'successful_molecules_{size}.csv'), index=False)
    print(f"Finished processing model size: {size}")
