import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from ase.io import read
from ase import units
from ase.optimize import BFGS
from mace.calculators.mace import MACECalculator
from mace.calculators import mace_off, mace_omol
import traceback



# ————————————————————————————————————————————
# CONFIGURATION
# ————————————————————————————————————————————
model_base_path_spice = "./MACE_SPICE_full_mu_alpha_early_seems_great.model"

# OFF models
off_models = {
    "off_small": "small",
    "off_medium": "medium",
    "off_large": "large"
}

# OMOL model
omol_models = {
    "omol_extra_large": mace_omol(model="extra_large", default_dtype="float64", device="cuda")
}

# Fine-tuned models
finetuned_paracetamol = MACECalculator(
    model_paths="../PARACETAMOL/MACE_model_swa.model",
    device="cuda",
    default_dtype="float64"
)
finetuned_aspirin = MACECalculator(
    model_paths="../ASPIRIN/MACE_model_swa.model",
    device="cuda",
    default_dtype="float64"
)

# SPICE polarizability model
polar_calc = MACECalculator(
    model_paths=model_base_path_spice,
    model_type="DipolePolarizabilityMACE",
    device="cuda",
    default_dtype="float64"
)

# Base calculators (always used)
base_calculators = {**omol_models, **{
    key: mace_off(model=size, default_dtype="float64", device="cuda")
    for key, size in off_models.items()
}}

# ————————————————————————————————————————————
# OUTPUT DIRECTORY
# ————————————————————————————————————————————
output_dir = "spectra_data_harm"
os.makedirs(output_dir, exist_ok=True)

# ————————————————————————————————————————————
# INPUT MOLECULES
# ————————————————————————————————————————————
files = [
    f for f in os.listdir("cifs")
    if os.path.isfile(os.path.join("cifs", f)) and (f.lower().endswith(".cif") or f.lower().endswith(".in"))
]
molecules = [{"name": os.path.splitext(f)[0], "xyz": os.path.join("cifs", f)} for f in sorted(files)]

# ————————————————————————————————————————————
# FUNCTIONS
# ————————————————————————————————————————————
def broaden_lorentz(freqs, intens, fwhm=10.0, x_min=0, x_max=4000, step=1.0):
    x = np.arange(x_min, x_max + step, step, dtype=np.float64)
    gamma = fwhm / 2.0
    y = np.zeros_like(x)
    for f, I in zip(freqs, intens):
        y += I * (gamma / ((x - f)**2 + gamma**2))
    return x, y

conv = units._hbar * units.m / np.sqrt(units._e * units._amu)
beta = 1.0 / (units.kB * 300)
nu0 = 1e7 / 632.8

class BFGSCount(BFGS):
    def run(self, fmax=None, steps=None):
        self.step_count = 0
        for s in range(steps if steps else 1000000):
            converged = super().run(fmax=fmax, steps=1)
            self.step_count += 1
            if converged:
                break
        return converged

# ————————————————————————————————————————————
# MAIN EXECUTION LOOP
# ————————————————————————————————————————————
success_records = []

for mol in molecules:
    name = mol["name"]
    print(f"\n=== Processing {name} ===")

    name_lower = name.lower()

    # Choose calculators depending on molecule
    calculators = dict(base_calculators)

    if ("para" in name_lower) or ("acetaminophen" in name_lower):
        calculators["finetuned_paracetamol"] = finetuned_paracetamol

    if "aspirin" in name_lower:
        calculators["finetuned_aspirin"] = finetuned_aspirin

    try:
        atoms0 = read(mol["xyz"])
        n_atoms = len(atoms0)
        masses = atoms0.get_masses()

        stacked_spectra = {}

        for model_key, calc in calculators.items():
            print(f" → Model: {model_key}")
            atoms = atoms0.copy()
            atoms.calc = calc

            # Geometry optimization
            dyn = BFGSCount(atoms)
            dyn.run(fmax=0.002, steps=5000)

            # Hessian
            H = calc.get_hessian(atoms).reshape(3*n_atoms, 3*n_atoms)
            H = 0.5 * (H + H.T)

            mw = np.repeat(masses**-0.5, 3)
            Hmw = (H * mw).T * mw

            omega2, vecs = np.linalg.eigh(Hmw)
            energies = conv * np.sqrt(np.abs(omega2))
            freqs = np.real(energies) / units.invcm

            modes = vecs.T.reshape(3*n_atoms, n_atoms, 3)
            modes *= masses[np.newaxis, :, np.newaxis]**-0.5

            intensities = []
            pos0 = atoms.get_positions().copy()

            for i, mode in enumerate(modes):
                dpos = mode * 1e-3

                atoms.set_positions(pos0 + dpos)
                a_p = polar_calc.get_property("polarizability", atoms)

                atoms.set_positions(pos0 - dpos)
                a_m = polar_calc.get_property("polarizability", atoms)

                atoms.set_positions(pos0)

                deriv = (a_p - a_m) / (2e-3)

                hbar_eVs = 6.582119569e-16        # eV·s
                omega = energies[i] / hbar_eVs    # rad/s   (because energies[i] = ħω)

                # ---- Placzek invariants ----
                axx, ayy, azz = deriv[0,0], deriv[1,1], deriv[2,2]
                axy, axz, ayz = deriv[0,1], deriv[0,2], deriv[1,2]

                G0 = (1/3) * (axx + ayy + azz)**2

                G2 = 0.5*((2*axy)**2 + (2*axz)**2 + (2*ayz)**2)
                G2 += (1/3)*((axx - ayy)**2 +(axx - azz)**2 + (ayy - azz)**2)


                pref_bose = 1.0 / (omega*(1 - np.exp(-beta * energies[i])))

                I_tot = pref_bose * (1.0/30.0) * (10*G0 + 7*G2)
                intensities.append(I_tot)
                #print(freqs[i],omega, I_tot, (1.0 / (omega * bose)))
            
            #exit()
            intensities = np.array(intensities)

            # Raw output
            pd.DataFrame({
                "freq_cm1": freqs,
                "intensity_raw": intensities,
            }).to_csv(os.path.join(output_dir, f"{name}_raw_{model_key}.csv"), index=False)

            # Broadened
            x, y = broaden_lorentz(freqs, intensities, fwhm=10.0)
            y /= y.max()

            pd.DataFrame({
                "cm1": x,
                f"intensity_{model_key}": y,
            }).to_csv(os.path.join(output_dir, f"{name}_broadened_{model_key}.csv"), index=False)

            stacked_spectra[model_key] = (x, y)

            # Individual plots
            plt.figure(figsize=(8,5))

            if "finetuned" in model_key:
                plt.plot(x, y, "-", label=f"{model_key} (fine-tuned)",
                         linewidth=2.8, color="black")
            else:
                plt.plot(x, y, "-", label=model_key)

            plt.xlim(0, 350)
            plt.xlabel("Raman shift (cm$^{-1}$)")
            plt.ylabel("Intensity")
            plt.title(f"{name} — {model_key}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_raman_{model_key}.png"), dpi=300)
            plt.close()

        # Combined stacked plot
        plt.figure(figsize=(8, 10))
        y_offset = 1.2

        for i, (model_key, (x, y)) in enumerate(stacked_spectra.items()):
            if "finetuned" in model_key:
                plt.plot(x, y + i*y_offset, "-",
                         label=f"{model_key} (fine-tuned)",
                         linewidth=2.8, color="black")
            else:
                plt.plot(x, y + i*y_offset, "-", label=model_key)

        plt.xlim(0, 350)
        plt.xlabel("Raman shift (cm$^{-1}$)")
        plt.ylabel("Intensity (stacked)")
        plt.title(f"{name} — Raman spectra (all models)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_raman_all_models.png"), dpi=300)
        plt.close()

        success_records.append({"name": name})

    except Exception as e:
        print(f"ERROR in {name}: {e}")
        with open(os.path.join(output_dir, "failed_molecules.txt"), "a") as f:
            f.write(f"{name}\n{e}\n{traceback.format_exc()}\n\n")

pd.DataFrame(success_records).to_csv(os.path.join(output_dir, "successful_molecules_all_models.csv"), index=False)
print("\nDONE.")
