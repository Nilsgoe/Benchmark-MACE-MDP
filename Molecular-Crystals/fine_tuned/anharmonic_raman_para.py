import os
import numpy as np
import pandas as pd
from ase.io import read, Trajectory
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory as TrajWriter
from mace.calculators.mace import MACECalculator

# —————————————————————————————————————————————
# CONFIGURATION
# —————————————————————————————————————————————
traj_steps      = 200000       # total desired production steps
timestep_fs     = 0.5          # MD step (fs)
temperature_K   = 298
equil_steps     = 10000        # short equilibration with Langevin
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

files = [
    f for f in os.listdir("cifs")
    if os.path.isfile(os.path.join("cifs", f)) and f.lower().endswith(".cif")
]
molecules = [{"name": os.path.splitext(f)[0], "cif": os.path.join("cifs", f)} for f in sorted(files)]
mae_calc=MACECalculator(model_paths="../PARACETAMOL/MACE_model_swa.model",device="cuda", default_dtype="float64")

dt_s = timestep_fs * units.fs  # timestep in ASE units

# —————————————————————————————————————————————
# MAIN LOOP
# —————————————————————————————————————————————
success_records = []

for mol in molecules:
    try:
        name = mol["name"]
        print(f"[MACE_ft-MD] Processing {name}...")

        traj_file = os.path.join(output_dir, f"{name}_nve.traj")

        # check if trajectory already exists
        n_done = 0
        atoms = None
        if os.path.exists(traj_file):
            with Trajectory(traj_file) as traj:
                n_done = len(traj)
                if n_done > 0:
                    atoms = traj[-1]  # continue from last snapshot
                    atoms.calc = mae_calc  # <-- reattach the calculator
            print(f"  → Found existing trajectory with {n_done} frames.")

        if n_done >= traj_steps:
            print(f"  ✓ {name} already finished ({n_done} / {traj_steps} frames). Skipping.")
            success_records.append({"name": name, "steps": n_done})
            continue

        # if no trajectory exists → start fresh
        if atoms is None:
            atoms = read(mol["cif"])
            atoms.calc = mae_calc

            # Geometry optimization
            print("  → Optimizing geometry...")
            dyn = BFGS(atoms, logfile=None)
            dyn.run(fmax=0.05, steps=5000)

            # Assign initial velocities
            MaxwellBoltzmannDistribution(atoms, temperature_K * units.kB)

            # Short Langevin equilibration
            print("  → Equilibrating...")
            dyn_lange = Langevin(
                atoms, dt_s, temperature_K * units.kB, friction=0.1,
                trajectory=os.path.join(output_dir, f"{name}_nvt.traj")
            )
            dyn_lange.run(equil_steps)

        # ensure calculator is attached before MD
        if atoms.calc is None:
            atoms.calc = mae_calc

        # open trajectory in append or write mode
        mode = "a" if os.path.exists(traj_file) else "w"
        traj_writer = TrajWriter(traj_file, mode=mode, atoms=atoms)

        # Switch to NVE Verlet dynamics
        remaining_steps = traj_steps - n_done
        print(f"  → Running NVE production ({remaining_steps} more steps)...")
        
        dyn_nve = VelocityVerlet(atoms, dt_s)
        #dyn_nve = Langevin(
        #        atoms, dt_s, temperature_K * units.kB, friction=0.001/units.fs,
        #        trajectory=os.path.join(output_dir, f"{name}_nvt_eq.traj")
        #    )
        
        # attach trajectory writing every step
        dyn_nve.attach(traj_writer.write, interval=1)

        # progress callback
        steps_per_ps = int(1.0 / timestep_fs * 1000)  # e.g., 2000 for 0.5 fs
        total_ps = traj_steps * timestep_fs / 1000.0

        def print_status():
            ps_done = (n_done + dyn_nve.nsteps) * timestep_fs / 1000.0
            print(f"    → NVE progress: {ps_done:.1f} / {total_ps:.1f} ps")

        dyn_nve.attach(print_status, interval=steps_per_ps)

        # run only the missing steps
        dyn_nve.run(remaining_steps)

        success_records.append({"name": name, "steps": traj_steps})
        print(f"  ✓ Finished {name}, trajectory saved.")

    except Exception as e:
        print(f"  !! Error with {name}: {e}")
        with open("failed_md.txt", "a") as f:
            f.write(f"{name}: {e}\n")

# —————————————————————————————————————————————
# SAVE SUMMARY
# —————————————————————————————————————————————
pd.DataFrame(success_records).to_csv("successful_md_runs.csv", index=False)
print("All done. Summary written to successful_md_runs.csv")
