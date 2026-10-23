from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import mace_off
import re
import os

xyz_file = "../datbase_IR-R-7193_wB97MD3.xyz"
output_xyz = "optimized_mace_off_large.xyz"
failed_log = "failed_optimize_large.txt"

calc = mace_off(model="large", default_dtype="float64", device="cuda")


def parse_name(comment: str) -> str:
    match = re.search(r'name="([^"]+)"', comment)
    return match.group(1).split("/")[-1] if match else "unknown"


all_atoms = read(xyz_file, index=":")

all_comments = []
with open(xyz_file) as f:
    lines = f.readlines()
i = 0
while i < len(lines):
    natoms = int(lines[i])
    all_comments.append(lines[i + 1].strip())
    i += natoms + 2

if os.path.exists(output_xyz):
    os.remove(output_xyz)

for idx, (atoms, comment) in enumerate(zip(all_atoms, all_comments)):
    name = parse_name(comment)
    print(f"[{idx + 1}/{len(all_atoms)}] Optimizing {name}")
    try:
        atoms.calc = calc
        dyn = BFGS(atoms, logfile=None)
        dyn.run(fmax=0.002, steps=5000)
        atoms.info["name"] = name
        write(output_xyz, atoms, format="extxyz", append=True)
    except Exception as e:
        print(f"  Failed: {e}")
        with open(failed_log, "a") as f:
            f.write(f"{name}: {type(e).__name__}: {e}\n")

print(f"Done. Optimized structures written to {output_xyz}")
