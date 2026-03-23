#!/usr/bin/env python3
import argparse
import numpy as np
from ase.io import Trajectory
from mace.calculators.mace import MACECalculator
import os
import glob

def process_traj(traj_path, csv_path, polar_calc, dt, continue_run, step):
    """Process a single trajectory file and save polarizabilities to CSV."""

    # Load only every nth frame
    traj = Trajectory(traj_path, "r")[::step]
    n_frames = len(traj)
    print(f"Found {n_frames} frames in {traj_path} (using every {step}th frame)")

    start_frame = 0
    if continue_run and os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            lines = f.readlines()
        start_frame = len(lines) - 1  # skip header
        print(f"Continuing from frame {start_frame}")
        fout = open(csv_path, "a")
    else:
        fout = open(csv_path, "w")
        header = [
            "image", "time_fs",
            "a00","a01","a02",
            "a10","a11","a12",
            "a20","a21","a22",
            "sh_xx","sh_yy","sh_zz",
            "sh_xy","sh_xz","sh_yz",
        ]
        fout.write(",".join(header) + "\n")

    for i in range(start_frame, n_frames):
        atoms = traj[i]
        atoms.calc = polar_calc
        flat = atoms.calc.get_property("polarizability", atoms)
        alpha = np.asarray(flat).reshape(3, 3)
        alpha_sh = atoms.calc.get_property("polarizability_sh", atoms)
        sh_xx, sh_yy, sh_zz, sh_xy, sh_xz, sh_yz = alpha_sh

        a00,a01,a02 = alpha[0,0], alpha[0,1], alpha[0,2]
        a10,a11,a12 = alpha[1,0], alpha[1,1], alpha[1,2]
        a20,a21,a22 = alpha[2,0], alpha[2,1], alpha[2,2]
        t = i * dt * step  # adjust time for skipped frames

        vals = [
            str(i * step),  # real frame index in original trajectory
            f"{t:.6f}",
            f"{a00:.8e}", f"{a01:.8e}", f"{a02:.8e}",
            f"{a10:.8e}", f"{a11:.8e}", f"{a12:.8e}",
            f"{a20:.8e}", f"{a21:.8e}", f"{a22:.8e}",
            f"{sh_xx:.8e}", f"{sh_yy:.8e}", f"{sh_zz:.8e}",
            f"{sh_xy:.8e}", f"{sh_xz:.8e}", f"{sh_yz:.8e}",
        ]
        fout.write(",".join(vals) + "\n")

        if (i + 1) % 20 == 0 or i == n_frames - 1:
            print(f"  wrote frame {i+1}/{n_frames}")

    fout.close()
    print(f"✅ Done — output written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute polarizabilities from one or more ASE trajectories"
    )
    parser.add_argument("traj_in", help="Input trajectory file OR directory containing *nve.traj files")
    parser.add_argument("--csv_out", help="Output CSV file (used only for single-file mode)")
    parser.add_argument("--dt", type=float, default=0.5, help="Timestep in fs")
    #parser.add_argument("--model", default="mace_mu_alpha_water_correct.model",
    #                    help="Path to MACE model file")
    parser.add_argument("--continue_run", action="store_true",
                        help="Continue from existing CSV instead of starting fresh")
    parser.add_argument("--steps", type=int, default=2,
                        help="Process every n-th frame (default: 2)")
    args = parser.parse_args()

    # Load MACE calculator once (reused for all trajs)
    polar_calc = MACECalculator(
        model_paths="./MACE_SPICE_full_mu_alpha_early_seems_great.model",
        model_type="DipolePolarizabilityMACE",
        device="cuda",
        default_dtype="float64"
    )

    # Handle directory or single file
    if os.path.isdir(args.traj_in):
        traj_files = sorted(glob.glob(os.path.join(args.traj_in, "*nve.traj")))
        if not traj_files:
            print(f"No *nve.traj files found in {args.traj_in}")
            return

        print(f"Found {len(traj_files)} trajectory files in directory {args.traj_in}")
        print(traj_files)
        for traj_path in traj_files:
            #if not "sulind" in traj_path and not  "tolfe" in traj_path:
            #    continue
            base = os.path.splitext(os.path.basename(traj_path))[0]
            csv_path = os.path.join(args.traj_in, f"{base}_polar.csv")
            print(f"\n=== Processing {traj_path} → {csv_path} ===")
            process_traj(traj_path, csv_path, polar_calc, args.dt, args.continue_run, args.steps)
    else:
        if not args.csv_out:
            raise ValueError("Please specify --csv_out when providing a single trajectory file.")
        process_traj(args.traj_in, args.csv_out, polar_calc, args.dt, args.continue_run, args.steps)


if __name__ == "__main__":
    main()
