#!/bin/bash
#SBATCH --job-name=mace_off_opt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=optimize_large_%j.out
#SBATCH --error=optimize_large_%j.err

source ~/.bashrc
conda activate mace  # <-- replace with your environment name

cd /work/Benchmark-MACE-MDP/IR-R-7193/mace_off

python optimize_large.py
