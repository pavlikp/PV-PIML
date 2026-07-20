#!/bin/bash
#SBATCH --job-name="ADR"
#SBATCH --output=outputs/ADR%J.out
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=62G

#SBATCH --exclude=n142,n143

#SBATCH --account=p1364-25-2

cd ..
srun /home/ppavlik/miniconda3/envs/pv/bin/python \
  run.py ADR10000 -n ADR_1/10000f