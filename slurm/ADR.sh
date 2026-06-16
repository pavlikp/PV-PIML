#!/bin/bash
#SBATCH --job-name="PV_ADR"
#SBATCH --output=outputs/PV_ADR%J.out
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=62G

#SBATCH --account=p1364-25-2

cd ..
srun /home/ppavlik/miniconda3/envs/pv/bin/python \
  run.py ADR10 -n ADR_Hybrid_1/10
