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
  run.py ADR -n ADR_Global -c /home/ppavlik/repos/PV-PIML/checkpoints/PV-PIML/ADR_GlobalLR+Bounded/epoch=23-step=24000.ckpt -t
