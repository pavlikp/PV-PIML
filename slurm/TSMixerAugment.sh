#!/bin/bash
#SBATCH --job-name="Augment"
#SBATCH --output=outputs/Augment%J.out
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=62G

#SBATCH --exclude=n142

#SBATCH --account=p1364-25-2

cd ..
srun /home/ppavlik/miniconda3/envs/pv/bin/python \
  run.py TSMixerADRAugment -n TSMixerAugment9 -s 9
