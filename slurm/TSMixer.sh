#!/bin/bash
#SBATCH --job-name="TSMixer"
#SBATCH --output=outputs/TSMixer%J.out
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
  run.py TSMixer_CZ -n TSMixer_Slovakia-in-Czechia -c /project/p1364-25-2/checkpoints/PV-PIML/TSMixer_Slovakia_final/epoch=98-step=82269.ckpt -t
