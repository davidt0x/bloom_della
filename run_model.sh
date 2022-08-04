#!/bin/bash
#SBATCH --time 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:2
#SBATCH --out=logs/%j.out

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1

module load anaconda3/2021.11
. activate bloom
python generate.py

