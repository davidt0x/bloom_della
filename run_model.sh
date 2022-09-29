#!/bin/bash
#SBATCH --time 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --out=logs/%j.out

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1

module load anaconda3/2021.11
. activate bloom
python embed.py

